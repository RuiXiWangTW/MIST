"""Native PyTorch DDP training for compact MSG and CANOPUS MIST models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from mist.data.augmentation import augment_peak_batch
from mist.data.compact import CompactFeatureDataset, make_compact_loader
from mist.data.training_sources import load_training_data_contract
from mist.metrics import (
    calibrate_threshold,
    fingerprint_loss,
    metric_summary,
    tanimoto_scores,
)
from mist.models.scaled_mist import ScaledMIST, count_parameters, load_mist_weights


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    required = {"seed", "data", "model", "loss", "optimizer", "train"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config is missing sections: {sorted(missing)}")
    return config


def _distributed_context(device_name: str = "auto") -> tuple[int, int, int, torch.device]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if device_name == "cuda" else torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    return rank, world_size, local_rank, device


def _seed_everything(seed: int, rank: int) -> None:
    rank_seed = int(seed) + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)


def _cache_provenance(dataset: CompactFeatureDataset) -> dict[str, Any]:
    manifest_path = dataset.cache_root / "manifest.json"
    output = {
        "path": str(dataset.cache_root),
        "rows": len(dataset),
        "manifest_sha256": _sha256(manifest_path),
        "kind": dataset.manifest.get("kind"),
        "dataset_name": dataset.manifest.get("dataset_name"),
        "split": dataset.manifest.get("split"),
    }
    selection = dataset.manifest.get("selection_manifest")
    if selection:
        selection_path = dataset.cache_root / selection
        output["selection_manifest_sha256"] = _sha256(selection_path)
    return output


class InfiniteLoader:
    """Endless deterministic source stream with a resumable batch cursor."""

    def __init__(self, loader, sampler):
        self.loader = loader
        self.sampler = sampler
        self.epoch = 0
        self.position = 0
        self.iterator = iter(loader)

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        if self.epoch < 0 or not 0 <= self.position <= len(self.sampler):
            raise ValueError("invalid training-source cursor")
        self.sampler.set_epoch(self.epoch)
        self.sampler.start_batch = self.position
        self.iterator = iter(self.loader)

    def next(self) -> dict[str, torch.Tensor]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.position = 0
            self.sampler.set_epoch(self.epoch)
            self.sampler.start_batch = 0
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        self.position += 1
        return batch


def _runtime_states(loaders, device, world_size):
    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "sources": [loader.state_dict() for loader in loaders],
    }
    if world_size == 1:
        return [state]
    states = [None] * world_size
    dist.all_gather_object(states, state)
    return states


def _restore_runtime(state, loaders, device):
    if len(state["sources"]) != len(loaders):
        raise ValueError("resume source count differs")
    for loader, cursor in zip(loaders, state["sources"]):
        loader.load_state_dict(cursor)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        if state["cuda_rng"] is None:
            raise ValueError("resume checkpoint has no CUDA random state")
        torch.cuda.set_rng_state(state["cuda_rng"], device)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in batch.items()
    }


def _cosine_learning_rate_factor(
    step: int, *, warmup_steps: int, total_steps: int
) -> float:
    """Return the schedule factor independently of an early frozen stop."""

    if total_steps < 1:
        raise ValueError("learning-rate schedule total_steps must be positive")
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _optimizer_class(name: str) -> type[torch.optim.Optimizer]:
    """Resolve an explicitly preregistered optimizer without silent fallback."""

    normalized = str(name).lower()
    optimizer_types = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    if normalized not in optimizer_types:
        raise ValueError("optimizer.name must be adam or adamw")
    return optimizer_types[normalized]


def _merge_batches(*batches: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not batches:
        raise ValueError("at least one training-source batch is required")
    keys = batches[0].keys()
    if any(batch.keys() != keys for batch in batches[1:]):
        raise ValueError("training-source batches have different feature keys")
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        value = model_state[name].detach()
        if ema_value.is_floating_point():
            ema_value.lerp_(value, 1.0 - decay)
        else:
            ema_value.copy_(value)


def _all_gather_rows(
    local: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor | None:
    if world_size == 1:
        return local.cpu()
    local_size = torch.tensor([len(local)], dtype=torch.int64, device=local.device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(sizes, local_size)
    row_counts = [int(value.item()) for value in sizes]
    max_rows = max(row_counts)
    padded = torch.zeros(
        (max_rows, *local.shape[1:]), dtype=local.dtype, device=local.device
    )
    padded[: len(local)] = local
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    if rank != 0:
        return None
    return torch.cat(
        [value[:rows].cpu() for value, rows in zip(gathered, row_counts)], dim=0
    )


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    loader,
    sampler,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[float, dict[str, Any] | None]:
    model.eval()
    sampler.set_epoch(0)
    probability_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for batch in loader:
        batch = _move_batch(batch, device)
        # Validation stays FP32 so scalar threshold calibration is not degraded
        # by BF16 probability ties.
        probabilities, _ = model(batch)
        probability_parts.append(probabilities.float().cpu())
        target_parts.append(batch["targets"].to(dtype=torch.uint8).cpu())
    dimension = model.output_size
    local_probabilities = (torch.cat(probability_parts) if probability_parts else torch.empty((0, dimension))).to(device)
    local_targets = (torch.cat(target_parts) if target_parts else torch.empty((0, dimension), dtype=torch.uint8)).to(device)
    probabilities = _all_gather_rows(local_probabilities, rank, world_size)
    targets = _all_gather_rows(local_targets, rank, world_size)

    result = None
    score = 0.0
    if rank == 0:
        assert probabilities is not None and targets is not None
        probabilities_np = probabilities.numpy()
        targets_np = targets.numpy()
        threshold, calibration = calibrate_threshold(probabilities_np, targets_np)
        scores = tanimoto_scores(probabilities_np, targets_np, threshold)
        summary = metric_summary(scores)
        score = float(summary["mean_tanimoto"])
        result = {"calibration": calibration, "spectrum_metrics": summary}
    score_tensor = torch.tensor([score], device=device)
    if world_size > 1:
        dist.broadcast(score_tensor, src=0)
    return float(score_tensor.item()), result


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict[str, Any],
    provenance: dict[str, Any],
    best_validation: float,
    runtime_states: list,
) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "format_version": 2,
            "runtime_states": runtime_states,
            "step": step,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "data_provenance": provenance,
            "best_validation_tanimoto": best_validation,
        },
        temporary,
    )
    os.replace(temporary, path)


def _save_evaluation_checkpoint(
    path: Path,
    *,
    step: int,
    ema_model: torch.nn.Module,
    config: dict[str, Any],
    provenance: dict[str, Any],
    best_validation: float,
) -> None:
    """Save the minimal EMA-only state needed for post-hoc val/test curves."""

    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "format_version": 2,
            "checkpoint_kind": "ema_evaluation_snapshot",
            "step": step,
            "ema_model": ema_model.state_dict(),
            "config": config,
            "data_provenance": provenance,
            "best_validation_tanimoto": best_validation,
        },
        temporary,
    )
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    resume_path = getattr(args, "resume", None)
    command_initialize_path = getattr(args, "initialize_from", None)
    configured_initialize_path = config["train"].get("initialize_from")
    if resume_path is not None and command_initialize_path is not None:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    initialize_path = (
        command_initialize_path
        if command_initialize_path is not None
        else None if resume_path is not None else configured_initialize_path
    )
    initialize_state = str(
        getattr(args, "initialize_state", None)
        or config["train"].get("initialize_state", "ema_model")
    )
    if initialize_state not in {"model", "ema_model"}:
        raise ValueError("initialization state must be model or ema_model")
    rank, world_size, local_rank, device = _distributed_context(getattr(args, "device", "auto"))
    _seed_everything(int(config["seed"]), rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_config = config["data"]
    workers = int(data_config.get("num_workers", 0))
    prefetch = int(data_config.get("prefetch_factor", 2))
    data_contract = load_training_data_contract(data_config)
    training_source_definitions = [
        {
            "name": source.name,
            "source_type": source.source_type,
            "dataset": source.dataset,
            "batch_size": source.batch_size,
            "seed_offset": source.seed_offset,
            "combine_part_tails": source.combine_part_tails,
        }
        for source in data_contract.training_sources
    ]
    target_val = data_contract.validation_source.dataset
    if len(target_val) == 0:
        raise ValueError("validation cache must contain at least one spectrum")
    validation_batch_size = data_contract.validation_source.batch_size
    batch_size = data_contract.per_device_batch_size
    observed_real_fraction = data_contract.real_fraction
    target_dataset_name = data_contract.dataset_name

    training_loaders = []
    training_samplers = []
    for source in training_source_definitions:
        loader, sampler = make_compact_loader(
            source["dataset"],
            batch_size=int(source["batch_size"]),
            rank=rank,
            world_size=world_size,
            seed=int(config["seed"]) + int(source["seed_offset"]),
            num_workers=workers,
            prefetch_factor=prefetch,
            drop_last=True,
            even_divisible=True,
            combine_part_tails=bool(source["combine_part_tails"]),
        )
        training_loaders.append(loader)
        training_samplers.append(sampler)
    val_loader, val_sampler = make_compact_loader(
        target_val,
        batch_size=validation_batch_size,
        rank=rank,
        world_size=world_size,
        seed=0,
        num_workers=max(0, workers // 2),
        prefetch_factor=prefetch,
        drop_last=False,
        even_divisible=False,
    )
    if any(len(sampler) == 0 for sampler in training_samplers):
        raise ValueError(
            "one configured cache produces zero batches; enable synthetic tail "
            "combining or lower the affected batch size"
        )

    source_provenance = []
    for source in training_source_definitions:
        cache = _cache_provenance(source["dataset"])
        source_record = {
            "name": source["name"],
            "source_type": source["source_type"],
            "batch_size": int(source["batch_size"]),
            "batch_fraction": int(source["batch_size"]) / batch_size,
            "seed_offset": int(source["seed_offset"]),
            "combine_part_tails": bool(source["combine_part_tails"]),
            "cache": cache,
        }
        source_provenance.append(source_record)
    provenance = {
        "dataset_name": target_dataset_name,
        "training_sources": source_provenance,
        "target_val": _cache_provenance(target_val),
        "test_cache_present": False,
    }

    train_config = config["train"]
    accumulation = int(train_config.get("gradient_accumulation", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    # PyTorch DDP's static-graph reducer cannot safely discover its graph when
    # the first backward pass is deliberately hidden inside ``no_sync``.  That
    # is exactly how gradient accumulation starts and, on torch 2.7, trips the
    # reducer's ``expect_autograd_hooks_`` internal assertion.  Keep the faster
    # static reducer for the ordinary one-microbatch path and use the regular
    # reducer whenever accumulation requires ``no_sync``.
    ddp_static_graph = bool(
        train_config.get("ddp_static_graph", accumulation == 1)
    )
    if world_size > 1 and accumulation > 1 and ddp_static_graph:
        raise ValueError(
            "train.ddp_static_graph=true is incompatible with distributed "
            "gradient accumulation; set it to false"
        )
    # The legacy FormulaTransformer intentionally retains an instrument
    # embedding parameter even when ``embed_instrument`` is disabled.  The
    # static reducer tolerates that fixed unused parameter set; the regular
    # reducer used for accumulation must discover it explicitly.
    ddp_find_unused_parameters = bool(
        train_config.get("ddp_find_unused_parameters", not ddp_static_graph)
    )
    expected_world_size = train_config.get("expected_world_size")
    if expected_world_size is not None and int(expected_world_size) != world_size:
        raise ValueError(
            f"config requires world_size={int(expected_world_size)}, got {world_size}"
        )
    loss_config = config["loss"]
    raw_model = ScaledMIST(**config["model"]).to(device)
    expected_parameter_count = train_config.get("expected_parameter_count")
    actual_parameter_count = count_parameters(raw_model)
    if (
        expected_parameter_count is not None
        and int(expected_parameter_count) != actual_parameter_count
    ):
        raise ValueError(
            "config/model parameter-count mismatch: "
            f"expected={int(expected_parameter_count):,}, "
            f"actual={actual_parameter_count:,}"
        )
    ema_model = deepcopy(raw_model).eval().requires_grad_(False)
    optimizer_config = config["optimizer"]
    optimizer_type = _optimizer_class(optimizer_config.get("name", "adamw"))
    optimizer = optimizer_type(
        raw_model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        # Preserve the existing scaled-training default while allowing a
        # preregistered unfused Adam control to match the native MIST family.
        fused=bool(optimizer_config.get("fused", True)) and device.type == "cuda",
    )
    max_steps = int(train_config["max_steps"])
    stop_after = getattr(args, "stop_after", None)
    end_step = min(max_steps, int(stop_after)) if stop_after is not None else max_steps
    if end_step < 1:
        raise ValueError("stop-after must be positive")
    lr_schedule_total_steps = int(
        train_config.get("lr_schedule_total_steps", max_steps)
    )
    if lr_schedule_total_steps < max_steps:
        raise ValueError("train.lr_schedule_total_steps cannot be below max_steps")
    warmup_steps = int(optimizer_config["warmup_steps"])

    def learning_rate_factor(step: int) -> float:
        return _cosine_learning_rate_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=lr_schedule_total_steps,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    start_step = 0
    best_validation = -math.inf
    initialization = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_kind") == "ema_evaluation_snapshot":
            raise ValueError("EMA evaluation snapshots cannot resume training")
        if checkpoint.get("config") != config:
            raise ValueError("resume checkpoint config differs from the requested config")
        checkpoint_provenance = checkpoint.get("data_provenance")
        if not isinstance(checkpoint_provenance, dict):
            raise ValueError("resume checkpoint lacks data provenance")
        base_checkpoint_provenance = {
            key: value
            for key, value in checkpoint_provenance.items()
            if key != "initialization"
        }
        if base_checkpoint_provenance != provenance:
            raise ValueError(
                "resume checkpoint data provenance differs from the current run"
            )
        if checkpoint_provenance.get("initialization") is not None:
            initialization = checkpoint_provenance["initialization"]
            provenance["initialization"] = initialization
        load_mist_weights(raw_model, checkpoint["model"])
        load_mist_weights(ema_model, checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        best_validation = float(checkpoint["best_validation_tanimoto"])
    elif initialize_path is not None:
        initialize_path = Path(initialize_path).resolve()
        checkpoint = torch.load(
            initialize_path, map_location="cpu", weights_only=False
        )
        parent_config = checkpoint.get("config")
        if not isinstance(parent_config, dict) or parent_config.get("model") != config.get(
            "model"
        ):
            raise ValueError("initialization checkpoint model architecture differs")
        parent_target = checkpoint.get("data_provenance", {}).get("target_val", {})
        if parent_target.get("dataset_name") != target_val.manifest.get("dataset_name"):
            raise ValueError("initialization checkpoint targets a different dataset")
        if initialize_state not in checkpoint:
            raise ValueError(
                f"initialization checkpoint lacks state {initialize_state!r}"
            )
        parent_state = checkpoint[initialize_state]
        load_mist_weights(raw_model, parent_state)
        load_mist_weights(ema_model, parent_state)
        initialization = {
            "mode": "weights_only_optimizer_and_scheduler_reset",
            "checkpoint": str(initialize_path),
            "checkpoint_sha256": _sha256(initialize_path),
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_kind": checkpoint.get("checkpoint_kind", "full_training"),
            "checkpoint_state": initialize_state,
            "parent_target_dataset": parent_target.get("dataset_name"),
        }
        provenance["initialization"] = initialization

    train_forward: torch.nn.Module = raw_model
    if bool(train_config.get("compile", False)):
        train_forward = torch.compile(raw_model, dynamic=False)
    if world_size > 1:
        train_forward = DistributedDataParallel(
            train_forward,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=ddp_find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=ddp_static_graph,
        )
        # DDP broadcasts the trainable model from rank 0 during construction.
        # A newly created EMA predates that broadcast, so synchronize its
        # starting point explicitly. Resumed EMA states were loaded identically
        # from the shared checkpoint and must be preserved.
        if resume_path is None:
            ema_model.load_state_dict(raw_model.state_dict(), strict=True)

    output_dir = Path(args.output_dir or train_config["output_dir"]).resolve()
    snapshot = output_dir / "run_manifest.json"
    if snapshot.exists() and resume_path is None:
        raise FileExistsError(
            f"run already exists at {output_dir}; use --resume or a new --output-dir"
        )
    # ``world_size`` can exceed the number of local devices for a multi-node
    # run.  Record one name per global rank instead of indexing this node's
    # CUDA devices with global ranks.
    local_gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    if world_size > 1:
        gathered_gpu_names: list[str | None] = [None] * world_size
        dist.all_gather_object(gathered_gpu_names, local_gpu_name)
        if any(value is None for value in gathered_gpu_names):
            raise RuntimeError("failed to gather one GPU name per distributed rank")
        gpu_names = [str(value) for value in gathered_gpu_names]
    else:
        gpu_names = [local_gpu_name]
    if resume_path is not None:
        if not snapshot.is_file():
            raise FileNotFoundError(f"resume run manifest is missing: {snapshot}")
        existing_run = json.loads(snapshot.read_text())
        resume_contract = {
            "config_sha256": _sha256(config_path),
            "data_provenance": provenance,
            "parameter_count": actual_parameter_count,
            "world_size": world_size,
            "gpu_names": gpu_names,
        }
        mismatched = [
            key
            for key, expected in resume_contract.items()
            if existing_run.get(key) != expected
        ]
        if mismatched:
            raise ValueError(
                "resume run manifest contract changed for: " + ", ".join(mismatched)
            )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_manifest = {
            "config": config,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "data_provenance": provenance,
            "parameter_count": count_parameters(raw_model),
            "world_size": world_size,
            "per_device_batch_size": batch_size,
            "micro_global_batch_size": batch_size * world_size,
            "gradient_accumulation": accumulation,
            "ddp_static_graph": ddp_static_graph if world_size > 1 else None,
            "ddp_find_unused_parameters": (
                ddp_find_unused_parameters if world_size > 1 else None
            ),
            "global_batch_size": batch_size * world_size * accumulation,
            "real_fraction": observed_real_fraction,
            "initialization": initialization,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_names": gpu_names,
            "created_at_unix": time.time(),
        }
        if not snapshot.exists():
            _json_atomic(snapshot, run_manifest)
    if world_size > 1:
        dist.barrier()

    training_infinite = [
        InfiniteLoader(loader, sampler)
        for loader, sampler in zip(training_loaders, training_samplers)
    ]
    if resume_path is not None:
        states = checkpoint.get("runtime_states")
        if not isinstance(states, list) or len(states) != world_size:
            raise ValueError("exact resume requires per-rank random states and source cursors; use a new run for older checkpoints")
        _restore_runtime(states[rank], training_infinite, device)
    augmentation = config.get("augmentation", {})
    precision = str(train_config.get("precision", "bf16"))
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    autocast_dtype = torch.bfloat16 if precision == "bf16" else None
    ema_decay = float(train_config.get("ema_decay", 0.999))
    validate_every = int(train_config["validate_every"])
    checkpoint_every = int(train_config["checkpoint_every"])
    log_every = int(train_config.get("log_every", 20))
    metrics_path = output_dir / "metrics.jsonl"
    started = time.time()
    cumulative_sample_exposures = start_step * batch_size * world_size * accumulation
    initial_sample_exposures = cumulative_sample_exposures
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, end_step + 1):
        train_forward.train()
        step_metrics = None
        for micro_step in range(accumulation):
            batch_parts = [loader.next() for loader in training_infinite]
            batch = _merge_batches(*batch_parts)
            batch = _move_batch(batch, device)
            batch = augment_peak_batch(
                batch,
                spectrum_probability=float(augmentation.get("spectrum_probability", 0.0)),
                peak_drop_probability=float(augmentation.get("peak_drop_probability", 0.0)),
                intensity_jitter_std=float(augmentation.get("intensity_jitter_std", 0.0)),
            )
            sync_context = (
                train_forward.no_sync()
                if world_size > 1 and micro_step < accumulation - 1
                else nullcontext()
            )
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else nullcontext()
            )
            with sync_context, autocast_context:
                predictions, auxiliary = train_forward(batch)
                loss, step_metrics = fingerprint_loss(
                    predictions,
                    batch["targets"],
                    auxiliary["intermediate_predictions"],
                    cosine_weight=float(loss_config["cosine_weight"]),
                    tanimoto_weight=float(loss_config["tanimoto_weight"]),
                    iterative_weight=float(loss_config["iterative_weight"]),
                )
                loss = loss / accumulation
            loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), float(train_config["gradient_clip_norm"])
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        _update_ema(ema_model, raw_model, ema_decay)
        cumulative_sample_exposures += batch_size * world_size * accumulation

        if step % log_every == 0 or step == 1:
            assert step_metrics is not None
            metric_names = [
                "loss",
                "cosine_loss",
                "soft_tanimoto_loss",
                "iterative_loss",
            ]
            values = torch.stack(
                [step_metrics[name] for name in metric_names]
                + [gradient_norm.detach()]
            ).float()
            if world_size > 1:
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= world_size
            if rank == 0:
                elapsed = time.time() - started
                record = {
                    "event": "train",
                    "step": step,
                    "loss": float(values[0]),
                    "cosine_loss": float(values[1]),
                    "soft_tanimoto_loss": float(values[2]),
                    "iterative_loss": float(values[3]),
                    "gradient_norm": float(values[-1]),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "cumulative_sample_exposures": cumulative_sample_exposures,
                    "samples_per_second": (
                        (cumulative_sample_exposures - initial_sample_exposures)
                        / max(elapsed, 1e-9)
                    ),
                    "elapsed_seconds": elapsed,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)

        validation_result = None
        if step % validate_every == 0 or step == end_step:
            validation_score, validation_result = _validate(
                ema_model,
                val_loader,
                val_sampler,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            runtime_states = _runtime_states(training_infinite, device, world_size)
            is_best = validation_score > best_validation
            best_validation = max(best_validation, validation_score)
            if rank == 0:
                record = {
                    "event": "validation",
                    "step": step,
                    "is_best": is_best,
                    "best_mean_tanimoto": best_validation,
                    "elapsed_seconds": time.time() - started,
                    **(validation_result or {}),
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                print(json.dumps(record, sort_keys=True), flush=True)
                if is_best:
                    _save_checkpoint(
                        output_dir / "best.pt",
                        step=step,
                        model=raw_model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        provenance=provenance,
                        best_validation=best_validation,
                        runtime_states=runtime_states,
                    )
                retain_evaluation = bool(
                    train_config.get("retain_validation_checkpoints", False)
                ) or (
                    bool(train_config.get("retain_endpoint_checkpoint", False))
                    and step == end_step
                )
                if retain_evaluation:
                    _save_evaluation_checkpoint(
                        output_dir / f"eval_step_{step:08d}.pt",
                        step=step,
                        ema_model=ema_model,
                        config=config,
                        provenance=provenance,
                        best_validation=best_validation,
                    )

        if step % checkpoint_every == 0 or step == end_step:
            runtime_states = _runtime_states(training_infinite, device, world_size)
        if rank == 0 and (step % checkpoint_every == 0 or step == end_step):
            _save_checkpoint(
                output_dir / "last.pt",
                step=step,
                model=raw_model,
                ema_model=ema_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                provenance=provenance,
                best_validation=best_validation,
                runtime_states=runtime_states,
            )
        if world_size > 1 and (
            step % validate_every == 0
            or step % checkpoint_every == 0
            or step == end_step
        ):
            dist.barrier()

    if rank == 0:
        print(
            f"training complete: steps={end_step} best_val_tanimoto={best_validation:.6f}",
            flush=True,
        )
    if world_size > 1:
        dist.destroy_process_group()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    start = parser.add_mutually_exclusive_group()
    start.add_argument("--resume", type=Path)
    start.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--initialize-state", choices=("model", "ema_model")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stop-after", type=int, help="Save and stop at this step without changing the learning-rate schedule")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    train(get_parser().parse_args())


if __name__ == "__main__":
    main()
