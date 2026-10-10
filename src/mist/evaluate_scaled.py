"""Validation calibration and MSG/CANOPUS test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from mist.data.compact import CompactFeatureDataset, make_compact_loader
from mist.metrics import calibrate_threshold, metric_summary, tanimoto_scores
from mist.models.scaled_mist import ScaledMIST, count_parameters
from mist.train_scaled import _all_gather_rows, _distributed_context, _move_batch


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


def _load_metadata(dataset: CompactFeatureDataset) -> tuple[np.ndarray, np.ndarray]:
    name_parts = []
    structure_parts = []
    for part_path, index_path in zip(
        dataset.part_paths, dataset.part_index_paths, strict=True
    ):
        name_path = part_path / "names.npy"
        structure_path = part_path / "structure_ids.npy"
        if not name_path.exists() or not structure_path.exists():
            raise FileNotFoundError(
                f"real-spectrum evaluation metadata is missing in {part_path}"
            )
        names = np.load(name_path, mmap_mode="r")
        structures = np.load(structure_path, mmap_mode="r")
        if index_path is not None:
            indices = np.load(index_path, mmap_mode="r")
            names = names[indices]
            structures = structures[indices]
        name_parts.append(np.asarray(names))
        structure_parts.append(np.asarray(structures))
    names = np.concatenate(name_parts).astype(str)
    structures = np.concatenate(structure_parts).astype(str)
    if len(names) != len(dataset) or len(structures) != len(dataset):
        raise ValueError("evaluation metadata row count mismatch")
    return names, structures


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model.eval()
    probability_parts = []
    target_parts = []
    index_parts = []
    for batch in loader:
        batch = _move_batch(batch, device)
        probabilities, _ = model(batch)
        probability_parts.append(probabilities.float().cpu())
        target_parts.append(batch["targets"].to(torch.uint8).cpu())
        index_parts.append(batch["row_indices"].cpu())
    dimension = model.output_size
    local_probabilities = (torch.cat(probability_parts) if probability_parts else torch.empty((0, dimension))).to(device)
    local_targets = (torch.cat(target_parts) if target_parts else torch.empty((0, dimension), dtype=torch.uint8)).to(device)
    local_indices = (torch.cat(index_parts) if index_parts else torch.empty(0, dtype=torch.int64)).to(device)
    probabilities = _all_gather_rows(local_probabilities, rank, world_size)
    targets = _all_gather_rows(local_targets, rank, world_size)
    indices = _all_gather_rows(local_indices, rank, world_size)
    if rank != 0:
        return None
    assert probabilities is not None and targets is not None and indices is not None
    order = torch.argsort(indices)
    sorted_indices = indices[order].numpy()
    if not np.array_equal(sorted_indices, np.arange(len(sorted_indices))):
        raise RuntimeError("distributed evaluation rows are missing or duplicated")
    return probabilities[order].numpy(), targets[order].numpy(), sorted_indices


def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = args.checkpoint.resolve()
    cache_path = args.cache.resolve()
    output_dir = args.output.resolve()
    rank, world_size, _local_rank, device = _distributed_context(getattr(args, "device", "auto"))
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = CompactFeatureDataset(cache_path)
    if dataset.manifest.get("kind") not in {"msg_real", "real_spectrum"}:
        raise ValueError("scaled evaluation accepts only a real-spectrum cache")
    split = dataset.manifest.get("split")
    if split not in {"val", "test"}:
        raise ValueError(f"evaluation cache must be val or test, got {split!r}")
    if split == "test" and not args.allow_test:
        raise PermissionError("locked test evaluation requires explicit --allow-test")
    if split == "val" and args.allow_test:
        raise ValueError("--allow-test was supplied for a validation cache")
    if split == "test" and args.threshold_file is None:
        raise ValueError("test evaluation requires --threshold-file from validation")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {summary_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_provenance = checkpoint.get("data_provenance", {}).get("target_val", {})
    target_dataset_name = target_provenance.get("dataset_name") or checkpoint.get("config", {}).get("data", {}).get("dataset_name")
    if target_dataset_name is not None and dataset.manifest.get("dataset_name") != target_dataset_name:
        raise ValueError(
            "evaluation dataset differs from the checkpoint's target dataset: "
            f"expected={target_dataset_name!r}, "
            f"got={dataset.manifest.get('dataset_name')!r}"
        )
    if (
        split == "val"
        and target_provenance.get("manifest_sha256") is not None
        and target_provenance["manifest_sha256"] != _sha256(cache_path / "manifest.json")
    ):
        raise ValueError("validation cache differs from the checkpoint's target validation cache")
    config = checkpoint["config"]
    model = ScaledMIST(**config["model"])
    state_name = "model" if args.raw_model else "ema_model"
    if state_name not in checkpoint:
        raise ValueError(f"checkpoint does not contain requested state {state_name!r}")
    model.load_state_dict(checkpoint[state_name], strict=True)
    model.to(device).eval()
    loader, sampler = make_compact_loader(
        dataset,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
        seed=0,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        drop_last=False,
        even_divisible=False,
    )
    sampler.set_epoch(0)
    started = time.time()
    predicted = _predict(
        model, loader, device=device, rank=rank, world_size=world_size
    )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert predicted is not None
        probabilities, targets, _ = predicted
        checkpoint_sha256 = _sha256(checkpoint_path)
        cache_manifest = cache_path / "manifest.json"
        common: dict[str, Any] = {
            "format_version": 1,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_state": state_name,
            "cache": str(cache_path),
            "cache_manifest_sha256": _sha256(cache_manifest),
            "parameter_count": count_parameters(model),
            "dataset_name": dataset.manifest.get("dataset_name"),
            "world_size": world_size,
            "elapsed_seconds": time.time() - started,
        }
        if split == "val":
            threshold, calibration = calibrate_threshold(probabilities, targets)
            scores = tanimoto_scores(probabilities, targets, threshold)
            names, structures = _load_metadata(dataset)
            predicted_bits = probabilities >= threshold
            per_spectrum = pd.DataFrame(
                {
                    "spec": names,
                    "structure_id": structures,
                    "tanimoto": scores,
                    "predicted_on_bits": predicted_bits.sum(axis=1),
                    "true_on_bits": targets.sum(axis=1),
                }
            )
            per_spectrum.to_csv(
                output_dir / "per_spectrum.tsv", sep="\t", index=False
            )
            per_compound = (
                per_spectrum.groupby("structure_id", sort=True)
                .agg(
                    n_spectra=("spec", "size"),
                    mean_tanimoto=("tanimoto", "mean"),
                    median_tanimoto=("tanimoto", "median"),
                )
                .reset_index()
            )
            per_compound.to_csv(
                output_dir / "per_underlying_compound.tsv", sep="\t", index=False
            )
            result = {
                **common,
                "threshold_source_split": "val",
                "threshold": threshold,
                "calibration": calibration,
                "spectrum_metrics": metric_summary(scores),
                "underlying_compound_metrics": metric_summary(
                    per_compound["mean_tanimoto"].to_numpy()
                ),
                "n_underlying_compounds": len(per_compound),
            }
            threshold_record = {
                "format_version": 1,
                "threshold_source_split": "val",
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_state": state_name,
                "validation_dataset_name": dataset.manifest.get("dataset_name"),
                "validation_cache_manifest_sha256": _sha256(cache_manifest),
                "threshold": threshold,
                "calibration": calibration,
            }
            _json_atomic(output_dir / "threshold.json", threshold_record)
        else:
            assert args.threshold_file is not None
            threshold_record = json.loads(args.threshold_file.read_text())
            if threshold_record.get("threshold_source_split") != "val":
                raise ValueError("threshold must have been calibrated on validation")
            if threshold_record.get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError("threshold and evaluated checkpoint hashes differ")
            if threshold_record.get("checkpoint_state") != state_name:
                raise ValueError("threshold and evaluated checkpoint states differ")
            if threshold_record.get("validation_dataset_name") != dataset.manifest.get(
                "dataset_name"
            ):
                raise ValueError("validation threshold and test dataset names differ")
            threshold = float(threshold_record["threshold"])
            scores = tanimoto_scores(probabilities, targets, threshold)
            names, structures = _load_metadata(dataset)
            predicted_bits = probabilities >= threshold
            per_spectrum = pd.DataFrame(
                {
                    "spec": names,
                    "structure_id": structures,
                    "tanimoto": scores,
                    "predicted_on_bits": predicted_bits.sum(axis=1),
                    "true_on_bits": targets.sum(axis=1),
                }
            )
            per_spectrum.to_csv(output_dir / "per_spectrum.tsv", sep="\t", index=False)
            per_compound = (
                per_spectrum.groupby("structure_id", sort=True)
                .agg(
                    n_spectra=("spec", "size"),
                    mean_tanimoto=("tanimoto", "mean"),
                    median_tanimoto=("tanimoto", "median"),
                )
                .reset_index()
            )
            per_compound.to_csv(
                output_dir / "per_underlying_compound.tsv", sep="\t", index=False
            )
            result = {
                **common,
                "threshold_source_split": "val",
                "threshold_file": str(args.threshold_file.resolve()),
                "threshold": threshold,
                "spectrum_metrics": metric_summary(scores),
                "underlying_compound_metrics": metric_summary(
                    per_compound["mean_tanimoto"].to_numpy()
                ),
                "n_underlying_compounds": len(per_compound),
            }
        _json_atomic(summary_path, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--raw-model", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    evaluate(get_parser().parse_args())


if __name__ == "__main__":
    main()
