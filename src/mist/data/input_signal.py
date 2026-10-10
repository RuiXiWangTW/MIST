"""Spectrum-only datasets and fingerprint prediction."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mist.data.compact import COMPACT_SCHEMA_VERSION
from mist.data.simulation_utils import sha256_file, write_json_atomic
from mist.models.scaled_mist import ScaledMIST, count_parameters


SPECTRUM_ARRAYS = (
    "form_vec.npy",
    "ion_vec.npy",
    "intens.npy",
    "peak_type.npy",
    "lengths.npy",
    "instruments.npy",
)


class SpectrumOnlyDataset(Dataset):
    """Read model inputs and names without opening target fingerprints."""

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root).resolve()
        self.manifest_path = self.cache_root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if int(self.manifest.get("schema_version", -1)) != COMPACT_SCHEMA_VERSION:
            raise ValueError("unsupported compact cache schema")
        entries = self.manifest.get("parts", [])
        if not entries:
            raise ValueError("spectrum cache contains no parts")
        self.part_paths = [
            (self.cache_root / entry["source"]).resolve()
            if entry.get("source") is not None
            else self.cache_root / entry["path"]
            for entry in entries
        ]
        self.index_paths = [
            (self.cache_root / entry["index"]).resolve()
            if entry.get("index") is not None
            else None
            for entry in entries
        ]
        self.part_sizes = [int(entry["rows"]) for entry in entries]
        self.offsets = np.concatenate(
            (
                np.asarray([0], dtype=np.int64),
                np.cumsum(self.part_sizes, dtype=np.int64),
            )
        )
        for path in self.part_paths:
            missing = [name for name in SPECTRUM_ARRAYS if not (path / name).is_file()]
            if missing:
                raise FileNotFoundError(f"spectrum part {path} is missing {missing}")
        self._arrays: dict[int, dict[str, np.ndarray]] = {}
        self._indices: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _resolve(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        part_index = bisect_right(self.offsets, index) - 1
        local = int(index - self.offsets[part_index])
        index_path = self.index_paths[part_index]
        if index_path is not None:
            values = self._indices.get(part_index)
            if values is None:
                values = np.load(index_path, mmap_mode="r")
                self._indices[part_index] = values
            local = int(values[local])
        return part_index, local

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        part_index, local = self._resolve(index)
        arrays = self._arrays.get(part_index)
        if arrays is None:
            arrays = {
                filename.removesuffix(".npy"): np.load(
                    self.part_paths[part_index] / filename, mmap_mode="r"
                )
                for filename in SPECTRUM_ARRAYS
            }
            self._arrays[part_index] = arrays
        return {
            **{name: values[local] for name, values in arrays.items()},
            "row_index": np.int64(index),
        }

    def names(self) -> np.ndarray:
        parts = []
        for path, index_path in zip(self.part_paths, self.index_paths, strict=True):
            values = np.load(path / "names.npy", mmap_mode="r")
            if index_path is not None:
                values = values[np.load(index_path, mmap_mode="r")]
            parts.append(np.asarray(values).astype(str))
        return np.concatenate(parts)


def collate_spectrum_only(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Collate model inputs while intentionally omitting target arrays."""

    return {
        "form_vec": torch.from_numpy(
            np.stack([row["form_vec"] for row in rows]).astype(np.float32)
        ),
        "ion_vec": torch.from_numpy(
            np.stack([row["ion_vec"] for row in rows]).astype(np.int64)
        ),
        "intens": torch.from_numpy(
            np.stack([row["intens"] for row in rows]).astype(np.float32)
        ),
        "types": torch.from_numpy(
            np.stack([row["peak_type"] for row in rows]).astype(np.int64)
        ),
        "num_peaks": torch.from_numpy(
            np.asarray([row["lengths"] for row in rows], dtype=np.int64)
        ),
        "instruments": torch.from_numpy(
            np.asarray([row["instruments"] for row in rows], dtype=np.int64)
        ),
        "row_indices": torch.from_numpy(
            np.asarray([row["row_index"] for row in rows], dtype=np.int64)
        ),
    }


def predict_input_signal(
    *,
    checkpoint_path: Path,
    cache_path: Path,
    output: Path,
    allow_test_input: bool,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict:
    """Predict fingerprints without loading target fingerprints or structures."""

    checkpoint_path = checkpoint_path.resolve()
    cache_path = cache_path.resolve()
    output = output.resolve()
    dataset = SpectrumOnlyDataset(cache_path)
    split = dataset.manifest.get("split")
    if split not in {"val", "test"}:
        raise ValueError("fingerprint prediction requires a validation or test cache")
    if (split == "test") != bool(allow_test_input):
        raise PermissionError("test input requires --allow-test-input")
    dataset_name = str(dataset.manifest.get("dataset_name"))
    if dataset.manifest.get("kind") != "real_spectrum" or dataset_name not in {
        "canopus_all",
        "msg_all",
    }:
        raise ValueError("prediction requires canonical MSG or CANOPUS real spectra")
    names = dataset.names()
    if len(set(names)) != len(names):
        raise ValueError("spectrum cache contains duplicate spectrum names")
    checkpoint_sha = sha256_file(checkpoint_path)
    cache_sha = sha256_file(cache_path / "manifest.json")
    existing = output / "manifest.json"
    if existing.is_file():
        result = json.loads(existing.read_text())
        if (
            result.get("checkpoint_sha256") != checkpoint_sha
            or result.get("cache_manifest_sha256") != cache_sha
        ):
            raise ValueError("existing predictions have incompatible provenance")
        return result
    if output.exists():
        raise FileExistsError(f"incomplete prediction output exists: {output}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_dataset = checkpoint["config"].get("data", {}).get("dataset_name")
    if checkpoint_dataset is not None and checkpoint_dataset != dataset_name:
        raise ValueError("prediction dataset differs from the checkpoint dataset")
    model = ScaledMIST(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    device = torch.device(device_name)
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_spectrum_only,
        pin_memory=False,
        prefetch_factor=2 if num_workers else None,
        persistent_workers=bool(num_workers),
    )
    probabilities = np.empty((len(dataset), 4096), dtype=np.float32)
    seen = np.zeros(len(dataset), dtype=bool)
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            indices = batch.pop("row_indices").numpy()
            predicted, _ = model(
                {key: value.to(device) for key, value in batch.items()}
            )
            probabilities[indices] = predicted.float().cpu().numpy()
            seen[indices] = True
    if not np.all(seen) or np.any(~np.isfinite(probabilities)):
        raise RuntimeError("fingerprint prediction is incomplete or nonfinite")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        np.save(stage / "probabilities.npy", probabilities, allow_pickle=False)
        np.save(stage / "spectrum_names.npy", names, allow_pickle=False)
        result = {
            "schema_version": 1,
            "kind": "mist_fingerprint_predictions",
            "dataset_name": dataset_name,
            "target_split": split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": int(checkpoint["step"]),
            "checkpoint_state": "ema_model",
            "cache": str(cache_path),
            "cache_manifest_sha256": cache_sha,
            "input_contract_signature": cache_sha,
            "spectra": len(names),
            "fingerprint_dimension": 4096,
            "target_ground_truth_fingerprint_arrays_read": False,
            "target_structure_identifiers_read": False,
                        "parameter_count": count_parameters(model),
            "probabilities_sha256": sha256_file(stage / "probabilities.npy"),
            "spectrum_names_sha256": sha256_file(stage / "spectrum_names.npy"),
            "elapsed_seconds": time.time() - started,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        }
        write_json_atomic(stage / "manifest.json", result)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test-input", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = predict_input_signal(
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        output=args.output,
        allow_test_input=args.allow_test_input,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
