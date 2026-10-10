"""Export a self-contained compact cache, materializing any indexed views."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from mist.data.compact import CompactFeatureDataset, REQUIRED_ARRAYS
from mist.data.simulation_utils import sha256_file, write_json_atomic


def export_cache(source: Path, output: Path, chunk_rows: int = 8192) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    dataset = CompactFeatureDataset(source)
    manifest = dataset.manifest
    if manifest.get("kind") not in {
        "real_spectrum",
        "simulated_spectrum",
        "simulated_spectrum_view",
    }:
        raise ValueError("expected a real or synthetic training cache")
    source_type = "real" if manifest["kind"] == "real_spectrum" else "simulated"
    expected_contract = {
        "schema_version": 1,
        "source_type": source_type,
        "dataset_name": manifest["dataset_name"],
        "split": manifest["split"],
        "conditioning": {"kind": "none"},
    }
    if manifest.get("source_contract") != expected_contract:
        raise ValueError("unexpected training data contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        parts = []
        for number, (path, index_path, rows) in enumerate(
            zip(
                dataset.part_paths,
                dataset.part_index_paths,
                dataset.part_sizes,
                strict=True,
            )
        ):
            destination = stage / f"part_{number:05d}"
            destination.mkdir()
            indices = np.load(index_path, mmap_mode="r") if index_path else None
            filenames = list(REQUIRED_ARRAYS)
            for filename in ("names.npy", "structure_ids.npy"):
                if (path / filename).is_file():
                    filenames.append(filename)
            if source_type == "real" and not {"names.npy", "structure_ids.npy"} <= set(
                filenames
            ):
                raise ValueError(
                    "real release caches require spectrum names and structure identifiers"
                )
            for filename in filenames:
                values = np.load(path / filename, mmap_mode="r")
                if len(values) != dataset.part_source_sizes[number]:
                    raise ValueError(f"row count mismatch: {path / filename}")
                target = np.lib.format.open_memmap(
                    destination / filename,
                    mode="w+",
                    dtype=values.dtype,
                    shape=(rows, *values.shape[1:]),
                )
                for start in range(0, rows, chunk_rows):
                    end = min(rows, start + chunk_rows)
                    selected = (
                        indices[start:end] if indices is not None else slice(start, end)
                    )
                    target[start:end] = values[selected]
                target.flush()
                del target
            parts.append({"path": destination.name, "rows": rows})
        result = {
            "schema_version": 1,
            "kind": f"{source_type if source_type == 'real' else 'simulated'}_spectrum",
            "dataset_name": manifest["dataset_name"],
            "split": manifest["split"],
            "max_peaks": manifest["max_peaks"],
            "fingerprint": manifest["fingerprint"],
            "source_contract": expected_contract,
            "parts": parts,
            "rows": len(dataset),
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
        }
        write_json_atomic(stage / "manifest.json", result)
        CompactFeatureDataset(stage)
        os.rename(stage, output)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    print(json.dumps(export_cache(args.source, args.output, args.chunk_rows), indent=2))


if __name__ == "__main__":
    main()
