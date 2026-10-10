"""Materialize compact MIST features from one canonical ``*_all`` dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from rdkit import rdBase

from mist.data.atlas import morgan4096_packed
from mist.data.compact import COMPACT_SCHEMA_VERSION
from mist.data.data import Spectra
from mist.data.datasets import (
    SPLIT_NAMES,
    SUPPORTED_DATASETS,
    DatasetLayout,
    load_dataset,
)
from mist.data.featurizers import PeakFormula
from mist.data.simulation_utils import sha256_file, write_json_atomic


def _save(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)


def _build_split(
    *,
    output_root: Path,
    dataset_name: str,
    split_name: str,
    records,
    featurizer: PeakFormula,
    provenance: dict,
    max_peaks: int,
) -> dict:
    final = output_root / split_name
    source_contract = {
        "schema_version": 1,
        "source_type": "real",
        "dataset_name": dataset_name,
        "split": split_name,
        "conditioning": {"kind": "none"},
    }
    expected = {
        "dataset_name": dataset_name,
        "split": split_name,
        "source_provenance": provenance,
        "source_contract": source_contract,
        "max_peaks": max_peaks,
    }
    if final.exists():
        existing = json.loads((final / "manifest.json").read_text())
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"incompatible existing real cache {final}: {mismatches}")
        return existing
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{split_name}.", dir=final.parent))
    part = temporary / "part_00000"
    part.mkdir()
    rows = len(records)
    width = max_peaks + 1
    arrays = {
        "form_vec": np.lib.format.open_memmap(
            part / "form_vec.npy", mode="w+", dtype=np.int16, shape=(rows, width, 18)
        ),
        "ion_vec": np.lib.format.open_memmap(
            part / "ion_vec.npy", mode="w+", dtype=np.uint8, shape=(rows, width)
        ),
        "intens": np.lib.format.open_memmap(
            part / "intens.npy", mode="w+", dtype=np.float16, shape=(rows, width)
        ),
        "peak_type": np.lib.format.open_memmap(
            part / "peak_type.npy", mode="w+", dtype=np.uint8, shape=(rows, width)
        ),
        "lengths": np.lib.format.open_memmap(
            part / "lengths.npy", mode="w+", dtype=np.uint8, shape=(rows,)
        ),
        "instruments": np.lib.format.open_memmap(
            part / "instruments.npy", mode="w+", dtype=np.uint8, shape=(rows,)
        ),
        "fingerprints": np.lib.format.open_memmap(
            part / "fingerprints.npy", mode="w+", dtype=np.uint8, shape=(rows, 512)
        ),
    }
    names: list[str] = []
    structure_ids: list[str] = []
    fingerprint_cache: dict[str, np.ndarray] = {}
    started = time.time()
    try:
        for index, record in enumerate(records):
            spectrum = Spectra(
                spectra_name=record.name,
                spectra_formula=record.formula,
                instrument=record.instrument,
            )
            feature = featurizer.featurize(spectrum, train_mode=False)
            length = len(feature["peak_type"])
            if length < 1 or length > width:
                raise RuntimeError(
                    f"unexpected feature width {length} for {record.name}; max={width}"
                )
            arrays["form_vec"][index] = 0
            arrays["ion_vec"][index] = 0
            arrays["intens"][index] = 0
            arrays["peak_type"][index] = 0
            arrays["form_vec"][index, :length] = feature["form_vec"]
            arrays["ion_vec"][index, :length] = feature["ion_vec"]
            arrays["intens"][index, :length] = feature["frag_intens"]
            arrays["peak_type"][index, :length] = feature["peak_type"]
            arrays["lengths"][index] = length
            arrays["instruments"][index] = int(feature["instrument"])
            fingerprint = fingerprint_cache.get(record.smiles)
            if fingerprint is None:
                fingerprint = morgan4096_packed(record.smiles)
                fingerprint_cache[record.smiles] = fingerprint
            arrays["fingerprints"][index] = fingerprint
            names.append(record.name)
            structure_ids.append(record.inchikey.split("-")[0][:14])
            if (index + 1) % 10_000 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"{dataset_name}/{split_name}: {index + 1:,}/{rows:,} "
                    f"rows_per_second={(index + 1) / elapsed:.1f}",
                    flush=True,
                )
        for values in arrays.values():
            values.flush()
        arrays.clear()
        name_width = max(1, max(map(len, names)))
        _save(part / "names.npy", np.asarray(names, dtype=f"U{name_width}"))
        _save(part / "structure_ids.npy", np.asarray(structure_ids, dtype="U14"))
        part_manifest = {
            "schema_version": COMPACT_SCHEMA_VERSION,
            "kind": "real_spectrum",
            "dataset_name": dataset_name,
            "split": split_name,
            "source_contract": source_contract,
            "rows": rows,
        }
        write_json_atomic(part / "manifest.json", part_manifest)
        manifest = {
            "schema_version": COMPACT_SCHEMA_VERSION,
            "kind": "real_spectrum",
            "dataset_name": dataset_name,
            "split": split_name,
            "source_contract": source_contract,
            "max_peaks": max_peaks,
            "fingerprint": "Morgan radius=2 fpSize=4096 includeChirality=false",
            "parts": [{"path": "part_00000", "rows": rows}],
            "rows": rows,
            "unique_smiles_fingerprinted": len(fingerprint_cache),
            "source_provenance": provenance,
            "created_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        }
        write_json_atomic(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
        return manifest
    except BaseException:
        arrays.clear()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_all_cache(args: argparse.Namespace) -> list[dict]:
    layout = DatasetLayout.resolve(args.dataset_name, args.data_dir)
    loaded = load_dataset(layout)
    output = args.output.resolve()
    provenance = {
        "dataset_root": str(layout.root),
        "labels_sha256": sha256_file(layout.labels),
        "spectra_sha256": sha256_file(layout.spectra),
        "assignments_sha256": sha256_file(layout.subformulae),
        "split": str(layout.split),
        "split_sha256": sha256_file(layout.split),
    }
    featurizer = PeakFormula(
        dataset_root=str(layout.root),
        max_peaks=args.max_peaks,
        subformula_ppm=args.subformula_ppm,
    )
    results = []
    requested = SPLIT_NAMES if args.splits is None else tuple(args.splits)
    for split_name in requested:
        results.append(
            _build_split(
                output_root=output,
                dataset_name=args.dataset_name,
                split_name=split_name,
                records=loaded[split_name],
                featurizer=featurizer,
                provenance=provenance,
                max_peaks=args.max_peaks,
            )
        )
    return results


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name", choices=tuple(sorted(SUPPORTED_DATASETS)), required=True
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", choices=SPLIT_NAMES, nargs="+")
    parser.add_argument("--max-peaks", type=int, default=10)
    parser.add_argument("--subformula-ppm", type=float, default=20.0)
    return parser


def main() -> None:
    results = build_all_cache(get_parser().parse_args())
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
