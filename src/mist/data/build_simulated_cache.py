"""Materialize compact MIST features from fixed GLACIER synthetic spectra."""

from __future__ import annotations

from collections import OrderedDict
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import h5py
import numpy as np

from mist.data.atlas import (
    ROOT_ION_NAMES,
    decode_packed_strings,
    morgan4096_packed,
    recover_spectrum,
)
from mist.data.compact import COMPACT_SCHEMA_VERSION
from mist.data.simulation_utils import (
    SimulationPlanError,
    load_plan_manifest,
    normalize_glacier_smiles,
    stable_json_hash,
    write_json_atomic,
)
from mist.utils.chem_utils import get_instr_idx


_CONTEXT: dict[str, Any] = {}
_TABLE_CACHE: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
    OrderedDict()
)


def _worker_init(context: dict[str, Any]) -> None:
    global _CONTEXT
    _CONTEXT = context


def _fingerprint_unique(index: int) -> tuple[int, np.ndarray, str | None]:
    try:
        return index, morgan4096_packed(_CONTEXT["unique_smiles"][index]), None
    except Exception as error:
        return index, np.empty(0, dtype=np.uint8), f"{type(error).__name__}: {error}"


def _recover_row(index: int) -> tuple[int, float, int, str | None]:
    context = _CONTEXT
    try:
        unique_index = int(context["inverse"][index])
        count = int(context["peak_counts"][index])
        recovered = recover_spectrum(
            smiles=context["unique_smiles"][unique_index],
            root_formula=context["unique_formulas"][unique_index],
            root_ion_index=int(context["root_ions"][index]),
            peak_mz=context["masses"][index, :count],
            intensities=context["intensities"][index, :count],
            mask_words=context["masks"][index, :count],
            max_peaks=int(context["max_peaks"]),
            mass_tolerance_da=float(context["mass_tolerance_da"]),
        )
        arrays = context["outputs"]
        arrays["form_vec"][index] = recovered.form_vec
        arrays["ion_vec"][index] = recovered.ion_vec
        arrays["intens"][index] = recovered.intens
        arrays["peak_type"][index] = recovered.peak_type
        arrays["lengths"][index] = recovered.length
        arrays["instruments"][index] = context["instruments"][index]
        return (
            index,
            recovered.max_mass_error,
            recovered.dropped_unrepresentable_peaks,
            None,
        )
    except Exception as error:
        return index, float("nan"), 0, f"{type(error).__name__}: {error}"


def _array_specs(rows: int, width: int) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    return {
        "form_vec.npy": ((rows, width, 18), np.dtype("<i2")),
        "ion_vec.npy": ((rows, width), np.dtype("u1")),
        "intens.npy": ((rows, width), np.dtype("<f2")),
        "peak_type.npy": ((rows, width), np.dtype("u1")),
        "lengths.npy": ((rows,), np.dtype("u1")),
        "instruments.npy": ((rows,), np.dtype("u1")),
        "fingerprints.npy": ((rows, 512), np.dtype("u1")),
    }


def _load_table(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = str(source)
    table = _TABLE_CACHE.get(key)
    if table is None:
        with h5py.File(source, "r") as handle:
            table = (
                np.asarray(handle["smiles_bytes"][:], dtype=np.uint8),
                np.asarray(handle["smiles_offsets"][:], dtype=np.uint64),
                np.asarray(handle["formula_bytes"][:], dtype=np.uint8),
                np.asarray(handle["formula_offsets"][:], dtype=np.uint64),
            )
        _TABLE_CACHE[key] = table
        if len(_TABLE_CACHE) > 2:
            _TABLE_CACHE.popitem(last=False)
    else:
        _TABLE_CACHE.move_to_end(key)
    return table


def _select_work_units(manifest: dict, rank: int, world_size: int) -> list[dict]:
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"invalid rank/world size: {rank}/{world_size}")
    lanes = [[] for _ in range(world_size)]
    lane_rows = [0] * world_size
    for unit in sorted(
        manifest["work_units"], key=lambda item: (-int(item["rows"]), int(item["work_id"]))
    ):
        owner = min(range(world_size), key=lambda value: (lane_rows[value], value))
        lanes[owner].append(unit)
        lane_rows[owner] += int(unit["rows"])
    return sorted(lanes[rank], key=lambda item: int(item["work_id"]))


def _validate_existing(
    path: Path,
    signature: str,
    rows: int,
    width: int,
) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("cache_signature") != signature:
        raise SimulationPlanError(f"incompatible existing compact synthetic part: {path}")
    if int(manifest.get("rows", -1)) != rows:
        raise SimulationPlanError(f"row mismatch in compact synthetic part: {path}")
    for filename, (shape, dtype) in _array_specs(rows, width).items():
        values = np.load(path / filename, mmap_mode="r")
        if values.shape != shape or values.dtype != dtype:
            raise SimulationPlanError(f"array contract mismatch: {path / filename}")
    return True


def _build_work_unit(
    *,
    simulation: Path,
    output: Path,
    plan: dict,
    unit: dict,
    settings: list[dict],
    workers: int,
    max_peaks: int,
    mass_tolerance_da: float,
    local_scratch: Path,
    cache_signature: str,
) -> dict:
    work_id = int(unit["work_id"])
    rows = int(unit["rows"])
    width = max_peaks + 1
    final = output / "parts" / f"work_{work_id:06d}"
    if _validate_existing(final, cache_signature, rows, width):
        return json.loads((final / "manifest.json").read_text())

    part_index = int(unit["part_index"])
    plan_part = simulation / "plan" / "parts" / f"part_{part_index:03d}"
    start = int(unit["start"])
    end = int(unit["end"])
    molecule_ids = np.asarray(
        np.load(plan_part / "molecule_index.npy", mmap_mode="r")[start:end],
        dtype=np.uint32,
    )
    setting_ids = np.asarray(
        np.load(plan_part / "setting_id.npy", mmap_mode="r")[start:end],
        dtype=np.uint16,
    )
    if not (len(molecule_ids) == len(setting_ids) == rows):
        raise SimulationPlanError(f"plan rows are misaligned for work {work_id}")
    source = simulation / plan["catalog_parts"][part_index]["source"]
    table = _load_table(source)
    unique_molecules, inverse = np.unique(molecule_ids, return_inverse=True)
    raw_smiles = decode_packed_strings(table[0], table[1], unique_molecules)
    unique_smiles = [normalize_glacier_smiles(value) for value in raw_smiles]
    unique_formulas = decode_packed_strings(table[2], table[3], unique_molecules)

    root_ions = np.empty(rows, dtype=np.uint8)
    instruments = np.empty(rows, dtype=np.uint8)
    for setting_id in np.unique(setting_ids):
        positions = setting_ids == setting_id
        setting = settings[int(setting_id)]
        try:
            root_ion = ROOT_ION_NAMES.index(str(setting["adduct"]))
        except ValueError as error:
            raise SimulationPlanError(
                f"unsupported MIST root ion {setting['adduct']!r}"
            ) from error
        root_ions[positions] = root_ion
        instruments[positions] = get_instr_idx(str(setting["instrument"]))

    prediction = simulation / "predictions" / f"work_{work_id:06d}"
    prediction_manifest = json.loads((prediction / "manifest.json").read_text())
    if prediction_manifest.get("plan_signature") != plan["plan_signature"]:
        raise SimulationPlanError(f"prediction does not match simulation plan: {prediction}")
    masses = np.load(prediction / "masses.npy", mmap_mode="r")
    intensities = np.load(prediction / "intensities.npy", mmap_mode="r")
    masks = np.load(prediction / "fragment_masks.npy", mmap_mode="r")
    peak_counts = np.load(prediction / "peak_counts.npy", mmap_mode="r")
    if len(masses) != rows:
        raise SimulationPlanError(f"prediction rows differ for work {work_id}")

    stage = Path(tempfile.mkdtemp(prefix=f"simulated_cache_{work_id:06d}.", dir=local_scratch))
    arrays = {
        filename.removesuffix(".npy"): np.lib.format.open_memmap(
            stage / filename, mode="w+", dtype=dtype, shape=shape
        )
        for filename, (shape, dtype) in _array_specs(rows, width).items()
    }
    context = {
        "unique_smiles": unique_smiles,
        "unique_formulas": unique_formulas,
        "inverse": inverse,
        "root_ions": root_ions,
        "instruments": instruments,
        "masses": masses,
        "intensities": intensities,
        "masks": masks,
        "peak_counts": peak_counts,
        "outputs": arrays,
        "max_peaks": max_peaks,
        "mass_tolerance_da": mass_tolerance_da,
    }
    started = time.time()
    maximum_error = 0.0
    dropped_peaks = 0
    errors: list[str] = []
    pool = mp.get_context("fork").Pool(
        processes=workers, initializer=_worker_init, initargs=(context,)
    )
    try:
        unique_fingerprints = np.empty((len(unique_smiles), 512), dtype=np.uint8)
        for unique_index, fingerprint, error in pool.imap_unordered(
            _fingerprint_unique, range(len(unique_smiles)), chunksize=64
        ):
            if error is not None:
                errors.append(f"molecule={int(unique_molecules[unique_index])}: {error}")
                if len(errors) >= 20:
                    break
            else:
                unique_fingerprints[unique_index] = fingerprint
        if errors:
            raise RuntimeError("fingerprint failures: " + "; ".join(errors))
        for chunk_start in range(0, rows, 65_536):
            chunk_end = min(rows, chunk_start + 65_536)
            arrays["fingerprints"][chunk_start:chunk_end] = unique_fingerprints[
                inverse[chunk_start:chunk_end]
            ]

        for completed, (index, error_da, dropped, error) in enumerate(
            pool.imap_unordered(_recover_row, range(rows), chunksize=64), start=1
        ):
            if error is not None:
                errors.append(f"row={index}: {error}")
                if len(errors) >= 20:
                    break
            else:
                maximum_error = max(maximum_error, error_da)
                dropped_peaks += dropped
            if completed % 100_000 == 0:
                elapsed = max(time.time() - started, 1e-9)
                print(
                    f"compact work {work_id:06d}: {completed:,}/{rows:,} "
                    f"rows_per_second={completed / elapsed:.1f}",
                    flush=True,
                )
        if errors:
            raise RuntimeError("feature recovery failures: " + "; ".join(errors))
        pool.close()
        pool.join()
        minimum_formula_count = int(arrays["form_vec"].min())
        maximum_formula_count = int(arrays["form_vec"].max())
        if minimum_formula_count < 0 or maximum_formula_count >= 255:
            raise RuntimeError(
                "recovered formula counts exceed the MIST pos-cos contract: "
                f"minimum={minimum_formula_count}, maximum={maximum_formula_count}"
            )
    except BaseException:
        pool.terminate()
        pool.join()
        arrays.clear()
        shutil.rmtree(stage, ignore_errors=True)
        raise

    for values in arrays.values():
        values.flush()
    arrays.clear()
    np.save(stage / "molecule_index.npy", molecule_ids, allow_pickle=False)
    np.save(stage / "setting_id.npy", setting_ids, allow_pickle=False)
    manifest = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "kind": "atlas_synthetic_part",
        "cache_signature": cache_signature,
        "simulation_plan_signature": plan["plan_signature"],
        "prediction_signature": prediction_manifest["prediction_signature"],
        "work_unit": unit,
        "rows": rows,
        "max_peaks": max_peaks,
        "mass_tolerance_da": mass_tolerance_da,
        "maximum_formula_mass_residual_da": maximum_error,
        "minimum_formula_count": minimum_formula_count,
        "maximum_formula_count": maximum_formula_count,
        "dropped_unrepresentable_peaks": dropped_peaks,
        "unique_molecules": len(unique_molecules),
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(stage / "manifest.json", manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    transfer = final.parent / f".{final.name}.{os.getpid()}.partial"
    if final.exists() or transfer.exists():
        raise FileExistsError(f"refusing to overwrite compact output {final}")
    shutil.copytree(stage, transfer)
    os.replace(transfer, final)
    shutil.rmtree(stage)
    return manifest


def run_worker(args: argparse.Namespace) -> None:
    simulation = args.simulation.resolve()
    output = args.output.resolve()
    plan = load_plan_manifest(simulation)
    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    units = _select_work_units(plan, rank, args.world_size)
    settings_frame = __import__("pandas").read_csv(
        simulation / plan["settings_path"], sep="\t", dtype=str, keep_default_na=False
    )
    settings = settings_frame.to_dict(orient="records")
    prediction_signatures = {
        json.loads(
            (
                simulation
                / "predictions"
                / f"work_{int(unit['work_id']):06d}"
                / "manifest.json"
            ).read_text()
        )["prediction_signature"]
        for unit in plan["work_units"]
    }
    if len(prediction_signatures) > 1:
        raise SimulationPlanError("Synthetic predictions contain mixed signatures")
    cache_inputs = {
        "schema_version": COMPACT_SCHEMA_VERSION,
        "kind": "atlas_synthetic",
        "simulation": str(simulation),
        "simulation_plan_signature": plan["plan_signature"],
        "prediction_signature": next(iter(prediction_signatures), None),
        "max_peaks": args.max_peaks,
        "mass_tolerance_da": args.mass_tolerance_da,
    }
    cache_signature = stable_json_hash(cache_inputs)
    local_scratch = args.local_scratch
    if local_scratch is None:
        local_scratch = Path(os.environ.get("TMPDIR", "/tmp"))
    local_scratch = local_scratch.resolve()
    local_scratch.mkdir(parents=True, exist_ok=True)
    completed_rows = 0
    for sequence, unit in enumerate(units, start=1):
        result = _build_work_unit(
            simulation=simulation,
            output=output,
            plan=plan,
            unit=unit,
            settings=settings,
            workers=args.workers,
            max_peaks=args.max_peaks,
            mass_tolerance_da=args.mass_tolerance_da,
            local_scratch=local_scratch,
            cache_signature=cache_signature,
        )
        completed_rows += int(result["rows"])
        print(
            f"compact lane {rank}/{args.world_size}: {sequence}/{len(units)} "
            f"work={int(unit['work_id']):06d} rows={int(result['rows']):,}",
            flush=True,
        )
    print(
        json.dumps(
            {
                **cache_inputs,
                "cache_signature": cache_signature,
                "rank": rank,
                "world_size": args.world_size,
                "assigned_work_units": len(units),
                "completed_rows": completed_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def finalize_cache(simulation: Path, output: Path) -> dict:
    """Publish the root manifest only after every planned part is complete."""
    simulation, output = simulation.resolve(), output.resolve()
    plan = load_plan_manifest(simulation)
    dataset_name = plan.get("dataset_name")
    if dataset_name not in {"msg_all", "canopus_all"}:
        raise SimulationPlanError("simulation plan requires dataset_name=msg_all or canopus_all")
    if plan.get("conditioning") != {"kind": "none"}:
        raise SimulationPlanError("simulation must be unconditioned")
    parts, signatures, widths = [], set(), set()
    for unit in plan["work_units"]:
        path = output / "parts" / f"work_{int(unit['work_id']):06d}"
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("simulation_plan_signature") != plan["plan_signature"]:
            raise SimulationPlanError(f"part belongs to a different simulation plan: {path}")
        rows, width = int(unit["rows"]), int(manifest["max_peaks"]) + 1
        _validate_existing(path, manifest["cache_signature"], rows, width)
        signatures.add(manifest["cache_signature"])
        widths.add(width)
        parts.append({"path": path.relative_to(output).as_posix(), "rows": rows})
    if len(signatures) != 1 or len(widths) != 1:
        raise SimulationPlanError("cache is empty or mixes incompatible part signatures")
    rows = sum(part["rows"] for part in parts)
    if rows != int(plan["physical_predictions"]):
        raise SimulationPlanError("part rows differ from the simulation plan")
    result = {
        "schema_version": 1, "kind": "simulated_spectrum",
        "dataset_name": dataset_name, "split": "train",
        "source_contract": {
            "schema_version": 1, "source_type": "simulated",
            "dataset_name": dataset_name, "split": "train", "conditioning": {"kind": "none"},
        },
        "max_peaks": next(iter(widths)) - 1,
        "fingerprint": "Morgan radius=2 fpSize=4096 includeChirality=false",
        "parts": parts, "rows": rows, "simulation_plan_signature": plan["plan_signature"],
        "compact_cache_signature": next(iter(signatures)),
    }
    destination = output / "manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != result:
        raise FileExistsError(f"incompatible existing manifest: {destination}")
    write_json_atomic(destination, result)
    return result


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-peaks", type=int, default=10)
    parser.add_argument("--mass-tolerance-da", type=float, default=2e-3)
    parser.add_argument("--local-scratch", type=Path)
    parser.add_argument("--finalize-only", action="store_true")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if args.workers < 1 or not 1 <= args.max_peaks <= 254:
        raise ValueError("workers must be positive and max-peaks must be between 1 and 254")
    if not args.finalize_only:
        run_worker(args)
    if args.finalize_only or args.world_size == 1:
        print(json.dumps(finalize_cache(args.simulation, args.output), indent=2))


if __name__ == "__main__":
    main()
