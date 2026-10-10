"""Build an exact-size, heldout-disjoint random PubChem catalog for GLACIER."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mist.data.atlas import ELEMENT_TO_INDEX, MONO_MASSES
from mist.data.simulation_utils import sha256_file, stable_json_hash, write_json_atomic

FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")
MAX_MIST_FORMULA_COUNT_EXCLUSIVE = 255


def _splitmix64(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.uint64).copy()
    output += np.uint64(0x9E3779B97F4A7C15)
    output = (output ^ (output >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    output = (output ^ (output >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return output ^ (output >> np.uint64(31))


def _priority(identity: np.ndarray, seed: int) -> np.ndarray:
    if identity.ndim != 2 or identity.shape[1] != 2:
        raise ValueError("identity_hash must have shape [rows,2]")
    rotated = (identity[:, 1] << np.uint64(17)) | (
        identity[:, 1] >> np.uint64(47)
    )
    return _splitmix64(identity[:, 0] ^ rotated ^ np.uint64(seed % (1 << 64)))


def _heldout_keys(labels_path: Path, split_path: Path) -> tuple[np.ndarray, dict]:
    labels = pd.read_csv(
        labels_path, sep="\t", dtype=str, usecols=["spec", "inchikey"]
    ).set_index("spec")
    split = pd.read_csv(split_path, sep="\t", dtype=str)
    selected = split.loc[split["split"].isin({"val", "test"})]
    missing = sorted(set(selected["name"]) - set(labels.index))
    if missing:
        raise ValueError(f"split names are missing from labels: {missing[:5]}")
    keys = np.asarray(
        sorted(
            {
                value.split("-", 1)[0].encode("ascii")
                for value in labels.loc[selected["name"], "inchikey"].astype(str)
            }
        ),
        dtype="S14",
    )
    validation = {
        "labels": str(labels_path.resolve()),
        "labels_sha256": sha256_file(labels_path),
        "split": str(split_path.resolve()),
        "split_sha256": sha256_file(split_path),
        "excluded_splits": ["test", "val"],
        "split_rows": len(selected),
        "unique_connectivities": len(keys),
    }
    return keys, validation


def _source_parts(root: Path) -> list[Path]:
    parts = sorted(root.resolve().glob("target_part_*.h5"))
    if not parts:
        raise FileNotFoundError(f"no PubChem parts in {root}")
    expected = [f"target_part_{index:03d}.h5" for index in range(len(parts))]
    if [path.name for path in parts] != expected:
        raise ValueError("source PubChem parts are not contiguous")
    return parts


def _connectivity(handle: h5py.File) -> np.ndarray:
    values = np.asarray(handle["inchikey"])
    return values.astype("S14", copy=False)


def _formula_counts(formula: bytes) -> list[int]:
    text = formula.decode("ascii")
    matches = list(FORMULA_PATTERN.finditer(text))
    if not matches or "".join(match.group(0) for match in matches) != text:
        raise ValueError(f"unsupported formula in source catalog: {text!r}")
    counts = []
    for match in matches:
        element, count_text = match.groups()
        if element not in ELEMENT_TO_INDEX:
            raise ValueError(f"unsupported element {element!r} in {text!r}")
        counts.append(1 if not count_text else int(count_text))
    return counts


def _formula_compatible_candidates(
    handle: h5py.File, rows: np.ndarray
) -> np.ndarray:
    if not len(rows):
        return np.empty(0, dtype=bool)
    formulas = _decode_selected(
        np.asarray(handle["formula_bytes"]),
        np.asarray(handle["formula_offsets"]),
        rows,
    )
    return np.fromiter(
        (
            max(_formula_counts(formula)) < MAX_MIST_FORMULA_COUNT_EXCLUSIVE
            for formula in formulas
        ),
        dtype=bool,
        count=len(formulas),
    )


def _eligible_mask(handle: h5py.File, heldout: np.ndarray) -> tuple[np.ndarray, int, int]:
    keys = _connectivity(handle)
    heldout_mask = np.isin(keys, heldout)
    atom_counts = np.asarray(handle["atom_count"], dtype=np.uint16)
    over_limit = atom_counts > 100
    return ~(heldout_mask | over_limit), int(heldout_mask.sum()), int(over_limit.sum())


def select_rows(
    parts: list[Path], heldout: np.ndarray, count: int, seed: int
) -> tuple[list[np.ndarray], dict]:
    """Select the exact global hash-priority sample with two bounded scans."""

    eligible_counts = []
    heldout_matches = 0
    over_atom_limit = 0
    for path in parts:
        with h5py.File(path, "r") as handle:
            eligible, part_heldout, part_over_limit = _eligible_mask(handle, heldout)
        heldout_matches += part_heldout
        over_atom_limit += part_over_limit
        eligible_counts.append(int(eligible.sum()))
    total = sum(eligible_counts)
    if count < 1 or count > total:
        raise ValueError(f"requested {count:,} rows from {total:,} eligible molecules")

    # A 10% oversampled deterministic cutoff keeps only ~2.2M candidates for
    # the requested 2M while making an undersized set astronomically unlikely.
    threshold_int = min(
        (1 << 64) - 1,
        math.ceil((1.10 * count / total) * (1 << 64)),
    )
    threshold = np.uint64(threshold_int)
    priority_parts = []
    secondary_parts = []
    part_parts = []
    row_parts = []
    formula_limit_rejections = 0
    candidates_before_formula_gate = 0
    for part_index, path in enumerate(parts):
        with h5py.File(path, "r") as handle:
            identity = np.asarray(handle["identity_hash"], dtype=np.uint64)
            eligible, _, _ = _eligible_mask(handle, heldout)
            priorities = _priority(identity, seed)
            rows = np.flatnonzero(eligible & (priorities < threshold)).astype(np.uint32)
            candidates_before_formula_gate += len(rows)
            compatible = _formula_compatible_candidates(handle, rows)
        formula_limit_rejections += int((~compatible).sum())
        rows = rows[compatible]
        priority_parts.append(priorities[rows])
        secondary_parts.append(identity[rows, 1])
        part_parts.append(np.full(len(rows), part_index, dtype=np.uint16))
        row_parts.append(rows)
    priorities = np.concatenate(priority_parts)
    secondary = np.concatenate(secondary_parts)
    part_ids = np.concatenate(part_parts)
    row_ids = np.concatenate(row_parts)
    if len(priorities) < count:
        # Deterministic small-catalog/rare-tail fallback. Production 2M/104M
        # selection is overwhelmingly covered by the 10% oversample.
        priority_parts = []
        secondary_parts = []
        part_parts = []
        row_parts = []
        formula_limit_rejections = 0
        candidates_before_formula_gate = 0
        for part_index, path in enumerate(parts):
            with h5py.File(path, "r") as handle:
                identity = np.asarray(handle["identity_hash"], dtype=np.uint64)
                eligible, _, _ = _eligible_mask(handle, heldout)
                rows = np.flatnonzero(eligible).astype(np.uint32)
                candidates_before_formula_gate += len(rows)
                compatible = _formula_compatible_candidates(handle, rows)
                priorities_all = _priority(identity, seed)
            formula_limit_rejections += int((~compatible).sum())
            rows = rows[compatible]
            priority_parts.append(priorities_all[rows])
            secondary_parts.append(identity[rows, 1])
            part_parts.append(np.full(len(rows), part_index, dtype=np.uint16))
            row_parts.append(rows)
        priorities = np.concatenate(priority_parts)
        secondary = np.concatenate(secondary_parts)
        part_ids = np.concatenate(part_parts)
        row_ids = np.concatenate(row_parts)
    order = np.lexsort((row_ids, part_ids, secondary, priorities))[:count]
    selected = []
    for part_index in range(len(parts)):
        rows = np.sort(row_ids[order][part_ids[order] == part_index])
        selected.append(rows.astype(np.uint32, copy=False))
    pairs = np.column_stack((part_ids[order], row_ids[order])).astype("<u4")
    return selected, {
        "kind": "global_identity_hash_priority_sample",
        "seed": seed,
        "requested_molecules": count,
        "eligible_catalog_molecules_before_formula_count_gate": total,
        "source_heldout_connectivity_rows": heldout_matches,
        "source_rows_over_100_atoms": over_atom_limit,
        "maximum_formula_element_count": MAX_MIST_FORMULA_COUNT_EXCLUSIVE - 1,
        "candidates_before_formula_count_gate": candidates_before_formula_gate,
        "candidate_rows_over_formula_count_limit": formula_limit_rejections,
        "oversample_factor": 1.10,
        "oversampled_candidates": len(priorities),
        "priority": "SplitMix64(word0 XOR rotl(word1,17) XOR seed)",
        "tie_break": "identity_word1_then_source_part_then_source_row",
        "selected_source_part_row_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
    }


def _decode_selected(
    payload: np.ndarray, offsets: np.ndarray, indices: np.ndarray
) -> list[bytes]:
    raw = memoryview(payload)
    return [
        bytes(raw[int(offsets[index]) : int(offsets[index + 1])])
        for index in indices
    ]


def _pack(values: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(len(values) + 1, dtype="<u8")
    offsets[0] = 0
    offsets[1:] = np.cumsum([len(value) for value in values], dtype=np.uint64)
    payload = np.frombuffer(b"".join(values), dtype=np.uint8).copy()
    return payload, offsets


def _formula_mass(formula: bytes, cache: dict[bytes, float]) -> float:
    value = cache.get(formula)
    if value is not None:
        return value
    text = formula.decode("ascii")
    matches = list(FORMULA_PATTERN.finditer(text))
    counts = _formula_counts(formula)
    mass = 0.0
    for match, count in zip(matches, counts, strict=True):
        element = match.group(1)
        mass += float(MONO_MASSES[ELEMENT_TO_INDEX[element]]) * count
    cache[formula] = mass
    return mass


def _write_part(
    output: Path,
    part_index: int,
    records: dict[str, list],
    signature: str,
) -> dict:
    final = output / f"target_part_{part_index:03d}.h5"
    fd, temporary_text = tempfile.mkstemp(
        prefix=f".{final.name}.", suffix=".partial", dir=output
    )
    os.close(fd)
    temporary = Path(temporary_text)
    smiles_bytes, smiles_offsets = _pack(records["smiles"])
    formula_bytes, formula_offsets = _pack(records["formula"])
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                {
                    "complete": 1,
                    "schema_version": 1,
                    "normalization": "atlas-pubchem-model",
                    "part_index": part_index,
                    "catalog_signature": signature,
                }
            )
            handle.create_dataset("atom_count", data=np.asarray(records["atom_count"], dtype="<u2"))
            handle.create_dataset("exact_mass", data=np.asarray(records["exact_mass"], dtype="<f4"))
            handle.create_dataset("smiles_bytes", data=smiles_bytes)
            handle.create_dataset("smiles_offsets", data=smiles_offsets)
            handle.create_dataset("formula_bytes", data=formula_bytes)
            handle.create_dataset("formula_offsets", data=formula_offsets)
            handle.create_dataset("inchikey", data=np.asarray(records["inchikey"], dtype="S27"))
            handle.create_dataset("identity_hash", data=np.asarray(records["identity_hash"], dtype="<u8"))
            handle.create_dataset("source_part", data=np.asarray(records["source_part"], dtype="<u2"))
            handle.create_dataset("source_row", data=np.asarray(records["source_row"], dtype="<u4"))
            handle.flush()
        os.replace(temporary, final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "part_index": part_index,
        "path": final.name,
        "source": str(final.resolve()),
        "rows": len(records["atom_count"]),
        "sha256": sha256_file(final),
        "size": final.stat().st_size,
    }


def build_catalog(
    *, source_root: Path, output: Path, labels: Path, split: Path,
    count: int, seed: int, output_part_size: int,
    source_part_limit: int | None = None,
) -> dict:
    source_root = source_root.resolve()
    output = output.resolve()
    labels = labels.resolve()
    split = split.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"catalog output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    parts = _source_parts(source_root)
    if source_part_limit is not None:
        if source_part_limit < 1:
            raise ValueError("source_part_limit must be positive")
        parts = parts[:source_part_limit]
    heldout, heldout_validation = _heldout_keys(labels, split)
    started = time.time()
    selected, selection = select_rows(parts, heldout, count, seed)
    inputs = {
        "schema_version": 1,
        "kind": "random_pubchem_heldout_disjoint_catalog",
        "source_root": str(source_root),
        "source_parts": len(parts),
        "source_part_limit": source_part_limit,
        "heldout_boundary": heldout_validation,
        "selection": selection,
        "output_part_size": output_part_size,
    }
    signature = stable_json_hash(inputs)
    output_parts = []
    formula_mass_cache: dict[bytes, float] = {}
    buffer: dict[str, list] = {
        key: [] for key in (
            "atom_count", "exact_mass", "smiles", "formula", "inchikey",
            "identity_hash", "source_part", "source_row",
        )
    }

    def flush() -> None:
        nonlocal buffer
        if not buffer["atom_count"]:
            return
        output_parts.append(_write_part(output, len(output_parts), buffer, signature))
        buffer = {key: [] for key in buffer}

    selected_connectivity_matches = 0
    for source_part, (path, indices) in enumerate(zip(parts, selected, strict=True)):
        if not len(indices):
            continue
        with h5py.File(path, "r") as handle:
            atom_count = np.asarray(handle["atom_count"])[indices]
            inchikey = np.asarray(handle["inchikey"])[indices]
            identity = np.asarray(handle["identity_hash"])[indices]
            smiles = _decode_selected(
                np.asarray(handle["smiles_bytes"]),
                np.asarray(handle["smiles_offsets"]),
                indices,
            )
            formulas = _decode_selected(
                np.asarray(handle["formula_bytes"]),
                np.asarray(handle["formula_offsets"]),
                indices,
            )
        selected_connectivity_matches += int(
            np.isin(inchikey.astype("S14", copy=False), heldout).sum()
        )
        for position, source_row in enumerate(indices):
            buffer["atom_count"].append(int(atom_count[position]))
            buffer["exact_mass"].append(
                _formula_mass(formulas[position], formula_mass_cache)
            )
            buffer["smiles"].append(smiles[position])
            buffer["formula"].append(formulas[position])
            buffer["inchikey"].append(bytes(inchikey[position]))
            buffer["identity_hash"].append(identity[position])
            buffer["source_part"].append(source_part)
            buffer["source_row"].append(int(source_row))
            if len(buffer["atom_count"]) == output_part_size:
                flush()
    flush()
    rows = sum(int(part["rows"]) for part in output_parts)
    if rows != count or selected_connectivity_matches:
        raise RuntimeError(
            f"catalog validation failed: rows={rows}, heldout_matches={selected_connectivity_matches}"
        )
    result = {
        **inputs,
        "catalog_signature": signature,
        "parts": output_parts,
        "rows": rows,
        "heldout_connectivity_matches_retained": 0,
        "formula_mass": "sum of RDKit most-common-isotope element masses; absolute-CE plan does not consume mass",
        "elapsed_seconds": time.time() - started,
    }
    write_json_atomic(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--output-part-size", type=int, default=100_000)
    parser.add_argument("--source-part-limit", type=int)
    args = parser.parse_args()
    result = build_catalog(
        source_root=args.source_root,
        output=args.output,
        labels=args.labels,
        split=args.split,
        count=args.count,
        seed=args.seed,
        output_part_size=args.output_part_size,
        source_part_limit=args.source_part_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
