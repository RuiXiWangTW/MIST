from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mist.data.random_pubchem import build_catalog


def _pack(values: list[bytes]):
    offsets = np.concatenate(
        (np.asarray([0], dtype=np.uint64), np.cumsum([len(v) for v in values], dtype=np.uint64))
    )
    return np.frombuffer(b"".join(values), dtype=np.uint8), offsets


def _decode(payload: np.ndarray, offsets: np.ndarray) -> list[bytes]:
    raw = memoryview(np.asarray(payload, dtype=np.uint8))
    return [
        bytes(raw[int(start) : int(end)])
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]


def _part(path: Path, part_index: int, keys: list[bytes]):
    smiles = [b"CC", b"CCC", b"CO", b"CN"]
    formulas = [b"C2H6", b"C3H255", b"CH4O", b"CH5N"] if part_index == 1 else [
        b"C2H6", b"C3H8", b"CH4O", b"CH5N"
    ]
    smiles_bytes, smiles_offsets = _pack(smiles)
    formula_bytes, formula_offsets = _pack(formulas)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(complete=1, schema_version=1)
        counts = [101, 3, 2, 2] if part_index == 1 else [2, 3, 2, 2]
        handle.create_dataset("atom_count", data=np.asarray(counts, dtype=np.uint16))
        handle.create_dataset("inchikey", data=np.asarray(keys, dtype="S27"))
        handle.create_dataset(
            "identity_hash",
            data=np.asarray([[part_index + 1, i + 11] for i in range(4)], dtype=np.uint64),
        )
        handle.create_dataset("smiles_bytes", data=smiles_bytes)
        handle.create_dataset("smiles_offsets", data=smiles_offsets)
        handle.create_dataset("formula_bytes", data=formula_bytes)
        handle.create_dataset("formula_offsets", data=formula_offsets)


def test_random_catalog_is_exact_size_and_heldout_disjoint(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    heldout_key = b"AAAAAAAAAAAAAA"
    _part(
        source / "target_part_000.h5",
        0,
        [heldout_key + b"-UHFFFAOYSA-N", b"B" * 27, b"C" * 27, b"D" * 27],
    )
    _part(
        source / "target_part_001.h5",
        1,
        [b"E" * 27, b"F" * 27, b"G" * 27, b"H" * 27],
    )
    labels = tmp_path / "labels.tsv"
    pd.DataFrame(
        {"spec": ["v", "t"], "inchikey": [heldout_key.decode() + "-UHFFFAOYSA-N", "Z" * 27]}
    ).to_csv(labels, sep="\t", index=False)
    split = tmp_path / "split.tsv"
    pd.DataFrame({"name": ["v", "t"], "split": ["val", "test"]}).to_csv(
        split, sep="\t", index=False
    )
    output = tmp_path / "output"
    result = build_catalog(
        source_root=source,
        output=output,
        labels=labels,
        split=split,
        count=5,
        seed=7,
        output_part_size=3,
        source_part_limit=None,
    )
    assert result["rows"] == 5
    assert result["heldout_connectivity_matches_retained"] == 0
    assert result["selection"]["maximum_formula_element_count"] == 254
    assert result["selection"]["candidate_rows_over_formula_count_limit"] == 1
    assert [part["rows"] for part in result["parts"]] == [3, 2]
    for part in result["parts"]:
        with h5py.File(part["source"], "r") as handle:
            assert not np.any(handle["inchikey"][:].astype("S14") == heldout_key)
            assert np.all(handle["atom_count"][:] <= 100)
            formulas = _decode(handle["formula_bytes"][:], handle["formula_offsets"][:])
            assert b"C3H255" not in formulas
