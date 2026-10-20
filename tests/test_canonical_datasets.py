import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from mist.data import build_all_cache, datasets, featurizers
from mist.data.compact import CompactFeatureDataset
from mist.data.data import Spectra
from mist.utils import chem_utils


def _tree(spec: str, collision: str, formula: str = "C2H6O") -> dict:
    fragment = "CH3O"
    mz = chem_utils.formula_mass(fragment) + chem_utils.ion_to_mass["[M+H]+"]
    return {
        "cand_form": formula,
        "spec_name": f"{spec}_collision {collision}",
        "cand_ion": "[M+H]+",
        "output_tbl": {
            "mz": [mz],
            "ms2_inten": [1.0],
            "formula": [""],
            "ions": [chem_utils.ion_to_mass["[M+H]+"]],
        },
    }


def _write_dataset(data_dir: Path, name: str) -> Path:
    root = data_dir / name
    (root / "subformulae").mkdir(parents=True)
    (root / "splits").mkdir()
    rows = [
        {
            "spec": "train_a",
            "formula": "C2H6O",
            "ionization": "[M+H]+",
            "instrument": "Orbitrap",
            "smiles": "CCO",
            "inchikey": "TRAIN",
            "collision_energies": "['10', '20']",
        },
        {
            "spec": "train_missing_ce",
            "formula": "C2H6O",
            "ionization": "[M+H]+",
            "instrument": "Unknown",
            "smiles": "CCO",
            "inchikey": "MISSING",
            "collision_energies": "",
        },
        {
            "spec": "train_assignment_without_ce",
            "formula": "C2H6O",
            "ionization": "[M+H]+",
            "instrument": "Orbitrap",
            "smiles": "CCO",
            "inchikey": "ASSIGNED",
            "collision_energies": "",
        },
        {
            "spec": "val_a",
            "formula": "C2H6O",
            "ionization": "[M+H]+",
            "instrument": "QTOF",
            "smiles": "CCO",
            "inchikey": "VAL",
            "collision_energies": "['30']",
        },
        {
            "spec": "test_a",
            "formula": "C2H6O",
            "ionization": "[M+H]+",
            "instrument": "Unknown",
            "smiles": "CCO",
            "inchikey": "TEST",
            "collision_energies": "['40']",
        },
    ]
    pd.DataFrame(rows).to_csv(root / "labels.tsv", sep="\t", index=False)
    split = pd.DataFrame(
        [
            ("train_a", "train"),
            ("train_missing_ce", "train"),
            ("train_assignment_without_ce", "train"),
            ("val_a", "val"),
            ("test_a", "test"),
        ],
        columns=["name", "split"],
    )
    if name == "msg_all":
        split.to_csv(root / "splits" / "split.tsv", sep="\t", index=False)
    else:
        source_splits = root / "splits" / "source_splits"
        source_splits.mkdir()
        split.to_csv(
            source_splits / "canopus_hplus_100_0.tsv", sep="\t", index=False
        )

    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(root / "spec_files.hdf5", "w") as handle:
        for row in rows:
            handle.create_dataset(
                f"{row['spec']}.ms", data=[">collision 10 eV\n1 1\n"], dtype=string_type
            )
    with h5py.File(root / "subformulae" / "no_subform.hdf5", "w") as handle:
        for spec, collision in (
            ("train_a", "10"),
            ("train_a", "20"),
            ("train_assignment_without_ce", "nan"),
            ("val_a", "30"),
            ("test_a", "40"),
        ):
            handle.create_dataset(
                f"{spec}_collision {collision}.json",
                data=[json.dumps(_tree(spec, collision))],
                dtype=string_type,
            )
    return root


def test_msg_loader_uses_assignments_without_requiring_ce(tmp_path):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "msg_all")
    layout = datasets.DatasetLayout.resolve("msg_all", data_dir)
    loaded = datasets.load_dataset(layout)

    assert layout.split == (data_dir / "msg_all" / "splits" / "split.tsv").resolve()
    assert [record.name for record in loaded["train"]] == [
        "train_a_collision 10",
        "train_a_collision 20",
        "train_assignment_without_ce_collision nan",
    ]
    assert len(loaded["val"]) == 1
    assert len(loaded["test"]) == 1
    assert loaded.missing_subformula == {"train": 1, "val": 0, "test": 0}


def test_canopus_loader_uses_hplus_100_0_split(tmp_path):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "canopus_all")
    layout = datasets.DatasetLayout.resolve("canopus_all", data_dir)

    assert layout.split.name == "canopus_hplus_100_0.tsv"
    assert layout.split.parent.name == "source_splits"
    loaded = datasets.load_dataset(layout)
    assert [record.name for record in loaded["train"]] == [
        "train_a", "train_missing_ce", "train_assignment_without_ce",
    ]
    assert loaded.missing_subformula == {"train": 0, "val": 0, "test": 0}
    limited = datasets.load_dataset(layout, max_records_per_split=1)
    assert [record.name for record in limited["train"]] == ["train_a"]


def test_split_schema_has_no_legacy_spec_column_fallback(tmp_path):
    data_dir = tmp_path / "data"
    root = _write_dataset(data_dir, "msg_all")
    split_path = root / "splits" / "split.tsv"
    split = pd.read_csv(split_path, sep="\t").rename(columns={"name": "spec"})
    split.to_csv(split_path, sep="\t", index=False)

    layout = datasets.DatasetLayout.resolve("msg_all", data_dir)
    with pytest.raises(datasets.DatasetContractError, match="exactly"):
        datasets.load_dataset(layout)


def test_peak_formula_assigns_canonical_no_subform_entry(tmp_path):
    data_dir = tmp_path / "data"
    _write_dataset(data_dir, "msg_all")
    layout = datasets.DatasetLayout.resolve("msg_all", data_dir)
    loaded = datasets.load_dataset(layout)
    featurizer = featurizers.PeakFormula(
        dataset_root=str(layout.root),
        max_peaks=10,
    )
    record = loaded["train"][0]
    feature = featurizer.featurize(
        Spectra(
            spectra_name=record.name,
            spectra_formula=record.formula,
            instrument=record.instrument,
        )
    )
    np.testing.assert_array_equal(
        feature["form_vec"][0], chem_utils.formula_to_dense("CH3O")
    )
    np.testing.assert_array_equal(
        feature["form_vec"][-1], chem_utils.formula_to_dense("C2H6O")
    )
    # Regression: max_peaks caps MS2 fragments only. The sole fragment must
    # not be dropped before the precursor/CLS row is appended.
    assert feature["form_vec"].shape[0] == 2
    assert feature["ion_vec"].tolist() == [0, 0]
    assert feature["peak_type"].tolist() == [0, 3]


def test_subformula_assignment_recovers_representable_peaks():
    formula = "C3H8O2"
    ion = "[M+H]+"
    fragments = ("CH3O", "C2H5O", "C3H7O2")
    masses = np.asarray(
        [
            chem_utils.formula_mass(fragment) + chem_utils.ion_to_mass[ion]
            for fragment in fragments
        ]
        + [999.0]
    )
    intensities = np.asarray([0.25, 0.5, 1.0, 0.1])
    tree = {
        "cand_form": formula,
        "cand_ion": ion,
        "output_tbl": {
            "mz": masses.tolist(),
            "ms2_inten": intensities.tolist(),
            "formula": [""] * len(masses),
            "ions": [chem_utils.ion_to_mass[ion]] * len(masses),
        },
    }

    actual = featurizers.assign_subformulae(tree, mass_diff_thresh=20.0)
    assert actual["cand_form"] == formula
    assert actual["cand_ion"] == ion
    assert actual["output_tbl"]["formula"] == list(fragments)
    assert actual["output_tbl"]["ions"] == [ion] * len(fragments)
    np.testing.assert_allclose(actual["output_tbl"]["mz"], masses[:3])
    np.testing.assert_allclose(actual["output_tbl"]["ms2_inten"], intensities[:3])
    np.testing.assert_allclose(actual["output_tbl"]["abs_mass_diff"], 0.0, atol=1e-10)


def test_only_named_canonical_datasets_are_supported(tmp_path):
    with pytest.raises(datasets.DatasetContractError, match="unsupported dataset"):
        datasets.DatasetLayout.resolve("legacy_dataset", tmp_path)


def test_merge_ms_spectrum_pools_before_normalization():
    text = """>parentmass 100
#INSTRUMENT TYPE Orbitrap

>collision 10
20.123441 9
30 1
50 0
60 -1

>collision 20
20.123442 4
40 16
20.123443 12
102 1000
70 0.000001
"""
    merged = featurizers.merge_ms_spectrum(text)
    np.testing.assert_allclose(
        merged,
        [[40, 1], [20.123441, np.sqrt(12 / 16)], [30, 0.25]],
    )


@pytest.mark.parametrize("text", ["", ">collision nan\n", ">collision 10\n20 0\n"])
def test_merge_ms_spectrum_handles_empty_peaks(text):
    assert featurizers.merge_ms_spectrum(text).shape == (0, 2)


@pytest.mark.parametrize("max_peaks", [2, 10])
def test_canopus_cache_merges_all_energies_in_every_split(tmp_path, max_peaks):
    data_dir = tmp_path / "data"
    root = _write_dataset(data_dir, "canopus_all")
    # CANOPUS uses raw .ms content, including spectra without assignment entries.
    (root / "subformulae" / "no_subform.hdf5").unlink()
    formulas = ["CH3O", "CH3", "C2H5O"]
    masses = [
        chem_utils.formula_mass(formula) + chem_utils.ion_to_mass["[M+H]+"]
        for formula in formulas
    ]
    text = (
        ">parentmass 47.049\n\n"
        f">collision 10\n{masses[0]} 9\n{masses[1]} 1\n\n"
        f">collision 20\n{masses[0]} 4\n{masses[2]} 16\n"
    )
    with h5py.File(root / "spec_files.hdf5", "r+") as handle:
        for key in handle:
            handle[key][0] = text

    output = tmp_path / "compact"
    args = build_all_cache.get_parser().parse_args([
        "--dataset-name", "canopus_all", "--data-dir", str(data_dir),
        "--output", str(output), "--max-peaks", str(max_peaks),
    ])
    manifests = build_all_cache.build_all_cache(args)
    assert [manifest["rows"] for manifest in manifests] == [3, 1, 1]
    expected_names = {
        "train": ["train_a", "train_missing_ce", "train_assignment_without_ce"],
        "val": ["val_a"],
        "test": ["test_a"],
    }
    selected = [2, 0, 1][:max_peaks]
    expected_forms = np.asarray([
        *(chem_utils.formula_to_dense(formulas[i]) for i in selected),
        chem_utils.formula_to_dense("C2H6O"),
    ])
    expected_intensities = [1.0, 0.75, 0.25][:max_peaks] + [1.0]
    for split, names in expected_names.items():
        cache = CompactFeatureDataset(output / split)
        part = output / split / "part_00000"
        assert np.load(part / "names.npy").tolist() == names
        for index in range(len(cache)):
            row = cache[index]
            length = len(expected_intensities)
            assert row["lengths"] == length
            np.testing.assert_array_equal(row["form_vec"][:length], expected_forms)
            np.testing.assert_allclose(row["intens"][:length], expected_intensities)
            np.testing.assert_array_equal(
                row["peak_type"][:length], [0] * len(selected) + [3],
            )
            assert row["fingerprints"].shape == (512,)
