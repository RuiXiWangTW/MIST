import json
from pathlib import Path

import numpy as np
import pytest

from mist.data.build_simulated_cache import _array_specs, finalize_cache
from mist.data.compact import CompactFeatureDataset


def test_finalization_requires_all_parts_and_publishes_relative_paths(tmp_path):
    simulation = tmp_path / "simulation"
    (simulation / "plan").mkdir(parents=True)
    plan = {"schema_version": 1, "kind": "glacier_fixed_pretraining_plan",
            "dataset_name": "canopus_all", "conditioning": {"kind": "none"},
            "plan_signature": "plan-signature", "physical_predictions": 4,
            "work_units": [{"work_id": i, "rows": 2} for i in range(2)]}
    (simulation / "plan/manifest.json").write_text(json.dumps(plan))
    output = tmp_path / "output"
    for i in range(2):
        part = output / "parts" / f"work_{i:06d}"
        part.mkdir(parents=True)
        for filename, (shape, dtype) in _array_specs(2, 3).items():
            np.save(part / filename, np.zeros(shape, dtype=dtype))
        (part / "manifest.json").write_text(json.dumps({
            "rows": 2, "max_peaks": 2, "cache_signature": "same-signature",
            "simulation_plan_signature": "plan-signature",
        }))
        if i == 0:
            with pytest.raises(FileNotFoundError):
                finalize_cache(simulation, output)
            assert not (output / "manifest.json").exists()
    manifest = finalize_cache(simulation, output)
    assert manifest["rows"] == 4
    assert all(not Path(p["path"]).is_absolute() for p in manifest["parts"])
    assert len(CompactFeatureDataset(output)) == 4
    assert finalize_cache(simulation, output) == manifest


def test_simulation_to_trainable_features(tmp_path):
    from argparse import Namespace
    import h5py
    from mist.data.atlas import morgan4096_packed
    from mist.data.build_simulated_cache import run_worker
    from mist.data.training_sources import _validate_feature_layout
    from mist.utils.chem_utils import formula_mass, ion_to_mass

    simulation = tmp_path / "simulation"
    part = simulation / "plan/parts/part_000"
    part.mkdir(parents=True)
    np.save(part / "molecule_index.npy", np.array([0, 0], dtype=np.uint32))
    np.save(part / "setting_id.npy", np.array([0, 0], dtype=np.uint16))
    with h5py.File(simulation / "catalog.h5", "w") as handle:
        for name, value in [("smiles", b"CCO"), ("formula", b"C2H6O")]:
            handle.create_dataset(f"{name}_bytes", data=np.frombuffer(value, dtype=np.uint8))
            handle.create_dataset(f"{name}_offsets", data=np.array([0, len(value)], dtype=np.uint64))
    (simulation / "plan/settings.tsv").write_text("adduct\tinstrument\n[M+H]+\tOrbitrap\n")
    plan = {
        "schema_version": 1, "kind": "glacier_fixed_pretraining_plan",
        "dataset_name": "canopus_all", "conditioning": {"kind": "none"},
        "plan_signature": "plan", "physical_predictions": 2,
        "settings_path": "plan/settings.tsv", "catalog_parts": [{"source": "catalog.h5"}],
        "work_units": [{"work_id": 0, "part_index": 0, "start": 0, "end": 2, "rows": 2}],
    }
    (simulation / "plan/manifest.json").write_text(json.dumps(plan))
    prediction = simulation / "predictions/work_000000"
    prediction.mkdir(parents=True)
    (prediction / "manifest.json").write_text(json.dumps({"plan_signature": "plan", "prediction_signature": "prediction"}))
    mass = formula_mass("C2H6O") + ion_to_mass["[M+H]+"]
    np.save(prediction / "masses.npy", np.full((2, 1), mass, dtype=np.float32))
    np.save(prediction / "intensities.npy", np.ones((2, 1), dtype=np.float32))
    np.save(prediction / "fragment_masks.npy", np.array([[[7, 0]], [[7, 0]]], dtype=np.uint64))
    np.save(prediction / "peak_counts.npy", np.ones(2, dtype=np.uint16))
    args = Namespace(simulation=simulation, output=tmp_path / "features", rank=0, world_size=1,
                     workers=1, max_peaks=2, mass_tolerance_da=.002, local_scratch=tmp_path / "scratch")
    run_worker(args)
    finalize_cache(simulation, args.output)
    data = CompactFeatureDataset(args.output)
    assert _validate_feature_layout(data) == 2
    assert len(data) == 2
    np.testing.assert_array_equal(data[0]["fingerprints"], morgan4096_packed("CCO"))
    assert np.count_nonzero(data[0]["peak_type"] == 3) == 1
    run_worker(args)  # completed parts can be safely reused
