import json
from pathlib import Path

import numpy as np
import pytest

from mist.data.training_sources import (
    FINGERPRINT_CONTRACT,
    TrainingSourceContractError,
    load_training_data_contract,
)


def _write_cache(
    root: Path,
    *,
    source_type: str,
    dataset_name: str = "canopus_all",
    split: str = "train",
    conditioning: dict | None = None,
    rows: int = 8,
) -> Path:
    part = root / "part_00000"
    part.mkdir(parents=True)
    width = 3
    arrays = {
        "form_vec": np.zeros((rows, width, 18), dtype=np.int16),
        "ion_vec": np.zeros((rows, width), dtype=np.uint8),
        "intens": np.ones((rows, width), dtype=np.float16),
        "peak_type": np.zeros((rows, width), dtype=np.uint8),
        "lengths": np.full(rows, width, dtype=np.uint8),
        "instruments": np.zeros(rows, dtype=np.uint8),
        "fingerprints": np.zeros((rows, 512), dtype=np.uint8),
    }
    arrays["peak_type"][:, -1] = 3
    arrays["form_vec"][:, :, 0] = np.arange(1, width + 1)
    arrays["form_vec"][:, :, 1] = np.arange(1, width + 1) * 2
    arrays["fingerprints"][:, 0] = 3
    arrays["fingerprints"][:, 1] = np.arange(rows, dtype=np.uint8)
    arrays["names"] = np.asarray([f"s{i}" for i in range(rows)])
    arrays["structure_ids"] = np.asarray([f"m{i}" for i in range(rows)])
    for name, values in arrays.items():
        np.save(part / f"{name}.npy", values, allow_pickle=False)
    contract = {
        "schema_version": 1,
        "source_type": source_type,
        "dataset_name": dataset_name,
        "split": split,
        "conditioning": conditioning or {"kind": "none"},
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": (
                    "real_spectrum" if source_type == "real" else "simulated_spectrum"
                ),
                "dataset_name": dataset_name,
                "split": split,
                "max_peaks": 2,
                "fingerprint": FINGERPRINT_CONTRACT,
                "source_contract": contract,
                "parts": [{"path": "part_00000", "rows": rows}],
                "rows": rows,
            }
        )
    )
    return root


def _config(tmp_path, synthetic=False):
    real = _write_cache(tmp_path / "real", source_type="real")
    val = _write_cache(tmp_path / "val", source_type="real", split="val", rows=1)
    sources = [
        {"name": "real", "source_type": "real", "cache": str(real), "batch_size": 2}
    ]
    if synthetic:
        cache = _write_cache(tmp_path / "synthetic", source_type="simulated")
        sources.append(
            {
                "name": "synthetic",
                "source_type": "simulated",
                "cache": str(cache),
                "batch_size": 2,
            }
        )
    return {
        "dataset_name": "canopus_all",
        "training_sources": sources,
        "validation_source": {"cache": str(val), "batch_size": 2},
        "num_workers": 0,
    }


@pytest.mark.parametrize("synthetic", [False, True])
def test_real_only_and_mixed_contracts(tmp_path, synthetic):
    contract = load_training_data_contract(_config(tmp_path, synthetic))
    assert len(contract.training_sources) == 1 + synthetic
    assert contract.real_fraction == (0.5 if synthetic else 1.0)


def test_rejects_nontraining_synthetic_contract(tmp_path):
    config = _config(tmp_path, True)
    path = Path(config["training_sources"][1]["cache"]) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_contract"]["conditioning"] = {"kind": "unsupported"}
    path.write_text(json.dumps(manifest))
    with pytest.raises(TrainingSourceContractError, match="unconditioned"):
        load_training_data_contract(config)


def test_export_indexed_cache_is_portable(tmp_path, monkeypatch):
    import shutil

    from mist.data.compact import CompactFeatureDataset
    from mist.data.export_cache import export_cache

    source = _write_cache(tmp_path / "input", source_type="real")
    np.save(source / "part_00000/names.npy", np.array([f"s{i}" for i in range(8)]))
    np.save(
        source / "part_00000/structure_ids.npy", np.array([f"m{i}" for i in range(8)])
    )
    np.save(source / "selection.npy", np.array([1, 4, 7], dtype=np.int64))
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["rows"] = 3
    manifest["parts"][0].update(rows=3, index="selection.npy")
    path.write_text(json.dumps(manifest))
    export_cache(source, tmp_path / "release", chunk_rows=2)
    shutil.rmtree(source)
    moved = tmp_path / "moved"
    (tmp_path / "release").rename(moved)
    monkeypatch.chdir(tmp_path.parent)
    data = CompactFeatureDataset(moved)
    assert len(data) == 3
    assert np.load(moved / "part_00000/names.npy").tolist() == ["s1", "s4", "s7"]
    assert str(tmp_path) not in (moved / "manifest.json").read_text()
