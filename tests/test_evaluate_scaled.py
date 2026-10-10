import json

import numpy as np

from mist.data.compact import REQUIRED_ARRAYS, CompactFeatureDataset
from mist.evaluate_scaled import _load_metadata


def test_load_metadata_respects_indexed_view(tmp_path):
    part = tmp_path / "part"
    part.mkdir()
    for filename in REQUIRED_ARRAYS:
        if filename == "fingerprints.npy":
            values = np.zeros((3, 512), dtype=np.uint8)
        else:
            values = np.zeros((3, 1), dtype=np.float32)
        np.save(part / filename, values)
    np.save(part / "names.npy", np.asarray(["input-a", "skip", "input-b"]))
    np.save(part / "structure_ids.npy", np.asarray(["mol-a", "skip", "mol-b"]))
    np.save(tmp_path / "indices.npy", np.asarray([0, 2], dtype=np.int64))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "real_spectrum",
                "dataset_name": "canopus_all",
                "split": "test",
                "rows": 2,
                "parts": [
                    {
                        "path": "logical-part",
                        "source": str(part),
                        "index": "indices.npy",
                        "rows": 2,
                    }
                ],
            }
        )
    )

    names, structures = _load_metadata(CompactFeatureDataset(tmp_path))

    assert names.tolist() == ["input-a", "input-b"]
    assert structures.tolist() == ["mol-a", "mol-b"]
