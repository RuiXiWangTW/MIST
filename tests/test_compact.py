import json
from pathlib import Path

import numpy as np

from mist.data.compact import (
    CompactFeatureDataset,
    LocalityBatchSampler,
    collate_compact,
)


def _write_part(path: Path, rows: int) -> None:
    path.mkdir(parents=True)
    arrays = {
        "form_vec": np.zeros((rows, 3, 18), dtype=np.int16),
        "ion_vec": np.zeros((rows, 3), dtype=np.uint8),
        "intens": np.ones((rows, 3), dtype=np.float16),
        "peak_type": np.zeros((rows, 3), dtype=np.uint8),
        "lengths": np.full(rows, 3, dtype=np.uint8),
        "instruments": np.full(rows, 2, dtype=np.uint8),
        "fingerprints": np.zeros((rows, 512), dtype=np.uint8),
    }
    arrays["fingerprints"][:, 0] = np.arange(rows, dtype=np.uint8)
    for name, values in arrays.items():
        np.save(path / f"{name}.npy", values, allow_pickle=False)


def test_dataset_and_collation(tmp_path):
    _write_part(tmp_path / "part_00000", 5)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parts": [{"path": "part_00000", "rows": 5}],
            }
        )
    )
    dataset = CompactFeatureDataset(tmp_path)
    batch = collate_compact([dataset[1], dataset[3]])
    assert len(dataset) == 5
    assert batch["form_vec"].shape == (2, 3, 18)
    assert batch["targets"].shape == (2, 4096)
    assert batch["row_indices"].tolist() == [1, 3]


def test_indexed_view_maps_sorted_logical_rows_to_source(tmp_path):
    source = tmp_path / "source_part"
    _write_part(source, 6)
    view = tmp_path / "view"
    view.mkdir()
    np.save(view / "indices.npy", np.asarray([1, 4, 5], dtype=np.uint32))
    (view / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "parts": [
                    {
                        "path": "logical_part_00000",
                        "source": str(source),
                        "index": "indices.npy",
                        "rows": 3,
                    }
                ],
            }
        )
    )
    dataset = CompactFeatureDataset(view)
    assert len(dataset) == 3
    assert int(dataset[0]["fingerprints"][0]) == 1
    assert int(dataset[1]["fingerprints"][0]) == 4
    assert int(dataset[2]["row_index"]) == 2


def test_distributed_sampler_is_disjoint_and_even():
    samplers = [
        LocalityBatchSampler(
            [23, 19],
            4,
            rank=rank,
            world_size=3,
            seed=8,
            drop_last=True,
            even_divisible=True,
        )
        for rank in range(3)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert [len(value) for value in batches] == [3, 3, 3]
    row_sets = [{row for batch in rank_batches for row in batch} for rank_batches in batches]
    assert not (row_sets[0] & row_sets[1])
    assert not (row_sets[0] & row_sets[2])
    assert not (row_sets[1] & row_sets[2])


def test_validation_sampler_covers_every_row_once():
    samplers = [
        LocalityBatchSampler(
            [10, 7],
            4,
            rank=rank,
            world_size=3,
            seed=0,
            drop_last=False,
            even_divisible=False,
        )
        for rank in range(3)
    ]
    rows = [row for sampler in samplers for batch in sampler for row in batch]
    assert sorted(rows) == list(range(17))


def test_tail_combining_sampler_uses_small_parts_without_duplicates():
    samplers = [
        LocalityBatchSampler(
            [2, 3, 4, 1],
            4,
            rank=rank,
            world_size=2,
            seed=3,
            drop_last=True,
            even_divisible=True,
            combine_part_tails=True,
        )
        for rank in range(2)
    ]
    batches = [list(sampler) for sampler in samplers]
    assert [len(value) for value in batches] == [1, 1]
    rows = [row for rank_batches in batches for batch in rank_batches for row in batch]
    assert len(rows) == 8
    assert len(set(rows)) == 8
