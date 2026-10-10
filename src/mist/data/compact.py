"""Memory-mapped fixed-width training data and locality-aware batching."""

from __future__ import annotations

from bisect import bisect_right
import json
from itertools import islice
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from mist.data.atlas import unpack_fingerprints


COMPACT_SCHEMA_VERSION = 1
REQUIRED_ARRAYS = (
    "form_vec.npy",
    "ion_vec.npy",
    "intens.npy",
    "peak_type.npy",
    "lengths.npy",
    "instruments.npy",
    "fingerprints.npy",
)


class CompactCacheError(ValueError):
    pass


def _read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


class CompactFeatureDataset(Dataset):
    """A logical dataset composed of row-aligned NumPy sidecar parts."""

    def __init__(self, cache_root: str | Path):
        self.cache_root = Path(cache_root).resolve()
        manifest_path = self.cache_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing compact cache manifest: {manifest_path}")
        self.manifest = _read_json(manifest_path)
        if int(self.manifest.get("schema_version", -1)) != COMPACT_SCHEMA_VERSION:
            raise CompactCacheError(
                f"unsupported compact schema: {self.manifest.get('schema_version')}"
            )
        part_entries = self.manifest.get("parts", [])
        if not part_entries:
            raise CompactCacheError(f"cache has no parts: {self.cache_root}")
        self.part_paths = [
            (self.cache_root / entry["source"]).resolve()
            if entry.get("source") is not None
            else self.cache_root / entry["path"]
            for entry in part_entries
        ]
        self.part_index_paths = [
            (self.cache_root / entry["index"]).resolve()
            if entry.get("index") is not None
            else None
            for entry in part_entries
        ]
        self.part_sizes = [int(entry["rows"]) for entry in part_entries]
        self.part_source_sizes = []
        for path, index_path, logical_rows in zip(
            self.part_paths, self.part_index_paths, self.part_sizes
        ):
            missing = [name for name in REQUIRED_ARRAYS if not (path / name).exists()]
            if missing:
                raise CompactCacheError(f"part {path} is missing {missing}")
            source_rows = len(np.load(path / REQUIRED_ARRAYS[0], mmap_mode="r"))
            self.part_source_sizes.append(source_rows)
            if index_path is not None:
                if not index_path.is_file():
                    raise CompactCacheError(f"part index does not exist: {index_path}")
                indices = np.load(index_path, mmap_mode="r")
                if indices.shape != (logical_rows,) or not np.issubdtype(
                    indices.dtype, np.integer
                ):
                    raise CompactCacheError(
                        f"invalid indexed-view array {index_path}: "
                        f"shape={indices.shape}, dtype={indices.dtype}"
                    )
                if len(indices):
                    if int(indices[0]) < 0 or int(indices[-1]) >= source_rows:
                        raise CompactCacheError(
                            f"indexed-view rows are outside source part {path}"
                        )
                    if np.any(indices[1:] <= indices[:-1]):
                        raise CompactCacheError(
                            f"indexed-view rows must be strict-sorted: {index_path}"
                        )
            elif source_rows != logical_rows:
                raise CompactCacheError(
                    f"part row count mismatch in {path}: "
                    f"manifest={logical_rows}, source={source_rows}"
                )
        self.part_offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(self.part_sizes, dtype=np.int64))
        )
        self._arrays: dict[int, dict[str, np.ndarray]] = {}
        self._indices: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return int(self.part_offsets[-1])

    def _load_part(self, part_index: int) -> dict[str, np.ndarray]:
        arrays = self._arrays.get(part_index)
        if arrays is None:
            path = self.part_paths[part_index]
            arrays = {
                filename.removesuffix(".npy"): np.load(path / filename, mmap_mode="r")
                for filename in REQUIRED_ARRAYS
            }
            expected = self.part_source_sizes[part_index]
            bad = {name: value.shape for name, value in arrays.items() if len(value) != expected}
            if bad:
                raise CompactCacheError(
                    f"row-count mismatch in {path}: expected={expected}, arrays={bad}"
                )
            self._arrays[part_index] = arrays
        return arrays

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        part_index = bisect_right(self.part_offsets, index) - 1
        return part_index, int(index - self.part_offsets[part_index])

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic]:
        part_index, local_index = self._resolve_index(index)
        arrays = self._load_part(part_index)
        index_path = self.part_index_paths[part_index]
        if index_path is not None:
            indices = self._indices.get(part_index)
            if indices is None:
                indices = np.load(index_path, mmap_mode="r")
                self._indices[part_index] = indices
            source_index = int(indices[local_index])
        else:
            source_index = local_index
        row = {name: values[source_index] for name, values in arrays.items()}
        row["row_index"] = np.int64(index)
        return row

    def take_array(self, name: str, indices: np.ndarray) -> np.ndarray:
        """Read arbitrary logical rows from one sidecar without Python row loops.

        The returned rows preserve the caller's order and work for both ordinary
        compact caches and indexed views without materializing complete feature
        dictionaries.
        """

        logical = np.asarray(indices, dtype=np.int64)
        if logical.ndim != 1:
            raise ValueError("compact row indices must be one-dimensional")
        if len(logical) and (int(logical.min()) < 0 or int(logical.max()) >= len(self)):
            raise IndexError("compact row index is outside the logical dataset")
        known = {filename.removesuffix(".npy") for filename in REQUIRED_ARRAYS}
        if name not in known:
            raise KeyError(f"unknown compact array: {name}")
        if not len(logical):
            sample = self._load_part(0)[name]
            return np.empty((0, *sample.shape[1:]), dtype=sample.dtype)
        part_ids = np.searchsorted(self.part_offsets[1:], logical, side="right")
        sample = self._load_part(int(part_ids[0]))[name]
        output = np.empty((len(logical), *sample.shape[1:]), dtype=sample.dtype)
        for part_id in np.unique(part_ids):
            part_id = int(part_id)
            positions = np.flatnonzero(part_ids == part_id)
            local = logical[positions] - self.part_offsets[part_id]
            index_path = self.part_index_paths[part_id]
            if index_path is not None:
                mapped = self._indices.get(part_id)
                if mapped is None:
                    mapped = np.load(index_path, mmap_mode="r")
                    self._indices[part_id] = mapped
                local = np.asarray(mapped[local], dtype=np.int64)
            output[positions] = self._load_part(part_id)[name][local]
        return output


def collate_compact(rows: Sequence[dict]) -> dict[str, torch.Tensor]:
    """Stack compact rows and unpack target fingerprints in one vectorized call."""

    if not rows:
        raise ValueError("cannot collate an empty batch")
    form_vec = np.stack([row["form_vec"] for row in rows]).astype(np.float32, copy=False)
    ion_vec = np.stack([row["ion_vec"] for row in rows]).astype(np.int64, copy=False)
    intens = np.stack([row["intens"] for row in rows]).astype(np.float32, copy=False)
    peak_type = np.stack([row["peak_type"] for row in rows]).astype(np.int64, copy=False)
    lengths = np.asarray([row["lengths"] for row in rows], dtype=np.int64)
    instruments = np.asarray([row["instruments"] for row in rows], dtype=np.int64)
    packed = np.stack([row["fingerprints"] for row in rows])
    targets = unpack_fingerprints(packed)
    row_indices = np.asarray([row["row_index"] for row in rows], dtype=np.int64)
    return {
        "form_vec": torch.from_numpy(form_vec),
        "ion_vec": torch.from_numpy(ion_vec),
        "intens": torch.from_numpy(intens),
        "types": torch.from_numpy(peak_type),
        "num_peaks": torch.from_numpy(lengths),
        "instruments": torch.from_numpy(instruments),
        "targets": torch.from_numpy(targets),
        "row_indices": torch.from_numpy(row_indices),
    }


class LocalityBatchSampler(Sampler[list[int]]):
    """Batch sequential rows from pre-randomized parts and partition batches by rank.

    Atlas rows are already globally randomized. Avoiding another sample-level
    permutation preserves large sequential reads from memory maps while part-order
    shuffling and an epoch-dependent rank rotation prevent a fixed rank assignment.
    """

    def __init__(
        self,
        part_sizes: Sequence[int],
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = True,
        even_divisible: bool | None = None,
        combine_part_tails: bool = False,
    ):
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid batch/distributed sampler arguments")
        self.part_sizes = tuple(int(value) for value in part_sizes)
        self.part_offsets = np.concatenate(
            (np.asarray([0], dtype=np.int64), np.cumsum(self.part_sizes, dtype=np.int64))
        )
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.even_divisible = world_size > 1 if even_divisible is None else even_divisible
        self.combine_part_tails = bool(combine_part_tails)
        self.epoch = 0
        self.start_batch = 0
        if self.combine_part_tails:
            total_rows = sum(self.part_sizes)
            self.total_batches = (
                total_rows // self.batch_size
                if self.drop_last
                else math.ceil(total_rows / self.batch_size)
            )
        else:
            self.total_batches = sum(
                size // self.batch_size
                if self.drop_last
                else math.ceil(size / self.batch_size)
                for size in self.part_sizes
            )
        self.usable_batches = (
            (self.total_batches // self.world_size) * self.world_size
            if self.even_divisible
            else self.total_batches
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.even_divisible:
            return self.usable_batches // self.world_size
        rotation = self.epoch % self.world_size
        wanted = (self.rank - rotation) % self.world_size
        if self.total_batches <= wanted:
            return 0
        return 1 + (self.total_batches - 1 - wanted) // self.world_size

    def __iter__(self) -> Iterator[list[int]]:
        return islice(self._batches(), self.start_batch, None)

    def _batches(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        part_order = rng.permutation(len(self.part_sizes))
        batch_counter = 0
        rotation = self.epoch % self.world_size
        if self.combine_part_tails:
            buffer: list[int] = []
            for part_index in part_order:
                part_size = self.part_sizes[int(part_index)]
                offset = int(self.part_offsets[int(part_index)])
                # Fixed-size locality blocks avoid constructing one enormous
                # Python range while still allowing small indexed-view tails to
                # combine into full distributed batches.  Shuffle the blocks so
                # a large physical part does not impose its molecule ordering on
                # a long contiguous stretch of optimizer steps.
                block_starts = np.arange(0, part_size, 8192, dtype=np.int64)
                for block_start_value in rng.permutation(block_starts):
                    block_start = int(block_start_value)
                    block_end = min(part_size, block_start + 8192)
                    cursor = block_start
                    while cursor < block_end:
                        needed = self.batch_size - len(buffer)
                        take = min(needed, block_end - cursor)
                        buffer.extend(range(offset + cursor, offset + cursor + take))
                        cursor += take
                        if len(buffer) == self.batch_size:
                            if batch_counter >= self.usable_batches:
                                return
                            owner = (batch_counter + rotation) % self.world_size
                            if owner == self.rank:
                                yield buffer
                            buffer = []
                            batch_counter += 1
            if buffer and not self.drop_last and batch_counter < self.usable_batches:
                owner = (batch_counter + rotation) % self.world_size
                if owner == self.rank:
                    yield buffer
            return
        for part_index in part_order:
            part_size = self.part_sizes[int(part_index)]
            offset = int(self.part_offsets[int(part_index)])
            starts = np.arange(0, part_size, self.batch_size)
            if self.drop_last and len(starts) and starts[-1] + self.batch_size > part_size:
                starts = starts[:-1]
            # Rows are random already; shuffle moderately sized contiguous blocks, not rows.
            block_size = max(1, 8192 // self.batch_size)
            block_starts = np.arange(0, len(starts), block_size)
            for block_index in rng.permutation(len(block_starts)):
                first = int(block_starts[block_index])
                last = min(first + block_size, len(starts))
                for start in starts[first:last]:
                    if batch_counter >= self.usable_batches:
                        return
                    owner = (batch_counter + rotation) % self.world_size
                    if owner == self.rank:
                        end = min(int(start) + self.batch_size, part_size)
                        yield list(range(offset + int(start), offset + end))
                    batch_counter += 1


def make_compact_loader(
    dataset: CompactFeatureDataset,
    *,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 0,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    even_divisible: bool | None = None,
    combine_part_tails: bool = False,
) -> tuple[DataLoader, LocalityBatchSampler]:
    sampler = LocalityBatchSampler(
        dataset.part_sizes,
        batch_size,
        rank=rank,
        world_size=world_size,
        seed=seed,
        drop_last=drop_last,
        even_divisible=even_divisible,
        combine_part_tails=combine_part_tails,
    )
    kwargs = {}
    if num_workers:
        kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_compact,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed + rank),
        **kwargs,
    )
    return loader, sampler
