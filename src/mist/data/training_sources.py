"""Canonical multi-source contract for scaled MIST training.

Real and simulated examples share the same fixed-width compact tensors.  Their
origin remains explicit in each cache manifest. Synthetic spectra must be
unconditioned, and validation always uses held-out real spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mist.data.compact import CompactFeatureDataset
from mist.data.datasets import SUPPORTED_DATASETS


SOURCE_CONTRACT_SCHEMA_VERSION = 1
SOURCE_TYPES = ("real", "simulated")
SIMULATED_KINDS = (
    "simulated_spectrum",
    "simulated_spectrum_view",
)
FINGERPRINT_CONTRACT = "Morgan radius=2 fpSize=4096 includeChirality=false"


class TrainingSourceContractError(ValueError):
    """Raised when a training cache/config violates the source contract."""


@dataclass(frozen=True)
class TrainingSource:
    name: str
    source_type: str
    cache_root: Path
    batch_size: int
    seed_offset: int
    combine_part_tails: bool
    dataset: CompactFeatureDataset


@dataclass(frozen=True)
class ValidationSource:
    cache_root: Path
    batch_size: int
    dataset: CompactFeatureDataset


@dataclass(frozen=True)
class TrainingDataContract:
    dataset_name: str
    training_sources: tuple[TrainingSource, ...]
    validation_source: ValidationSource
    max_peaks: int

    @property
    def per_device_batch_size(self) -> int:
        return sum(source.batch_size for source in self.training_sources)

    @property
    def real_fraction(self) -> float:
        real = sum(
            source.batch_size
            for source in self.training_sources
            if source.source_type == "real"
        )
        return real / self.per_device_batch_size


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingSourceContractError(f"{label} must be a mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing or extra:
        raise TrainingSourceContractError(
            f"{label} keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _source_contract(dataset: CompactFeatureDataset) -> Mapping[str, Any]:
    contract = _mapping(
        dataset.manifest.get("source_contract"),
        f"{dataset.cache_root}/manifest.json source_contract",
    )
    if int(contract.get("schema_version", -1)) != SOURCE_CONTRACT_SCHEMA_VERSION:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: unsupported source contract schema"
        )
    return contract


def _validate_feature_layout(dataset: CompactFeatureDataset) -> int:
    manifest = dataset.manifest
    try:
        max_peaks = int(manifest["max_peaks"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: max_peaks is required"
        ) from error
    if max_peaks < 1:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: max_peaks must be positive"
        )
    if manifest.get("fingerprint") != FINGERPRINT_CONTRACT:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: unexpected fingerprint contract"
        )
    width = max_peaks + 1
    expected = {
        "form_vec.npy": ((width, 18), np.dtype("<i2")),
        "ion_vec.npy": ((width,), np.dtype("u1")),
        "intens.npy": ((width,), np.dtype("<f2")),
        "peak_type.npy": ((width,), np.dtype("u1")),
        "lengths.npy": ((), np.dtype("u1")),
        "instruments.npy": ((), np.dtype("u1")),
        "fingerprints.npy": ((512,), np.dtype("u1")),
    }
    for part_index, (path, rows) in enumerate(
        zip(dataset.part_paths, dataset.part_source_sizes)
    ):
        for filename, (tail, dtype) in expected.items():
            values = np.load(path / filename, mmap_mode="r")
            if values.shape != (rows, *tail) or values.dtype != dtype:
                raise TrainingSourceContractError(
                    f"{dataset.cache_root}: part {part_index} {filename} has "
                    f"shape={values.shape}, dtype={values.dtype}; expected "
                    f"shape={(rows, *tail)}, dtype={dtype}"
                )
    return max_peaks


def _validate_real(
    dataset: CompactFeatureDataset,
    contract: Mapping[str, Any],
    *,
    dataset_name: str,
    expected_split: str,
) -> None:
    if dataset.manifest.get("kind") != "real_spectrum":
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: real source must have kind=real_spectrum"
        )
    expected = {
        "schema_version": SOURCE_CONTRACT_SCHEMA_VERSION,
        "source_type": "real",
        "dataset_name": dataset_name,
        "split": expected_split,
        "conditioning": {"kind": "none"},
    }
    if dict(contract) != expected:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: real source contract differs from {expected}"
        )
    if dataset.manifest.get("dataset_name") != dataset_name:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: real dataset_name differs from {dataset_name}"
        )
    if dataset.manifest.get("split") != expected_split:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: real split differs from {expected_split}"
        )


def _validate_simulated(
    dataset: CompactFeatureDataset,
    contract: Mapping[str, Any],
    *,
    dataset_name: str,
) -> None:
    if dataset.manifest.get("kind") not in SIMULATED_KINDS:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: unsupported simulated cache kind"
        )
    if contract.get("source_type") != "simulated":
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: simulated source_type is required"
        )
    if contract.get("dataset_name") != dataset_name or contract.get("split") != "train":
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: simulated source must train {dataset_name}"
        )
    if dataset.manifest.get("dataset_name") != dataset_name or dataset.manifest.get("split") != "train":
        raise TrainingSourceContractError("synthetic manifest dataset/split differs from its source contract")
    conditioning = _mapping(
        contract.get("conditioning"), f"{dataset.cache_root} conditioning"
    )
    if dict(conditioning) != {"kind": "none"}:
        raise TrainingSourceContractError(
            f"{dataset.cache_root}: synthetic training requires unconditioned spectra"
        )


def _validate_source(
    dataset: CompactFeatureDataset,
    *,
    source_type: str,
    dataset_name: str,
    usage: str,
) -> int:
    contract = _source_contract(dataset)
    if source_type == "real":
        _validate_real(
            dataset,
            contract,
            dataset_name=dataset_name,
            expected_split="train" if usage == "train" else "val",
        )
    elif source_type == "simulated" and usage == "train":
        _validate_simulated(
            dataset,
            contract,
            dataset_name=dataset_name,
        )
    else:
        raise TrainingSourceContractError(
            f"source_type={source_type!r} is invalid for usage={usage!r}"
        )
    return _validate_feature_layout(dataset)


def load_training_data_contract(data_config: Mapping[str, Any]) -> TrainingDataContract:
    """Resolve and validate the canonical multi-source training configuration."""

    dataset_name = str(data_config.get("dataset_name", ""))
    if dataset_name not in SUPPORTED_DATASETS:
        raise TrainingSourceContractError(
            "data.dataset_name must be msg_all or canopus_all"
        )
    raw_sources = data_config.get("training_sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise TrainingSourceContractError("data.training_sources must be a list")
    if not raw_sources:
        raise TrainingSourceContractError("data.training_sources cannot be empty")

    sources: list[TrainingSource] = []
    names: set[str] = set()
    max_peaks: int | None = None
    for index, raw_source in enumerate(raw_sources):
        source_config = _mapping(raw_source, f"training_sources[{index}]")
        _exact_keys(
            source_config,
            label=f"training_sources[{index}]",
            required={"name", "source_type", "cache", "batch_size"},
            optional={"seed_offset", "combine_part_tails"},
        )
        name = str(source_config["name"])
        if not name or name in names:
            raise TrainingSourceContractError(
                f"training source names must be nonempty and unique: {name!r}"
            )
        names.add(name)
        source_type = str(source_config["source_type"])
        if source_type not in SOURCE_TYPES:
            raise TrainingSourceContractError(
                f"training source {name!r} has invalid source_type"
            )
        batch_size = int(source_config["batch_size"])
        if batch_size < 1:
            raise TrainingSourceContractError(
                f"training source {name!r} batch_size must be positive"
            )
        cache_root = Path(source_config["cache"]).resolve()
        dataset = CompactFeatureDataset(cache_root)
        source_max_peaks = _validate_source(
            dataset,
            source_type=source_type,
            dataset_name=dataset_name,
            usage="train",
        )
        if max_peaks is None:
            max_peaks = source_max_peaks
        elif source_max_peaks != max_peaks:
            raise TrainingSourceContractError("training sources use mixed max_peaks")
        sources.append(
            TrainingSource(
                name=name,
                source_type=source_type,
                cache_root=cache_root,
                batch_size=batch_size,
                seed_offset=int(source_config.get("seed_offset", index * 1009)),
                combine_part_tails=bool(
                    source_config.get("combine_part_tails", source_type == "simulated")
                ),
                dataset=dataset,
            )
        )
    if not any(source.source_type == "real" for source in sources):
        raise TrainingSourceContractError(
            "at least one canonical real train source is required"
        )

    validation_config = _mapping(
        data_config.get("validation_source"), "data.validation_source"
    )
    _exact_keys(
        validation_config,
        label="data.validation_source",
        required={"cache", "batch_size"},
    )
    validation_batch_size = int(validation_config["batch_size"])
    if validation_batch_size < 1:
        raise TrainingSourceContractError(
            "validation_source batch_size must be positive"
        )
    validation_root = Path(validation_config["cache"]).resolve()
    validation_dataset = CompactFeatureDataset(validation_root)
    validation_max_peaks = _validate_source(
        validation_dataset,
        source_type="real",
        dataset_name=dataset_name,
        usage="validation",
    )
    if validation_max_peaks != max_peaks:
        raise TrainingSourceContractError(
            "training and validation sources use mixed max_peaks"
        )
    return TrainingDataContract(
        dataset_name=dataset_name,
        training_sources=tuple(sources),
        validation_source=ValidationSource(
            cache_root=validation_root,
            batch_size=validation_batch_size,
            dataset=validation_dataset,
        ),
        max_peaks=int(max_peaks),
    )
