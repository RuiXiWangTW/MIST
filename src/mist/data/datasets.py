"""Canonical MIST datasets backed by the ``*_all`` HDF5 layout.

The training loader deliberately supports one on-disk contract. A dataset is
selected by name below and contains ``labels.tsv``, ``spec_files.hdf5``, a
fixed split table, and, except for CANOPUS, ``subformulae/no_subform.hdf5``.
CANOPUS uses one record per source spectrum, merging all collision energies
during featurization. MSG and Enveda records use the subformula-assignment keys.
Enveda's split table uses ``spec`` and ``Fold_0``; the loader reads those as
``name`` and ``split``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_DATASETS: Mapping[str, Path] = {
    "msg_all": Path("splits/split.tsv"),
    "canopus_all": Path("splits/source_splits/canopus_hplus_100_0.tsv"),
    "enveda": Path("splits/split_1.tsv"),
}
REQUIRED_LABEL_COLUMNS = (
    "spec",
    "formula",
    "ionization",
    "instrument",
    "smiles",
    "inchikey",
)
SPLIT_NAMES = ("train", "val", "test")


class DatasetContractError(ValueError):
    """Raised when a dataset does not satisfy the canonical ``*_all`` layout."""


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved paths for one supported dataset."""

    name: str
    root: Path
    labels: Path
    spectra: Path
    subformulae: Path
    split: Path

    @classmethod
    def resolve(cls, dataset_name: str, data_dir: str | Path = "data") -> DatasetLayout:
        if dataset_name not in SUPPORTED_DATASETS:
            choices = ", ".join(sorted(SUPPORTED_DATASETS))
            raise DatasetContractError(
                f"unsupported dataset {dataset_name!r}; expected one of: {choices}"
            )
        root = (Path(data_dir) / dataset_name).resolve()
        layout = cls(
            name=dataset_name,
            root=root,
            labels=root / "labels.tsv",
            spectra=root / "spec_files.hdf5",
            subformulae=root / "subformulae" / "no_subform.hdf5",
            split=root / SUPPORTED_DATASETS[dataset_name],
        )
        layout.validate()
        return layout

    @property
    def split_name(self) -> str:
        return self.split.stem

    @property
    def merge_collision_energies(self) -> bool:
        return self.name == "canopus_all"

    def validate(self) -> None:
        hdf5_paths = (self.spectra,)
        if not self.merge_collision_energies:
            hdf5_paths += (self.subformulae,)
        required = (self.labels, self.split, *hdf5_paths)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{self.name} does not satisfy the *_all dataset contract; "
                f"missing files: {missing}"
            )
        for path in hdf5_paths:
            try:
                with h5py.File(path, "r"):
                    pass
            except OSError as error:
                raise DatasetContractError(f"cannot open HDF5 file {path}") from error


@dataclass(frozen=True)
class SpectrumRecord:
    """One source spectrum or collision-specific training example."""

    name: str
    source_spec: str
    formula: str
    ionization: str
    instrument: str
    smiles: str
    inchikey: str


@dataclass(frozen=True)
class LoadedDataset:
    """Canonical records and validation counts for one dataset."""

    layout: DatasetLayout
    records: Mapping[str, tuple[SpectrumRecord, ...]]
    missing_subformula: Mapping[str, int]

    def __getitem__(self, split: str) -> tuple[SpectrumRecord, ...]:
        return self.records[split]


def _assignment_source(key: str, path: Path) -> str:
    """Return the source spectrum encoded by a canonical assignment key."""

    suffix = ".json"
    marker = "_collision "
    if not key.endswith(suffix):
        raise DatasetContractError(
            f"unexpected subformula key {key!r} in {path}; expected a .json key"
        )
    stem = key.removesuffix(suffix)
    if marker not in stem:
        raise DatasetContractError(
            f"unexpected subformula key {key!r} in {path}; "
            f"expected '<spec>{marker}<value>.json'"
        )
    source_spec, collision_label = stem.rsplit(marker, 1)
    if not source_spec or not collision_label:
        raise DatasetContractError(f"malformed subformula key {key!r} in {path}")
    return source_spec


def _index_assignments(
    layout: DatasetLayout, label_names: set[str]
) -> dict[str, tuple[str, ...]]:
    """Index assignment keys by label without interpreting collision energy."""

    assignments: dict[str, list[str]] = {}
    unknown_sources: set[str] = set()
    with h5py.File(layout.subformulae, "r") as subformula_h5:
        for key in subformula_h5:
            source_spec = _assignment_source(key, layout.subformulae)
            if source_spec not in label_names:
                unknown_sources.add(source_spec)
                continue
            assignments.setdefault(source_spec, []).append(key)

    if unknown_sources:
        preview = sorted(unknown_sources)[:5]
        raise DatasetContractError(
            f"{layout.subformulae} contains assignments for "
            f"{len(unknown_sources)} names absent from labels; first entries: {preview}"
        )
    return {
        source_spec: tuple(sorted(keys))
        for source_spec, keys in assignments.items()
    }


def _read_tables(layout: DatasetLayout) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(
        layout.labels,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    missing_columns = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels]
    if missing_columns:
        raise DatasetContractError(
            f"{layout.labels} is missing required columns: {missing_columns}"
        )
    if labels["spec"].eq("").any() or labels["spec"].duplicated().any():
        raise DatasetContractError(f"{layout.labels} contains empty or duplicate spec IDs")

    split = pd.read_csv(
        layout.split,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if list(split.columns) == ["spec", "Fold_0"]:
        split = split.rename(columns={"spec": "name", "Fold_0": "split"})
    if list(split.columns) != ["name", "split"]:
        raise DatasetContractError(
            f"{layout.split} must have exactly ['name', 'split'] columns, "
            f"got {list(split.columns)}"
        )
    if split["name"].eq("").any() or split["name"].duplicated().any():
        raise DatasetContractError(f"{layout.split} contains empty or duplicate names")
    unexpected = sorted(set(split["split"]) - set(SPLIT_NAMES))
    if unexpected:
        raise DatasetContractError(
            f"{layout.split} contains unsupported split labels: {unexpected}"
        )
    missing_folds = sorted(set(SPLIT_NAMES) - set(split["split"]))
    if missing_folds:
        raise DatasetContractError(f"{layout.split} is missing folds: {missing_folds}")
    unknown_names = sorted(set(split["name"]) - set(labels["spec"]))
    if unknown_names:
        preview = unknown_names[:5]
        raise DatasetContractError(
            f"{layout.split} references {len(unknown_names)} names absent from labels; "
            f"first entries: {preview}"
        )
    return labels, split


def load_dataset(
    layout: DatasetLayout,
    *,
    max_records_per_split: int | None = None,
) -> LoadedDataset:
    """Load source-level CANOPUS or collision-specific MSG records."""

    if max_records_per_split is not None and max_records_per_split < 1:
        raise ValueError("max_records_per_split must be positive")
    labels, split_table = _read_tables(layout)
    label_rows = labels.set_index("spec", drop=False).to_dict(orient="index")
    assignments = (
        {}
        if layout.merge_collision_energies
        else _index_assignments(layout, set(label_rows))
    )

    records: dict[str, list[SpectrumRecord]] = {name: [] for name in SPLIT_NAMES}
    missing_subformula = {name: 0 for name in SPLIT_NAMES}
    checked_spectra: set[str] = set()

    with h5py.File(layout.spectra, "r") as spectra_h5:
        for source_spec, split_name in split_table[["name", "split"]].itertuples(
            index=False, name=None
        ):
            if (
                max_records_per_split is not None
                and len(records[split_name]) >= max_records_per_split
            ):
                continue
            row = label_rows[source_spec]
            names = (
                (source_spec,)
                if layout.merge_collision_energies
                else tuple(
                    key.removesuffix(".json")
                    for key in assignments.get(source_spec, ())
                )
            )
            if not names:
                missing_subformula[split_name] += 1
                continue

            spectrum_key = f"{source_spec}.ms"
            if spectrum_key not in checked_spectra:
                if spectrum_key not in spectra_h5:
                    raise DatasetContractError(
                        f"{layout.spectra} is missing spectrum key {spectrum_key!r}"
                    )
                checked_spectra.add(spectrum_key)

            for name in names:
                if (
                    max_records_per_split is not None
                    and len(records[split_name]) >= max_records_per_split
                ):
                    break
                for required_value in ("formula", "ionization", "smiles"):
                    if not row[required_value]:
                        raise DatasetContractError(
                            f"empty {required_value} for {source_spec!r} in {layout.labels}"
                        )
                records[split_name].append(
                    SpectrumRecord(
                        name=name,
                        source_spec=source_spec,
                        formula=row["formula"],
                        ionization=row["ionization"],
                        instrument=row["instrument"] or "Unknown",
                        smiles=row["smiles"],
                        inchikey=row["inchikey"],
                    )
                )

    frozen_records = {name: tuple(records[name]) for name in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        if not frozen_records[split_name]:
            raise DatasetContractError(
                f"{layout.name} produced no usable {split_name} entries"
            )
        logger.info(
            "%s %s: %d entries (%d labels without subformula assignments)",
            layout.name,
            split_name,
            len(frozen_records[split_name]),
            missing_subformula[split_name],
        )
    return LoadedDataset(
        layout=layout,
        records=frozen_records,
        missing_subformula=missing_subformula,
    )
