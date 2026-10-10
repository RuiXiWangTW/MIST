"""Peak-formula featurization for canonical compact MIST caches."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from mist.data.data import Spectra
from mist.utils import chem_utils


class HDF5JsonStore:
    """Process-local, read-only access to UTF-8 JSON entries in HDF5."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._handle: h5py.File | None = None
        self._pid: int | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    def _file(self) -> h5py.File:
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = pid
        return self._handle

    def read(self, key: str) -> str:
        handle = self._file()
        if key not in handle:
            raise KeyError(f"missing HDF5 key {key!r} in {self.path}")
        value = handle[key][()]
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ValueError(
                    f"expected one string at {self.path}:{key}, got {value.shape}"
                )
            value = value.reshape(-1)[0]
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise TypeError(f"expected UTF-8 text at {self.path}:{key}")
        return value


@lru_cache(maxsize=512)
def _heavy_formula_space(formula: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = chem_utils.formula_to_dense(formula).astype(np.int16)
    hydrogen_index = chem_utils.element_to_ind["H"]
    heavy_indices = np.flatnonzero(root > 0)
    heavy_indices = heavy_indices[heavy_indices != hydrogen_index]
    if len(heavy_indices) == 0:
        heavy = np.zeros((1, len(root)), dtype=np.int16)
    else:
        shape = tuple(int(root[index]) + 1 for index in heavy_indices)
        counts = np.indices(shape, dtype=np.int16).reshape(len(shape), -1).T
        heavy = np.zeros((len(counts), len(root)), dtype=np.int16)
        heavy[:, heavy_indices] = counts

    heavy_masses = heavy.astype(np.float64) @ chem_utils.VALID_MONO_MASSES
    rdbe_without_h = 1.0 + 0.5 * (
        heavy.astype(np.float64) @ chem_utils.rdbe_mult
    )
    max_h_from_rdbe = np.floor(2.0 * rdbe_without_h + 1e-12).astype(np.int32)
    max_h = np.minimum(max_h_from_rdbe, int(root[hydrogen_index]))
    valid = max_h >= 0
    return heavy[valid], heavy_masses[valid], max_h[valid]


def assign_subformulae(tree: dict, mass_diff_thresh: float = 20.0) -> dict:
    """Assign precursor-consistent subformulae to unassigned peaks."""

    output_table = tree.get("output_tbl")
    if output_table is None:
        return {
            "cand_form": tree["cand_form"],
            "cand_ion": tree["cand_ion"],
            "output_tbl": None,
        }

    formula = tree["cand_form"]
    ion = tree["cand_ion"]
    if ion not in chem_utils.ION_LST:
        raise ValueError(f"unsupported canonical adduct {ion!r}")
    heavy, heavy_masses, max_h = _heavy_formula_space(formula)
    hydrogen_index = chem_utils.element_to_ind["H"]
    hydrogen_mass = float(chem_utils.VALID_MONO_MASSES[hydrogen_index])
    ion_mass = float(chem_utils.ion_to_mass[ion])
    masses = np.asarray(output_table["mz"], dtype=np.float64)
    intensities = np.asarray(output_table["ms2_inten"], dtype=np.float64)
    if masses.shape != intensities.shape or masses.ndim != 1:
        raise ValueError("subformula m/z and intensity arrays must be aligned vectors")

    assigned = {
        "mz": [],
        "ms2_inten": [],
        "mono_mass": [],
        "abs_mass_diff": [],
        "mass_diff": [],
        "formula": [],
        "ions": [],
    }
    for mass, intensity in zip(masses, intensities):
        ideal_h = (mass - ion_mass - heavy_masses) / hydrogen_mass
        hydrogen_counts = np.floor(ideal_h + 0.5).astype(np.int32)
        hydrogen_counts = np.clip(hydrogen_counts, 0, max_h)
        represented = heavy_masses + hydrogen_counts * hydrogen_mass + ion_mass
        errors = np.abs(mass - represented)
        best = int(np.argmin(errors))
        abs_error = float(errors[best])
        ppm_error = abs_error / max(float(mass), 200.0) * 1e6
        if ppm_error >= mass_diff_thresh:
            continue
        form_vec = heavy[best].copy()
        form_vec[hydrogen_index] = hydrogen_counts[best]
        assigned["mz"].append(float(mass))
        assigned["ms2_inten"].append(float(intensity))
        assigned["mono_mass"].append(float(represented[best]))
        assigned["abs_mass_diff"].append(abs_error)
        assigned["mass_diff"].append(ppm_error)
        assigned["formula"].append(chem_utils.vec_to_formula(form_vec))
        assigned["ions"].append(ion)

    return {
        "cand_form": formula,
        "cand_ion": ion,
        "output_tbl": assigned if assigned["mz"] else None,
    }


class PeakFormula:
    """Convert one canonical HDF5 subformula entry to MIST tensors."""

    PEAK_TYPES = {"fragment": 0, "cls": 3}

    def __init__(
        self,
        *,
        dataset_root: str | Path,
        max_peaks: int = 10,
        subformula_ppm: float = 20.0,
        **_: object,
    ) -> None:
        self.max_peaks = int(max_peaks)
        self.subformula_ppm = float(subformula_ppm)
        path = Path(dataset_root).resolve() / "subformulae" / "no_subform.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"canonical subformula HDF5 does not exist: {path}")
        self.subformula_store = HDF5JsonStore(path)

    def _peak_dict(self, spectrum: Spectra) -> dict:
        name = spectrum.get_spec_name()
        tree = json.loads(self.subformula_store.read(f"{name}.json"))
        table = tree.get("output_tbl")
        if table is not None:
            count = len(table.get("mz", []))
            formulas = table.get("formula", [])
            ions = table.get("ions", [])
            if (
                len(formulas) != count
                or len(ions) != count
                or any(not isinstance(value, str) or not value for value in formulas)
                or any(value not in chem_utils.ION_LST for value in ions)
            ):
                tree = assign_subformulae(tree, self.subformula_ppm)

        if tree["cand_form"] != spectrum.get_spectra_formula():
            raise ValueError(
                f"formula mismatch for {name}: labels={spectrum.get_spectra_formula()!r}, "
                f"subformulae={tree['cand_form']!r}"
            )
        table = tree["output_tbl"]
        if table is None:
            fragments: list[str] = []
            intensities: list[float] = []
            ions: list[str] = []
        else:
            order = np.argsort(np.asarray(table["ms2_inten"], dtype=float))[::-1]
            order = order[: self.max_peaks]
            fragments = np.asarray(table["formula"], dtype=str)[order].tolist()
            intensities = np.asarray(table["ms2_inten"], dtype=float)[order].tolist()
            ions = np.asarray(table["ions"], dtype=str)[order].tolist()
        return {
            "fragments": fragments,
            "intensities": intensities,
            "ions": ions,
            "root_formula": tree["cand_form"],
            "root_ion": tree["cand_ion"],
        }

    def featurize(self, spectrum: Spectra, train_mode: bool = False) -> dict:
        if train_mode:
            raise ValueError("compact-cache construction does not augment on CPU")
        peaks = self._peak_dict(spectrum)
        forms = [chem_utils.formula_to_dense(value) for value in peaks["fragments"]]
        forms.append(chem_utils.formula_to_dense(peaks["root_formula"]))
        ions = [chem_utils.get_ion_idx(value) for value in peaks["ions"]]
        ions.append(chem_utils.get_ion_idx(peaks["root_ion"]))
        intensities = [*peaks["intensities"], 1.0]
        peak_types = [self.PEAK_TYPES["fragment"]] * len(peaks["fragments"])
        peak_types.append(self.PEAK_TYPES["cls"])
        return {
            "form_vec": np.asarray(forms, dtype=np.int16),
            "ion_vec": np.asarray(ions, dtype=np.uint8),
            "frag_intens": np.asarray(intensities, dtype=np.float32),
            "peak_type": np.asarray(peak_types, dtype=np.uint8),
            "name": spectrum.get_spec_name(),
            "instrument": chem_utils.get_instr_idx(spectrum.get_instrument()),
        }
