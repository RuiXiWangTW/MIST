"""Chemical vocabularies shared by compact data and the MIST model."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from rdkit import Chem

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")
VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]
element_to_ind = {element: index for index, element in enumerate(VALID_ELEMENTS)}
VALID_MONO_MASSES = np.asarray(
    [Chem.GetPeriodicTable().GetMostCommonIsotopeMass(element) for element in VALID_ELEMENTS]
)
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, VALID_MONO_MASSES, strict=True))
NORM_VEC = np.asarray(
    [81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6]
)

ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
]
ion_to_idx = {ion: index for index, ion in enumerate(ION_LST)}
ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + 4 * ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "[M-H4O2+H]+": -2 * ELEMENT_TO_MASS["O"] - 3 * ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
}

instrument_to_type = defaultdict(lambda: "unknown")
instrument_to_type.update(
    {
        "Orbitrap": "orbitrap",
        "QTOF": "qtof",
        "Unknown": "unknown",
        "Thermo Finnigan Velos Orbitrap": "orbitrap",
        "Thermo Finnigan Elite Orbitrap": "orbitrap",
        "Orbitrap Fusion Lumos": "orbitrap",
        "Q-ToF (LCMS)": "qtof",
        "Unknown (LCMS)": "unknown",
        "ion trap": "iontrap",
        "FTICR (LCMS)": "fticr",
        "Bruker Q-ToF (LCMS)": "qtof",
        "Orbitrap (LCMS)": "orbitrap",
    }
)
instruments = sorted(set(instrument_to_type.values()))
instrument_to_idx = {name: index for index, name in enumerate(instruments)}
max_instr_idx = len(instruments) + 1

rdbe_mult = np.zeros(len(VALID_ELEMENTS), dtype=np.float64)
for element, weight in zip(
    ("C", "N", "P", "H", "Cl", "Br", "I", "F"),
    (2, 1, 1, -1, -1, -1, -1, -1),
    strict=True,
):
    rdbe_mult[element_to_ind[element]] = weight


def get_ion_idx(ionization: str) -> int:
    return ion_to_idx[ionization]


def get_instr_idx(instrument: str) -> int:
    return instrument_to_idx[instrument_to_type.get(instrument, "unknown")]


def formula_to_dense(formula: str) -> np.ndarray:
    vector = np.zeros(len(VALID_ELEMENTS), dtype=np.float64)
    matches = list(CHEM_FORMULA_PATTERN.finditer(formula))
    if not matches or "".join(match.group(0) for match in matches) != formula:
        raise ValueError(f"invalid molecular formula {formula!r}")
    for match in matches:
        element, count_text = match.groups()
        if element not in element_to_ind:
            raise ValueError(f"unsupported element {element!r} in {formula!r}")
        vector[element_to_ind[element]] += 1 if count_text == "" else int(count_text)
    return vector


def vec_to_formula(vector: np.ndarray) -> str:
    return "".join(
        element if int(vector[index]) == 1 else f"{element}{int(vector[index])}"
        for index, element in enumerate(VALID_ELEMENTS)
        if vector[index] > 0
    )


def formula_mass(formula: str) -> float:
    return float(formula_to_dense(formula) @ VALID_MONO_MASSES)
