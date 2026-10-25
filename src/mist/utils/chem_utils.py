"""Chemical vocabularies shared by compact data and the MIST model."""

from __future__ import annotations

import re

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

# Slots 0-6 are the released MIST vocabulary. Later slots follow that order so
# an adduct spelling keeps one index. Released checkpoints one-hot seven adducts
# plus one unused slot.
ION_LST = [
    "[M+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H2O+H]+",
    "[M+H3N+H]+",
    "[M]+",
    "[M-H4O2+H]+",
    "[M+H-H3N]+",
    "[M+H-CH2O2]+",
    "[M+CHO2]-",
    "[M-H]-",
    "[M+Cl]-",
    "[M-H2O-H]-",
    "[M-H-CO2]-",
]
LEGACY_ADDUCT_DIM = 8
ion_to_idx = {ion: index for index, ion in enumerate(ION_LST)}
_ADDUCT_ALIASES = {
    "[M+NH4]+": "[M+H3N+H]+",
    "[M+H-H2O]+": "[M-H2O+H]+",
    "[M+H-2H2O]+": "[M-H4O2+H]+",
    "[M-2H2O+H]+": "[M-H4O2+H]+",
    "[M-H3N+H]+": "[M+H-H3N]+",
    "[M-NH3+H]+": "[M+H-H3N]+",
    "[M+H-NH3]+": "[M+H-H3N]+",
    "[M+NH3+H]+": "[M+H3N+H]+",
    "[M-H-H2O]-": "[M-H2O-H]-",
    "[M+CH2O2-H]-": "[M+CHO2]-",
    "M+H": "[M+H]+",
    "M+Na": "[M+Na]+",
    "M+H-H2O": "[M-H2O+H]+",
    "M-H2O+H": "[M-H2O+H]+",
    "M+NH4": "[M+H3N+H]+",
    "M-2H2O+H": "[M-H4O2+H]+",
}


def _explicit_charge(ion: str) -> str:
    """Spell ``[M]+`` as ``[M]1+`` when the charge has no explicit count."""

    if "]" not in ion:
        return ion
    adduct, charge = ion.split("]", 1)
    if charge and not charge[0].isnumeric():
        return adduct + "]1" + charge
    return ion


def _close_adduct_aliases(remap: dict[str, str]) -> dict[str, str]:
    """Add explicit-charge and NH3/H3N spellings of every known adduct."""

    closed = dict(remap)
    changed = True
    while changed:
        changed = False
        for adduct, canonical in list(closed.items()):
            variants = [_explicit_charge(adduct)]
            if "H3N" in adduct:
                variants.append(adduct.replace("H3N", "NH3"))
            if "NH3" in adduct:
                variants.append(adduct.replace("NH3", "H3N"))
            for variant in variants:
                if variant not in closed:
                    closed[variant] = canonical
                    changed = True
    return closed


ion_remap = _close_adduct_aliases({ion: ion for ion in ION_LST} | _ADDUCT_ALIASES)
ion_to_mass = {
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + 4 * ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "[M-H4O2+H]+": -2 * ELEMENT_TO_MASS["O"] - 3 * ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-H3N]+": -ELEMENT_TO_MASS["N"] - 2 * ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-CH2O2]+": (
        -ELEMENT_TO_MASS["C"] - 2 * ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS
    ),
    "[M+CHO2]-": (
        ELEMENT_TO_MASS["C"] + ELEMENT_TO_MASS["H"] + 2 * ELEMENT_TO_MASS["O"] + ELECTRON_MASS
    ),
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M-H2O-H]-": -ELEMENT_TO_MASS["O"] - 3 * ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M-H-CO2]-": (
        -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["H"] - 2 * ELEMENT_TO_MASS["O"] + ELECTRON_MASS
    ),
}

instrument_to_type = {
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
    "timsTOF": "timstof",
}
# Indices 0-4 are the historical alphabetical order. Index 5 is the unused slot
# shipped in released checkpoints. timsTOF starts at 8 so those columns stay put.
instrument_to_idx = {
    "fticr": 0,
    "iontrap": 1,
    "orbitrap": 2,
    "qtof": 3,
    "unknown": 4,
    "timstof": 8,
}
LEGACY_INSTR_DIM = 6
max_instr_idx = max(instrument_to_idx.values()) + 2

rdbe_mult = np.zeros(len(VALID_ELEMENTS), dtype=np.float64)
for element, weight in zip(
    ("C", "N", "P", "H", "Cl", "Br", "I", "F"),
    (2, 1, 1, -1, -1, -1, -1, -1),
    strict=True,
):
    rdbe_mult[element_to_ind[element]] = weight


def standardize_adduct(adduct: str) -> str:
    """Map an adduct spelling onto an ``ION_LST`` entry."""

    if adduct is None or (isinstance(adduct, float) and np.isnan(adduct)):
        raise ValueError("adduct is missing")
    canonical = ion_remap.get(str(adduct).replace(" ", ""), str(adduct).replace(" ", ""))
    if canonical not in ion_to_idx:
        raise ValueError(f"unsupported adduct {adduct!r}")
    return canonical


def get_ion_idx(ionization: str) -> int:
    return ion_to_idx[standardize_adduct(ionization)]


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
