"""Read-only adapters for the compact GLACIER PubChem Atlas.

The Atlas stores a heavy-atom mask and an exact GLACIER m/z for every peak.
GLACIER constructs each peak mass from the selected atoms (including the
hydrogens attached to each atom), an integer hydrogen shift, and either the
spectrum adduct or an electron-only ``[M]+`` shift.  The resulting MIST mass
representation can be recovered without changing the Atlas.  For protonated
spectra, ``formula + [M+H]+`` is mass-identical to ``formula+H + [M]+``; we use
the former as a deterministic canonical representation whenever it is valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


SCHEMA_VERSION = 1
FORM_ELEMENTS = (
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
)
ELEMENT_TO_INDEX = {element: index for index, element in enumerate(FORM_ELEMENTS)}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")
ELECTRON_MASS = 0.00054858
M_PLUS_ION_INDEX = 5
ROOT_ION_NAMES = ("[M+H]+", "[M+Na]+")

_PERIODIC_TABLE = Chem.GetPeriodicTable()
MONO_MASSES = np.asarray(
    [_PERIODIC_TABLE.GetMostCommonIsotopeMass(element) for element in FORM_ELEMENTS],
    dtype=np.float64,
)
H_MASS = float(MONO_MASSES[ELEMENT_TO_INDEX["H"]])
ROOT_ION_SHIFTS = np.asarray(
    (
        H_MASS - ELECTRON_MASS,
        float(MONO_MASSES[ELEMENT_TO_INDEX["Na"]]) - ELECTRON_MASS,
    ),
    dtype=np.float64,
)
M_PLUS_SHIFT = -ELECTRON_MASS

_MORGAN_4096 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


class AtlasFeatureError(ValueError):
    """Raised when a row cannot be represented by the MIST feature contract."""


@dataclass(frozen=True)
class RecoveredSpectrum:
    """Fixed-width MIST features recovered for one Atlas spectrum."""

    form_vec: np.ndarray
    ion_vec: np.ndarray
    intens: np.ndarray
    peak_type: np.ndarray
    length: int
    max_mass_error: float
    dropped_unrepresentable_peaks: int


def parse_formula(formula: str) -> np.ndarray:
    """Parse a molecular formula into the canonical 18-element MIST vector."""

    matches = list(FORMULA_PATTERN.finditer(formula))
    if not matches or "".join(match.group(0) for match in matches) != formula:
        raise AtlasFeatureError(f"invalid molecular formula: {formula!r}")
    output = np.zeros(len(FORM_ELEMENTS), dtype=np.int16)
    for match in matches:
        element, count_text = match.groups()
        if element not in ELEMENT_TO_INDEX:
            raise AtlasFeatureError(f"unsupported element {element!r} in {formula!r}")
        count = 1 if count_text == "" else int(count_text)
        if count > np.iinfo(np.int16).max:
            raise AtlasFeatureError(f"formula count overflows int16: {formula!r}")
        output[ELEMENT_TO_INDEX[element]] += count
    return output


@lru_cache(maxsize=4096)
def _molecule_atom_features(smiles: str) -> tuple[Chem.Mol, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AtlasFeatureError(f"RDKit could not parse SMILES: {smiles!r}")
    num_atoms = mol.GetNumAtoms()
    if num_atoms > 128:
        raise AtlasFeatureError(f"Atlas mask supports at most 128 atoms, got {num_atoms}")

    atom_forms = np.zeros((num_atoms, len(FORM_ELEMENTS)), dtype=np.int16)
    for atom_index, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        element_index = ELEMENT_TO_INDEX.get(symbol)
        if element_index is None:
            raise AtlasFeatureError(f"unsupported atom {symbol!r} in {smiles!r}")
        atom_forms[atom_index, element_index] = 1
        atom_forms[atom_index, ELEMENT_TO_INDEX["H"]] = atom.GetTotalNumHs(
            includeNeighbors=True
        )
    atom_masses = atom_forms.astype(np.float64) @ MONO_MASSES
    return mol, atom_forms, atom_masses


def unpack_fragment_masks(mask_words: np.ndarray, num_atoms: int) -> np.ndarray:
    """Decode two little-endian uint64 words into ``[peaks, atoms]`` booleans."""

    words = np.asarray(mask_words, dtype=np.uint64)
    if words.ndim != 2 or words.shape[1] != 2:
        raise AtlasFeatureError(f"expected [num_peaks, 2] masks, got {words.shape}")
    bit_positions = np.arange(64, dtype=np.uint64)
    low = ((words[:, 0, None] >> bit_positions[None, :]) & 1).astype(bool)
    high = ((words[:, 1, None] >> bit_positions[None, :]) & 1).astype(bool)
    return np.concatenate((low, high), axis=1)[:, :num_atoms]


def _recover_peak_formula_candidates(
    smiles: str,
    root_ion_index: int,
    peak_mz: np.ndarray,
    mask_words: np.ndarray,
    *,
    mass_tolerance_da: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if root_ion_index not in (0, 1):
        raise AtlasFeatureError(f"Atlas root ion index must be 0 or 1, got {root_ion_index}")
    mz = np.asarray(peak_mz, dtype=np.float64)
    if mz.ndim != 1:
        raise AtlasFeatureError(f"peak_mz must be one-dimensional, got {mz.shape}")
    if len(mz) != len(mask_words):
        raise AtlasFeatureError("peak m/z and fragment-mask lengths differ")
    if len(mz) == 0:
        return (
            np.empty((0, len(FORM_ELEMENTS)), dtype=np.int16),
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=bool),
        )

    _, atom_forms, atom_masses = _molecule_atom_features(smiles)
    membership = unpack_fragment_masks(mask_words, len(atom_forms))
    if np.any(~membership.any(axis=1)):
        raise AtlasFeatureError("encountered an empty fragment mask")

    base_forms = membership.astype(np.int16) @ atom_forms
    base_masses = membership.astype(np.float64) @ atom_masses
    shifts = np.asarray((ROOT_ION_SHIFTS[root_ion_index], M_PLUS_SHIFT))
    hydrogen_delta = np.rint((mz[:, None] - base_masses[:, None] - shifts) / H_MASS).astype(
        np.int16
    )
    residual = np.abs(
        mz[:, None]
        - base_masses[:, None]
        - shifts[None, :]
        - hydrogen_delta.astype(np.float64) * H_MASS
    )

    h_index = ELEMENT_TO_INDEX["H"]
    final_h = base_forms[:, h_index, None] + hydrogen_delta
    residual[final_h < 0] = np.inf
    if root_ion_index == 0:
        # [M+H]+ differs from [M]+ by exactly one hydrogen mass.  m/z alone
        # cannot recover the generator's mode bit, so prefer a stable root-ion
        # representation wherever its formula has a nonnegative H count.
        residual[final_h[:, 0] >= 0, 1] = np.inf
    mode = residual.argmin(axis=1)
    row = np.arange(len(mz))
    selected_error = residual[row, mode]
    valid = np.isfinite(selected_error) & (selected_error <= mass_tolerance_da)

    output = base_forms.copy()
    output[:, h_index] += hydrogen_delta[row, mode]
    ion_vec = np.where(mode == 0, root_ion_index, M_PLUS_ION_INDEX).astype(np.uint8)
    return output, ion_vec, selected_error, valid


def recover_peak_formula_vectors(
    smiles: str,
    root_ion_index: int,
    peak_mz: np.ndarray,
    mask_words: np.ndarray,
    *,
    mass_tolerance_da: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover valid peak formulas, failing if any requested peak is invalid.

    For each peak we test the two GLACIER mass families: the spectrum's root
    adduct and electron-only ``[M]+``.  The integer hydrogen shift is rounded
    analytically and the lower-residual valid formula is selected.  The exact
    proton/``[M]+`` degeneracy is canonicalized to the proton representation.
    """

    output, ion_vec, selected_error, valid = _recover_peak_formula_candidates(
        smiles,
        root_ion_index,
        peak_mz,
        mask_words,
        mass_tolerance_da=mass_tolerance_da,
    )
    if not np.all(valid):
        worst = int(np.argmax(selected_error))
        mz = np.asarray(peak_mz)
        raise AtlasFeatureError(
            "could not invert GLACIER mass construction: "
            f"peak={worst} mz={mz[worst]:.8f} residual={selected_error[worst]:.6g}"
        )

    return output, ion_vec, selected_error


def recover_spectrum(
    *,
    smiles: str,
    root_formula: str,
    root_ion_index: int,
    peak_mz: np.ndarray,
    intensities: np.ndarray,
    mask_words: np.ndarray,
    max_peaks: int = 10,
    mass_tolerance_da: float = 2e-3,
) -> RecoveredSpectrum:
    """Convert one Atlas row into fixed-width MIST inputs including the CLS row."""

    if max_peaks < 1:
        raise ValueError("max_peaks must be positive")
    mz = np.asarray(peak_mz)
    intens = np.asarray(intensities, dtype=np.float32)
    masks = np.asarray(mask_words)
    if not (len(mz) == len(intens) == len(masks)):
        raise AtlasFeatureError("peak arrays have different lengths")
    if np.any(~np.isfinite(mz)) or np.any(~np.isfinite(intens)):
        raise AtlasFeatureError("non-finite peak value")

    order = np.argsort(-intens, kind="stable")[:max_peaks]
    dropped_unrepresentable = 0
    try:
        selected_forms, selected_ions, errors = recover_peak_formula_vectors(
            smiles,
            root_ion_index,
            mz[order],
            masks[order],
            mass_tolerance_da=mass_tolerance_da,
        )
    except AtlasFeatureError:
        all_forms, all_ions, all_errors, valid = _recover_peak_formula_candidates(
            smiles,
            root_ion_index,
            mz,
            masks,
            mass_tolerance_da=mass_tolerance_da,
        )
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            raise AtlasFeatureError("spectrum has no nonnegative representable peak formula")
        order = valid_indices[
            np.argsort(-intens[valid_indices], kind="stable")[:max_peaks]
        ]
        selected_forms = all_forms[order]
        selected_ions = all_ions[order]
        errors = all_errors[order]
        dropped_unrepresentable = int(np.count_nonzero(~valid))
    num_fragments = len(order)
    width = max_peaks + 1
    forms = np.zeros((width, len(FORM_ELEMENTS)), dtype=np.int16)
    ions = np.zeros(width, dtype=np.uint8)
    output_intens = np.zeros(width, dtype=np.float32)
    peak_type = np.zeros(width, dtype=np.uint8)
    if num_fragments:
        forms[:num_fragments] = selected_forms
        ions[:num_fragments] = selected_ions
        output_intens[:num_fragments] = intens[order]

    cls_index = num_fragments
    forms[cls_index] = parse_formula(root_formula)
    ions[cls_index] = root_ion_index
    output_intens[cls_index] = 1.0
    peak_type[cls_index] = 3
    return RecoveredSpectrum(
        form_vec=forms,
        ion_vec=ions,
        intens=output_intens,
        peak_type=peak_type,
        length=cls_index + 1,
        max_mass_error=float(errors.max()) if len(errors) else 0.0,
        dropped_unrepresentable_peaks=dropped_unrepresentable,
    )


def morgan4096_packed(mol_or_smiles: Chem.Mol | str) -> np.ndarray:
    """Return a radius-2 Morgan fingerprint packed into 512 little-bit-order bytes."""

    mol = Chem.MolFromSmiles(mol_or_smiles) if isinstance(mol_or_smiles, str) else mol_or_smiles
    if mol is None:
        raise AtlasFeatureError("cannot fingerprint an invalid molecule")
    bit_vector = _MORGAN_4096.GetFingerprint(mol)
    dense = np.zeros(4096, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, dense)
    return np.packbits(dense, bitorder="little")


def unpack_fingerprints(packed: np.ndarray) -> np.ndarray:
    """Unpack one or more 4096-bit fingerprints to float32 training targets."""

    values = np.asarray(packed, dtype=np.uint8)
    if values.shape[-1] != 512:
        raise ValueError(f"expected 512 packed bytes, got {values.shape}")
    return np.unpackbits(values, axis=-1, count=4096, bitorder="little").astype(np.float32)


def compatible_formula(formula: str) -> bool:
    """Return whether a formula obeys the fixed MIST element vocabulary."""

    try:
        parse_formula(formula)
    except AtlasFeatureError:
        return False
    return True


def decode_packed_strings(
    byte_values: np.ndarray, offsets: np.ndarray, indices: Iterable[int]
) -> list[str]:
    """Decode selected strings from Atlas byte/offset storage."""

    raw = memoryview(np.asarray(byte_values, dtype=np.uint8)).cast("B")
    off = np.asarray(offsets, dtype=np.uint64)
    return [bytes(raw[int(off[i]) : int(off[i + 1])]).decode("utf-8") for i in indices]
