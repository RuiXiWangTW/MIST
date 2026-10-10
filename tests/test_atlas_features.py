import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from mist.data.atlas import (
    ELEMENT_TO_INDEX,
    H_MASS,
    M_PLUS_SHIFT,
    MONO_MASSES,
    ROOT_ION_SHIFTS,
    morgan4096_packed,
    parse_formula,
    recover_spectrum,
    unpack_fingerprints,
)


def test_formula_vocabulary_order_is_locked():
    vector = parse_formula("C2H6ClNO")
    assert vector.dtype == np.int16
    assert vector[ELEMENT_TO_INDEX["C"]] == 2
    assert vector[ELEMENT_TO_INDEX["H"]] == 6
    assert vector[ELEMENT_TO_INDEX["Cl"]] == 1
    assert vector[ELEMENT_TO_INDEX["N"]] == 1
    assert vector[ELEMENT_TO_INDEX["O"]] == 1
    assert vector.sum() == 11


def test_glacier_mass_families_are_inverted_exactly():
    # RDKit atom order for CCO is C, C, O, with implicit H counts 3, 2, 1.
    masks = np.asarray([[0b011, 0], [0b100, 0]], dtype=np.uint64)
    c2h5 = parse_formula("C2H5").astype(np.float64) @ MONO_MASSES
    oh = parse_formula("HO").astype(np.float64) @ MONO_MASSES
    mz = np.asarray(
        [
            c2h5 + ROOT_ION_SHIFTS[0] - H_MASS,
            oh + M_PLUS_SHIFT + 2 * H_MASS,
        ],
        dtype=np.float64,
    )
    recovered = recover_spectrum(
        smiles="CCO",
        root_formula="C2H6O",
        root_ion_index=0,
        peak_mz=mz,
        intensities=np.asarray([0.4, 0.9]),
        mask_words=masks,
        max_peaks=10,
        mass_tolerance_da=1e-8,
    )
    # Intensity sorting puts the mass-equivalent OH2 + [M+H]+ peak first.
    assert np.array_equal(recovered.form_vec[0], parse_formula("H2O"))
    assert recovered.ion_vec[0] == 0
    assert np.array_equal(recovered.form_vec[1], parse_formula("C2H4"))
    assert recovered.ion_vec[1] == 0
    assert recovered.peak_type[2] == 3
    assert np.array_equal(recovered.form_vec[2], parse_formula("C2H6O"))
    assert recovered.length == 3
    assert recovered.max_mass_error < 1e-10
    assert recovered.dropped_unrepresentable_peaks == 0

    sodium_spectrum = recover_spectrum(
        smiles="CCO",
        root_formula="C2H6O",
        root_ion_index=1,
        peak_mz=np.asarray([oh + M_PLUS_SHIFT + 2 * H_MASS]),
        intensities=np.asarray([1.0]),
        mask_words=masks[1:],
        max_peaks=10,
        mass_tolerance_da=1e-8,
    )
    assert np.array_equal(sodium_spectrum.form_vec[0], parse_formula("H3O"))
    assert sodium_spectrum.ion_vec[0] == 5


def test_packed_morgan_matches_rdkit_dense_bits():
    smiles = "CC(=O)N[C@@H](C)C(=O)O"
    packed = morgan4096_packed(smiles)
    unpacked = unpack_fingerprints(packed)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(smiles))
    expected = np.zeros(4096, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, expected)
    assert packed.shape == (512,)
    assert np.array_equal(unpacked, expected)


def test_unphysical_negative_hydrogen_peak_is_dropped_not_the_spectrum():
    base_mass = parse_formula("AsSe").astype(np.float64) @ MONO_MASSES
    masks = np.asarray([[0b11, 0], [0b11, 0]], dtype=np.uint64)
    recovered = recover_spectrum(
        smiles="[As]=[Se]",
        root_formula="AsSe",
        root_ion_index=0,
        peak_mz=np.asarray(
            [base_mass + ROOT_ION_SHIFTS[0] - 7 * H_MASS, base_mass + M_PLUS_SHIFT]
        ),
        intensities=np.asarray([1.0, 0.5]),
        mask_words=masks,
        max_peaks=10,
        mass_tolerance_da=1e-8,
    )
    assert recovered.dropped_unrepresentable_peaks == 1
    assert recovered.length == 2
    assert np.array_equal(recovered.form_vec[0], parse_formula("AsSe"))
    assert recovered.ion_vec[0] == 5
