import torch

from mist.models.scaled_mist import ScaledMIST, count_parameters, load_mist_weights
from mist.utils.chem_utils import (
    LEGACY_ADDUCT_DIM,
    LEGACY_INSTR_DIM,
    get_instr_idx,
    get_ion_idx,
    ion_to_idx,
)


def test_published_recipe_keeps_the_released_parameter_count():
    kwargs = dict(
        hidden_size=1024,
        peak_attn_layers=4,
        num_heads=16,
        refine_layers=4,
        spectra_dropout=0.1,
        pairwise_featurization=True,
        no_diffs=True,
        form_embedder="pos-cos",
        output_size=4096,
    )
    assert count_parameters(ScaledMIST(**kwargs)) == 72_296_996
    assert count_parameters(ScaledMIST(**kwargs, extended_vocabulary=True)) == 72_308_324


def test_released_adduct_and_instrument_indices_stay_put():
    released = [
        "[M+H]+",
        "[M+Na]+",
        "[M+K]+",
        "[M-H2O+H]+",
        "[M+H3N+H]+",
        "[M]+",
        "[M-H4O2+H]+",
    ]
    assert [ion_to_idx[ion] for ion in released] == list(range(len(released)))
    assert get_ion_idx("[M+NH4]+") == ion_to_idx["[M+H3N+H]+"]
    assert get_ion_idx("[M+H-2H2O]+") == ion_to_idx["[M-H4O2+H]+"]
    assert get_ion_idx("[M-H]-") == ion_to_idx["[M-H]-"]
    assert get_ion_idx("[M+CHO2]-") == ion_to_idx["[M+CHO2]-"]
    assert get_ion_idx("[M+Cl]-") == ion_to_idx["[M+Cl]-"]
    assert get_ion_idx("[M-H2O-H]-") == ion_to_idx["[M-H2O-H]-"]
    assert get_instr_idx("Orbitrap") == 2
    assert get_instr_idx("QTOF") == 3
    assert get_instr_idx("Unknown") == 4
    assert get_instr_idx("timsTOF") == 8


def test_legacy_checkpoint_matches_the_original_vocabulary():
    torch.manual_seed(0)
    kwargs = {
        "hidden_size": 32,
        "num_heads": 4,
        "peak_attn_layers": 1,
        "refine_layers": 1,
        "embed_instrument": True,
        "extended_vocabulary": True,
    }
    reference = ScaledMIST(**kwargs).eval()
    encoder = reference.encoder
    prefix = encoder.formula_dim * 2 + encoder.num_types
    legacy_width = prefix + LEGACY_ADDUCT_DIM + encoder.inten_feats + LEGACY_INSTR_DIM
    new_inten = prefix + encoder.adduct_dim
    new_instr = new_inten + encoder.inten_feats
    with torch.no_grad():
        weight = encoder.intermediate_layer.input_layer.weight
        weight[:, prefix + LEGACY_ADDUCT_DIM : new_inten] = 0
        weight[:, new_instr + LEGACY_INSTR_DIM :] = 0
        shrunk = weight.new_empty(weight.shape[0], legacy_width)
        shrunk[:, : prefix + LEGACY_ADDUCT_DIM] = weight[:, : prefix + LEGACY_ADDUCT_DIM]
        old_inten = prefix + LEGACY_ADDUCT_DIM
        shrunk[:, old_inten : old_inten + encoder.inten_feats] = weight[
            :, new_inten : new_inten + encoder.inten_feats
        ]
        old_instr = old_inten + encoder.inten_feats
        shrunk[:, old_instr : old_instr + LEGACY_INSTR_DIM] = weight[
            :, new_instr : new_instr + LEGACY_INSTR_DIM
        ]

    legacy = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    legacy["encoder.intermediate_layer.input_layer.weight"] = shrunk
    embed = encoder.instrument_embedder.detach()
    legacy["encoder.instrument_embedder"] = embed[
        :LEGACY_INSTR_DIM, :LEGACY_INSTR_DIM
    ].clone()

    restored = ScaledMIST(**kwargs).eval()
    load_mist_weights(restored, legacy)
    peaks = 3
    form = torch.zeros(1, peaks, 18)
    form[0, :, 0] = 2
    form[0, :, 1] = 6
    batch = {
        "form_vec": form,
        "types": torch.tensor([[0, 0, 3]]),
        "ion_vec": torch.tensor([[0, 1, 0]]),
        "intens": torch.tensor([[0.4, 0.2, 1.0]]),
        "instruments": torch.tensor([get_instr_idx("Orbitrap")]),
        "num_peaks": torch.tensor([peaks]),
    }
    with torch.no_grad():
        original, _ = reference(batch)
        loaded, _ = restored(batch)
    torch.testing.assert_close(original, loaded, rtol=0.0, atol=0.0)
