"""A lean, explicitly constructed MIST encoder for native PyTorch training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mist.models.modules import FormulaTransformer, FPGrowingModule
from mist.utils.chem_utils import (
    ION_LST,
    LEGACY_ADDUCT_DIM,
    LEGACY_INSTR_DIM,
    max_instr_idx,
)


class ScaledMIST(nn.Module):
    """Formula Transformer plus iterative fingerprint unfolding.

    This preserves the two strongest MIST inductive biases—formula attention
    and coarse-to-fine fingerprint prediction—while omitting the unused MAGMa
    fragment head from the historical training graph.
    """

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        peak_attn_layers: int = 3,
        num_heads: int = 12,
        refine_layers: int = 4,
        spectra_dropout: float = 0.1,
        pairwise_featurization: bool = True,
        no_diffs: bool = True,
        form_embedder: str = "pos-cos",
        output_size: int = 4096,
        set_pooling: str = "cls",
        embed_instrument: bool = False,
        extended_vocabulary: bool = False,
        **unused: Any,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if output_size != 4096:
            raise ValueError("MIST predicts a 4096-bit Morgan fingerprint")
        if refine_layers < 0:
            raise ValueError("refine_layers must be nonnegative")
        self.output_size = int(output_size)
        self.refine_layers = int(refine_layers)
        self.encoder = FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            num_heads=num_heads,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            no_diffs=no_diffs,
            form_embedder=form_embedder,
            set_pooling=set_pooling,
            embed_instrument=embed_instrument,
            inten_transform="float",
            output_size=output_size,
            adduct_dim=(len(ION_LST) + 1 if extended_vocabulary else LEGACY_ADDUCT_DIM),
            instr_dim=(max_instr_idx if extended_vocabulary else LEGACY_INSTR_DIM),
        )
        self.decoder = FPGrowingModule(
            hidden_input_dim=hidden_size,
            final_target_dim=output_size,
            num_splits=refine_layers,
            reduce_factor=2,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        encoded = self.encoder(batch)
        predictions = self.decoder(encoded)
        return predictions[-1], {
            "intermediate_predictions": predictions[:-1],
            "prediction_dims": self.decoder.output_dims,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _expand_legacy_vocabulary(
    state: Mapping[str, torch.Tensor], model: nn.Module
) -> dict[str, torch.Tensor]:
    """Place a released 7-adduct / 5-instrument checkpoint into the wider input.

    New adduct and instrument columns stay zero, so spectra that use the original
    vocabulary produce the same fingerprint. A state that already matches the
    current module is returned unchanged.
    """

    encoder = model.encoder
    current = model.state_dict()
    expanded = dict(state)
    prefix = encoder.formula_dim * 2 + encoder.num_types
    legacy_width = prefix + LEGACY_ADDUCT_DIM + encoder.inten_feats + LEGACY_INSTR_DIM
    weight_key = "encoder.intermediate_layer.input_layer.weight"
    weight = expanded.get(weight_key)
    target = current.get(weight_key)
    if (
        torch.is_tensor(weight)
        and torch.is_tensor(target)
        and weight.shape != target.shape
        and weight.shape[0] == target.shape[0]
        and weight.shape[1] == legacy_width
    ):
        new_weight = weight.new_zeros(target.shape)
        new_weight[:, :prefix] = weight[:, :prefix]
        new_weight[:, prefix : prefix + LEGACY_ADDUCT_DIM] = weight[
            :, prefix : prefix + LEGACY_ADDUCT_DIM
        ]
        old_inten = prefix + LEGACY_ADDUCT_DIM
        new_inten = prefix + encoder.adduct_dim
        new_weight[:, new_inten : new_inten + encoder.inten_feats] = weight[
            :, old_inten : old_inten + encoder.inten_feats
        ]
        old_instr = old_inten + encoder.inten_feats
        new_instr = new_inten + encoder.inten_feats
        new_weight[:, new_instr : new_instr + LEGACY_INSTR_DIM] = weight[
            :, old_instr : old_instr + LEGACY_INSTR_DIM
        ]
        expanded[weight_key] = new_weight

    embed_key = "encoder.instrument_embedder"
    embed = expanded.get(embed_key)
    target_embed = current.get(embed_key)
    if (
        torch.is_tensor(embed)
        and torch.is_tensor(target_embed)
        and embed.shape == (LEGACY_INSTR_DIM, LEGACY_INSTR_DIM)
        and target_embed.shape[0] == target_embed.shape[1]
        and target_embed.shape[0] > LEGACY_INSTR_DIM
    ):
        new_embed = torch.eye(target_embed.shape[0], dtype=embed.dtype)
        new_embed[:LEGACY_INSTR_DIM, :LEGACY_INSTR_DIM] = embed
        expanded[embed_key] = new_embed
    return expanded


def load_mist_weights(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load MIST weights, expanding released checkpoints into the current vocabulary."""

    model.load_state_dict(_expand_legacy_vocabulary(state, model), strict=True)
