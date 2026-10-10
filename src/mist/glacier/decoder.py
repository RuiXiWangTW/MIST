"""Intensity decoder layers for GLACIER checkpoint inference."""

import torch
from torch import nn
import torch.nn.functional as F


class _MembershipDecoderLayer(nn.Module):
    """Post-norm transformer decoder layer with membership-factorized cross-attention.

    Mathematically equivalent to a standard decoder layer run on a per-fragment "tagged"
    memory ``memory + group_embed[membership]`` (membership one of in-fragment / complement
    / root token), but WITHOUT materializing that ``[B, F, N, H]`` memory or projecting K/V
    per fragment.

    Linearity of the K/V projections is exploited: for a fragment ``f`` and atom ``a`` with
    membership ``g(f,a)``,
        ``K[f,a] = W_k (mem[a] + group[g]) + b_k = (W_k mem[a] + b_k) + W_k group[g]``
    so the base term ``W_k mem[a] + b_k`` is projected ONCE from the shared ``[B, N, H]``
    memory and the membership only adds the (2-vector) ``W_k group[g]`` tag. Same for V.
    This drops the ``F`` (== max_frags) factor from the cross-attention K/V footprint, so
    this branch costs ~the same as the masked decoder instead of ``max_frags`` times more.
    """

    def __init__(
        self, d_model, nhead, dim_feedforward, dropout, self_attn_mode="diagonal"
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        if self_attn_mode not in ("cross_fragment", "diagonal", "none"):
            raise ValueError(
                f"self_attn_mode must be one of 'cross_fragment', 'diagonal', 'none'; got {self_attn_mode!r}"
            )
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** (-0.5)
        self.attn_dropout_p = dropout
        self.self_attn_mode = self_attn_mode
        if self_attn_mode == "none":
            self.norm_in = nn.LayerNorm(d_model)
        else:
            if self_attn_mode == "diagonal":
                self.self_transform = nn.Linear(d_model, d_model)
            else:
                self.self_attn = nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                )
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self, tgt, memory, group_embed, membership, atom_pad, tgt_key_padding_mask
    ):
        """tgt [B,F,H]; memory [B,N,H]; group_embed [G,H] (row 0 = in-fragment tag, row 1 =
        complement tag, row 2 = root-token tag); membership [B,F,N] long (0 = in-fragment
        atom, 1 = complement atom, 2 = root token) selecting each memory position's tag;
        atom_pad [B,N] bool (True == padding atom); tgt_key_padding_mask [B,F] bool (True ==
        padding)."""
        B, Fq, H = tgt.shape
        N = memory.shape[1]
        G = group_embed.shape[0]
        nh, hd = (self.nhead, self.head_dim)
        if self.self_attn_mode == "none":
            x = self.norm_in(tgt)
        else:
            if self.self_attn_mode == "cross_fragment":
                sa = self.self_attn(
                    tgt,
                    tgt,
                    tgt,
                    key_padding_mask=tgt_key_padding_mask,
                    need_weights=False,
                )[0]
            else:
                sa = self.self_transform(tgt)
            x = self.norm1(tgt + self.dropout1(sa))
        q = self.q_proj(x).view(B, Fq, nh, hd).transpose(1, 2)
        k_base = self.k_proj(memory).view(B, N, nh, hd).transpose(1, 2)
        v_base = self.v_proj(memory).view(B, N, nh, hd).transpose(1, 2)
        k_tag = F.linear(group_embed, self.k_proj.weight).view(G, nh, hd)
        v_tag = F.linear(group_embed, self.v_proj.weight).view(G, nh, hd)
        base = torch.einsum("bhfd,bhnd->bhfn", q, k_base)
        q_kt = torch.einsum("bhfd,ghd->gbhf", q, k_tag)
        group_onehot = (
            F.one_hot(membership, num_classes=G).permute(3, 0, 1, 2).to(q.dtype)
        )
        tag_logit = torch.einsum("gbhf,gbfn->bhfn", q_kt, group_onehot)
        logits = (base + tag_logit) * self.scale
        logits = logits.masked_fill(atom_pad[:, None, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout_p, training=self.training)
        ctx = torch.einsum("bhfn,bhnd->bhfd", attn, v_base)
        attn_g = torch.einsum("bhfn,gbfn->gbhf", attn, group_onehot)
        ctx = ctx + torch.einsum("gbhf,ghd->bhfd", attn_g, v_tag)
        ctx = ctx.transpose(1, 2).reshape(B, Fq, H)
        ctx = self.out_proj(ctx)
        x = self.norm2(x + self.dropout2(ctx))
        ff = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(ff))
        return x


class MembershipIntenDecoder(nn.Module):
    """Stack of membership-factorized decoder layers (drop-in intensity decoder).

    Each fragment query attends to every atom while a learned in-fragment vs complement
    tag is folded into the keys/values -- equivalent to tagging the memory then running a
    standard decoder, but without the ``max_frags`` memory blowup (see
    ``_MembershipDecoderLayer``).
    """

    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        self_attn_mode="diagonal",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _MembershipDecoderLayer(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    self_attn_mode=self_attn_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, tgt, memory, group_embed, membership, atom_pad, tgt_key_padding_mask
    ):
        x = tgt
        for layer in self.layers:
            x = layer(
                x, memory, group_embed, membership, atom_pad, tgt_key_padding_mask
            )
        return x


class _CrossAttnDecoderLayer(nn.Module):
    """Post-norm decoder layer WITHOUT self-attention: cross-attention + feed-forward.

    Mirrors ``nn.TransformerDecoderLayer`` minus the self-attention sub-layer, so fragment
    queries are updated purely from the (masked) memory and the per-fragment feed-forward.
    Used by :class:`CrossAttnDecoder` to realize the ``self_attn_mode="none"`` simplification
    on the hard-masked (non-membership) intensity-decoder path. A single input LayerNorm is
    applied to the queries to keep the cross-attention input distribution stable.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        x = self.norm_in(tgt)
        ca = self.cross_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        x = self.norm_cross(x + self.dropout_cross(ca))
        ff = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        x = self.norm_ff(x + self.dropout_ff(ff))
        return x


class CrossAttnDecoder(nn.Module):
    """Stack of cross-attention-only decoder layers (no fragment self-attention).

    Drop-in replacement for the ``nn.TransformerDecoder`` used by the hard-masked intensity
    decoder when ``self_attn_mode="none"``: each fragment attends only to its own atoms
    (via ``memory_mask``) and is otherwise transformed per-fragment, so fragments never
    exchange information.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _CrossAttnDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x


class _DiagonalDecoderLayer(nn.Module):
    """Post-norm decoder layer whose "self-attention" is an equivalent per-fragment linear map.

    Diagonal self-attention -- each fragment attending only to itself -- is a degenerate
    attention whose output reduces to a single affine map of the query token,
    ``out_proj(W_v x_f)`` (the softmax over one key is always 1, so the query/key projections
    drop out). We therefore implement it directly with one ``nn.Linear`` instead of running a
    diagonal-masked attention, which is wasteful and would also produce NaN rows for padded
    fragments. The rest of the layer matches ``nn.TransformerDecoderLayer`` (post-norm, ReLU
    feed-forward): cross-attention to the (masked) atom memory, then a feed-forward block.
    Fragments never exchange information.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_transform = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        sa = self.self_transform(tgt)
        x = self.norm1(tgt + self.dropout1(sa))
        ca = self.cross_attn(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        x = self.norm2(x + self.dropout2(ca))
        ff = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(ff))
        return x


class DiagonalIntenDecoder(nn.Module):
    """Intensity decoder equivalent to diagonal self-attention, without the degenerate attention.

    Equivalent to a standard decoder with diagonal-masked self-attention (each fragment
    attends only to itself) but implemented without running that degenerate attention: the
    diagonal self-attention is replaced by an equivalent per-fragment linear map (see
    :class:`_DiagonalDecoderLayer`). Each fragment cross-attends only to its own atoms via
    ``memory_mask``, so fragments never exchange information. Each layer has its own weights.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DiagonalDecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x
