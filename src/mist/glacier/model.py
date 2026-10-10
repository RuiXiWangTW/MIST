"""Inference architecture for GLACIER checkpoints with membership attention.

Derived from the MIT-licensed ms-pred implementation. See NOTICE.md.
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation
import ms_pred.nn_utils as nn_utils
from ms_pred.graphormer.graphormer_graph_encoder import GraphormerGraphEncoder
from .features import TreeProcessor
from .decoder import MembershipIntenDecoder, CrossAttnDecoder, DiagonalIntenDecoder


class CheckpointGLACIER(nn.Module):
    def __init__(
        self,
        frag_decoder_layers: int = 3,
        frag_encoder_layers: int = 0,
        frag_dropout: float = 0.1,
        node_feats: int = 128,
        edge_feats: int = 12,
        multi_hop_max_dist: int = 5,
        num_edge_dis: int = 10,
        max_breakpoints: int = 40,
        embed_adduct: bool = False,
        embed_collision: bool = False,
        embed_elem_group: bool = False,
        embed_instrument: bool = False,
        encode_forms: bool = False,
        max_broken_bonds: int = 6,
        pe_embed_k: int = 0,
        enable_decoder_norm: bool = False,
        inten_decoder_layers: int = 3,
        inten_encoder_layers: int = 3,
        inten_dropout: float = 0,
        graphormer_dropout: float = 0.05,
        graphormer_layers: int = 5,
        hidden_size: int = 512,
        num_bins: int = 15000,
        upper_limit: int = 1500,
        inten_attend_full: bool = False,
        inten_self_attn_mode: str = "diagonal",
        inten_broken_bond_feats: bool = True,
        encoder_type: str = "graphormer",
        **kwargs,
    ):
        super().__init__()
        self.frag_decoder_layers = frag_decoder_layers
        self.frag_encoder_layers = frag_encoder_layers
        self.frag_dropout = frag_dropout
        self.node_feats = node_feats
        self.edge_feats = edge_feats
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_edge_dis = num_edge_dis
        self.max_breakpoints = max_breakpoints
        self.embed_adduct = embed_adduct
        self.embed_collision = embed_collision
        self.embed_elem_group = embed_elem_group
        self.encode_forms = encode_forms
        assert encode_forms, "JointModel requires encode_forms=True"
        self.embed_instrument = embed_instrument
        self.max_broken_bonds = max_broken_bonds
        self.pe_embed_k = pe_embed_k
        self.enable_decoder_norm = enable_decoder_norm
        self.inten_decoder_layers = inten_decoder_layers
        self.inten_encoder_layers = inten_encoder_layers
        self.inten_dropout = inten_dropout
        self.inten_broken_bond_feats = inten_broken_bond_feats
        self.graphormer_dropout = graphormer_dropout
        self.graphormer_layers = graphormer_layers
        self.hidden_size = hidden_size
        self.nhead = 8
        self.num_bins = num_bins
        self.upper_limit = upper_limit
        self.inten_attend_full = inten_attend_full
        if inten_self_attn_mode not in ("cross_fragment", "diagonal", "none"):
            raise ValueError(
                f"inten_self_attn_mode must be one of 'cross_fragment', 'diagonal', 'none'; got {inten_self_attn_mode!r}"
            )
        self.inten_self_attn_mode = inten_self_attn_mode
        if encoder_type != "graphormer":
            raise ValueError(
                f"only the Graphormer architecture is supported; got {encoder_type!r}"
            )
        self.encoder_type = encoder_type
        self.root_encode = "graphormer"
        self.tree_processor = TreeProcessor(
            pe_embed_k=pe_embed_k,
            root_encode=self.root_encode,
            embed_elem_group=embed_elem_group,
            multi_hop_max_dist=multi_hop_max_dist,
        )
        adduct_shift = 0
        if self.embed_adduct:
            adduct_types = len(set(common.ion2onehot_pos.values()))
            onehot_types = torch.eye(adduct_types)
            if self.embed_elem_group:
                adduct_modes = len(
                    set([j for i in common.ion_pos2extra_multihot.values() for j in i])
                )
                multihot_modes = torch.zeros((adduct_types, adduct_modes))
                for i in range(adduct_types):
                    for j in common.ion_pos2extra_multihot[i]:
                        multihot_modes[i, j] = 1
                adduct_embedder = torch.cat((onehot_types, multihot_modes), dim=-1)
                self.adduct_embedder = nn.Parameter(adduct_embedder.float())
                self.adduct_embedder.requires_grad = False
                adduct_shift = adduct_types + adduct_modes
            else:
                self.adduct_embedder = nn.Parameter(onehot_types.float())
                self.adduct_embedder.requires_grad = False
                adduct_shift = adduct_types
        collision_shift = 0
        if self.embed_collision:
            pe_dim = common.COLLISION_PE_DIM
            pe_scalar = common.COLLISION_PE_SCALAR
            pe_power = 2 * torch.arange(pe_dim // 2) / pe_dim
            self.collision_embedder_denominators = nn.Parameter(
                torch.pow(pe_scalar, pe_power)
            )
            self.collision_embedder_denominators.requires_grad = False
            collision_shift = pe_dim
            self.collision_embed_merged = nn.Parameter(torch.zeros(pe_dim))
            self.collision_embed_merged.requires_grad = False
        instrument_shift = 0
        if self.embed_instrument:
            instrument_types = len(set(common.instrument2onehot_pos.values()))
            onehot_types = torch.eye(instrument_types)
            self.instrument_embedder = nn.Parameter(onehot_types.float())
            self.instrument_embedder.requires_grad = False
            instrument_shift = instrument_types
        encoder_node_feats = (
            node_feats + adduct_shift + collision_shift + instrument_shift
        )
        self.root_module = GraphormerGraphEncoder(
            num_atom_features=encoder_node_feats,
            num_degree=8,
            num_edge_features=edge_feats,
            num_spatial=1025,
            num_edge_dis=self.num_edge_dis,
            edge_type="multi_hop",
            multi_hop_max_dist=self.multi_hop_max_dist,
            num_encoder_layers=self.graphormer_layers,
            embedding_dim=self.hidden_size,
            ffn_embedding_dim=4 * self.hidden_size,
            num_attention_heads=self.nhead,
            dropout=self.graphormer_dropout,
            attention_dropout=self.graphormer_dropout,
            activation_dropout=self.graphormer_dropout,
            apply_graphormer_init=True,
        )
        self.formula_in_dim = 0
        if self.encode_forms:
            self.embedder = nn_utils.get_embedder("abs-sines")
            self.formula_dim = common.NORM_VEC.shape[0]
            self.formula_in_dim = self.formula_dim * self.embedder.num_dim
            self.formula_mapper = nn.Linear(
                self.formula_in_dim + self.hidden_size, self.hidden_size
            )
        self.inten_collision_dim = collision_shift
        self.broken_bond_feat_dim = (
            self.tree_processor.get_edge_feats() if self.inten_broken_bond_feats else 0
        )
        self.broken_bond_embed_dim = (
            self.broken_bond_feat_dim * self.embedder.num_dim
            if self.inten_broken_bond_feats
            else 0
        )
        token_size = (
            self.hidden_size
            + 2 * self.formula_in_dim
            + (self.max_broken_bonds + 1)
            + self.inten_collision_dim
            + self.broken_bond_embed_dim
        )
        self.token_mapper = nn.Linear(token_size, self.hidden_size)
        self.enable_decoder_norm = enable_decoder_norm
        self.fragment_decoder = nn_utils.SlotDecoder(
            hidden_dim=hidden_size,
            num_slots=max_breakpoints,
            nhead=self.nhead,
            num_layers=self.frag_decoder_layers,
            dropout=frag_dropout,
            enable_norm=self.enable_decoder_norm,
        )
        if self.frag_encoder_layers > 0:
            fragment_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=self.nhead,
                dim_feedforward=hidden_size * 4,
                dropout=frag_dropout,
                batch_first=True,
            )
            self.fragment_encoder = nn.TransformerEncoder(
                fragment_encoder_layer, num_layers=self.frag_encoder_layers
            )
        self.frag_card_mapper = nn.Linear(
            hidden_size, fragmentation.FRAGMENT_ENGINE_PARAMS["max_tree_depth"] + 1
        )
        if self.inten_attend_full:
            self.inten_group_embed = nn.Embedding(5, self.hidden_size)
            self.inten_decoder = MembershipIntenDecoder(
                d_model=self.hidden_size,
                nhead=self.nhead,
                num_layers=self.inten_decoder_layers,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.inten_dropout,
                self_attn_mode=self.inten_self_attn_mode,
            )
        elif self.inten_self_attn_mode == "none":
            self.inten_decoder = CrossAttnDecoder(
                d_model=self.hidden_size,
                nhead=self.nhead,
                num_layers=self.inten_decoder_layers,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.inten_dropout,
            )
        elif self.inten_self_attn_mode == "diagonal":
            self.inten_decoder = DiagonalIntenDecoder(
                d_model=self.hidden_size,
                nhead=self.nhead,
                num_layers=self.inten_decoder_layers,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.inten_dropout,
            )
        else:
            inten_decoder_layer = nn.TransformerDecoderLayer(
                self.hidden_size,
                nhead=self.nhead,
                batch_first=True,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.inten_dropout,
            )
            self.inten_decoder = nn.TransformerDecoder(
                inten_decoder_layer, self.inten_decoder_layers
            )
        if self.inten_encoder_layers > 0:
            inten_encoder_layer = nn.TransformerEncoderLayer(
                self.hidden_size,
                nhead=self.nhead,
                batch_first=True,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.inten_dropout,
            )
            self.inten_encoder = nn.TransformerEncoder(
                inten_encoder_layer, self.inten_encoder_layers
            )
        self.inten_activation = nn.Sigmoid()
        self.output_size = self.max_broken_bonds * 2 + 1
        self.output_map = nn.Linear(self.hidden_size, self.output_size * 2)

    def molecular_embedding(
        self,
        adducts,
        collision_engs,
        graphormer_input=None,
        instruments=None,
        root_graph=None,
    ):
        embed_adducts = self.adduct_embedder[adducts.long()]
        batch_size = collision_engs.shape[0]
        if graphormer_input is not None:
            original_node_features = node_features = graphormer_input["x"]
            max_nodes = node_features.shape[1]
            if self.embed_adduct:
                embed_adducts_expanded = embed_adducts.unsqueeze(1).expand(
                    batch_size, max_nodes, -1
                )
                node_features = torch.cat(
                    [node_features, embed_adducts_expanded], dim=-1
                )
            if self.embed_collision:
                embed_collision = torch.cat(
                    (
                        torch.sin(
                            collision_engs.unsqueeze(1)
                            / self.collision_embedder_denominators.unsqueeze(0)
                        ),
                        torch.cos(
                            collision_engs.unsqueeze(1)
                            / self.collision_embedder_denominators.unsqueeze(0)
                        ),
                    ),
                    dim=1,
                )
                embed_collision = torch.where(
                    torch.isnan(embed_collision),
                    self.collision_embed_merged.unsqueeze(0),
                    embed_collision,
                )
                embed_collision_expanded = embed_collision.unsqueeze(1).expand(
                    batch_size, max_nodes, -1
                )
                node_features = torch.cat(
                    [node_features, embed_collision_expanded], dim=-1
                )
            if self.embed_instrument:
                embed_instruments = self.instrument_embedder[instruments.long()]
                embed_instruments_expanded = embed_instruments.unsqueeze(1).expand(
                    batch_size, max_nodes, -1
                )
                node_features = torch.cat(
                    [node_features, embed_instruments_expanded], dim=-1
                )
            graphormer_input["x"] = node_features
            inner_states, graph_rep = self.root_module(graphormer_input)
            final_layer_output = inner_states[-1]
            node_embeddings = final_layer_output[1:].transpose(0, 1)
            root_tokens = graph_rep.unsqueeze(1)
            graphormer_input["x"] = original_node_features
        else:
            raise ValueError("graphormer_input is required")
        return {"root_tokens": root_tokens, "node_embeddings": node_embeddings}

    def _broken_bond_feature_matrix(self, graphormer_input, root_graph, num_atoms):
        return graphormer_input["attn_edge_type"]

    def breakpoint_forward(
        self, node_embeddings, root_tokens, num_atoms, root_form_vecs, last_only=False
    ):
        batch_size, max_nodes, _ = node_embeddings.shape
        device = node_embeddings.device
        node_mask = torch.arange(max_nodes, device=device).unsqueeze(
            0
        ) >= num_atoms.unsqueeze(1)
        frag_mask = F.pad(node_mask, (1, 0, 0, 0), mode="constant", value=0).bool()
        if self.encode_forms:
            encoded_form = self.embedder(root_form_vecs)[:, None, :]
            root_tokens = self.formula_mapper(
                torch.cat((root_tokens, encoded_form), dim=-1)
            )
            frag_vecs = self.fragment_decoder(
                root_tokens, node_embeddings, memory_key_padding_mask=frag_mask
            )
        n_layers = frag_vecs.shape[0]
        if self.frag_encoder_layers > 0:
            frag_vecs_flatten = frag_vecs.reshape(
                -1, self.max_breakpoints, self.hidden_size
            )
            frag_vecs_encoded = self.fragment_encoder(frag_vecs_flatten)
            frag_vecs = frag_vecs_encoded.reshape(
                n_layers, batch_size, self.max_breakpoints, self.hidden_size
            )
        frag_card_logits = self.frag_card_mapper(frag_vecs)
        frag_logits = torch.einsum("nbij,bkj->nbik", frag_vecs, node_embeddings)
        return {"frag_logits": frag_logits, "frag_card_logits": frag_card_logits}

    def inten_calculation(
        self,
        root_tokens,
        node_embeddings,
        frag_targs,
        num_frag_targs,
        root_form_vecs,
        atom_form_vecs,
        num_atoms,
        atom_hs,
        total_hs,
        adj_matrices,
        adduct_mass_shifts,
        masses,
        collision_engs=None,
        bond_feats=None,
    ):
        batch_size = root_tokens.shape[0]
        device = frag_targs.device
        atom_form_vecs_padded = nn_utils.pad_packed_tensor(atom_form_vecs, num_atoms, 0)
        frag_targs_padded = nn_utils.pad_packed_tensor(frag_targs, num_frag_targs, True)
        frag_to_mol = torch.arange(batch_size, device=device).repeat_interleave(
            num_frag_targs
        )
        expanded_atom_form_vecs = torch.repeat_interleave(
            atom_form_vecs_padded, num_frag_targs, dim=0
        )
        frag_masks = frag_targs.bool()
        mol_atom_hs = atom_hs[frag_to_mol]
        frag_hs = (frag_masks.float() * mol_atom_hs.float()).sum(dim=1)
        mol_total_hs = total_hs[frag_to_mol].float()
        max_remove = torch.clamp(frag_hs, max=self.max_broken_bonds)
        max_add = torch.clamp(mol_total_hs - frag_hs, max=self.max_broken_bonds)
        frag_mask_padded = nn_utils.pad_packed_tensor(
            frag_targs.float(), num_frag_targs, 0
        )
        node_degrees = adj_matrices.sum(dim=-1)
        adj_times_mask = torch.einsum("bij,bkj->bki", adj_matrices, frag_mask_padded)
        term_deg = (frag_mask_padded * node_degrees.unsqueeze(1)).sum(dim=-1)
        term_quad = (frag_mask_padded * adj_times_mask).sum(dim=-1)
        num_broken_padded = term_deg - term_quad
        max_add_padded = nn_utils.pad_packed_tensor(max_add, num_frag_targs, 0)
        max_remove_padded = nn_utils.pad_packed_tensor(max_remove, num_frag_targs, 0)
        masked_form_vecs = expanded_atom_form_vecs * frag_targs.unsqueeze(-1).float()
        fragment_form_vecs_flat = torch.sum(masked_form_vecs, dim=1)
        max_frags = frag_targs_padded.shape[1]
        fragment_form_vecs = nn_utils.pad_packed_tensor(
            fragment_form_vecs_flat, num_frag_targs, 0
        )
        root_tokens_expanded = root_tokens.expand(
            batch_size, max_frags, self.hidden_size
        )
        diffs = root_form_vecs[:, None, :] - fragment_form_vecs
        form_encodings = self.embedder(fragment_form_vecs)
        diff_encodings = self.embedder(diffs)
        num_broken_clamped = torch.clamp(
            num_broken_padded, max=self.max_broken_bonds
        ).long()
        broken_bonds_embedded = F.one_hot(
            num_broken_clamped, num_classes=self.max_broken_bonds + 1
        ).float()
        token_list = [
            root_tokens_expanded,
            form_encodings,
            diff_encodings,
            broken_bonds_embedded,
        ]
        if self.embed_collision:
            if collision_engs is None:
                raise ValueError("collision_engs is required when embed_collision=True")
            collision_encoding = self._collision_embedding(collision_engs)
            collision_encoding = collision_encoding[:, None, :].expand(
                batch_size, max_frags, -1
            )
            token_list.append(collision_encoding)
        if self.inten_broken_bond_feats:
            if bond_feats is None:
                raise ValueError(
                    "bond_feats is required when inten_broken_bond_feats=True"
                )
            deg_f = bond_feats.sum(dim=2)
            term_deg_f = torch.einsum("bkn,bnf->bkf", frag_mask_padded, deg_f)
            quad_cols = []
            for f in range(bond_feats.shape[-1]):
                ef_m = torch.einsum(
                    "bij,bkj->bki", bond_feats[..., f], frag_mask_padded
                )
                quad_cols.append((frag_mask_padded * ef_m).sum(dim=-1))
            term_quad_f = torch.stack(quad_cols, dim=-1)
            crossing_feats = term_deg_f - term_quad_f
            crossing_feats = self.embedder(crossing_feats)
            token_list.append(crossing_feats)
        root_token_embedded = self.token_mapper(torch.cat(token_list, dim=-1))
        frag_mask = torch.arange(max_frags, device=device).unsqueeze(
            0
        ) >= num_frag_targs.unsqueeze(-1)
        memory = torch.cat([root_tokens, node_embeddings], dim=1)
        if self.inten_attend_full:
            n_nodes = node_embeddings.shape[1]
            atom_pad = torch.arange(n_nodes, device=device).unsqueeze(
                0
            ) >= num_atoms.unsqueeze(1)
            in_frag = frag_targs_padded.bool()
            membership = torch.where(in_frag, 0, 1)
            out_frag_neighbors = node_degrees.unsqueeze(1) - adj_times_mask
            boundary_in = in_frag & (out_frag_neighbors > 1e-06)
            boundary_out = ~in_frag & (adj_times_mask > 1e-06)
            membership = torch.where(boundary_in, 3, membership)
            membership = torch.where(boundary_out, 4, membership)
            root_membership = torch.full(
                (batch_size, max_frags, 1), 2, dtype=membership.dtype, device=device
            )
            membership = torch.cat([root_membership, membership], dim=2)
            root_pad = torch.zeros((batch_size, 1), dtype=atom_pad.dtype, device=device)
            atom_pad = torch.cat([root_pad, atom_pad], dim=1)
            hidden = self.inten_decoder(
                tgt=root_token_embedded,
                memory=memory,
                group_embed=self.inten_group_embed.weight,
                membership=membership,
                atom_pad=atom_pad,
                tgt_key_padding_mask=frag_mask,
            )
        else:
            frag_targs_padded = torch.repeat_interleave(
                frag_targs_padded, self.nhead, dim=0
            )
            memory_mask = ~frag_targs_padded
            root_col = torch.zeros(
                (*memory_mask.shape[:-1], 1), dtype=memory_mask.dtype, device=device
            )
            memory_mask = torch.cat([root_col, memory_mask], dim=-1)
            if self.inten_self_attn_mode == "none":
                hidden = self.inten_decoder(
                    tgt=root_token_embedded, memory=memory, memory_mask=memory_mask
                )
            elif self.inten_self_attn_mode == "diagonal":
                hidden = self.inten_decoder(
                    tgt=root_token_embedded, memory=memory, memory_mask=memory_mask
                )
            else:
                hidden = self.inten_decoder(
                    tgt=root_token_embedded,
                    memory=memory,
                    memory_mask=memory_mask,
                    tgt_key_padding_mask=frag_mask,
                )
        if self.inten_encoder_layers > 0:
            hidden = self.inten_encoder(hidden, src_key_padding_mask=frag_mask)
        hydrogen_shift = (
            torch.arange(
                -self.max_broken_bonds, self.max_broken_bonds + 1, device=device
            )
            * common.ELEMENT_TO_MASS["H"]
        )
        masses_expanded = masses[frag_to_mol]
        frag_targs_f = frag_targs.float()
        net_fragment_mass_flat = (masses_expanded * frag_targs_f).sum(dim=-1)
        net_fragment_mass = nn_utils.pad_packed_tensor(
            net_fragment_mass_flat, num_frag_targs, 0
        )
        fragment_mass = (
            net_fragment_mass[:, :, None, None]
            + hydrogen_shift[None, None, None, :]
            + adduct_mass_shifts[:, None, :, None]
        )
        fragment_mass = torch.where(
            fragment_mass > 0, fragment_mass, torch.zeros_like(fragment_mass)
        )
        max_inten_shift = (self.output_size - 1) / 2
        max_break_ar = torch.arange(self.output_size, device=device)[None, None, :]
        max_breaks_ub = max_add_padded + max_inten_shift
        max_breaks_lb = -max_remove_padded + max_inten_shift
        ub_mask = max_break_ar <= max_breaks_ub[:, :, None]
        lb_mask = max_break_ar >= max_breaks_lb[:, :, None]
        valid_pos = torch.logical_and(ub_mask, lb_mask)
        valid_pos = torch.logical_and(valid_pos, ~frag_mask[:, :, None]).unsqueeze(-2)
        valid_pos = valid_pos.expand(
            batch_size, max_frags, 2, self.output_size
        ).reshape(batch_size, max_frags, -1)
        masses = fragment_mass.reshape(batch_size, max_frags, -1)
        output = self.output_map(hidden)
        output_unbinned = self.inten_activation(output)
        output_unbinned = output_unbinned.masked_fill(~valid_pos, 0)
        output_unbinned = output_unbinned.reshape(batch_size, -1)
        masses = masses.reshape(batch_size, -1)
        inten_output = torch.stack((masses, output_unbinned), dim=-1)
        return {"output": inten_output}

    def forward(
        self,
        graphormer_input,
        num_atoms,
        adducts,
        collision_engs,
        root_form_vecs,
        masses,
        adduct_mass_shifts,
        atom_form_vecs,
        adj_matrices,
        atom_hs,
        total_hs,
        instruments=None,
        root_graph=None,
    ):
        if self.training:
            raise RuntimeError(
                "GLACIER compatibility model is for inference; call eval()"
            )
        embedded = self.molecular_embedding(
            adducts, collision_engs, graphormer_input, instruments
        )
        root_tokens, nodes = (embedded["root_tokens"], embedded["node_embeddings"])
        predicted = self.breakpoint_forward(
            nodes, root_tokens, num_atoms, root_form_vecs, last_only=True
        )
        breakpoints = self.breakpoint_inference(
            predicted["frag_logits"][-1], predicted["frag_card_logits"][-1], num_atoms
        )
        fragments, counts = self.breakpoints_to_patterns(
            breakpoints, adj_matrices, num_atoms
        )
        fragments = nn_utils.pack_padded_tensor(fragments, counts).bool()
        intensities = self.inten_calculation(
            root_tokens,
            nodes,
            fragments,
            counts,
            root_form_vecs,
            atom_form_vecs,
            num_atoms,
            atom_hs,
            total_hs,
            adj_matrices,
            adduct_mass_shifts,
            masses,
            collision_engs=collision_engs,
            bond_feats=graphormer_input["attn_edge_type"]
            if self.inten_broken_bond_feats
            else None,
        )
        return {
            "inten_pred_end_to_end": intensities,
            "frags_pred": {"fragments": fragments, "fragment_count": counts},
        }

    def breakpoint_inference(
        self, breakpoint_logit, breakpoint_card, num_atoms, debug=False
    ):
        """
        Returns a binary tensor of shape [B, N, N_atom] with k positive entries per last axis,
        where k is the predicted cardinality for each pattern (from breakpoint_card).
        Batchified implementation.
        """
        k = torch.argmax(breakpoint_card, dim=-1)
        B, N, N_atom = breakpoint_logit.shape
        atom_pad = torch.arange(N_atom, device=breakpoint_logit.device).unsqueeze(
            0
        ) >= num_atoms.unsqueeze(1)
        breakpoint_preds = breakpoint_logit.masked_fill(
            atom_pad.unsqueeze(1), -1000000000.0
        )
        flat_preds = breakpoint_preds.reshape(-1, N_atom)
        flat_k = k.reshape(-1)
        patterns = torch.zeros_like(flat_preds, dtype=torch.bool)
        max_flat_k = flat_k.max().item()
        if max_flat_k > 0:
            topk_vals, topk_idx = torch.topk(flat_preds, k=max_flat_k, dim=-1)
            arange_k = torch.arange(
                max_flat_k, device=breakpoint_preds.device
            ).unsqueeze(0)
            valid_mask = arange_k < flat_k.unsqueeze(1)
            batch_idx = (
                torch.arange(flat_preds.shape[0], device=breakpoint_preds.device)
                .unsqueeze(1)
                .expand(-1, max_flat_k)
            )
            patterns[batch_idx[valid_mask], topk_idx[valid_mask]] = True
        out = patterns.view(B, N, N_atom)
        return out

    def breakpoints_to_patterns(self, mask, A, num_nodes=None):
        B, M, N = mask.shape
        BM = B * M
        A_exp = A.unsqueeze(1).expand(-1, M, -1, -1).reshape(BM, N, N)
        mask_exp = mask.reshape(BM, N)
        R = self.compute_reachability(A_exp, mask_exp, num_nodes=num_nodes)
        chunk_bits = 32
        num_chunks = (N + chunk_bits - 1) // chunk_bits
        padded_N = num_chunks * chunk_bits
        powers = 1 << torch.arange(chunk_bits, device=R.device, dtype=torch.int64)
        R_int = R.to(torch.int64)
        if padded_N > N:
            R_int = F.pad(R_int, (0, padded_N - N))
        R_words = (R_int.view(BM, N, num_chunks, chunk_bits) * powers).sum(dim=-1)
        batch_ids = torch.arange(B, device=R.device).repeat_interleave(M)
        batch_col = batch_ids[:, None, None].expand(BM, N, 1)
        flat_rows = torch.cat([batch_col, R_words], dim=-1).reshape(
            BM * N, 1 + num_chunks
        )
        unique_rows = torch.unique(flat_rows, dim=0)
        non_empty = (unique_rows[:, 1:] != 0).any(dim=-1)
        unique_rows = unique_rows[non_empty]
        batch_ids_unique = unique_rows[:, 0]
        pattern_counts = torch.bincount(batch_ids_unique, minlength=B)
        bits = unique_rows[:, 1:].unsqueeze(-1) & powers > 0
        unique_patterns = bits.reshape(-1, padded_N)[:, :N].to(torch.int64)
        padded_pattern = nn_utils.pad_packed_tensor(unique_patterns, pattern_counts, 0)
        debug_mask = (
            torch.arange(mask.shape[-1], device=mask.device)[None, :]
            < num_nodes[:, None]
        )
        debug_mask = torch.logical_and(debug_mask, pattern_counts[:, None] == 0)
        padded_pattern[:, 0, :] = torch.logical_or(padded_pattern[:, 0, :], debug_mask)
        pattern_counts = torch.clamp(pattern_counts, min=1)
        return (padded_pattern, pattern_counts)

    def compute_reachability(
        self, A: torch.Tensor, mask: torch.Tensor = None, num_nodes: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute transitive closure (reachability matrix) for a batch of adjacency matrices.

        Args:
            A: (B, N, N) adjacency matrices (bool or int)
            mask: (B, N) boolean tensor of nodes to keep (optional)
                If provided, edges touching unkept nodes will be zeroed out.

        Returns:
            R: (B, N, N) boolean reachability matrices,
            where R[b,i,j] == True means node i can reach node j (including itself).
        """
        device = A.device
        B, N, _ = A.shape
        A = A > 0
        if mask is not None:
            row_mask = mask.unsqueeze(-1)
            col_mask = mask.unsqueeze(-2)
            A = A & ~row_mask & ~col_mask
        if N > 1:
            num_steps = math.ceil(math.log2(N))
            row_chunk = max(1, min(N, 64 * 1024 * 1024 // max(1, B * N * N)))
            for _ in range(num_steps):
                nxt = A.clone()
                for r0 in range(0, N, row_chunk):
                    A_rows = A[:, r0 : r0 + row_chunk, :]
                    reach2 = (A_rows.unsqueeze(-1) & A.unsqueeze(1)).any(dim=2)
                    nxt[:, r0 : r0 + row_chunk, :] |= reach2
                if torch.equal(nxt, A):
                    break
                A = nxt
        if num_nodes is not None:
            row_idx = torch.arange(N, device=device).unsqueeze(0)
            col_idx = torch.arange(N, device=device).unsqueeze(0)
            valid_row = row_idx < num_nodes.unsqueeze(1)
            valid_col = col_idx < num_nodes.unsqueeze(1)
            valid_matrix = valid_row.unsqueeze(-1) & valid_col.unsqueeze(-2)
            valid_matrix = torch.repeat_interleave(
                valid_matrix, repeats=A.shape[0] // valid_matrix.shape[0], dim=0
            )
            A = A & valid_matrix
        return A

    def predict_mol(self, smi, collision_eng, adduct, device, instrument=None):
        if not getattr(self, "_predict_prepared", False):
            self.eval()
            self.requires_grad_(False)
            self._predict_prepared = True
        root_smi = smi
        if type(root_smi) is str:
            batched_input = False
            root_smi = [root_smi]
            collision_eng = [collision_eng]
            adduct = [adduct]
            if self.embed_instrument:
                instrument = [instrument]
        else:
            batched_input = True
        batch_size = len(root_smi)

        def to_tensor(x):
            return (
                torch.tensor(x, device=device, dtype=torch.float)
                if x is not None
                else x
            )

        instruments = (
            to_tensor([common.instrument2onehot_pos[i] for i in instrument])
            if self.embed_instrument
            else None
        )
        adducts = to_tensor([common.ion2onehot_pos[a] for a in adduct])
        collision_engs = to_tensor(collision_eng)
        mols = [Chem.MolFromSmiles(rsmi) for rsmi in root_smi]
        adj_matrices = [
            Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True) for mol in mols
        ]
        adj_matrices = [
            torch.from_numpy(adj_matrix).float().to(device)
            for adj_matrix in adj_matrices
        ]
        root_graph = None
        graphormer_batch = None
        graphormer_inputs = [
            self.tree_processor.create_graphormer_input(
                mol=m, multi_hop_max_dist=self.tree_processor.multi_hop_max_dist
            )
            for m in mols
        ]
        num_atoms = torch.tensor(
            [gf["num_atoms"] for gf in graphormer_inputs],
            dtype=torch.long,
            device=device,
        )
        max_nodes_gf = max((gf["x"].shape[0] for gf in graphormer_inputs))
        max_dist = max((gf["edge_input"].shape[2] for gf in graphormer_inputs))
        node_feat_dim = graphormer_inputs[0]["x"].shape[1]
        edge_feat_dim = graphormer_inputs[0]["attn_edge_type"].shape[2]
        x_batch = torch.zeros(
            [batch_size, max_nodes_gf, node_feat_dim],
            dtype=torch.float32,
            device=device,
        )
        attn_bias_batch = torch.full(
            [batch_size, max_nodes_gf + 1, max_nodes_gf + 1],
            -99999,
            dtype=torch.float32,
            device=device,
        )
        attn_edge_type_batch = torch.zeros(
            [batch_size, max_nodes_gf, max_nodes_gf, edge_feat_dim],
            dtype=torch.float32,
            device=device,
        )
        spatial_pos_batch = torch.zeros(
            [batch_size, max_nodes_gf, max_nodes_gf], dtype=torch.long, device=device
        )
        degree_batch = torch.zeros(
            [batch_size, max_nodes_gf], dtype=torch.long, device=device
        )
        edge_input_batch = torch.zeros(
            [batch_size, max_nodes_gf, max_nodes_gf, max_dist, edge_feat_dim],
            dtype=torch.float32,
            device=device,
        )
        for i, gf_input in enumerate(graphormer_inputs):
            num_nodes = gf_input["x"].shape[0]
            edge_dist = gf_input["edge_input"].shape[2]
            x_batch[i, :num_nodes] = gf_input["x"].to(device)
            attn_bias_batch[i, : num_nodes + 1, : num_nodes + 1] = gf_input[
                "attn_bias"
            ].to(device)
            attn_edge_type_batch[i, :num_nodes, :num_nodes] = gf_input[
                "attn_edge_type"
            ].to(device)
            spatial_pos_batch[i, :num_nodes, :num_nodes] = gf_input["spatial_pos"].to(
                device
            )
            degree_batch[i, :num_nodes] = gf_input["degree"].to(device)
            edge_input_batch[i, :num_nodes, :num_nodes, :edge_dist] = gf_input[
                "edge_input"
            ].to(device)
        graphormer_batch = {
            "x": x_batch,
            "attn_bias": attn_bias_batch,
            "attn_edge_type": attn_edge_type_batch,
            "spatial_pos": spatial_pos_batch,
            "degree": degree_batch,
            "edge_input": edge_input_batch,
        }
        max_nodes = torch.max(num_atoms).item()
        adj_matrices_batch = torch.zeros(
            [batch_size, max_nodes, max_nodes], dtype=torch.float32, device=device
        )
        for i, adj in enumerate(adj_matrices):
            adj_matrices_batch[i, : adj.shape[0], : adj.shape[1]] = adj
        adduct_mass_shift = torch.tensor(
            [
                [
                    common.ion2mass[mol_adduct],
                    -common.ELECTRON_MASS
                    if common.is_positive_adduct(mol_adduct)
                    else common.ELECTRON_MASS,
                ]
                for mol_adduct in adduct
            ],
            device=device,
        )
        engines = [
            fragmentation.FragmentEngine(
                mol_str=rsmi, mol_str_type="smiles", mol_str_canonicalized=True
            )
            for rsmi in root_smi
        ]
        total_atom_masses = [
            torch.from_numpy(engine.atom_weights_h).to(device) for engine in engines
        ]
        masses_padded = torch.nn.utils.rnn.pad_sequence(
            total_atom_masses, batch_first=True
        )
        root_forms = [common.form_from_smi(rsmi) for rsmi in root_smi]
        root_form_vecs = torch.stack(
            [
                torch.from_numpy(common.formula_to_dense(root_form))
                for root_form in root_forms
            ]
        ).to(device, non_blocking=True)
        atom_hs_list = [
            torch.tensor(engine.atom_hs, device=device) for engine in engines
        ]
        atom_hs_padded = torch.nn.utils.rnn.pad_sequence(atom_hs_list, batch_first=True)
        total_hs = torch.tensor(
            [engine.total_hs for engine in engines], device=device, dtype=torch.long
        )
        atom_symbols_batch = [engine.atom_symbols for engine in engines]
        atom_form_vecs_np = [
            [common.formula_to_dense(f"{s}H{h}") for s, h in zip(atom_symbols, num_hs)]
            for atom_symbols, num_hs in zip(atom_symbols_batch, atom_hs_list)
        ]
        atom_form_vecs_padded = torch.nn.utils.rnn.pad_sequence(
            [
                torch.from_numpy(np.stack(atom_form_vec_np, axis=0)).to(device)
                for atom_form_vec_np in atom_form_vecs_np
            ],
            batch_first=True,
        )
        atom_form_vecs = nn_utils.pack_padded_tensor(
            atom_form_vecs_padded, lengths=num_atoms
        )
        with torch.inference_mode():
            pred = self.forward(
                graphormer_batch,
                num_atoms,
                adducts,
                collision_engs,
                root_form_vecs=root_form_vecs,
                masses=masses_padded,
                adduct_mass_shifts=adduct_mass_shift,
                atom_form_vecs=atom_form_vecs,
                adj_matrices=adj_matrices_batch,
                atom_hs=atom_hs_padded,
                total_hs=total_hs,
                instruments=instruments if self.embed_instrument else None,
                root_graph=root_graph,
            )
            out = self._assemble_spec_frag(pred, adduct_mass_shift)
            if batched_input:
                return out
            else:
                return {k: v[0] for k, v in out.items()}

    def _assemble_spec_frag(self, pred, adduct_mass_shifts):
        """Turn a raw forward() output into per-molecule sparse spec rows + frag masks.

        Shared by ``predict_mol`` (raw-SMILES path) and ``predict_inten_frag_batch``
        (DataLoader path). Returns ``{"spec": [Tensor[num_rows, 2], ...],
        "frag": [Tensor[num_rows, num_atoms], ...]}`` with one entry per molecule.
        """
        inten_out = pred["inten_pred_end_to_end"]
        frags_pred = pred["frags_pred"]
        output = inten_out["output"]
        num_shifts = adduct_mass_shifts.shape[1] * self.output_size
        out = {"spec": [], "frag": []}
        fragments = nn_utils.pad_packed_tensor(
            frags_pred["fragments"], frags_pred["fragment_count"], 0
        )
        for i, n in enumerate(frags_pred["fragment_count"]):
            num_rows = int(n.item()) * num_shifts
            spec_pred = output[i, :num_rows, :]
            out_frag = fragments[i, :n].repeat_interleave(num_shifts, dim=0)
            out["spec"].append(spec_pred)
            out["frag"].append(out_frag)
        return out

    def boundary_nodes(self, A: torch.Tensor, subgraph_mask: torch.Tensor):
        """
        A: (B, N, N) bool
        subgraph_mask: (B, M, N) bool
        """
        B, N, _ = A.shape
        _, M, _ = subgraph_mask.shape
        device = A.device
        neighbors = (subgraph_mask.unsqueeze(-1) & A.unsqueeze(1)).any(dim=2)
        boundary_mask = neighbors & ~subgraph_mask
        chunk_size = 64
        num_chunks = (N + chunk_size - 1) // chunk_size
        padded_N = num_chunks * chunk_size
        pad_width = padded_N - N
        if pad_width > 0:
            boundary_mask = torch.nn.functional.pad(boundary_mask, (0, pad_width))
        boundary_mask = boundary_mask.view(B, M, num_chunks, chunk_size)
        powers = 1 << torch.arange(chunk_size, device=device, dtype=torch.int64)
        chunks = (boundary_mask.to(torch.int64) * powers).sum(dim=-1)
        batch_ids = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, 1)
        combined = torch.cat([batch_ids, chunks], dim=-1)
        combined_flat = combined.view(-1, 1 + num_chunks)
        unique_combined = torch.unique(combined_flat, dim=0)
        unique_batch_ids = unique_combined[:, 0]
        unique_chunks = unique_combined[:, 1:]
        unique_boundary_patterns = torch.bincount(unique_batch_ids, minlength=B)
        bits = unique_chunks.unsqueeze(-1) & powers > 0
        recovered = bits.view(-1, padded_N)[..., :N]
        max_unique = unique_boundary_patterns.max().item()
        boundary_mask_out = torch.zeros(
            B, max_unique, N, dtype=torch.float, device=device
        )
        counts = unique_boundary_patterns
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts
        row_indices = (
            torch.arange(unique_batch_ids.shape[0], device=device)
            - starts[unique_batch_ids]
        )
        boundary_mask_out[unique_batch_ids, row_indices] = recovered.float()
        return {
            "boundary_mask": boundary_mask_out,
            "unique_boundary_patterns": unique_boundary_patterns,
        }

    def _collision_embedding(self, collision_engs):
        """Sinusoidal collision-energy embedding [B, collision_dim] (NaN -> merged)."""
        embed_collision = torch.cat(
            (
                torch.sin(
                    collision_engs.unsqueeze(1)
                    / self.collision_embedder_denominators.unsqueeze(0)
                ),
                torch.cos(
                    collision_engs.unsqueeze(1)
                    / self.collision_embedder_denominators.unsqueeze(0)
                ),
            ),
            dim=1,
        )
        return torch.where(
            torch.isnan(embed_collision),
            self.collision_embed_merged.unsqueeze(0),
            embed_collision,
        )
