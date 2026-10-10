"""Molecular features for the released GLACIER checkpoint architecture."""

from typing import Any

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from ms_pred import common, nn_utils
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem.EState import EStateIndices
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class TreeProcessor:
    @staticmethod
    def _to_heavy_mol(mol):
        return None if mol is None else Chem.RemoveHs(mol)

    def __init__(
        self,
        pe_embed_k: int = 10,
        root_encode: str = "gnn",
        embed_elem_group: bool = False,
        multi_hop_max_dist: int = 5,
    ):
        self.pe_embed_k = pe_embed_k
        self.root_encode = root_encode
        self.embed_elem_group = embed_elem_group
        self.multi_hop_max_dist = multi_hop_max_dist

    def get_pe_for_tensor(self, mol: Any, node_features: torch.Tensor):
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        src, dst = np.nonzero(adj)
        eweights = adj[src, dst]
        tmp = dgl.graph((src, dst), num_nodes=adj.shape[0])
        tmp.ndata["h"] = node_features
        tmp.edata["e_ind"] = torch.tensor(eweights, dtype=torch.float32)
        pe_embeds = nn_utils.random_walk_pe(
            tmp, k=self.pe_embed_k, eweight_name="e_ind"
        )
        return pe_embeds

    def _feat_meta(self) -> dict:
        """Cached lookup tables for vectorized atom/bond featurization."""
        fm = getattr(self, "_fm", None)
        if fm is not None:
            return fm
        fm = {
            "valid_atom_num": np.asarray(common.VALID_ATOM_NUM),
            "hyb_to_idx": {h: i for i, h in enumerate(common.COMMON_HYBRIDIZATION)},
            "n_hyb": len(common.COMMON_HYBRIDIZATION),
            "h_cats": np.arange(common.COMMON_MAX_HYDROGEN_COUNTS + 1),
            "charge_cats": np.arange(
                -common.MAX_ABS_FORMAL_CHARGE, common.MAX_ABS_FORMAL_CHARGE + 1
            ),
            "deg_cats": np.arange(common.MAX_COMMON_DEGREE + 1),
            "rs_index": {s: i for i, s in enumerate(common.COMMON_RING_SIZES)},
            "n_rs": len(common.COMMON_RING_SIZES),
            "bt_to_idx": {b: i for i, b in enumerate(common.COMMON_BOND_TYPES)},
            "n_bt": len(common.COMMON_BOND_TYPES),
            "chi_to_idx": {c: i for i, c in enumerate(common.COMMON_CHIRALITY)},
            "n_chi": len(common.COMMON_CHIRALITY),
            "stereo_to_idx": {s: i for i, s in enumerate(common.VALID_BOND_STEREO)},
            "n_stereo": len(common.VALID_BOND_STEREO),
            "rotatable_smarts": Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"),
        }
        self._fm = fm
        return fm

    @staticmethod
    def _gasteiger_charges(mol) -> np.ndarray:
        """Per-atom Gasteiger partial charges, computed once and cached on the mol.

        Returns finite float32 charges (NaN/inf -> 0). Idempotent: skips recompute if
        the charges are already present (so atom and bond featurizers share one pass).
        """
        n = mol.GetNumAtoms()
        q = np.zeros(n, dtype=np.float32)
        if n == 0:
            return q
        if not mol.GetAtomWithIdx(0).HasProp("_GasteigerCharge"):
            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except Exception:  # noqa: BLE001 - preserve the checkpoint featurizer's zero fallback
                return q
        for i in range(n):
            a = mol.GetAtomWithIdx(i)
            if a.HasProp("_GasteigerCharge"):
                try:
                    v = a.GetDoubleProp("_GasteigerCharge")
                except Exception:  # noqa: BLE001 - preserve the checkpoint featurizer's zero fallback
                    v = 0.0
                if np.isfinite(v):
                    q[i] = v
        return q

    def _atom_feature_matrix(self, mol, num_atoms, atom_ring_counts, atom_ring_sizes):
        """Vectorized per-atom feature matrix (full feature set).

        A single light pass gathers raw RDKit attributes; the one-hot expansion is
        done with numpy. Includes the full chemical feature set: atomic number,
        heavy-atom degree, formal charge, hybridization, aromaticity, hydrogen
        count, ring presence, ring-size membership, fused-ring flag, chirality, plus
        two continuous electronic descriptors that drive bond cleavage / charge
        localization (the Gasteiger partial charge and the electrotopological E-state
        index). Binary attributes carry both their positive and complementary "not"
        columns, and the integer one-hots (degree / formal charge / hydrogen count)
        carry an explicit overflow bin. Heavy-atom degree is also returned separately
        for the graphormer degree-centrality input.
        """
        fm = self._feat_meta()
        n = num_atoms
        an = np.empty(n, dtype=np.int64)
        deg = np.empty(n, dtype=np.int64)
        hyb_idx = np.full(n, -1, dtype=np.int64)
        nh = np.empty(n, dtype=np.int64)
        fc = np.empty(n, dtype=np.int64)
        chi_idx = np.full(n, -1, dtype=np.int64)
        arom = np.zeros(n, dtype=bool)
        rs_mat = np.zeros((n, fm["n_rs"]), dtype=bool)
        elem_rows = None
        if self.embed_elem_group:
            elem_rows = np.zeros((n, common.ELEMENT_GROUP_DIM), dtype=np.float32)
        rs_index = fm["rs_index"]
        for i in range(n):
            atom = mol.GetAtomWithIdx(i)
            an[i] = atom.GetAtomicNum()
            deg[i] = atom.GetTotalDegree() - atom.GetTotalNumHs()
            hyb_idx[i] = fm["hyb_to_idx"].get(atom.GetHybridization(), -1)
            nh[i] = atom.GetTotalNumHs()
            fc[i] = atom.GetFormalCharge()
            chi_idx[i] = fm["chi_to_idx"].get(atom.GetChiralTag(), -1)
            arom[i] = atom.GetIsAromatic()
            for s in atom_ring_sizes[i]:
                k = rs_index.get(s)
                if k is not None:
                    rs_mat[i, k] = True
            if elem_rows is not None:
                sym = atom.GetSymbol()
                if sym in common.ELEMENT_TO_GROUP:
                    elem_rows[i] = common.element_to_group[sym]
        arc = np.asarray(atom_ring_counts, dtype=np.int64)
        in_ring = (arc >= 1)[:, None]
        fused = (arc >= 2)[:, None]
        gasteiger = self._gasteiger_charges(mol)
        try:
            estate = np.asarray(EStateIndices(mol), dtype=np.float32)
        except Exception:  # noqa: BLE001 - preserve the checkpoint featurizer's zero fallback
            estate = np.zeros(n, dtype=np.float32)
        if estate.shape[0] != n:
            estate = np.zeros(n, dtype=np.float32)
        estate = np.nan_to_num(estate, nan=0.0, posinf=0.0, neginf=0.0)
        estate = estate * 0.1

        def f(x):
            return np.asarray(x, dtype=np.float32)

        cols = []
        eq = an[:, None] == fm["valid_atom_num"][None, :]
        cols.append(f(eq))
        cols.append(f(~eq.any(axis=1, keepdims=True)))
        cols.append(f(deg[:, None] == fm["deg_cats"][None, :]))
        cols.append(f((deg > common.MAX_COMMON_DEGREE)[:, None]))
        cols.append(f(fc[:, None] == fm["charge_cats"][None, :]))
        cols.append(f((np.abs(fc) > common.MAX_ABS_FORMAL_CHARGE)[:, None]))
        cols.append(f(hyb_idx[:, None] == np.arange(fm["n_hyb"])[None, :]))
        cols.append(f(hyb_idx[:, None] == -1))
        cols.append(f(arom[:, None]))
        cols.append(f(nh[:, None] == fm["h_cats"][None, :]))
        cols.append(f((nh > common.COMMON_MAX_HYDROGEN_COUNTS)[:, None]))
        cols.append(f(~in_ring))
        cols.append(f(in_ring))
        cols.append(f(rs_mat))
        cols.append(f(~rs_mat))
        cols.append(f(~fused))
        cols.append(f(fused))
        cols.append(f(chi_idx[:, None] == np.arange(fm["n_chi"])[None, :]))
        cols.append(f(gasteiger[:, None]))
        cols.append(f(estate[:, None]))
        if elem_rows is not None:
            cols.append(elem_rows)
        node_features = np.concatenate(cols, axis=1)
        return (node_features, deg)

    def _bond_feature_arrays(self, mol, bond_ring_counts, bond_ring_sizes):
        """Vectorized per-bond feature matrix (full feature set).

        Returns (begin_idx, end_idx, bond_feats) with one row per bond. Includes bond
        type, conjugation, ring presence, ring-size membership, fused-ring flag and
        bond stereo, each with its complementary "not" column, plus two
        fragmentation-relevant features: a rotatable-bond flag (single, acyclic,
        non-terminal bonds are the canonical low-barrier cleavage sites) and the
        Gasteiger charge gap |q_i - q_j| (bond polarity -> heterolytic cleavage
        propensity).
        """
        fm = self._feat_meta()
        m = mol.GetNumBonds()
        begin = np.empty(m, dtype=np.int64)
        end = np.empty(m, dtype=np.int64)
        bt_idx = np.full(m, -1, dtype=np.int64)
        stereo_idx = np.full(m, -1, dtype=np.int64)
        conj = np.zeros(m, dtype=bool)
        in_ring = np.zeros(m, dtype=bool)
        fused = np.zeros(m, dtype=bool)
        rotatable = np.zeros(m, dtype=bool)
        rs_mat = np.zeros((m, fm["n_rs"]), dtype=bool)
        rs_index = fm["rs_index"]
        rot_pairs = set()
        rot_smarts = fm.get("rotatable_smarts")
        if rot_smarts is not None:
            for a, b_ in mol.GetSubstructMatches(rot_smarts):
                rot_pairs.add((a, b_) if a <= b_ else (b_, a))
        gasteiger = self._gasteiger_charges(mol)
        polarity = np.zeros(m, dtype=np.float32)
        for b, bond in enumerate(mol.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            begin[b] = i
            end[b] = j
            bt_idx[b] = fm["bt_to_idx"].get(bond.GetBondType(), -1)
            stereo_idx[b] = fm["stereo_to_idx"].get(bond.GetStereo(), -1)
            conj[b] = bond.GetIsConjugated()
            key = (i, j) if i <= j else (j, i)
            for s in bond_ring_sizes.get(key, ()):
                k = rs_index.get(s)
                if k is not None:
                    rs_mat[b, k] = True
            ring_count = bond_ring_counts.get(key, 0)
            in_ring[b] = ring_count >= 1
            fused[b] = ring_count >= 2
            rotatable[b] = key in rot_pairs
            polarity[b] = abs(gasteiger[i] - gasteiger[j])

        def f(x):
            return np.asarray(x, dtype=np.float32)

        cols = []
        cols.append(f(bt_idx[:, None] == np.arange(fm["n_bt"])[None, :]))
        cols.append(f(bt_idx[:, None] == -1))
        cols.append(f(~conj[:, None]))
        cols.append(f(conj[:, None]))
        cols.append(f(~in_ring[:, None]))
        cols.append(f(in_ring[:, None]))
        cols.append(f(rs_mat))
        cols.append(f(~rs_mat))
        cols.append(f(~fused[:, None]))
        cols.append(f(fused[:, None]))
        cols.append(f(stereo_idx[:, None] == np.arange(fm["n_stereo"])[None, :]))
        cols.append(f(stereo_idx[:, None] == -1))
        cols.append(f(rotatable[:, None]))
        cols.append(f(polarity[:, None]))
        bond_feats = np.concatenate(cols, axis=1)
        return (begin, end, bond_feats)

    def get_node_feats(self) -> int:
        if self.root_encode in ("gnn", "graphormer"):
            nf = 0
            nf += len(common.VALID_ELEMENTS) + 1
            nf += common.MAX_COMMON_DEGREE + 1 + 1
            nf += 2 * common.MAX_ABS_FORMAL_CHARGE + 1 + 1
            nf += len(common.COMMON_HYBRIDIZATION) + 1
            nf += 1
            nf += common.COMMON_MAX_HYDROGEN_COUNTS + 1 + 1
            nf += 2
            nf += 2 * len(common.COMMON_RING_SIZES)
            nf += 2
            nf += len(common.COMMON_CHIRALITY)
            nf += 1
            nf += 1
            if self.embed_elem_group:
                nf += common.ELEMENT_GROUP_DIM
            if self.root_encode == "graphormer":
                nf += self.pe_embed_k
            return nf
        raise NotImplementedError(
            f"Unsupported root encoding method: {self.root_encode}"
        )

    def get_edge_feats(self) -> int:
        nf = 0
        nf += len(common.COMMON_BOND_TYPES) + 1
        nf += 2
        nf += 2
        nf += 2 * len(common.COMMON_RING_SIZES)
        nf += 2
        nf += len(common.VALID_BOND_STEREO) + 1
        nf += 1
        nf += 1
        return nf

    def create_graphormer_input(
        self, mol: Any, spatial_pos_max: int = 1024, multi_hop_max_dist: int = 5
    ) -> dict[str, Any]:
        mol = self._to_heavy_mol(mol)
        if mol is None:
            raise ValueError("RDKit molecule is None after hydrogen removal")
        num_atoms = mol.GetNumAtoms()
        sssr = Chem.GetSymmSSSR(mol)
        atom_ring_counts = [0] * num_atoms
        atom_ring_sizes = [set() for _ in range(num_atoms)]
        bond_ring_counts = {}
        bond_ring_sizes = {}
        for bond in mol.GetBonds():
            key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            bond_ring_counts[key] = 0
            bond_ring_sizes[key] = set()
        for ring in sssr:
            ring_atoms = list(ring)
            r_len = len(ring_atoms)
            for a in ring_atoms:
                atom_ring_counts[a] += 1
                atom_ring_sizes[a].add(r_len)
            for k in range(r_len):
                a1 = ring_atoms[k]
                a2 = ring_atoms[(k + 1) % r_len]
                key = tuple(sorted((a1, a2)))
                if key in bond_ring_counts:
                    bond_ring_counts[key] += 1
                    bond_ring_sizes[key].add(r_len)
        node_features, degrees_arr = self._atom_feature_matrix(
            mol, num_atoms, atom_ring_counts, atom_ring_sizes
        )
        x = torch.from_numpy(node_features).float()
        if self.pe_embed_k > 0:
            pe_embeds = self.get_pe_for_tensor(mol, x)
            x = torch.cat((x, pe_embeds), -1).float()
        adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=False).astype(np.bool_)
        n = num_atoms
        dist = shortest_path(
            csr_matrix(adj_matrix), method="D", unweighted=True, directed=False
        )
        reachable = np.isfinite(dist)
        parents = np.full((n, n), -1, dtype=np.int16)
        node = np.arange(n)
        parents[node, node] = node.astype(np.int16)
        nbr_p, nbr_t = np.nonzero(adj_matrix)
        for e in np.argsort(-nbr_p, kind="stable"):
            p = int(nbr_p[e])
            t = int(nbr_t[e])
            match = reachable[:, t] & (dist[:, p] == dist[:, t] - 1)
            parents[match, t] = p
        path_len = np.where(reachable, dist, -1).astype(np.intp)
        dist_matrix = np.clip(
            np.where(reachable, dist, 32767), 0, spatial_pos_max - 1
        ).astype(np.int16)
        spatial_pos = torch.from_numpy(dist_matrix).long()
        bond_feat_dim = self.get_edge_feats()
        begin_idx, end_idx, bond_feats = self._bond_feature_arrays(
            mol, bond_ring_counts, bond_ring_sizes
        )
        attn_edge_type_np = np.zeros(
            (num_atoms, num_atoms, bond_feat_dim), dtype=np.float32
        )
        if bond_feats.shape[0]:
            attn_edge_type_np[begin_idx, end_idx] = bond_feats
            attn_edge_type_np[end_idx, begin_idx] = bond_feats
        attn_edge_type = torch.from_numpy(attn_edge_type_np)
        degree = torch.from_numpy(degrees_arr).long()
        bond_feat_np = attn_edge_type.numpy().astype(np.int64)
        edge_input_np = np.zeros(
            [num_atoms, num_atoms, multi_hop_max_dist, bond_feat_dim], dtype=np.int64
        )
        P = parents.astype(np.intp)
        s_col = np.arange(num_atoms, dtype=np.intp)[:, None]
        t_row = np.broadcast_to(
            np.arange(num_atoms, dtype=np.intp)[None, :], (num_atoms, num_atoms)
        )
        A = np.where(reachable, t_row, s_col).astype(np.intp)
        max_path = int(path_len.max()) if path_len.size else 0
        for k in range(max_path):
            A_next = np.take_along_axis(P, A, axis=1)
            slot = path_len - 1 - k
            valid = (k < path_len) & (slot >= 0) & (slot < multi_hop_max_dist)
            ss, tt = np.nonzero(valid)
            if ss.size:
                edge_input_np[ss, tt, slot[ss, tt]] = bond_feat_np[
                    A_next[ss, tt], A[ss, tt]
                ]
            A = A_next
        edge_input = torch.from_numpy(edge_input_np)
        attn_bias = torch.where(spatial_pos >= spatial_pos_max, -99999, 0.0)
        attn_bias = F.pad(attn_bias, (1, 0, 1, 0), value=0.0)
        return {
            "x": x,
            "attn_bias": attn_bias,
            "attn_edge_type": attn_edge_type,
            "spatial_pos": spatial_pos,
            "degree": degree,
            "edge_input": edge_input,
            "num_atoms": num_atoms,
        }
