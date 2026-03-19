import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import HeteroConv, Linear, global_mean_pool
from torch_geometric.nn import GINEConv


class DDGHeteroModel(nn.Module):
    """
    protein_mode:
      - mode1: x_aa
      - mode2: x_aa + esm_wt
      - mode3: x_aa + (esm_wt - esm_mt)
      - mode4: x_aa + esm_wt + esm_mt
    """

    def __init__(
        self,
        metadata,
        hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        edge_dim: int = 20,
        prop_dim: int = 8,

        mut_nei_radius: float = 8.0,
        mut_nei_sigma: float = 2.5,

        use_mut_nei: bool = False,
        use_mut_pocket_nei: bool = True,

        protein_mode: str = "mode2",
    ):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.prop_dim = prop_dim

        self.mut_nei_radius = float(mut_nei_radius)
        self.mut_nei_sigma = float(mut_nei_sigma)
        self.use_mut_nei = bool(use_mut_nei)
        self.use_mut_pocket_nei = bool(use_mut_pocket_nei)

        protein_mode = str(protein_mode).lower().strip()
        if protein_mode not in {"mode1", "mode2", "mode3", "mode4"}:
            raise ValueError(f"protein_mode must be one of mode1/mode2/mode3/mode4, got {protein_mode}")
        self.protein_mode = protein_mode

        node_types, edge_types = metadata

        self.node_lin = nn.ModuleDict({nt: Linear(-1, hidden) for nt in node_types})
        self.mut_delta_lin = nn.Linear(20, hidden)
        self.mut_prop_lin = nn.Linear(prop_dim, hidden)
        self.mut_mask_lin = nn.Linear(1, hidden)
        self.edge_lin = nn.ModuleDict()
        for et in edge_types:
            self.edge_lin[self._etype_str(et)] = nn.Linear(edge_dim, hidden)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for et in edge_types:
                nn_gine = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                conv_dict[et] = GINEConv(nn_gine, train_eps=True)
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        extra = 0
        if self.use_mut_nei:
            extra += hidden
        if self.use_mut_pocket_nei:
            extra += hidden

        in_dim = hidden * 5 + extra + 20 + prop_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def _etype_str(et):
        return f"{et[0]}__{et[1]}__{et[2]}"

    def _get_protein_x_aa(self, batch):
        if "x_aa" in batch["protein"]:
            return batch["protein"].x_aa
        if "x" in batch["protein"]:
            return batch["protein"].x
        raise KeyError("protein has neither x_aa nor x")

    def _build_protein_input(self, batch):
        x_aa = self._get_protein_x_aa(batch).float()
        if self.protein_mode == "mode1":
            return x_aa

        if ("esm_wt" not in batch["protein"]) or ("esm_mt" not in batch["protein"]):
            raise KeyError("protein_mode requires protein.esm_wt and protein.esm_mt in graph")

        esm_wt = batch["protein"].esm_wt.float()
        esm_mt = batch["protein"].esm_mt.float()

        if self.protein_mode == "mode2":
            return torch.cat([x_aa, esm_wt], dim=1)
        if self.protein_mode == "mode3":
            return torch.cat([x_aa, (esm_wt - esm_mt)], dim=1)
        return torch.cat([x_aa, esm_wt, esm_mt], dim=1)

    def forward(self, batch):

        x_dict = dict(batch.x_dict)
        x_dict["protein"] = self._build_protein_input(batch)

        edge_index_dict = batch.edge_index_dict

        h = {}
        for nt, x in x_dict.items():
            h[nt] = F.relu(self.node_lin[nt](x))
        prot_batch = batch["protein"].batch
        mut_mask = batch["protein"].mut_mask.float()
        mut_delta = batch.mut_delta
        if mut_delta.dim() == 3:
            mut_delta = mut_delta.squeeze(1)

        if hasattr(batch, "mut_prop_delta") and (batch.mut_prop_delta is not None):
            mut_prop_delta = batch.mut_prop_delta
            if mut_prop_delta.dim() == 3:
                mut_prop_delta = mut_prop_delta.squeeze(1)
        else:
            mut_prop_delta = mut_delta.new_zeros((mut_delta.size(0), self.prop_dim))

        mut_emb_graph = self.mut_delta_lin(mut_delta) + self.mut_prop_lin(mut_prop_delta)
        h["protein"] = h["protein"] + self.mut_mask_lin(mut_mask) + (mut_emb_graph[prot_batch] * mut_mask)

        edge_attr_dict = {}
        for et, _ in edge_index_dict.items():
            store = batch[et]
            if not hasattr(store, "edge_attr") or store.edge_attr is None:
                continue
            if store.edge_attr.numel() == 0:
                continue
            edge_attr_dict[et] = self.edge_lin[self._etype_str(et)](store.edge_attr.float())

        for conv in self.convs:
            h = conv(h, edge_index_dict, edge_attr_dict=edge_attr_dict)
            for nt in h:
                h[nt] = F.relu(h[nt])
                h[nt] = F.dropout(h[nt], p=self.dropout, training=self.training)

        lig_batch = batch["ligand"].batch
        prot_g = global_mean_pool(h["protein"], prot_batch)
        lig_g = global_mean_pool(h["ligand"], lig_batch)

        if "master" in h and ("batch" in batch["master"]):
            master_g = global_mean_pool(h["master"], batch["master"].batch)
        else:
            B = prot_g.size(0)
            master_g = prot_g.new_zeros((B, self.hidden))

        pocket_w = torch.zeros((h["protein"].size(0), 1), device=h["protein"].device)
        et_rc = ("protein", "rev_contact", "ligand")
        if et_rc in batch.edge_types:
            ei = batch[et_rc].edge_index
            if ei is not None and ei.numel() > 0:
                pidx = ei[0].unique()
                pocket_w.index_fill_(0, pidx, 1.0)

        pocket_sum = scatter_sum(h["protein"] * pocket_w, prot_batch, dim=0)
        pocket_cnt = scatter_sum(pocket_w, prot_batch, dim=0).clamp(min=1.0)
        pocket_g = pocket_sum / pocket_cnt

        mut_node = scatter_sum(h["protein"] * mut_mask, prot_batch, dim=0)

        feats_extra = []

        if (self.use_mut_nei or self.use_mut_pocket_nei) and ("pos" in batch["protein"]):
            prot_pos = batch["protein"].pos
            mut_cnt = scatter_sum(mut_mask, prot_batch, dim=0)
            valid = (mut_cnt.squeeze(-1) > 0.5)
            mut_pos = scatter_sum(prot_pos * mut_mask, prot_batch, dim=0) / mut_cnt.clamp(min=1.0)

            d = torch.norm(prot_pos - mut_pos[prot_batch], dim=1)
            sigma = prot_pos.new_tensor(self.mut_nei_sigma)
            radius = prot_pos.new_tensor(self.mut_nei_radius)

            w_base = torch.exp(-0.5 * (d / sigma).pow(2))
            w_base = w_base * (d <= radius).float()
            w_base = w_base.unsqueeze(-1)

            if self.use_mut_nei:
                nei_sum = scatter_sum(h["protein"] * w_base, prot_batch, dim=0)
                nei_den = scatter_sum(w_base, prot_batch, dim=0).clamp(min=1e-6)
                mut_nei_g = nei_sum / nei_den
                if (~valid).any():
                    mut_nei_g = mut_nei_g.clone()
                    mut_nei_g[~valid] = 0.0
                feats_extra.append(mut_nei_g)

            if self.use_mut_pocket_nei:
                w_mp = w_base * pocket_w
                mp_sum = scatter_sum(h["protein"] * w_mp, prot_batch, dim=0)
                mp_den = scatter_sum(w_mp, prot_batch, dim=0).clamp(min=1e-6)
                mut_pocket_nei_g = mp_sum / mp_den

                mp_valid = (scatter_sum(pocket_w, prot_batch, dim=0).squeeze(-1) > 0.5) & valid
                if (~mp_valid).any():
                    mut_pocket_nei_g = mut_pocket_nei_g.clone()
                    mut_pocket_nei_g[~mp_valid] = 0.0
                feats_extra.append(mut_pocket_nei_g)

        else:
            B = prot_g.size(0)
            if self.use_mut_nei:
                feats_extra.append(prot_g.new_zeros((B, self.hidden)))
            if self.use_mut_pocket_nei:
                feats_extra.append(prot_g.new_zeros((B, self.hidden)))

        feat = torch.cat(
            [prot_g, pocket_g, lig_g, master_g, mut_node] + feats_extra + [mut_delta, mut_prop_delta],
            dim=1
        )
        out = self.mlp(feat).squeeze(-1)
        return out
