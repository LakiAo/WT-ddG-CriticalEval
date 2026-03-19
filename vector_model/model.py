import torch
import torch.nn as nn
import torch.nn.functional as F


class DDGVectorModel(nn.Module):

    def __init__(
        self,
        esm_dim: int = 1152,
        fp_dim: int = 1024,
        prop_dim: int = 8,
        hidden: int = 512,
        dropout: float = 0.25,
        use_gated_injection: bool = True,
    ):
        super().__init__()
        self.esm_dim = int(esm_dim)
        self.fp_dim = int(fp_dim)
        self.prop_dim = int(prop_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.use_gated_injection = bool(use_gated_injection)

        self.esm_lin = nn.Linear(self.esm_dim, hidden)
        self.fp_lin = nn.Linear(self.fp_dim, hidden)

        self.mut_delta_lin = nn.Linear(20, hidden)
        self.mut_prop_lin = nn.Linear(prop_dim, hidden)

        if self.use_gated_injection:
            self.gate = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.Sigmoid(),
            )

        in_dim = hidden * (4 if self.use_gated_injection else 3)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 2:
            raise ValueError(f"Expected x as [B,D], got {tuple(x.shape)}")

        B, D = x.shape
        need = self.esm_dim + 20 + self.prop_dim + self.fp_dim
        if D != need:
            raise ValueError(f"Input dim mismatch: got {D}, expected {need}")

        off = 0
        wt_esm = x[:, off:off + self.esm_dim]; off += self.esm_dim
        mut_delta = x[:, off:off + 20]; off += 20
        mut_prop = x[:, off:off + self.prop_dim]; off += self.prop_dim
        fp = x[:, off:off + self.fp_dim]; off += self.fp_dim

        prot_h = F.relu(self.esm_lin(wt_esm))
        lig_h = F.relu(self.fp_lin(fp))

        mut_h = self.mut_delta_lin(mut_delta) + self.mut_prop_lin(mut_prop)
        mut_h = F.relu(mut_h)

        prot_h = F.dropout(prot_h, p=self.dropout, training=self.training)
        lig_h = F.dropout(lig_h, p=self.dropout, training=self.training)
        mut_h = F.dropout(mut_h, p=self.dropout, training=self.training)

        if self.use_gated_injection:
            g = self.gate(torch.cat([prot_h, mut_h], dim=1))
            injected = prot_h + g * mut_h
            feat = torch.cat([prot_h, lig_h, mut_h, injected], dim=1)
        else:
            feat = torch.cat([prot_h, lig_h, mut_h], dim=1)

        out = self.mlp(feat).squeeze(-1)
        return out
