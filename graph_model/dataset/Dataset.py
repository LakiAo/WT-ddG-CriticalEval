import os
import torch
import pandas as pd
from torch_geometric.data import Dataset

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

_AA_PROP_RAW = {
    "A": [0.0,  1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "R": [1.0, -4.5, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    "N": [0.0, -3.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "D": [-1.0,-3.5, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "C": [0.0,  2.5, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    "Q": [0.0, -3.5, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "E": [-1.0,-3.5, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "G": [0.0, -0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "H": [0.5, -3.2, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    "I": [0.0,  4.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "L": [0.0,  3.8, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "K": [1.0, -3.9, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    "M": [0.0,  1.9, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "F": [0.0,  2.8, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "P": [0.0, -1.6, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "S": [0.0, -0.8, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    "T": [0.0, -0.7, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    "W": [0.0, -0.9, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    "Y": [0.0, -1.3, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    "V": [0.0,  4.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")
_PROP_MAT = torch.tensor([_AA_PROP_RAW[a] for a in _AA_ORDER], dtype=torch.float32)  # [20,8]
_PROP_MEAN = _PROP_MAT.mean(dim=0, keepdim=True)
_PROP_STD = _PROP_MAT.std(dim=0, keepdim=True).clamp(min=1e-6)
AA_PROP_Z = {a: ((_PROP_MAT[i:i+1] - _PROP_MEAN) / _PROP_STD).squeeze(0) for i, a in enumerate(_AA_ORDER)}
PROP_DIM = int(_PROP_MAT.size(1))


class DDGDataset(Dataset):

    def __init__(
        self,
        graph_dir: str,
        csv_file: str,
        root: str | None = None,
        cache_graphs: bool = True,
        strict: bool = True,
        fuse_protein_features: bool = False,
        drop_global_edges: bool = True,
        add_mut_physchem: bool = True,
    ):
        self.graph_dir = graph_dir
        self.csv_file = csv_file
        self.cache_graphs = cache_graphs
        self.strict = strict

        self.fuse_protein_features = fuse_protein_features
        self.drop_global_edges = drop_global_edges
        self.add_mut_physchem = add_mut_physchem

        df = pd.read_csv(csv_file)

        required = {"graph_id", "resnum", "wt_aa", "mut_aa", "ddg"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        df["graph_id"] = df["graph_id"].astype(str)
        df["resnum"] = df["resnum"].astype(int)
        df["wt_aa"] = df["wt_aa"].astype(str).str.upper().str.strip()
        df["mut_aa"] = df["mut_aa"].astype(str).str.upper().str.strip()
        df["ddg"] = df["ddg"].astype(float)

        df["graph_path"] = df["graph_id"].apply(lambda gid: os.path.join(graph_dir, f"{gid}_graph.pth"))
        exists_mask = df["graph_path"].apply(os.path.exists)
        if exists_mask.sum() < len(df):
            bad = df.loc[~exists_mask, "graph_id"].tolist()[:10]
            msg = f"{len(df) - int(exists_mask.sum())} graphs not found. Examples: {bad}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print("[Warn]", msg)
                df = df[exists_mask].reset_index(drop=True)

        def _one2three(x):
            return ONE_TO_THREE.get(x, None)

        df["wt_aa3"] = df["wt_aa"].apply(_one2three)
        df["mut_aa3"] = df["mut_aa"].apply(_one2three)
        aa_ok = df["wt_aa3"].notna() & df["mut_aa3"].notna()
        if aa_ok.sum() < len(df):
            bad = df.loc[~aa_ok, ["graph_id", "wt_aa", "mut_aa"]].head(10).to_dict("records")
            msg = f"{len(df)-int(aa_ok.sum())} rows have invalid AA code. Examples: {bad}"
            if strict:
                raise ValueError(msg)
            else:
                print("[Warn]", msg)
                df = df[aa_ok].reset_index(drop=True)

        df["wt_idx"] = df["wt_aa3"].apply(lambda a3: AA_TO_IDX[a3])
        df["mut_idx"] = df["mut_aa3"].apply(lambda a3: AA_TO_IDX[a3])

        if self.add_mut_physchem:
            df["wt_prop"] = df["wt_aa"].apply(lambda a1: AA_PROP_Z[a1].tolist())
            df["mut_prop"] = df["mut_aa"].apply(lambda a1: AA_PROP_Z[a1].tolist())

        self.df = df.reset_index(drop=True)

        self._cache = {} if cache_graphs else None
        super().__init__(root or graph_dir)

    def len(self):
        return len(self.df)

    def get(self, idx: int):
        row = self.df.iloc[idx]
        graph_id = row["graph_id"]
        resnum = int(row["resnum"])
        wt_idx = int(row["wt_idx"])
        mut_idx = int(row["mut_idx"])
        ddg = float(row["ddg"])

        data = self._load_graph_cached(graph_id, row["graph_path"])
        batch_data = data.clone()

        if self.drop_global_edges:
            for k in ("global_edge_index", "global_edge_attr"):
                if hasattr(batch_data, k):
                    delattr(batch_data, k)

        if ("x_aa" not in batch_data["protein"]) and ("x" in batch_data["protein"]):
            batch_data["protein"].x_aa = batch_data["protein"].x

        if ("x" not in batch_data["protein"]) and ("x_aa" in batch_data["protein"]):
            batch_data["protein"].x = batch_data["protein"].x_aa

        if "resnum" not in batch_data["protein"]:
            raise KeyError(f"[{graph_id}] data['protein'].resnum not found in graph")

        mask_bool = (batch_data["protein"].resnum == resnum)
        hit = int(mask_bool.sum().item())
        if hit != 1:
            msg = f"[{graph_id}] resnum={resnum} matched {hit} protein nodes (expected exactly 1)"
            if self.strict:
                raise ValueError(msg)
            mask_bool = torch.zeros_like(batch_data["protein"].resnum, dtype=torch.bool)

        batch_data["protein"].mut_mask = mask_bool.float().unsqueeze(-1)

        wt_onehot = torch.zeros(20, dtype=torch.float32)
        mut_onehot = torch.zeros(20, dtype=torch.float32)
        wt_onehot[wt_idx] = 1.0
        mut_onehot[mut_idx] = 1.0

        batch_data.mut_wt = wt_onehot.unsqueeze(0)
        batch_data.mut_mut = mut_onehot.unsqueeze(0)
        batch_data.mut_delta = (mut_onehot - wt_onehot).unsqueeze(0)

        if self.add_mut_physchem:
            wt_prop = torch.tensor(row["wt_prop"], dtype=torch.float32)
            mut_prop = torch.tensor(row["mut_prop"], dtype=torch.float32)
            batch_data.mut_prop_wt = wt_prop.unsqueeze(0)
            batch_data.mut_prop_mut = mut_prop.unsqueeze(0)
            batch_data.mut_prop_delta = (mut_prop - wt_prop).unsqueeze(0)

        batch_data.mut_resnum = torch.tensor([resnum], dtype=torch.long)
        batch_data.sample_id = f"{graph_id}|{row['wt_aa']}{resnum}{row['mut_aa']}"

        batch_data.y = torch.tensor([ddg], dtype=torch.float32)
        return batch_data

    def _load_graph_cached(self, graph_id: str, path: str):
        if self.cache_graphs and graph_id in self._cache:
            return self._cache[graph_id]

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load graph {path}: {e}")

        if self.fuse_protein_features:
            feats = []
            if "x_aa" in data["protein"]:
                feats.append(data["protein"].x_aa.float())
            elif "x" in data["protein"]:
                feats.append(data["protein"].x.float())

            if "esm_wt" in data["protein"]:
                feats.append(data["protein"].esm_wt.float())
            if "esm_mt" in data["protein"]:
                feats.append(data["protein"].esm_mt.float())

            if len(feats) == 0:
                raise ValueError(f"[{graph_id}] No protein features found to fuse.")
            data["protein"].x = torch.cat(feats, dim=1)

        if self.cache_graphs:
            self._cache[graph_id] = data
        return data
