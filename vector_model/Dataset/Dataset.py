import os
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import rdFingerprintGenerator

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
_PROP_MAT = torch.tensor([_AA_PROP_RAW[a] for a in _AA_ORDER], dtype=torch.float32)
_PROP_MEAN = _PROP_MAT.mean(dim=0, keepdim=True)
_PROP_STD = _PROP_MAT.std(dim=0, keepdim=True).clamp(min=1e-6)
AA_PROP_Z = {a: ((_PROP_MAT[i:i+1] - _PROP_MEAN) / _PROP_STD).squeeze(0) for i, a in enumerate(_AA_ORDER)}


def _onehot20(idx: int) -> torch.Tensor:
    v = torch.zeros(20, dtype=torch.float32)
    v[idx] = 1.0
    return v


def _read_first_mol_from_sdf(sdf_path: str) -> Chem.Mol:

    suppl = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=False, strictParsing=True)
    for m in suppl:
        if m is not None:
            return m
    raise ValueError(f"No valid molecule in SDF: {sdf_path}")


class DDGVectorDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        ecfp_bits: int = 1024,
        ecfp_radius: int = 2,#ECFP4
        prop_dim: int = 8,

        cache_esm: bool = True,
        cache_ecfp: bool = True,
        strict: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.ecfp_bits = int(ecfp_bits)
        self.ecfp_radius = int(ecfp_radius)
        self.prop_dim = int(prop_dim)
        self.cache_esm = bool(cache_esm)
        self.cache_ecfp = bool(cache_ecfp)
        self.strict = bool(strict)

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

        df["sdf_path"] = df["graph_id"].apply(lambda gid: os.path.join(self.data_dir, f"{gid}.sdf"))
        df["esm_path"] = df["graph_id"].apply(lambda gid: os.path.join(self.data_dir, f"{gid}_esmc_600m.pt"))

        sdf_ok = df["sdf_path"].apply(os.path.exists)
        esm_ok = df["esm_path"].apply(os.path.exists)
        ok = sdf_ok & esm_ok

        if ok.sum() < len(df):
            bad = df.loc[~ok, ["graph_id", "sdf_path", "esm_path"]].head(10).to_dict("records")
            msg = f"{len(df)-int(ok.sum())} rows missing sdf/esm. Examples: {bad}"
            if self.strict:
                raise FileNotFoundError(msg)
            else:
                print("[Warn]", msg)
                df = df[ok].reset_index(drop=True)

        def _one2three(x):
            return ONE_TO_THREE.get(x, None)

        df["wt_aa3"] = df["wt_aa"].apply(_one2three)
        df["mut_aa3"] = df["mut_aa"].apply(_one2three)
        aa_ok = df["wt_aa3"].notna() & df["mut_aa3"].notna()

        if aa_ok.sum() < len(df):
            bad = df.loc[~aa_ok, ["graph_id", "wt_aa", "mut_aa"]].head(10).to_dict("records")
            msg = f"{len(df)-int(aa_ok.sum())} rows have invalid AA code. Examples: {bad}"
            if self.strict:
                raise ValueError(msg)
            else:
                print("[Warn]", msg)
                df = df[aa_ok].reset_index(drop=True)

        df["wt_idx"] = df["wt_aa3"].apply(lambda a3: AA_TO_IDX[a3])
        df["mut_idx"] = df["mut_aa3"].apply(lambda a3: AA_TO_IDX[a3])

        df["wt_prop"] = df["wt_aa"].apply(lambda a1: AA_PROP_Z[a1].tolist())
        df["mut_prop"] = df["mut_aa"].apply(lambda a1: AA_PROP_Z[a1].tolist())

        self.df = df.reset_index(drop=True)

        self._morgan = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.ecfp_radius,
            fpSize=self.ecfp_bits
        )

        self._esm_cache: Dict[str, torch.Tensor] = {} if self.cache_esm else None
        self._ecfp_cache: Dict[str, torch.Tensor] = {} if self.cache_ecfp else None

        self._feature_dim: Optional[int] = None

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            x0, _, _ = self[0]
            self._feature_dim = int(x0.numel())
        return int(self._feature_dim)

    def __len__(self) -> int:
        return len(self.df)

    def _load_wt_esm_pooled(self, graph_id: str, esm_path: str) -> torch.Tensor:
        if self.cache_esm and (graph_id in self._esm_cache):
            return self._esm_cache[graph_id]

        obj = torch.load(esm_path, map_location="cpu", weights_only=False) if "weights_only" in torch.load.__code__.co_varnames else torch.load(esm_path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            t = obj.float()
        else:
            t = torch.tensor(obj, dtype=torch.float32)

        if t.dim() == 2:
            pooled = t.mean(dim=0)
        elif t.dim() == 1:
            pooled = t
        else:
            raise ValueError(f"[{graph_id}] bad ESM tensor shape: {tuple(t.shape)} in {esm_path}")

        pooled = pooled.contiguous()

        if self.cache_esm:
            self._esm_cache[graph_id] = pooled
        return pooled

    def _load_ecfp(self, graph_id: str, sdf_path: str) -> torch.Tensor:
        if self.cache_ecfp and (graph_id in self._ecfp_cache):
            return self._ecfp_cache[graph_id]

        mol = _read_first_mol_from_sdf(sdf_path)

        fp = self._morgan.GetFingerprint(mol)
        arr = np.zeros((self.ecfp_bits,), dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        t = torch.from_numpy(arr).float().contiguous()

        if self.cache_ecfp:
            self._ecfp_cache[graph_id] = t
        return t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        graph_id = str(row["graph_id"])
        resnum = int(row["resnum"])
        wt_idx = int(row["wt_idx"])
        mut_idx = int(row["mut_idx"])
        ddg = float(row["ddg"])

        sdf_path = str(row["sdf_path"])
        esm_path = str(row["esm_path"])

        wt_esm = self._load_wt_esm_pooled(graph_id, esm_path)

        wt_one = _onehot20(wt_idx)
        mut_one = _onehot20(mut_idx)
        mut_delta = (mut_one - wt_one)

        wt_prop = torch.tensor(row["wt_prop"], dtype=torch.float32)
        mut_prop = torch.tensor(row["mut_prop"], dtype=torch.float32)
        mut_prop_delta = (mut_prop - wt_prop)

        ecfp = self._load_ecfp(graph_id, sdf_path)

        x = torch.cat([wt_esm, mut_delta, mut_prop_delta, ecfp], dim=0).float().contiguous()
        y = torch.tensor([ddg], dtype=torch.float32)

        meta = {
            "graph_id": graph_id,
            "resnum": resnum,
            "wt_aa": str(row["wt_aa"]),
            "mut_aa": str(row["mut_aa"]),
            "sample_id": f"{graph_id}|{row['wt_aa']}{resnum}{row['mut_aa']}",
            "sdf_path": sdf_path,
            "esm_path": esm_path,
        }
        return x, y, meta
