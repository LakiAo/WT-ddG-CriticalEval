# infer.py
import os
import argparse
import math
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from dataset.Dataset import DDGDataset
from model import DDGHeteroModel

'''
python inference.py \
  --graph_dir ./pth \
  --csv ./test.csv \
  --ckpt ./runs/best.pt \
  --out_dir ./outdir
'''

def _rankdata_torch(x: torch.Tensor) -> torch.Tensor:
    idx = torch.argsort(x)
    ranks = torch.empty_like(idx, dtype=torch.float32)
    ranks[idx] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred = pred.detach().cpu().view(-1)
    y = y.detach().cpu().view(-1)

    mae = (pred - y).abs().mean().item()
    rmse = math.sqrt(((pred - y) ** 2).mean().item())

    vx = pred - pred.mean()
    vy = y - y.mean()
    denom = (vx.pow(2).sum().sqrt() * vy.pow(2).sum().sqrt()).item()
    pearson = (vx * vy).sum().item() / denom if denom > 1e-12 else 0.0

    rp = _rankdata_torch(pred)
    ry = _rankdata_torch(y)
    v1 = rp - rp.mean()
    v2 = ry - ry.mean()
    denom_s = (v1.pow(2).sum().sqrt() * v2.pow(2).sum().sqrt()).item()
    spearman = (v1 * v2).sum().item() / denom_s if denom_s > 1e-12 else 0.0

    std_true = y.std().item() if y.numel() > 1 else 0.0
    std_pred = pred.std().item() if pred.numel() > 1 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "std_true": std_true,
        "std_pred": std_pred,
        "n": float(y.numel()),
        "y_true_min": y.min().item() if y.numel() else 0.0,
        "y_true_max": y.max().item() if y.numel() else 0.0,
        "y_pred_min": pred.min().item() if pred.numel() else 0.0,
        "y_pred_max": pred.max().item() if pred.numel() else 0.0,
    }


@torch.no_grad()
def predict_loop(
    model,
    loader,
    device,
    use_zscore_y: bool,
    y_mean: float,
    y_std: float,
) -> torch.Tensor:
    model.eval()
    preds = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)  # [B]

        pred = out * y_std + y_mean if use_zscore_y else out
        preds.append(pred.detach().cpu())

    return torch.cat(preds, dim=0).view(-1)


def _bool(x: str) -> bool:
    return str(x).lower() in ["true", "1", "yes", "y", "t"]


def _load_ckpt(ckpt_path: str, map_location="cpu") -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"bad checkpoint (missing 'model'): {ckpt_path}")
    return ckpt


def _build_model_from_ckpt(
    ckpt: dict,
    metadata,
    override_protein_mode: Optional[str] = None,
    override_hidden: Optional[int] = None,
    override_layers: Optional[int] = None,
    override_dropout: Optional[float] = None,
):
    ckpt_args = ckpt.get("args", {}) or {}

    hidden = int(override_hidden if override_hidden is not None else ckpt_args.get("hidden", 256))
    num_layers = int(override_layers if override_layers is not None else ckpt_args.get("layers", 3))
    dropout = float(override_dropout if override_dropout is not None else ckpt_args.get("dropout", 0.2))
    protein_mode = str(override_protein_mode if override_protein_mode is not None else ckpt_args.get("protein_mode", "mode2"))

    model = DDGHeteroModel(
        metadata=metadata,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
        protein_mode=protein_mode,
    )
    return model, {"hidden": hidden, "num_layers": num_layers, "dropout": dropout, "protein_mode": protein_mode}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--csv", required=True, help="csv to infer (can contain ddg or not)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="runs/infer")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--protein_mode", type=str, default=None, choices=[None, "mode1", "mode2", "mode3", "mode4"])
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--zscore_y", type=_bool, default=None)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_csv is None:
        args.out_csv = os.path.join(args.out_dir, "predictions.csv")

    if args.device == "cuda" and (not torch.cuda.is_available()):
        print("[Warn] cuda requested but not available, fallback to cpu.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt = _load_ckpt(args.ckpt, map_location="cpu")

    ckpt_use_z = bool(ckpt.get("zscore_y", False))
    ckpt_mu = float(ckpt.get("y_mean", 0.0))
    ckpt_sd = float(ckpt.get("y_std", 1.0))
    if not np.isfinite(ckpt_sd) or ckpt_sd <= 0:
        ckpt_sd = 1.0

    use_zscore_y = ckpt_use_z if args.zscore_y is None else bool(args.zscore_y)

    print("device:", device)
    print("[CKPT]", args.ckpt)
    print(f"[Norm] zscore_y={use_zscore_y} (ckpt={ckpt_use_z}) | mean={ckpt_mu:.6f} std={ckpt_sd:.6f}")

    ds = DDGDataset(
        graph_dir=args.graph_dir,
        csv_file=args.csv,
        strict=True,
        fuse_protein_features=False,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    metadata = ds[0].metadata()
    model, mcfg = _build_model_from_ckpt(
        ckpt,
        metadata=metadata,
        override_protein_mode=args.protein_mode,
        override_hidden=args.hidden,
        override_layers=args.layers,
        override_dropout=args.dropout,
    )
    print("[Model]", mcfg)

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    pred = predict_loop(
        model,
        loader,
        device=device,
        use_zscore_y=use_zscore_y,
        y_mean=ckpt_mu,
        y_std=ckpt_sd,
    )

    out_df = ds.df.copy()
    out_df["pred_ddg"] = pred.numpy()
    has_ddg = False
    if "ddg" in out_df.columns:
        try:
            y_np = out_df["ddg"].astype(float).values
            if np.isfinite(y_np).all():
                has_ddg = True
        except Exception:
            has_ddg = False

    if has_ddg:
        y = torch.tensor(out_df["ddg"].astype(float).values, dtype=torch.float32)
        p = torch.tensor(out_df["pred_ddg"].values, dtype=torch.float32)
        m = metrics(p, y)
        print(
            f"[Eval@CSV] N={int(m['n'])} | MAE {m['mae']:.4f} RMSE {m['rmse']:.4f} "
            f"P {m['pearson']:.3f} S {m['spearman']:.3f} stdP {m['std_pred']:.3f}"
        )
        metric_path = os.path.join(args.out_dir, "infer_metrics.json")
        try:
            import json
            with open(metric_path, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)
            print("[OK] wrote:", metric_path)
        except Exception as e:
            print("[Warn] failed to write infer_metrics.json:", e)

    out_df.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)


if __name__ == "__main__":
    main()
