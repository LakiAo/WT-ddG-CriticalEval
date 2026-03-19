import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Dataset.Dataset import DDGVectorDataset
from model import DDGVectorModel

'''
python inference.py \
  --data_dir ./output \
  --csv test.csv \
  --ckpt ./runs/best.pt \
  --out_dir ./outdir
'''

def _load_ckpt(ckpt_path: str, map_location="cpu") -> dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "model" not in ckpt:
        raise KeyError(f"bad checkpoint (missing 'model'): {ckpt_path}")
    return ckpt


def _bool(x: str) -> bool:
    return str(x).lower() in ["true", "1", "yes", "y", "t"]


@torch.no_grad()
def predict_loop(model, loader, device, use_zscore_y: bool, y_mean: float, y_std: float) -> torch.Tensor:
    model.eval()
    preds = []
    for x, _y, _meta in loader:
        x = x.to(device)
        out = model(x).view(-1)
        pred = out * y_std + y_mean if use_zscore_y else out
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).view(-1)


def _make_tmp_csv_for_dataset(in_csv: str, out_dir: str) -> str:
    df = pd.read_csv(in_csv)
    if "ddg" not in df.columns:
        df["ddg"] = np.nan
    tmp_path = os.path.join(out_dir, "_tmp_for_dataset.csv")
    df.to_csv(tmp_path, index=False)
    return tmp_path


def _build_model_from_ckpt(
    ckpt: dict,
    esm_dim: int,
    fp_dim: int,
    prop_dim: int,
    override_hidden: Optional[int] = None,
    override_dropout: Optional[float] = None,
    override_no_gated_injection: Optional[bool] = None,
) -> DDGVectorModel:
    ckpt_args = ckpt.get("args", {}) or {}

    hidden = int(override_hidden if override_hidden is not None else ckpt_args.get("hidden", 512))
    dropout = float(override_dropout if override_dropout is not None else ckpt_args.get("dropout", 0.25))

    no_gated = bool(ckpt_args.get("no_gated_injection", False))
    if override_no_gated_injection is not None:
        no_gated = bool(override_no_gated_injection)

    model = DDGVectorModel(
        esm_dim=int(esm_dim),
        fp_dim=int(fp_dim),
        prop_dim=int(prop_dim),
        hidden=int(hidden),
        dropout=float(dropout),
        use_gated_injection=(not no_gated),
    )
    return model


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs/infer_vec")
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--ecfp_bits", type=int, default=1024)
    ap.add_argument("--ecfp_radius", type=int, default=2)
    ap.add_argument("--prop_dim", type=int, default=8)
    ap.add_argument("--zscore_y", type=_bool, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--no_gated_injection", type=_bool, default=None)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_csv is None:
        args.out_csv = os.path.join(args.out_dir, "predictions.csv")

    if args.device == "cuda" and (not torch.cuda.is_available()):
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    in_df = pd.read_csv(args.csv)
    tmp_csv = _make_tmp_csv_for_dataset(args.csv, args.out_dir)
    ckpt = _load_ckpt(args.ckpt, map_location="cpu")

    ckpt_use_z = bool(ckpt.get("zscore_y", False))
    ckpt_mu = float(ckpt.get("y_mean", 0.0))
    ckpt_sd = float(ckpt.get("y_std", 1.0))
    if not np.isfinite(ckpt_sd) or ckpt_sd <= 0:
        ckpt_sd = 1.0

    use_zscore_y = ckpt_use_z if args.zscore_y is None else bool(args.zscore_y)

    ds = DDGVectorDataset(
        data_dir=args.data_dir,
        csv_file=tmp_csv,
        ecfp_bits=int(args.ecfp_bits),
        ecfp_radius=int(args.ecfp_radius),
        prop_dim=int(args.prop_dim),
        strict=True,
        cache_esm=True,
        cache_ecfp=True,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    feat_dim = int(ds.feature_dim)
    esm_dim = feat_dim - (20 + int(args.prop_dim) + int(args.ecfp_bits))
    if esm_dim <= 0:
        raise RuntimeError(
            f"Bad inferred esm_dim={esm_dim} from feature_dim={feat_dim}. "
            f"Check concat order in DDGVectorDataset."
        )

    model = _build_model_from_ckpt(
        ckpt=ckpt,
        esm_dim=esm_dim,
        fp_dim=int(args.ecfp_bits),
        prop_dim=int(args.prop_dim),
        override_hidden=args.hidden,
        override_dropout=args.dropout,
        override_no_gated_injection=(None if args.no_gated_injection is None else bool(args.no_gated_injection)),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    pred = predict_loop(model, loader, device, use_zscore_y=use_zscore_y, y_mean=ckpt_mu, y_std=ckpt_sd)

    if len(in_df) != int(pred.numel()):
        raise RuntimeError(f"Row mismatch: input has {len(in_df)} rows, pred has {pred.numel()} rows")

    out_df = in_df.copy()
    out_df["pred_ddg"] = pred.numpy()

    out_df.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)


if __name__ == "__main__":
    main()
