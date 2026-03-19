import os
import argparse
import math
import csv
import random
import json
import shutil
from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset.Dataset import DDGDataset
from model import DDGHeteroModel

'''
ROOT_OUT=./runs/kfold_uniprot_runs
SPLIT_ROOT=./kfold_uniprot

for mode in mode1 mode2 mode3 mode4; do
  for fold in fold_0 fold_1 fold_2 fold_3 fold_4; do
    python ./kfold_train.py \
      --graph_dir ./pth \
      --train_csv ${SPLIT_ROOT}/${fold}/train.csv \
      --val_csv   ${SPLIT_ROOT}/${fold}/val.csv \
      --test_csv  ${SPLIT_ROOT}/${fold}/test.csv \
      --out_dir ${ROOT_OUT} \
      --protein_mode ${mode} \
      --kfold_layout \
      --seed 42 \
      --hidden 128 \
      --layers 2 \
      --dropout 0.35 \
      --lr 3e-4 \
      --save_warmup 2 \
      --min_epochs 10 \
      --patience 20 \
      --topk 1 \
      --score_mode pearson \
      --tie_by rmse \
      --std_pred_min 0.10
  done
done
' > kfold.log 2>&1 &
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
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "std_true": float(std_true),
        "std_pred": float(std_pred),
        "n": float(y.numel()),
        "y_true_min": float(y.min().item()) if y.numel() else 0.0,
        "y_true_max": float(y.max().item()) if y.numel() else 0.0,
        "y_pred_min": float(pred.min().item()) if pred.numel() else 0.0,
        "y_pred_max": float(pred.max().item()) if pred.numel() else 0.0,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    use_zscore_y: bool,
    y_mean: float,
    y_std: float,
    debug: bool = False,
    tag: str = "",
) -> Dict[str, float]:
    model.eval()
    preds, ys = [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        y = batch.y.to(device).view(-1)

        pred = out * y_std + y_mean if use_zscore_y else out

        preds.append(pred)
        ys.append(y)

    pred = torch.cat(preds, dim=0).view(-1)
    y = torch.cat(ys, dim=0).view(-1)

    m = metrics(pred, y)

    if debug:
        print(f"[{tag}] N:", int(m["n"]))
        print(f"[{tag}] std(y_true):", m["std_true"])
        print(f"[{tag}] std(y_pred):", m["std_pred"])
        print(f"[{tag}] y_true min/max:", m["y_true_min"], m["y_true_max"])
        print(f"[{tag}] y_pred min/max:", m["y_pred_min"], m["y_pred_max"])
        print(f"[{tag}] pearson:", m["pearson"])
        print(f"[{tag}] spearman:", m["spearman"])

    return m


def _bool(x: str) -> bool:
    return str(x).lower() in ["true", "1", "yes", "y", "t"]


def _infer_fold_name_from_csv(train_csv: str) -> str:
    d = os.path.basename(os.path.dirname(train_csv))
    return d if d else "fold_unknown"


def _infer_split_name_from_csv(train_csv: str) -> str:
    d1 = os.path.dirname(train_csv)
    d2 = os.path.dirname(d1)
    d3 = os.path.dirname(d2)
    base = os.path.basename(d3)
    return base if base else "splits_unknown"


def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument(
        "--protein_mode",
        type=str,
        default="mode2",
        choices=["mode1", "mode2", "mode3", "mode4"]
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out_dir", type=str, default="runs/ddg_baseline")
    ap.add_argument("--kfold_layout", action="store_true")
    ap.add_argument("--fold_name", type=str, default="")
    ap.add_argument("--split_name", type=str, default="")
    ap.add_argument("--zscore_y", default=True, type=_bool)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min_epochs", type=int, default=10)
    ap.add_argument("--save_warmup", type=int, default=10)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    ap.add_argument("--huber_delta", type=float, default=3.0)
    ap.add_argument("--score_mode", type=str, default="pearson",
                    choices=["pearson", "pearson_rmse"])
    ap.add_argument("--score_lambda", type=float, default=0.02)
    ap.add_argument("--tie_by", type=str, default="rmse", choices=["rmse", "mae"])
    ap.add_argument("--std_pred_min", type=float, default=0.10)
    args = ap.parse_args()

    fold_name = args.fold_name.strip() if args.fold_name.strip() else _infer_fold_name_from_csv(args.train_csv)
    split_name = args.split_name.strip() if args.split_name.strip() else _infer_split_name_from_csv(args.train_csv)

    run_out_dir = args.out_dir
    if args.kfold_layout:
        run_out_dir = os.path.join(args.out_dir, args.protein_mode, fold_name)
    _safe_makedirs(run_out_dir)

    seed = int(args.seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("[Config] protein_mode:", args.protein_mode)
    print("[Config] out_dir:", run_out_dir)
    print("[Config] split_name:", split_name, "| fold_name:", fold_name)
    print("[Config] save_warmup:", int(args.save_warmup), "| topk:", int(args.topk))
    print("[Config] score_mode:", args.score_mode, "| lambda:", float(args.score_lambda), "| tie_by:", args.tie_by)
    print("[Config] early_stop fixed by topk-score | min_epochs:", int(args.min_epochs), "| patience:", int(args.patience))
    print("[Config] std_pred_min:", float(args.std_pred_min))

    train_ds = DDGDataset(graph_dir=args.graph_dir, csv_file=args.train_csv, strict=True,
                          fuse_protein_features=False)
    val_ds = DDGDataset(graph_dir=args.graph_dir, csv_file=args.val_csv, strict=True,
                        fuse_protein_features=False)
    test_ds = DDGDataset(graph_dir=args.graph_dir, csv_file=args.test_csv, strict=True,
                         fuse_protein_features=False)

    if args.zscore_y:
        y_mean = float(train_ds.df["ddg"].astype(float).mean())
        y_std = float(train_ds.df["ddg"].astype(float).std())
        if not np.isfinite(y_std) or y_std < 1e-8:
            raise ValueError(f"Bad train y_std={y_std}. Cannot z-score.")
        print(f"[ZScore] train mean={y_mean:.6f}, std={y_std:.6f}")
    else:
        y_mean, y_std = 0.0, 1.0
        print("[ZScore] disabled")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    metadata = train_ds[0].metadata()
    model = DDGHeteroModel(
        metadata=metadata,
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        protein_mode=args.protein_mode,
    ).to(device)

    huber_delta_z = None
    if args.loss == "mse":
        loss_fn = nn.MSELoss()
        loss_name = "mse"
    else:
        huber_delta_z = float(args.huber_delta) / float(y_std) if args.zscore_y else float(args.huber_delta)
        loss_fn = nn.HuberLoss(delta=huber_delta_z)
        loss_name = "huber"
        print(f"[Huber] delta(orig)={args.huber_delta} -> delta(z)={huber_delta_z:.6f} (zscore_y={args.zscore_y})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    topk = max(1, int(args.topk))
    topk_paths: List[str] = [os.path.join(run_out_dir, f"best_by_score_{i+1}.pt") for i in range(topk)]
    topk_scores: List[float] = [-1e18 for _ in range(topk)]
    topk_epochs: List[int] = [-1 for _ in range(topk)]
    topk_ties: List[float] = [float("inf") for _ in range(topk)]

    def gate_ok(m: Dict[str, float]) -> bool:
        return float(m["std_pred"]) >= float(args.std_pred_min)

    def compute_score_and_tie(m: Dict[str, float]) -> Tuple[float, float]:

        p = float(m["pearson"])
        rmse = float(m["rmse"])
        mae = float(m["mae"])

        if args.score_mode == "pearson":
            score = p
        else:
            score = p - float(args.score_lambda) * rmse

        tie_value = rmse if args.tie_by == "rmse" else mae
        return float(score), float(tie_value)

    def better_than(a_score: float, a_tie: float, b_score: float, b_tie: float) -> bool:

        if a_score > b_score + 1e-12:
            return True
        if abs(a_score - b_score) <= 1e-12 and a_tie < b_tie - 1e-12:
            return True
        return False

    def try_insert_topk(score: float, tie_value: float, epoch: int, payload: dict) -> bool:

        nonlocal topk_scores, topk_epochs, topk_ties

        pos = None
        for i in range(topk):
            if better_than(score, tie_value, topk_scores[i], topk_ties[i]):
                pos = i
                break
        if pos is None:
            return False

        for j in range(topk - 1, pos, -1):
            topk_scores[j] = topk_scores[j - 1]
            topk_ties[j] = topk_ties[j - 1]
            topk_epochs[j] = topk_epochs[j - 1]
            if os.path.exists(topk_paths[j - 1]):
                try:
                    shutil.copy2(topk_paths[j - 1], topk_paths[j])
                except Exception:
                    pass

        topk_scores[pos] = float(score)
        topk_ties[pos] = float(tie_value)
        topk_epochs[pos] = int(epoch)
        torch.save(payload, topk_paths[pos])
        return True

    log_path = os.path.join(run_out_dir, "val_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "train_loss",
            "val_mae", "val_rmse", "val_pearson", "val_spearman", "val_std_pred",
            "val_score", "val_tie"
        ])

    bad_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device)

            out = model(batch).view(-1)
            y = batch.y.to(device).view(-1)

            if args.zscore_y:
                y_z = (y - y_mean) / y_std
                loss = loss_fn(out, y_z)
            else:
                loss = loss_fn(out, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = int(y.numel())
            total_loss += loss.item() * bs
            n += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{(total_loss / max(1, n)):.4f}"})

        train_loss = total_loss / max(1, n)

        val_m = evaluate(model, val_loader, device, use_zscore_y=args.zscore_y, y_mean=y_mean, y_std=y_std, debug=False)
        val_score, val_tie = compute_score_and_tie(val_m)
        g_ok = gate_ok(val_m)

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
            f"val MAE {val_m['mae']:.4f} RMSE {val_m['rmse']:.4f} "
            f"P {val_m['pearson']:.3f} S {val_m['spearman']:.3f} "
            f"stdP {val_m['std_pred']:.3f} score {val_score:.3f} tie {val_tie:.4f}"
        )

        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                epoch, train_loss,
                val_m["mae"], val_m["rmse"], val_m["pearson"], val_m["spearman"], val_m["std_pred"],
                val_score, val_tie
            ])

        ckpt_payload = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val": val_m,
            "val_score": float(val_score),
            "val_tie": float(val_tie),
            "y_mean": y_mean,
            "y_std": y_std,
            "zscore_y": bool(args.zscore_y),
            "loss": loss_name,
            "huber_delta_orig": float(args.huber_delta),
            "huber_delta_z": float(huber_delta_z) if huber_delta_z is not None else None,
            "resolved": {"run_out_dir": run_out_dir, "fold_name": fold_name, "split_name": split_name},
        }

        can_save = (epoch > int(args.save_warmup))

        inserted = False
        if can_save and g_ok:
            inserted = try_insert_topk(val_score, val_tie, epoch, ckpt_payload)

        if epoch < int(args.min_epochs):
            continue

        improved = True if (not g_ok) else bool(inserted)

        bad_count = 0 if improved else (bad_count + 1)
        if int(args.patience) > 0 and bad_count >= int(args.patience):
            print(f"[EarlyStop] No improvement in top-{topk} by score for {args.patience} epochs. Stop at epoch {epoch}.")
            break

    print("\n=== Saved top-k checkpoints by score ===")
    for i in range(topk):
        print(f"best_by_score_{i+1}: {topk_paths[i]} | epoch {topk_epochs[i]} | val_score {topk_scores[i]:.6f} | tie {topk_ties[i]:.6f}")

    def _load_norm_from_ckpt(ckpt: dict) -> Tuple[bool, float, float]:
        use_z = bool(ckpt.get("zscore_y", False))
        mu = float(ckpt.get("y_mean", 0.0))
        sd = float(ckpt.get("y_std", 1.0))
        return use_z, mu, sd

    def test_ckpt(path: str, name: str) -> Optional[Dict[str, float]]:
        if not os.path.exists(path):
            print(f"[Skip] {name}: checkpoint not found -> {path}")
            return None
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        use_z, mu, sd = _load_norm_from_ckpt(ckpt)
        test_m = evaluate(model, test_loader, device, use_zscore_y=use_z, y_mean=mu, y_std=sd, debug=False, tag=f"TEST|{name}")
        test_score, test_tie = compute_score_and_tie(test_m)

        print(
            f"[TEST|{name}] MAE {test_m['mae']:.4f} RMSE {test_m['rmse']:.4f} "
            f"Pearson {test_m['pearson']:.3f} Spearman {test_m['spearman']:.3f} "
            f"stdP {test_m['std_pred']:.3f} score {test_score:.3f} tie {test_tie:.4f}"
        )

        out = dict(test_m)
        out["score"] = float(test_score)
        out["tie"] = float(test_tie)
        return out

    test_results: List[Optional[Dict[str, float]]] = []
    for i in range(topk):
        test_results.append(test_ckpt(topk_paths[i], f"best_by_score_{i+1}"))

    best_block = {}
    for i in range(topk):
        best_block[f"best_by_score_{i+1}"] = {
            "epoch": int(topk_epochs[i]),
            "val_score": float(topk_scores[i]),
            "val_tie": float(topk_ties[i]),
            "test": test_results[i],
        }

    metrics_payload = {
        "split_name": split_name,
        "fold_name": fold_name,
        "protein_mode": args.protein_mode,
        "score_mode": args.score_mode,
        "score_lambda": float(args.score_lambda),
        "tie_by": args.tie_by,
        "std_pred_min": float(args.std_pred_min),
        "topk": int(topk),
        "best": best_block,
        "paths": {
            "run_out_dir": run_out_dir,
            "val_log": log_path,
            "topk_paths": topk_paths,
        }
    }

    metrics_path = os.path.join(run_out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print("[OK] wrote:", metrics_path)

    def _flat(prefix: str, m: Optional[Dict[str, float]]) -> Dict[str, float]:
        keys = ["mae", "rmse", "pearson", "spearman", "std_pred", "score", "tie"]
        if m is None:
            return {f"{prefix}_{k}": float("nan") for k in keys}
        return {f"{prefix}_{k}": float(m.get(k, float("nan"))) for k in keys}

    row = {
        "split_name": split_name,
        "fold_name": fold_name,
        "protein_mode": args.protein_mode,
        "seed": int(args.seed),
        "zscore_y": int(bool(args.zscore_y)),
        "min_epochs": int(args.min_epochs),
        "patience": int(args.patience),
        "save_warmup": int(args.save_warmup),
        "topk": int(topk),
        "score_mode": args.score_mode,
        "score_lambda": float(args.score_lambda),
        "tie_by": args.tie_by,
        "std_pred_min": float(args.std_pred_min),
    }

    for i in range(topk):
        row[f"best_epoch_score_{i+1}"] = int(topk_epochs[i])
        row[f"val_score_{i+1}"] = float(topk_scores[i])
        row[f"val_tie_{i+1}"] = float(topk_ties[i])
        row.update(_flat(f"test_best_by_score_{i+1}", test_results[i]))

    metrics_csv = os.path.join(run_out_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print("[OK] wrote:", metrics_csv)

    print("\n[OK] wrote:", log_path)


if __name__ == "__main__":
    main()
