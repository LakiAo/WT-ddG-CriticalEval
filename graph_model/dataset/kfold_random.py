import os
import json
import argparse
import numpy as np
import pandas as pd
'''
python ./dataset/kfold_random.py \ 
  --csv ./dataset/mdrdb_graph_mut_ddg.csv \ 
  --out_dir ./dataset/kfold_random \ 
  --seed 42 --k 5 --val_frac 0.15 --ddg_abs_max 8.0 --save_outliers
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./dataset/mdrdb_graph_mut_ddg.csv")
    ap.add_argument("--out_dir", default="./dataset/kfold_random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--ddg_abs_max", type=float, default=8.0)
    ap.add_argument("--save_outliers", action="store_true")
    ap.add_argument("--shuffle", action="store_true", default=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"graph_id", "resnum", "wt_aa", "mut_aa", "ddg", "uniprot_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip().str.upper()
    df = df[df["uniprot_id"] != ""].reset_index(drop=True)

    df["ddg"] = pd.to_numeric(df["ddg"], errors="coerce")
    before_nan = len(df)
    df = df.dropna(subset=["ddg"]).reset_index(drop=True)
    if len(df) != before_nan:
        print(f"[Warn] dropped NaN ddg rows: {before_nan - len(df)}")

    if args.ddg_abs_max and args.ddg_abs_max > 0:
        thr = float(args.ddg_abs_max)
        outliers = df[df["ddg"].abs() > thr].copy()
        if len(outliers) > 0:
            print(f"[Outlier] abs(ddg)>{thr}: {len(outliers)} rows (will drop)")
            if args.save_outliers:
                out_path = os.path.join(args.out_dir, f"outliers_abs_ddg_gt_{thr:g}.csv")
                outliers.to_csv(out_path, index=False)
                print("[Outlier] wrote:", out_path)
            df = df[df["ddg"].abs() <= thr].reset_index(drop=True)
        else:
            print(f"[Outlier] abs(ddg)>{thr}: 0 rows")

    rng = np.random.default_rng(args.seed)
    n = len(df)
    if n < args.k:
        raise ValueError(f"Too few rows after filtering: n={n}, k={args.k}")

    idx = np.arange(n)
    if args.shuffle:
        rng.shuffle(idx)

    k = int(args.k)
    val_frac = float(args.val_frac)
    if not (0.0 < val_frac < 0.5):
        raise ValueError("--val_frac should be in (0, 0.5)")

    folds = np.array_split(idx, k)

    all_meta = {
        "seed": int(args.seed),
        "k": k,
        "val_frac": val_frac,
        "n_total": int(n),
        "fold_sizes": [int(len(f)) for f in folds],
        "folds": [],
    }

    def _stat(name, d):
        y = d["ddg"].astype(float)
        return {
            "name": name,
            "n": int(len(d)),
            "unique_uniprot": int(d["uniprot_id"].nunique()),
            "mean": float(y.mean()),
            "std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
            "min": float(y.min()),
            "max": float(y.max()),
        }

    for i in range(k):
        fold_dir = os.path.join(args.out_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        test_idx = folds[i]
        trainval_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        tv = trainval_idx.copy()
        rng_i = np.random.default_rng(args.seed + 1000 + i)
        rng_i.shuffle(tv)

        n_tv = len(tv)
        n_val = max(1, int(round(n_tv * val_frac)))
        n_train = max(1, n_tv - n_val)

        if n_tv < 2:
            raise ValueError(f"Fold {i}: too few trainval samples: {n_tv}")

        val_idx = tv[:n_val]
        train_idx = tv[n_val:n_val + n_train]

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)
        test_df  = df.iloc[test_idx].reset_index(drop=True)

        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        fold_meta = {
            "fold": int(i),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "stats": {
                "train": _stat("train", train_df),
                "val": _stat("val", val_df),
                "test": _stat("test", test_df),
            }
        }
        all_meta["folds"].append(fold_meta)

        print(f"\n[Fold {i}] wrote -> {fold_dir}")
        print("  train/val/test:", len(train_df), len(val_df), len(test_df))
        print("  test unique uniprot:", int(test_df["uniprot_id"].nunique()))

    meta_path = os.path.join(args.out_dir, "kfold_splits_idx.json")
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    print("\n[OK] k-fold random row split done")
    print("[OK] wrote:", args.out_dir)
    print("[OK] meta :", meta_path)


if __name__ == "__main__":
    main()
