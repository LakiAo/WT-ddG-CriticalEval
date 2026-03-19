import os
import json
import argparse
import numpy as np
import pandas as pd

'''
python ./dataset/kfold_uniprot.py \ 
  --csv ./dataset/mdrdb_graph_mut_ddg.csv \ 
  --out_dir ./dataset/kfold_uniprot \ 
  --seed 42 --k 5 --val_frac 0.15 --ddg_abs_max 8.0 --save_outliers
'''

def _check_disjoint(a, b, name_a, name_b):
    inter = set(a) & set(b)
    if inter:
        sample = list(sorted(inter))[:10]
        raise RuntimeError(
            f"[LEAK] {name_a} ∩ {name_b} not empty! size={len(inter)} examples={sample}"
        )


def _check_fold_no_leak(train_df, val_df, test_df, col):
    tu = set(train_df[col].astype(str).str.upper().str.strip().tolist())
    vu = set(val_df[col].astype(str).str.upper().str.strip().tolist())
    su = set(test_df[col].astype(str).str.upper().str.strip().tolist())
    _check_disjoint(tu, vu, "train_uniprot", "val_uniprot")
    _check_disjoint(tu, su, "train_uniprot", "test_uniprot")
    _check_disjoint(vu, su, "val_uniprot", "test_uniprot")


def _shuffle_ties_by_weight(items_sorted, weights, rng):
    i = 0
    while i < len(items_sorted):
        w = weights.get(items_sorted[i], 0)
        j = i + 1
        while j < len(items_sorted) and weights.get(items_sorted[j], 0) == w:
            j += 1
        if j - i > 1:
            block = items_sorted[i:j]
            rng.shuffle(block)
            items_sorted[i:j] = block
        i = j
    return items_sorted


def _pack_by_uniprot_count_then_rows(uniprots, weights, k, rng):

    U = len(uniprots)
    base = U // k
    rem = U % k
    caps = [base + 1 if i < rem else base for i in range(k)]

    items_sorted = sorted(uniprots, key=lambda x: weights.get(x, 0), reverse=True)
    items_sorted = _shuffle_ties_by_weight(items_sorted, weights, rng)

    folds = [[] for _ in range(k)]
    fold_rows = [0] * k
    fold_cnt = [0] * k

    for u in items_sorted:
        w = int(weights.get(u, 0))

        candidates = [i for i in range(k) if fold_cnt[i] < caps[i]]
        if not candidates:
            raise RuntimeError("No candidate fold has remaining capacity. This should not happen.")

        min_rows = min(fold_rows[i] for i in candidates)
        best = [i for i in candidates if fold_rows[i] == min_rows]

        idx = int(rng.choice(best))
        folds[idx].append(u)
        fold_rows[idx] += w
        fold_cnt[idx] += 1

    return folds, fold_rows, caps


def _choose_val_fixed_count_close_rows(candidates, weights, target_cnt, target_rows, rng, refine_iters=2000):

    cand = list(candidates)

    if target_cnt <= 0:
        return [], 0
    if target_cnt >= len(cand):
        total = int(sum(weights.get(u, 0) for u in cand))
        return cand, total

    cand_sorted = sorted(cand, key=lambda x: weights.get(x, 0), reverse=True)
    cand_sorted = _shuffle_ties_by_weight(cand_sorted, weights, rng)

    chosen = []
    chosen_set = set()
    total = 0

    for i, u in enumerate(cand_sorted):
        w = int(weights.get(u, 0))
        remaining = len(cand_sorted) - i
        need = target_cnt - len(chosen)

        if need <= 0:
            break

        if remaining == need:
            chosen.append(u)
            chosen_set.add(u)
            total += w
            continue

        cur_dev = abs(total - target_rows)
        new_dev = abs((total + w) - target_rows)

        if total < target_rows:
            if new_dev <= cur_dev:
                chosen.append(u); chosen_set.add(u); total += w
        else:
            if new_dev <= cur_dev:
                chosen.append(u); chosen_set.add(u); total += w

    if len(chosen) < target_cnt:
        for u in cand_sorted:
            if u in chosen_set:
                continue
            chosen.append(u)
            chosen_set.add(u)
            total += int(weights.get(u, 0))
            if len(chosen) >= target_cnt:
                break

    rest = [u for u in cand_sorted if u not in chosen_set]
    best_total = total
    best_dev = abs(best_total - target_rows)

    if len(rest) > 0 and refine_iters > 0:
        for _ in range(refine_iters):
            a = chosen[int(rng.integers(0, len(chosen)))]
            b = rest[int(rng.integers(0, len(rest)))]
            wa = int(weights.get(a, 0))
            wb = int(weights.get(b, 0))
            new_total = best_total - wa + wb
            new_dev = abs(new_total - target_rows)
            if new_dev < best_dev:
                chosen.remove(a)
                chosen.append(b)
                rest.remove(b)
                rest.append(a)
                chosen_set.remove(a)
                chosen_set.add(b)
                best_total = new_total
                best_dev = new_dev

    return chosen, int(best_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="./dataset/mdrdb_graph_mut_ddg.csv")
    ap.add_argument("--out_dir", default="./dataset/kfold_uniprot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--ddg_abs_max", type=float, default=8.0)
    ap.add_argument("--save_outliers", action="store_true")
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--uniprot_col", type=str, default="uniprot_id")
    ap.add_argument("--also_check_graph_id", action="store_true")
    ap.add_argument("--val_refine_iters", type=int, default=4000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    required = {"graph_id", "resnum", "wt_aa", "mut_aa", "ddg", args.uniprot_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df[args.uniprot_col] = df[args.uniprot_col].astype(str).str.strip().str.upper()
    df = df[df[args.uniprot_col] != ""].reset_index(drop=True)

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

    g = df.groupby(args.uniprot_col).size()
    weights = {u: int(c) for u, c in g.items()}
    uniprots = list(weights.keys())

    if len(uniprots) < args.k:
        raise ValueError(f"unique {args.uniprot_col} ({len(uniprots)}) < k ({args.k}). Reduce k.")

    if not (0.0 < float(args.val_frac) < 0.5):
        raise ValueError("--val_frac should be in (0, 0.5)")

    rng = np.random.default_rng(args.seed)

    folds, fold_rows, caps = _pack_by_uniprot_count_then_rows(uniprots, weights, args.k, rng=rng)

    total_rows = int(len(df))
    total_uni = int(len(uniprots))
    target_rows = total_rows / float(args.k)
    target_cnt = total_uni / float(args.k)

    summary = {
        "k": int(args.k),
        "seed": int(args.seed),
        "uniprot_col": args.uniprot_col,
        "val_frac": float(args.val_frac),
        "ddg_abs_max": float(args.ddg_abs_max),
        "total_rows": int(total_rows),
        "total_uniprot": int(total_uni),
        "targets": {"rows_per_fold": float(target_rows), "uniprot_per_fold": float(target_cnt)},
        "folds_balance": [],
        "folds": [],
    }

    print("=== Test folds (UNIPROT-count first, rows second) ===")
    for i in range(args.k):
        rec = {
            "fold": int(i),
            "cap_uniprot": int(caps[i]),
            "n_uniprot": int(len(folds[i])),
            "rows": int(fold_rows[i]),
            "rows_delta_to_target": float(fold_rows[i] - target_rows),
        }
        summary["folds_balance"].append(rec)
        print("[FOLD]", rec)

    for i in range(args.k):
        test_ids = list(folds[i])
        remain_ids = [u for j in range(args.k) if j != i for u in folds[j]]

        test_rows = int(sum(weights[u] for u in test_ids))
        remain_rows = int(total_rows - test_rows)

        target_val_rows = int(round(remain_rows * float(args.val_frac)))
        target_val_rows = max(1, target_val_rows)

        target_val_cnt = int(round(len(remain_ids) * float(args.val_frac)))
        target_val_cnt = max(1, target_val_cnt)

        rng_i = np.random.default_rng(args.seed + 1000 + i)

        val_ids, val_rows = _choose_val_fixed_count_close_rows(
            remain_ids, weights,
            target_cnt=target_val_cnt,
            target_rows=target_val_rows,
            rng=rng_i,
            refine_iters=int(args.val_refine_iters),
        )
        val_set = set(val_ids)
        train_ids = [u for u in remain_ids if u not in val_set]

        _check_disjoint(train_ids, val_ids, "train_uniprot", "val_uniprot")
        _check_disjoint(train_ids, test_ids, "train_uniprot", "test_uniprot")
        _check_disjoint(val_ids, test_ids, "val_uniprot", "test_uniprot")

        fold_dir = os.path.join(args.out_dir, f"fold_{i}")
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, "splits.json"), "w") as f:
            json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

        train_df = df[df[args.uniprot_col].isin(train_ids)].reset_index(drop=True)
        val_df   = df[df[args.uniprot_col].isin(val_ids)].reset_index(drop=True)
        test_df  = df[df[args.uniprot_col].isin(test_ids)].reset_index(drop=True)

        _check_fold_no_leak(train_df, val_df, test_df, args.uniprot_col)

        if args.also_check_graph_id:
            tg = set(train_df["graph_id"].astype(str).tolist())
            vg = set(val_df["graph_id"].astype(str).tolist())
            sg = set(test_df["graph_id"].astype(str).tolist())
            _check_disjoint(tg, vg, "train_graph_id", "val_graph_id")
            _check_disjoint(tg, sg, "train_graph_id", "test_graph_id")
            _check_disjoint(vg, sg, "val_graph_id", "test_graph_id")

        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

        info = {
            "fold": int(i),
            "train_uniprot": int(len(train_ids)),
            "val_uniprot": int(len(val_ids)),
            "test_uniprot": int(len(test_ids)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "target_val_uniprot": int(target_val_cnt),
            "target_val_rows": int(target_val_rows),
            "actual_val_rows": int(val_rows),
            "remain_rows": int(remain_rows),
        }
        summary["folds"].append(info)
        print("[OK]", info)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] wrote k-fold splits to: {args.out_dir}")
    print("Note: test folds are balanced by uniprot count (hard), rows are secondary.")
    print("      val is balanced by uniprot count first, rows second (with swap refinement).")


if __name__ == "__main__":
    main()