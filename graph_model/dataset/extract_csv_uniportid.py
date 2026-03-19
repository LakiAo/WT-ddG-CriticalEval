import os
import re
import glob
import argparse
import pandas as pd

AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
    "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
    "THR":"T","TRP":"W","TYR":"Y","VAL":"V",
}

def extract_graph_ids(pth_dir: str):
    pattern = re.compile(r"^(MdrDBcore\d+)_graph\.pth$", re.IGNORECASE)
    ids = set()
    for fp in glob.glob(os.path.join(pth_dir, "*.pth")):
        base = os.path.basename(fp)
        m = pattern.match(base)
        if m:
            gid = m.group(1)
            gid = "MdrDBcore" + gid[len("MdrDBcore"):]
            ids.add(gid)
    return ids

def parse_single_mutation(mutation_field: str):
    if mutation_field is None or (isinstance(mutation_field, float) and pd.isna(mutation_field)):
        return None
    s = str(mutation_field).strip()
    if not s:
        return None

    s = s.strip("()[]{}")

    m1 = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", s)
    if m1:
        wt, resnum, mut = m1.group(1).upper(), int(m1.group(2)), m1.group(3).upper()
        return (resnum, wt, mut)

    m3 = re.match(r"^([A-Za-z]{3})(\d+)([A-Za-z]{3})$", s)
    if m3:
        wt3, resnum, mut3 = m3.group(1).upper(), int(m3.group(2)), m3.group(3).upper()
        if wt3 in AA3_TO_1 and mut3 in AA3_TO_1:
            return (resnum, AA3_TO_1[wt3], AA3_TO_1[mut3])

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="./MdrDB_CoreSet_release_v1.0.2022.tsv")
    ap.add_argument("--pth_dir", default="./pth")
    ap.add_argument("--out_csv", default="./mdrdb_graph_mut_ddg.csv")
    ap.add_argument("--drop_invalid", action="store_true")
    args = ap.parse_args()

    graph_ids = extract_graph_ids(args.pth_dir)
    if not graph_ids:
        raise RuntimeError(f"No matching *.pth found in {args.pth_dir}")

    df = pd.read_csv(args.tsv, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    uniprot_col = None
    for c in ["UNIPROT_ID", "UNIPROT", "UniProt"]:
        if c in df.columns:
            uniprot_col = c
            break
    if uniprot_col is None:
        raise KeyError(f"No UNIPROT column found. Available: {list(df.columns)}")

    required_cols = ["SAMPLE_ID", uniprot_col, "MUTATION", "DDG.EXP"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"TSV missing columns: {missing}")

    df = df[df["SAMPLE_ID"].isin(graph_ids)].copy()

    rows = []
    for _, r in df.iterrows():
        gid = str(r["SAMPLE_ID"]).strip()
        uniprot_id = str(r[uniprot_col]).strip()

        mut = parse_single_mutation(r["MUTATION"])
        if mut is None:
            resnum = wt = mutaa = None
        else:
            resnum, wt, mutaa = mut

        ddg_raw = r["DDG.EXP"]
        ddg = None
        if ddg_raw is not None and not (isinstance(ddg_raw, float) and pd.isna(ddg_raw)):
            s = str(ddg_raw).strip()
            if s != "":
                try:
                    ddg = float(s)
                except ValueError:
                    ddg = None

        rows.append({
            "graph_id": gid,
            "resnum": resnum,
            "wt_aa": wt,
            "mut_aa": mutaa,
            "ddg": ddg,
            "uniprot_id": uniprot_id
        })

    out = pd.DataFrame(rows, columns=["graph_id", "resnum", "wt_aa", "mut_aa", "ddg", "uniprot_id"])

    if args.drop_invalid:
        out = out.dropna(subset=["resnum", "wt_aa", "mut_aa", "ddg"])

    out["resnum"] = pd.to_numeric(out["resnum"], errors="coerce").astype("Int64")
    out["ddg"] = pd.to_numeric(out["ddg"], errors="coerce")
    out["uniprot_id"] = out["uniprot_id"].fillna("").astype(str)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"[OK] matched_samples={df.shape[0]} output_rows={out.shape[0]} -> {args.out_csv}")

if __name__ == "__main__":
    main()
