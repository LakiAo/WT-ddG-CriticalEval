import os
import time
import csv
import argparse

import torch
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser

from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from esm.sdk.api import ESMProteinTensor, LogitsConfig

from utils.f_parse_pdb_general import parse_pdb

"""
python -m datapreprocess.esmc_embedding_mt \
  --data_dir ./output \
  --meta_tsv ./MdrDB_CoreSet_release_v1.0.2022.tsv
"""

def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--meta_tsv",type=str,default="./MdrDB_CoreSet_release_v1.0.2022.tsv")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


_CANON_AA = set("ACDEFGHIKLMNPQRSTVWY")

def _norm_aa(ch: str) -> str:
    ch = (ch or "").strip().upper()
    return ch if ch in _CANON_AA else "X"

def load_sample_mut_map(tsv_path: str) -> dict:
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"meta_tsv not found: {tsv_path}")

    m = {}
    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("TSV header not found (empty file?)")

        if "SAMPLE_ID" not in reader.fieldnames or "MUTATION" not in reader.fieldnames:
            raise ValueError(f"TSV must contain SAMPLE_ID and MUTATION, got: {reader.fieldnames}")

        for row in reader:
            sid = (row.get("SAMPLE_ID") or "").strip()
            mut = (row.get("MUTATION") or "").strip()
            if sid and mut:
                if sid not in m:
                    m[sid] = mut
                su = sid.upper()
                if su not in m:
                    m[su] = mut
    return m

def parse_single_mutation_no_re(mutation: str):

    mutation = (mutation or "").strip()
    if len(mutation) < 3:
        raise ValueError(f"Bad MUTATION: {mutation}")

    wt = _norm_aa(mutation[0])
    mt = _norm_aa(mutation[-1])
    mid = mutation[1:-1]

    if not mid.isdigit():
        raise ValueError(f"Bad MUTATION position (not digits): {mutation}")

    if wt == "X" or mt == "X":
        raise ValueError(f"Bad MUTATION AA (non-canonical): {mutation}")

    return wt, mid, mt


def init_esmc(device):
    os.environ["INFRA_PROVIDER"] = "True"
    client = ESMC.from_pretrained("esmc_600m", device=device)
    tokenizer = EsmSequenceTokenizer("esmc_600m")
    embedding_dim = 1152
    return client, tokenizer, embedding_dim


def get_esmc_embedding_for_sequence(sequence: str, client, tokenizer, device, crop_special_tokens: bool = True) -> torch.Tensor:
    ids = tokenizer.encode(sequence)
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)

    tokens = ids.to(device)
    protein_tensor = ESMProteinTensor(sequence=tokens)

    with torch.no_grad():
        out = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
        emb = out.embeddings[0]
        if crop_special_tokens:
            emb = emb[1:-1, :]
    return emb


def get_single_chain_key(prot: dict):

    keys = []
    for k in prot:
        aa = prot[k].get("aa_seq", "")
        if aa:
            keys.append(k)
    if len(keys) != 1:
        return None
    return keys[0]


def build_resnum_to_seqpos(chain_dict: dict):

    aa_seq = chain_dict.get("aa_seq", "")
    aa_residues = chain_dict.get("aa_residues", None)
    if aa_residues is None:
        raise KeyError("chain_dict has no 'aa_residues' (need resnum mapping)")

    items = sorted(aa_residues.items(), key=lambda x: x[0])  # 按遍历顺序
    L = min(len(aa_seq), len(items))

    resnum2pos = {}
    dup_resnums = set()

    for idx in range(L):
        _, rinfo = items[idx]
        resnum = (rinfo.get("resnum") or "").strip()
        if not resnum:
            continue
        if resnum in resnum2pos:
            dup_resnums.add(resnum)
            resnum2pos[resnum] = None
        else:
            resnum2pos[resnum] = idx + 1  # 1-based

    return aa_seq, resnum2pos, (len(aa_seq), len(items)), dup_resnums


def main():
    args = arg_parser()
    data_dir = args.data_dir
    meta_tsv = args.meta_tsv
    overwrite = args.overwrite

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sample_mut_map = load_sample_mut_map(meta_tsv)

    client, tokenizer, embedding_dim = init_esmc(device)
    parser = PDBParser(PERMISSIVE=1, QUIET=True)

    pdb_files = [f for f in os.scandir(data_dir) if f.is_file() and f.name.lower().endswith(".pdb")]
    print(f"Found {len(pdb_files)} pdb files in {data_dir}")

    stats = {
        "total_pdb": 0,
        "already_exists": 0,
        "no_mutation_in_tsv": 0,
        "bad_mutation_format": 0,
        "parse_pdb_failed": 0,
        "multi_chain_ignored": 0,
        "mapping_build_failed": 0,
        "resnum_not_found_or_ambiguous": 0,
        "wt_mismatch": 0,
        "embedding_failed": 0,
        "save_failed": 0,
        "success": 0,
        "mapping_len_mismatch_warn": 0,
        "duplicate_resnum_warn": 0,
    }

    tic = time.time()

    for pdb_file in tqdm(pdb_files):
        stats["total_pdb"] += 1

        pdb_path = pdb_file.path
        base = os.path.splitext(pdb_file.name)[0]

        save_path = os.path.join(data_dir, base + "mt_esmc_600m.pt")
        if (not overwrite) and os.path.exists(save_path):
            stats["already_exists"] += 1
            continue

        mutation = sample_mut_map.get(base) or sample_mut_map.get(base.upper())
        if not mutation:
            stats["no_mutation_in_tsv"] += 1
            continue

        try:
            wt, pos_num_str, mt = parse_single_mutation_no_re(mutation)
        except Exception:
            stats["bad_mutation_format"] += 1
            continue

        try:
            with open(pdb_path) as fh:
                prot = parse_pdb(parser, base, fh)
        except Exception:
            stats["parse_pdb_failed"] += 1
            continue

        chain_key = get_single_chain_key(prot)
        if chain_key is None:
            stats["multi_chain_ignored"] += 1
            continue

        chain_dict = prot[chain_key]

        try:
            aa_seq, resnum2pos, lens_info, dup_resnums = build_resnum_to_seqpos(chain_dict)
        except Exception:
            stats["mapping_build_failed"] += 1
            continue

        aa_len, items_len = lens_info
        if aa_len != items_len:
            stats["mapping_len_mismatch_warn"] += 1
        if len(dup_resnums) > 0:
            stats["duplicate_resnum_warn"] += 1

        seq_pos = resnum2pos.get(pos_num_str, None)
        if seq_pos is None:
            stats["resnum_not_found_or_ambiguous"] += 1
            continue

        if seq_pos > len(aa_seq):
            stats["resnum_not_found_or_ambiguous"] += 1
            continue

        observed = _norm_aa(aa_seq[seq_pos - 1])
        if observed != wt:
            stats["wt_mismatch"] += 1
            continue

        seq_list = list(aa_seq)
        seq_list[seq_pos - 1] = mt
        mut_seq = "".join(seq_list)

        try:
            emb = get_esmc_embedding_for_sequence(mut_seq, client, tokenizer, device)
            emb_all = emb.cpu().float()
        except Exception:
            stats["embedding_failed"] += 1
            continue

        try:
            torch.save(emb_all, save_path)
            stats["success"] += 1
        except Exception:
            stats["save_failed"] += 1

    elapsed = time.time() - tic

    print("\n========== Summary ==========")
    print(f"Data dir: {data_dir}")
    print(f"Meta TSV: {meta_tsv}")
    print(f"Total PDB: {stats['total_pdb']}")
    print(f"Success: {stats['success']}")
    print(f"Already exists (skipped): {stats['already_exists']}")
    print(f"No MUTATION in TSV: {stats['no_mutation_in_tsv']}")
    print(f"Bad MUTATION format: {stats['bad_mutation_format']}")
    print(f"parse_pdb failed: {stats['parse_pdb_failed']}")
    print(f"Multi-chain ignored: {stats['multi_chain_ignored']}")
    print(f"Mapping build failed: {stats['mapping_build_failed']}")
    print(f"Resnum not found/ambiguous: {stats['resnum_not_found_or_ambiguous']}")
    print(f"WT mismatch: {stats['wt_mismatch']}")
    print(f"Embedding failed: {stats['embedding_failed']}")
    print(f"Save failed: {stats['save_failed']}")
    print("--- warnings (not fatal) ---")
    print(f"Mapping length mismatch warns: {stats['mapping_len_mismatch_warn']}")
    print(f"Duplicate resnum warns: {stats['duplicate_resnum_warn']}")
    print(f"Done. Total time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
