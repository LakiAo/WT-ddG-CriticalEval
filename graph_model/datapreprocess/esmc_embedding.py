import os
import re
import time
import argparse

import torch
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser

from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from esm.sdk.api import ESMProteinTensor, LogitsConfig

from utils.f_parse_pdb_general import parse_pdb


"""
This script computes residue embeddings using ESMC 600M for all *.pdb files in a specified directory, and saves them as <same_filename>_esmc_600m.pt.
python -m datapreprocess.esmc_embedding     --data_dir ./output
"""


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Compute ESMC embeddings for all PDB files in a directory."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )
    return parser.parse_args()

def init_esmc(device):
    os.environ["INFRA_PROVIDER"] = "True"
    client = ESMC.from_pretrained("esmc_600m", device=device)
    tokenizer = EsmSequenceTokenizer("esmc_600m")
    embedding_dim = 1152
    return client, tokenizer, embedding_dim


def get_esmc_embedding_for_sequence(
    sequence: str,
    client,
    tokenizer,
    device,
    crop_special_tokens: bool = True,
) -> torch.Tensor:

    ids = tokenizer.encode(sequence)
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)

    tokens = ids.to(device)
    protein_tensor = ESMProteinTensor(sequence=tokens)

    with torch.no_grad():
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        emb = logits_output.embeddings
        emb = emb[0]

        if crop_special_tokens:
            emb = emb[1:-1, :]

    return emb

def main():
    args = arg_parser()
    data_dir = args.data_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    client, tokenizer, embedding_dim = init_esmc(device)
    parser = PDBParser(PERMISSIVE=1, QUIET=True)

    pdb_files = [
        f for f in os.scandir(data_dir)
        if f.is_file() and f.name.lower().endswith(".pdb")
    ]
    print(f"Found {len(pdb_files)} pdb files in {data_dir}")

    tic = time.time()

    for pdb_file in tqdm(pdb_files):
        pdb_path = pdb_file.path
        base = os.path.splitext(pdb_file.name)[0]

        save_path = os.path.join(data_dir, base + "_esmc_600m.pt")

        if os.path.exists(save_path):
            print(f"[Skip] {pdb_file.name} → already has _esmc_600m.pt")
            continue

        print(f"[Processing] {pdb_file.name}")

        try:
            with open(pdb_path) as fh:
                prot = parse_pdb(parser, base, fh)
        except Exception as e:
            print(f"[Error] parse_pdb failed for {pdb_file.name}: {e}")
            continue

        emb_all = torch.empty(0, embedding_dim, dtype=torch.float32)

        try:
            for chain in prot:
                aa_seq = prot[chain]["aa_seq"]
                if len(aa_seq) == 0:
                    continue

                emb_chain = get_esmc_embedding_for_sequence(
                    aa_seq, client, tokenizer, device
                )

                emb_all = torch.vstack((emb_all, emb_chain.cpu().float()))

        except Exception as e:
            print(f"[Error] embedding failed for {pdb_file.name}: {e}")
            continue

        try:
            torch.save(emb_all, save_path)
            print(f"[Saved] {save_path}")
        except Exception as e:
            print(f"[Error] saving failed for {pdb_file.name}: {e}")

    elapsed = time.time() - tic
    print(f"Done. Total time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
