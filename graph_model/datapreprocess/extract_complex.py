import os
import sys
import argparse
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

'''
python WT-ddG-CriticalEval/graph_model/datapreprocess/extract_complex.py \ --data_dir /path/to/data \ --tsv_path /path/to/MdrDB_CoreSet_release_v1.0.2022.tsv \ --output_dir /path/to/output
'''

SAMPLE_ID_COL = "SAMPLE_ID"
SMILES_COL = "SMILES"

WATER_RESNAMES = {"HOH", "WAT", "H2O"}
METAL_ELEMENTS = {
    "LI", "NA", "K", "RB", "CS",
    "MG", "CA", "SR", "BA",
    "MN", "FE", "CO", "NI", "CU", "ZN",
    "CD", "HG",
    "AL", "CR", "PT", "PD", "AG", "AU",
}
METAL_RESNAMES = METAL_ELEMENTS.copy()
HALIDE_RESNAMES = {"CL", "BR", "I", "F"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract protein-ligand pairs from PDB dataset.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tsv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    return parser.parse_args()


def load_smiles_table(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    if SAMPLE_ID_COL not in df.columns:
        raise ValueError(f"Column not found in TSV: {SAMPLE_ID_COL}")
    if SMILES_COL not in df.columns:
        raise ValueError(f"Column not found in TSV: {SMILES_COL}")
    df = df.set_index(SAMPLE_ID_COL)
    return df


def find_complex_pdb_in_dir(dir_path: str) -> str | None:
    for fname in os.listdir(dir_path):
        lower = fname.lower()
        if not lower.endswith(".pdb"):
            continue
        if not fname.startswith("WT_"):
            continue
        if "complex" not in lower:
            continue
        return os.path.join(dir_path, fname)
    return None


def is_water_or_metal_or_halide(line: str) -> bool:
    resname = line[17:20].strip().upper()
    element = line[76:78].strip().upper()
    if resname in WATER_RESNAMES:
        return True
    if resname in METAL_RESNAMES:
        return True
    if element in METAL_ELEMENTS:
        return True
    if resname in HALIDE_RESNAMES:
        return True
    return False


def extract_ligand_groups_from_pdb(pdb_path: str):
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    hetatm_lines = [l for l in lines if l.startswith("HETATM")]
    filtered = [l for l in hetatm_lines if not is_water_or_metal_or_halide(l)]

    groups = defaultdict(list)
    for line in filtered:
        resname = line[17:20].strip()
        chain = line[21].strip()
        resseq = line[22:26].strip()
        icode = line[26].strip()
        key = (resname, chain, resseq, icode)
        groups[key].append(line)

    return groups


def sdf_from_mol(mol, out_path: str):
    mol_H = Chem.AddHs(mol, addCoords=True)
    writer = Chem.SDWriter(out_path)
    writer.write(mol_H)
    writer.close()


def canonical_smiles_loose(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def morgan_fp(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def ligand_mol_from_pdb_with_template(pdb_block: str, template_smiles: str | None):
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False)
    if mol is None:
        return None

    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None

    frag = max(frags, key=lambda m: m.GetNumAtoms())

    if template_smiles:
        templ = Chem.MolFromSmiles(template_smiles)
        if templ is not None:
            try:
                templ_noh = Chem.RemoveHs(templ)
                frag_noh = Chem.RemoveHs(frag)
                frag_noh = AllChem.AssignBondOrdersFromTemplate(templ_noh, frag_noh)
                return frag_noh
            except Exception:
                pass
            try:
                templ_h = Chem.AddHs(templ)
                frag_h = Chem.AddHs(frag)
                frag_h = AllChem.AssignBondOrdersFromTemplate(templ_h, frag_h)
                frag_h = Chem.RemoveHs(frag_h)
                return frag_h
            except Exception:
                pass

    frag = Chem.RemoveHs(frag)
    return frag


def write_receptor_without_ligand(original_pdb: str,
                                  ligand_keys,
                                  out_path: str,
                                  remove_water_and_metals: bool = True):

    ligand_keys = set(ligand_keys)

    with open(original_pdb, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            record = line[:6].strip()

            if record not in ("ATOM", "HETATM"):
                fout.write(line)
                continue

            if record == "ATOM":
                fout.write(line)
                continue

            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = line[22:26].strip()
            icode = line[26].strip()
            key = (resname, chain, resseq, icode)

            if key in ligand_keys:
                continue

            if remove_water_and_metals and is_water_or_metal_or_halide(line):
                continue

            fout.write(line)


def process_one_sample(sample_dir: str, df_smiles: pd.DataFrame, output_dir: str):
    sample_id = os.path.basename(sample_dir)
    print(f"\n=== Processing sample: {sample_id} ===")

    if sample_id not in df_smiles.index:
        print(f"  [WARN] SAMPLE_ID = {sample_id} not found in TSV. Skipping.")
        return None

    smiles_raw = df_smiles.loc[sample_id, SMILES_COL]
    tmpl_mol = Chem.MolFromSmiles(smiles_raw)
    if tmpl_mol is None:
        print(f"  [WARN] Failed to parse SMILES: {smiles_raw!r}. Skipping.")
        return None

    tmpl_mol = Chem.RemoveHs(tmpl_mol)
    tmpl_smiles_loose = canonical_smiles_loose(tmpl_mol)
    tmpl_fp = morgan_fp(tmpl_mol)

    pdb_path = find_complex_pdb_in_dir(sample_dir)
    if pdb_path is None:
        print("  [WARN] WT_*_complex*.pdb not found. Skipping.")
        return None

    ligand_groups = extract_ligand_groups_from_pdb(pdb_path)
    if not ligand_groups:
        return None

    for key, atom_lines in ligand_groups.items():
        pdb_block = "HEADER\n" + "".join(atom_lines) + "END\n"
        lig_mol = ligand_mol_from_pdb_with_template(pdb_block, smiles_raw)
        if lig_mol is None:
            continue

        lig_fp = morgan_fp(lig_mol)
        sim = DataStructs.TanimotoSimilarity(tmpl_fp, lig_fp)

        if sim == 1.0:
            os.makedirs(output_dir, exist_ok=True)
            protein_pdb_out = os.path.join(output_dir, f"{sample_id}.pdb")
            ligand_sdf_out = os.path.join(output_dir, f"{sample_id}.sdf")

            write_receptor_without_ligand(pdb_path, [key], protein_pdb_out)
            sdf_from_mol(lig_mol, ligand_sdf_out)

            print(f"  [DONE] {sample_id}")
            return sample_id, sim

    return None


def main():
    args = parse_args()

    DATA_DIR = args.data_dir
    TSV_PATH = args.tsv_path
    OUTPUT_DIR = args.output_dir

    df_smiles = load_smiles_table(TSV_PATH)

    subdirs = [
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    for sd in subdirs:
        process_one_sample(sd, df_smiles, OUTPUT_DIR)


if __name__ == "__main__":
    main()