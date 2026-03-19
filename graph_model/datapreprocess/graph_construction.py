# ------------------------------------------------------------------------------
# This file contains code adapted from the GEMS project.
# Original Author: David Graber
# Original Repository: https://github.com/camlab-ethz/GEMS
# License: MIT License
# ------------------------------------------------------------------------------

import os
import glob
import argparse
import numpy as np
import re

from Bio.PDB.PDBParser import PDBParser
from utils.f_parse_pdb_general import parse_pdb
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.data import HeteroData

SELF_LOOP_FEATURE_VECTOR = torch.tensor(
    [0., 1., 0.,
     0., 0., 0., 0.,
     0., 0., 0., 0., 0.,
     0., 0.,
     0., 0., 0., 0., 0., 0.], dtype=torch.float
)

AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL"
]

HETATM_SMILES_DICT = {
    'ZN': '[Zn+2]', 'MG': '[Mg+2]', 'NA': '[Na+1]', 'MN': '[Mn+2]',
    'CA': '[Ca+2]', 'K': '[K+1]', 'NI': '[Ni+2]', 'FE': '[Fe+2]',
    'CO': '[Co+2]', 'HG': '[Hg+2]', 'CD': '[Cd+2]', 'CU': '[Cu+2]',
    'CS': '[Cs+1]', 'AU': '[Au+1]', 'LI': '[Li+1]', 'GA': '[Ga+3]',
    'IN': '[In+3]', 'BA': '[Ba+2]', 'RB': '[Rb+1]', 'SR': '[Sr+2]',
    'CL': '[Cl-1]'
}

ALL_ATOMS = ['B', 'C', 'N', 'O', 'P', 'S', 'Se', 'metal', 'halogen']
HALOGENS = ['F', 'Cl', 'Br', 'I', 'At']
METALS = [
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re',
    'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 'Pa',
    'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi', 'As', 'Si', 'Sb', 'Te'
]

NUM_ATOMFEATURES = 40
NUM_EDGEFEATURES = 20

MIN_NEAR_AA_DIST = 5.0

def arg_parser():

    parser = argparse.ArgumentParser(description="Whole-Protein HeteroGraph Builder (Robust + Embedding Align)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--replace', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument('--masternode', default=True, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument('--protein_embeddings', default=True, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument('--on_embedding_mismatch', type=str, default='skip_complex',
                        choices=['skip_complex', 'disable_embeddings', 'error'])
    parser.add_argument('--include_hetatm', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument('--protein_seq_edges', default=True, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument('--protein_knn_edges', default=True, type=lambda x: x.lower() in ['true', '1', 'yes'])
    parser.add_argument('--protein_knn_k', type=int, default=16)
    parser.add_argument('--contact_max_len', type=float, default=7.0)

    return parser.parse_args()

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set{allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def parse_sdf_file(file_path):

    suppl = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=True, strictParsing=True)
    return [mol for mol in suppl if mol is not None]


def sanitize_ligand_mol(mol: Chem.Mol):

    if mol is None:
        return None
    try:
        mol2 = Chem.RemoveHs(mol, sanitize=True)
        if mol2 is None:
            return None
        if mol2.GetNumConformers() == 0:
            return None
        return mol2
    except Exception:
        return None


def get_atom_features(mol, all_atoms, padding_len=0):

    x = []
    for atom in mol.GetAtoms():
        padding = [0 for _ in range(padding_len)]
        symbol = atom.GetSymbol()
        if symbol in METALS:
            symbol = 'metal'
        elif symbol in HALOGENS:
            symbol = 'halogen'
        if symbol == 'H':
            continue

        atom_encoding = one_of_k_encoding(symbol, all_atoms)
        ringm = [atom.IsInRing()]
        hybr = one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP2D,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
        ])
        charge = [float(atom.GetFormalCharge())]
        aromatic = [atom.GetIsAromatic()]
        mass = [atom.GetMass() / 100]
        numHs = one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        degree = one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 'OTHER'])
        chirality = one_of_k_encoding_unk(str(atom.GetChiralTag()), [
            'CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'OTHER'
        ])

        results = atom_encoding + ringm + hybr + charge + aromatic + mass + numHs + degree + chirality + padding
        x.append(results)
    return np.array(x, dtype=np.float32)


def make_undirected_with_self_loops(edge_index, edge_attr, undirected=True, self_loops=True, num_nodes=None):
    if undirected:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    if self_loops:
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=SELF_LOOP_FEATURE_VECTOR, num_nodes=num_nodes
        )
    return edge_index, edge_attr


def edge_index_and_attr(mol, pos, undirected=True, self_loops=True):
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        edge_index[0].append(a1)
        edge_index[1].append(a2)

        feat = []
        feat.extend(one_of_k_encoding('covalent', ['covalent', 'self-loop', 'non-covalent']))
        length = np.linalg.norm(pos[a1] - pos[a2])
        feat.extend([(length / 10)] * 4)
        feat.extend(one_of_k_encoding(bond.GetBondTypeAsDouble(), [0., 1.0, 1.5, 2.0, 3.0]))
        feat.append(bond.GetIsConjugated())
        feat.append(bond.IsInRing())
        allowed_stereo = [
            Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS
        ]
        feat.extend(one_of_k_encoding(bond.GetStereo(), allowed_stereo))
        edge_attr.append(feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    if len(edge_attr) == 0:
        edge_attr = torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float)
    else:
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return make_undirected_with_self_loops(edge_index, edge_attr, undirected=undirected, self_loops=self_loops)


def calculate_cbeta_position(ca_coords, c_coords, n_coords):
    ca, c, n = np.array(ca_coords), np.array(c_coords), np.array(n_coords)
    bond_length_ca_cb = 1.54
    bond_angle_n_ca_cb = np.deg2rad(109.5)
    bond_angle_c_ca_cb = np.deg2rad(109.5)

    u_n_ca = (n - ca) / np.linalg.norm(n - ca)
    u_c_ca = (c - ca) / np.linalg.norm(c - ca)
    u_orth = np.cross(u_n_ca, u_c_ca)
    norm_orth = np.linalg.norm(u_orth)
    if norm_orth < 1e-6:
        return ca
    u_orth /= norm_orth

    u_plane = np.cross(u_orth, u_n_ca)
    u_plane /= np.linalg.norm(u_plane)

    cb = ca + bond_length_ca_cb * (
        np.cos(bond_angle_n_ca_cb) * u_n_ca +
        np.sin(bond_angle_n_ca_cb) * (np.cos(bond_angle_c_ca_cb) * u_plane + np.sin(bond_angle_c_ca_cb) * u_orth)
    )
    return cb


def residue_representative_coord(res_dict, resname, amino_acids):
    if resname in amino_acids:
        if 'atoms' in res_dict and 'coords' in res_dict and 'CA' in res_dict['atoms']:
            idx = res_dict['atoms'].index('CA')
            return np.array(res_dict['coords'][idx], dtype=np.float32)
        if 'coords' in res_dict and len(res_dict['coords']) > 0:
            return np.mean(res_dict['coords'], axis=0).astype(np.float32)

    coords = res_dict.get('hetatmcoords', res_dict.get('coords', None))
    if coords is not None and len(coords) > 0:
        return np.mean(coords, axis=0).astype(np.float32)
    return None


def build_protein_seq_edges(chain_to_locals_aa, protein_pos: torch.Tensor):
    src, dst = [], []
    for _, lst in chain_to_locals_aa.items():
        if len(lst) < 2:
            continue
        for i in range(len(lst) - 1):
            a, b = lst[i], lst[i + 1]
            dist_sq = ((protein_pos[a] - protein_pos[b]) ** 2).sum()
            if dist_sq < 225.0:
                src.extend([a, b])
                dst.extend([b, a])

    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    dist = torch.norm(protein_pos[edge_index[0]] - protein_pos[edge_index[1]], dim=1).view(-1, 1) / 10.0
    edge_attr = torch.zeros((dist.size(0), NUM_EDGEFEATURES), dtype=torch.float)
    edge_attr[:, 2] = 1.0
    edge_attr[:, 3:7] = dist.repeat(1, 4)
    return edge_index, edge_attr


def build_protein_knn_edges(protein_pos: torch.Tensor, k: int = 16):
    N = protein_pos.size(0)
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float)
    k = min(k, N - 1)

    dist_mat = torch.cdist(protein_pos, protein_pos)
    values, indices = dist_mat.topk(k + 1, largest=False)
    knn_indices = indices[:, 1:]
    knn_values = values[:, 1:]

    src = torch.arange(N, dtype=torch.long).view(-1, 1).repeat(1, k).reshape(-1)
    dst = knn_indices.reshape(-1)
    edge_index = torch.stack([src, dst], dim=0)

    dist = knn_values.reshape(-1, 1) / 10.0
    edge_attr = torch.zeros((dist.size(0), NUM_EDGEFEATURES), dtype=torch.float)
    edge_attr[:, 2] = 1.0
    edge_attr[:, 3:7] = dist.repeat(1, 4)

    return to_undirected(edge_index, edge_attr)


def build_self_loop_relation(num_nodes: int, feature_vec: torch.Tensor):
    if num_nodes <= 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, feature_vec.numel()), dtype=torch.float)
    idx = torch.arange(num_nodes, dtype=torch.long)
    edge_index = torch.stack([idx, idx], dim=0)
    edge_attr = feature_vec.view(1, -1).repeat(num_nodes, 1).to(torch.float)
    return edge_index, edge_attr


def parse_resnum_for_order(resnum_raw: str):
    s = str(resnum_raw).strip()
    if not s:
        return (10**18, '~~')

    m = re.match(r'^(-?\d+)\s*([A-Za-z]?)$', s)
    if m:
        resseq = int(m.group(1))
        icode = m.group(2) if m.group(2) else ''
        return (resseq, icode)

    m2 = re.match(r'^(-?\d+)(.*)$', s)
    if m2:
        resseq = int(m2.group(1))
        icode = m2.group(2).strip()
        return (resseq, icode)

    return (10**18, s)


def iter_residues_by_resnum(res_map: dict):
    items = list(res_map.items())
    items.sort(key=lambda kv: parse_resnum_for_order(kv[1].get('resnum', '')))
    return items


class SkipComplexException(Exception):
    pass


def load_and_align_embeddings(emb_path: str,
                             aa_keep_mask: np.ndarray,
                             kept_aa_count: int,
                             on_mismatch: str):
    emb_data = torch.load(emb_path)
    if isinstance(emb_data, torch.Tensor):
        aa_emb = emb_data.detach().cpu().numpy().astype(np.float32)
    else:
        aa_emb = np.array(emb_data, dtype=np.float32)

    if aa_emb.ndim != 2:
        msg = f"Embedding has wrong shape: {aa_emb.shape}"
        if on_mismatch == 'error':
            raise ValueError(msg)
        if on_mismatch == 'disable_embeddings':
            print(f"  - [Warn] {msg}. Disable embeddings for this complex.", flush=True)
            return None
        raise SkipComplexException(msg)

    L = aa_emb.shape[0]
    total_aa = int(aa_keep_mask.shape[0])

    if L == kept_aa_count:
        return aa_emb

    if L == total_aa:
        aa_kept = aa_emb[aa_keep_mask]
        if aa_kept.shape[0] != kept_aa_count:
            msg = f"Embedding masking failed: got {aa_kept.shape[0]} kept rows, expected {kept_aa_count}"
            if on_mismatch == 'error':
                raise ValueError(msg)
            if on_mismatch == 'disable_embeddings':
                print(f"  - [Warn] {msg}. Disable embeddings for this complex.", flush=True)
                return None
            raise SkipComplexException(msg)
        return aa_kept

    msg = (f"Embedding length mismatch: emb_len={L}, kept_AA={kept_aa_count}, "
           f"total_AA_including_insertion={total_aa}. Cannot safely align.")
    if on_mismatch == 'error':
        raise ValueError(msg)
    if on_mismatch == 'disable_embeddings':
        print(f"  - [Warn] {msg} Disable embeddings for this complex.", flush=True)
        return None
    raise SkipComplexException(msg)

def main():
    args = arg_parser()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Whole-Protein HeteroGraphs (Robust Order + Embedding Align) ===", flush=True)
    print(f"Input: {args.data_dir} | Output: {args.output_dir}", flush=True)
    print(f"Rule: skip ligand if no AA residue within {MIN_NEAR_AA_DIST:.1f} Å", flush=True)
    print(f"Embedding mismatch policy: {args.on_embedding_mismatch}", flush=True)
    print(f"Rule: skip complex if AA-chain count != 1", flush=True)
    print(f"Rule: ligand RemoveHs enforced to prevent bond index OOB", flush=True)
    print(f"Embedding naming: WT={{{'{base}'}}}_esmc_600m.pt | MT={{{'{base}'}}}mt_esmc_600m.pt", flush=True)

    pdb_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pdb")))
    parser = PDBParser(PERMISSIVE=1, QUIET=True)

    success_count = 0
    fail_count = 0
    skipped_ligand_count = 0
    skipped_multichain_count = 0
    skipped_bad_ligand_count = 0

    for i, pdb_path in enumerate(pdb_files):
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        sdf_path = os.path.join(args.data_dir, base_name + ".sdf")

        emb_wt_path = os.path.join(args.data_dir, base_name + "_esmc_600m.pt")
        emb_mt_path = os.path.join(args.data_dir, base_name + "mt_esmc_600m.pt")

        if not os.path.exists(sdf_path):
            continue

        if args.protein_embeddings:
            if (not os.path.exists(emb_wt_path)) or (not os.path.exists(emb_mt_path)):
                continue

        print(f"Processing {base_name} ({i + 1}/{len(pdb_files)})", flush=True)

        try:
            with open(pdb_path) as f:
                protein_dict = parse_pdb(parser, base_name, f)

            aa_chain_keys = []
            aa_chain_ids = []
            for k, v in protein_dict.items():
                comp = v.get('composition', [False, False])
                has_aa = bool(comp[0]) and len(v.get('aa_residues', {})) > 0
                if has_aa:
                    aa_chain_keys.append(k)
                    aa_chain_ids.append(v.get('chain_id', str(k)))

            if len(aa_chain_keys) != 1:
                print(f"  - Skip {base_name}: AA chains != 1, found {len(aa_chain_keys)} chains {aa_chain_ids}", flush=True)
                skipped_multichain_count += 1
                continue

            residues_dict = {}
            res_idx_counter = 1

            aa_keep_flags = []
            kept_aa_count = 0

            all_prot_atoms_coords = []
            all_prot_atoms_residx = []

            for chain in protein_dict:
                chain_id = protein_dict[chain].get('chain_id', str(chain))

                aa_res = protein_dict[chain].get('aa_residues', {})
                for _, res in iter_residues_by_resnum(aa_res):
                    resnum_raw = str(res.get('resnum', '')).strip()
                    is_insertion = bool(resnum_raw and re.match(r'^\d+.*[A-Za-z].*$', resnum_raw))

                    aa_keep_flags.append(not is_insertion)

                    if is_insertion:
                        continue

                    res['_is_aa'] = True
                    res['_chain_id'] = chain_id
                    res['_aa_counter'] = kept_aa_count
                    kept_aa_count += 1

                    residues_dict[res_idx_counter] = res

                    coords = res.get('coords', [])
                    if len(coords) > 0:
                        coords_np = np.array(coords, dtype=np.float32)
                        all_prot_atoms_coords.append(coords_np)
                        all_prot_atoms_residx.extend([res_idx_counter] * coords_np.shape[0])

                    res_idx_counter += 1

                if args.include_hetatm:
                    het_res = protein_dict[chain].get('hetatm_residues', {})
                    for _, res in iter_residues_by_resnum(het_res):
                        resnum_raw = str(res.get('resnum', '')).strip()
                        if resnum_raw and re.match(r'^\d+.*[A-Za-z].*$', resnum_raw):
                            continue

                        res['_is_aa'] = False
                        res['_chain_id'] = chain_id
                        residues_dict[res_idx_counter] = res

                        coords = res.get('hetatmcoords', [])
                        if len(coords) > 0:
                            coords_np = np.array(coords, dtype=np.float32)
                            all_prot_atoms_coords.append(coords_np)
                            all_prot_atoms_residx.extend([res_idx_counter] * coords_np.shape[0])

                        res_idx_counter += 1

            if not all_prot_atoms_coords:
                raise SkipComplexException("No protein atoms found.")

            all_prot_atoms_coords_np = np.vstack(all_prot_atoms_coords).astype(np.float32)
            all_prot_atoms_residx_np = np.array(all_prot_atoms_residx, dtype=np.int64)

            aa_keep_mask = np.array(aa_keep_flags, dtype=bool)

            aa_emb_wt_kept = None
            aa_emb_mt_kept = None
            if args.protein_embeddings:
                aa_emb_wt_kept = load_and_align_embeddings(
                    emb_path=emb_wt_path,
                    aa_keep_mask=aa_keep_mask,
                    kept_aa_count=kept_aa_count,
                    on_mismatch=args.on_embedding_mismatch
                )
                aa_emb_mt_kept = load_and_align_embeddings(
                    emb_path=emb_mt_path,
                    aa_keep_mask=aa_keep_mask,
                    kept_aa_count=kept_aa_count,
                    on_mismatch=args.on_embedding_mismatch
                )

                if (aa_emb_wt_kept is None) or (aa_emb_mt_kept is None):
                    aa_emb_wt_kept = None
                    aa_emb_mt_kept = None
                else:
                    if aa_emb_wt_kept.shape != aa_emb_mt_kept.shape:
                        msg = f"WT/MT embedding shape mismatch: wt={aa_emb_wt_kept.shape}, mt={aa_emb_mt_kept.shape}"
                        if args.on_embedding_mismatch == 'error':
                            raise ValueError(msg)
                        if args.on_embedding_mismatch == 'disable_embeddings':
                            print(f"  - [Warn] {msg}. Disable embeddings for this complex.", flush=True)
                            aa_emb_wt_kept = None
                            aa_emb_mt_kept = None
                        else:
                            raise SkipComplexException(msg)

            ligands = parse_sdf_file(sdf_path)
            if not ligands:
                raise SkipComplexException("No ligand found.")

            for l_idx, ligand_mol in enumerate(ligands):
                id_with_lig = f"{base_name}_L{l_idx + 1:05}" if len(ligands) > 1 else base_name
                save_path = os.path.join(args.output_dir, f"{id_with_lig}_graph.pth")
                if (not args.replace) and os.path.exists(save_path):
                    continue

                ligand_mol = sanitize_ligand_mol(ligand_mol)
                if ligand_mol is None:
                    print(f"  - Skip {id_with_lig}: ligand RemoveHs/sanitize failed or no conformer", flush=True)
                    skipped_bad_ligand_count += 1
                    continue

                ligand_pos = ligand_mol.GetConformer().GetPositions().astype(np.float32)
                if ligand_pos.shape[0] < 2:
                    print(f"  - Skip {id_with_lig}: Ligand too small", flush=True)
                    skipped_ligand_count += 1
                    continue

                diff = all_prot_atoms_coords_np[np.newaxis, :, :] - ligand_pos[:, np.newaxis, :]
                dists = np.linalg.norm(diff, axis=2)

                near_atom_mask_5 = (dists <= MIN_NEAR_AA_DIST).any(axis=0)
                near_rids = np.unique(all_prot_atoms_residx_np[near_atom_mask_5])

                near_aa = 0
                for rid in near_rids:
                    res = residues_dict.get(int(rid))
                    if res and res.get('_is_aa', False) and (res.get('resname', 'UNK') in AMINO_ACIDS):
                        near_aa += 1

                if near_aa == 0:
                    print(f"  - Skip {id_with_lig}: no AA residue within {MIN_NEAR_AA_DIST:.1f} Å", flush=True)
                    skipped_ligand_count += 1
                    continue

                close_mask = dists <= (args.contact_max_len + 1.0)
                connections = []
                for atom_mask in close_mask:
                    res_indices = np.unique(all_prot_atoms_residx_np[atom_mask])
                    connections.append(res_indices)

                x_lig = get_atom_features(ligand_mol, ALL_ATOMS, padding_len=len(AMINO_ACIDS))
                n_l_nodes = x_lig.shape[0]

                if n_l_nodes != ligand_mol.GetNumAtoms() or n_l_nodes != ligand_pos.shape[0]:
                    print(f"  - Skip {id_with_lig}: ligand atom count mismatch "
                          f"(features={n_l_nodes}, mol={ligand_mol.GetNumAtoms()}, pos={ligand_pos.shape[0]})", flush=True)
                    skipped_bad_ligand_count += 1
                    continue

                ei_lig, ea_lig = edge_index_and_attr(ligand_mol, ligand_pos, undirected=False, self_loops=False)

                # protein nodes
                pos_prot_list = []
                x_aa_list = []
                resnum_list = []
                x_wt_list = []
                x_mt_list = []
                chain_to_locals_aa = {}
                rid_to_local_idx = {}

                esm_dim = None
                if (aa_emb_wt_kept is not None) and (aa_emb_mt_kept is not None):
                    esm_dim = int(aa_emb_wt_kept.shape[1])

                for rid in sorted(residues_dict.keys()):
                    res = residues_dict[rid]
                    resname = res.get('resname', 'UNK')

                    coord = residue_representative_coord(res, resname, AMINO_ACIDS)
                    if coord is None or np.isnan(coord).any():
                        continue

                    aa20 = np.zeros((len(AMINO_ACIDS),), dtype=np.float32)
                    if res.get('_is_aa') and resname in AMINO_ACIDS:
                        aa20[:] = np.array(one_of_k_encoding(resname, AMINO_ACIDS), dtype=np.float32)

                    if esm_dim is not None:
                        ew = np.zeros((esm_dim,), dtype=np.float32)
                        em = np.zeros((esm_dim,), dtype=np.float32)
                        if res.get('_is_aa'):
                            idx = int(res.get('_aa_counter', -1))
                            if 0 <= idx < aa_emb_wt_kept.shape[0]:
                                ew = aa_emb_wt_kept[idx]
                                em = aa_emb_mt_kept[idx]
                        x_wt_list.append(ew)
                        x_mt_list.append(em)

                    curr = len(pos_prot_list)
                    rid_to_local_idx[rid] = curr
                    pos_prot_list.append(coord.astype(np.float32))
                    x_aa_list.append(aa20)

                    rn = -1
                    if res.get('resnum'):
                        m = re.match(r'^(\d+)', str(res['resnum']).strip())
                        if m:
                            rn = int(m.group(1))
                    resnum_list.append(rn)

                    if res.get('_is_aa'):
                        chain_to_locals_aa.setdefault(res.get('_chain_id', 'UNK'), []).append(curr)

                pos_prot = np.stack(pos_prot_list).astype(np.float32)
                x_aa = np.stack(x_aa_list).astype(np.float32)
                n_p_nodes = pos_prot.shape[0]
                pos_prot_t = torch.tensor(pos_prot, dtype=torch.float)

                x_wt = None
                x_mt = None
                if esm_dim is not None:
                    x_wt = np.stack(x_wt_list).astype(np.float32)
                    x_mt = np.stack(x_mt_list).astype(np.float32)

                # protein edges
                ei_seq = torch.empty((2, 0), dtype=torch.long)
                ea_seq = torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float)
                ei_knn = torch.empty((2, 0), dtype=torch.long)
                ea_knn = torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float)

                if args.protein_seq_edges:
                    ei_seq, ea_seq = build_protein_seq_edges(chain_to_locals_aa, pos_prot_t)
                if args.protein_knn_edges:
                    ei_knn, ea_knn = build_protein_knn_edges(pos_prot_t, k=args.protein_knn_k)

                # contact edges
                ei_cont_list, ea_cont_list = [[], []], []
                for lig_atm_idx, neighbors in enumerate(connections):
                    for rid in neighbors:
                        rid = int(rid)
                        if rid not in rid_to_local_idx:
                            continue
                        prot_idx = rid_to_local_idx[rid]
                        ei_cont_list[0].append(lig_atm_idx)
                        ei_cont_list[1].append(prot_idx)

                        res = residues_dict.get(rid, None)
                        if res is not None and res.get('_is_aa') and ('atoms' in res) and ('coords' in res):
                            try:
                                atoms = res['atoms']
                                coords = np.array(res['coords'], dtype=np.float32)
                                ca = coords[atoms.index('CA')]
                                n_ = coords[atoms.index('N')]
                                c_ = coords[atoms.index('C')]
                                cb = calculate_cbeta_position(ca, c_, n_)
                                atm_ca = float(np.linalg.norm(ligand_pos[lig_atm_idx] - ca))
                                atm_n = float(np.linalg.norm(ligand_pos[lig_atm_idx] - n_))
                                atm_c = float(np.linalg.norm(ligand_pos[lig_atm_idx] - c_))
                                atm_cb = float(np.linalg.norm(ligand_pos[lig_atm_idx] - cb))
                                feat = [0., 0., 1., atm_ca / 10, atm_n / 10, atm_c / 10, atm_cb / 10] + [0.] * 13
                            except Exception:
                                d = float(np.linalg.norm(ligand_pos[lig_atm_idx] - pos_prot[prot_idx]))
                                feat = [0., 0., 1., d / 10, d / 10, d / 10, d / 10] + [0.] * 13
                        else:
                            d = float(np.linalg.norm(ligand_pos[lig_atm_idx] - pos_prot[prot_idx]))
                            feat = [0., 0., 1., d / 10, d / 10, d / 10, d / 10] + [0.] * 13

                        ea_cont_list.append(feat)

                ei_cont = torch.tensor(ei_cont_list, dtype=torch.long)
                ea_cont = torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float) if len(ea_cont_list) == 0 else torch.tensor(ea_cont_list, dtype=torch.float)

                data = HeteroData()
                data.id = id_with_lig

                data['ligand'].x = torch.tensor(x_lig[:, :NUM_ATOMFEATURES], dtype=torch.float)
                data['ligand'].pos = torch.tensor(ligand_pos, dtype=torch.float)

                data['protein'].x_aa = torch.tensor(x_aa, dtype=torch.float)
                data['protein'].x = data['protein'].x_aa
                data['protein'].pos = pos_prot_t
                data['protein'].resnum = torch.tensor(resnum_list, dtype=torch.long)

                if (x_wt is not None) and (x_mt is not None):
                    data['protein'].esm_wt = torch.tensor(x_wt, dtype=torch.float)
                    data['protein'].esm_mt = torch.tensor(x_mt, dtype=torch.float)

                ei_b, ea_b = to_undirected(ei_lig, ea_lig)
                data['ligand', 'bond', 'ligand'].edge_index = ei_b
                data['ligand', 'bond', 'ligand'].edge_attr = ea_b

                data['ligand', 'contact', 'protein'].edge_index = ei_cont
                data['ligand', 'contact', 'protein'].edge_attr = ea_cont

                if ei_cont.numel() > 0:
                    rev = torch.stack([ei_cont[1], ei_cont[0]], dim=0)
                    data['protein', 'rev_contact', 'ligand'].edge_index = rev
                    data['protein', 'rev_contact', 'ligand'].edge_attr = ea_cont.clone()
                else:
                    data['protein', 'rev_contact', 'ligand'].edge_index = torch.empty((2, 0), dtype=torch.long)
                    data['protein', 'rev_contact', 'ligand'].edge_attr = torch.empty((0, NUM_EDGEFEATURES), dtype=torch.float)

                if args.protein_seq_edges:
                    data['protein', 'seq', 'protein'].edge_index = ei_seq
                    data['protein', 'seq', 'protein'].edge_attr = ea_seq
                if args.protein_knn_edges:
                    data['protein', 'knn', 'protein'].edge_index = ei_knn
                    data['protein', 'knn', 'protein'].edge_attr = ea_knn

                sl_l_i, sl_l_a = build_self_loop_relation(n_l_nodes, SELF_LOOP_FEATURE_VECTOR)
                data['ligand', 'self', 'ligand'].edge_index = sl_l_i
                data['ligand', 'self', 'ligand'].edge_attr = sl_l_a

                sl_p_i, sl_p_a = build_self_loop_relation(n_p_nodes, SELF_LOOP_FEATURE_VECTOR)
                data['protein', 'self', 'protein'].edge_index = sl_p_i
                data['protein', 'self', 'protein'].edge_attr = sl_p_a

                if args.masternode:
                    data['master'].x = torch.zeros((1, 1), dtype=torch.float)
                    center = np.mean(np.vstack((ligand_pos, pos_prot)), axis=0)
                    data['master'].pos = torch.tensor(center.reshape(1, 3), dtype=torch.float)

                    data['master', 'self', 'master'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    data['master', 'self', 'master'].edge_attr = SELF_LOOP_FEATURE_VECTOR.view(1, -1)

                    l2m = torch.stack([torch.arange(n_l_nodes), torch.zeros(n_l_nodes, dtype=torch.long)], dim=0)
                    data['ligand', 'to_master', 'master'].edge_index = l2m
                    data['ligand', 'to_master', 'master'].edge_attr = torch.zeros((n_l_nodes, NUM_EDGEFEATURES), dtype=torch.float)

                    p2m = torch.stack([torch.arange(n_p_nodes), torch.zeros(n_p_nodes, dtype=torch.long)], dim=0)
                    data['protein', 'to_master', 'master'].edge_index = p2m
                    data['protein', 'to_master', 'master'].edge_attr = torch.zeros((n_p_nodes, NUM_EDGEFEATURES), dtype=torch.float)

                # global backup
                ei_seq_g = ei_seq + n_l_nodes if ei_seq.numel() > 0 else ei_seq
                ei_knn_g = ei_knn + n_l_nodes if ei_knn.numel() > 0 else ei_knn
                ei_cont_g = ei_cont.clone()
                if ei_cont_g.numel() > 0:
                    ei_cont_g[1] += n_l_nodes

                global_idx = torch.cat([ei_lig, ei_cont_g, ei_seq_g, ei_knn_g], dim=1)
                global_attr = torch.cat([ea_lig, ea_cont, ea_seq, ea_knn], dim=0)

                global_idx, global_attr = make_undirected_with_self_loops(
                    global_idx, global_attr, undirected=True, self_loops=True, num_nodes=(n_l_nodes + n_p_nodes)
                )
                data.global_edge_index = global_idx
                data.global_edge_attr = global_attr

                torch.save(data, save_path)
                print(f"Saved: {id_with_lig}", flush=True)
                success_count += 1

        except Exception as e:
            print(f"FAILED {base_name}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            fail_count += 1

    print(
        f"\nDone. Success: {success_count}, Failed: {fail_count}, "
        f"SkippedLigands: {skipped_ligand_count}, SkippedMultiChain: {skipped_multichain_count}, "
        f"SkippedBadLigand(H/OOB): {skipped_bad_ligand_count}"
    )


if __name__ == "__main__":
    main()
