# ------------------------------------------------------------------------------
# This file contains code adapted from the GEMS project.
# Original Author: David Graber
# Original Repository: https://github.com/camlab-ethz/GEMS
# License: MIT License
# ------------------------------------------------------------------------------

from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1
import numpy as np

def parse_pdb(parser, protein_id, filepath):
    
    '''
    This function uses BioPython's PDBParser to load a PDB file
    and extracts the protein chains, residue information, atom types,
    and atomic coordinates. All data are stored in a nested dictionary
    and returned.

    The structure of the dictionary is as follows:

    protein = {
        0: {
            "aa_seq": "YH",
            "chain_id": "A",
            "coords": np.array([[x,y,z], [x,y,z], ...]),
            "residues": {
                1: {
                    'resname': 'TYR',
                    'atom_indices': [0,1,2,3,4, ...],
                    'atoms': ['CA', 'C', 'N', 'O', ...]
                },

                2: {
                    'resname': 'HIS',
                    'atom_indices': [12,13,14,15,16, ...],
                    'atoms': ['CA', 'C', 'N', 'O', ...]
                }
            }
        },

        1: {
            "aa_seq": "VN",
            "chain_id": "B",
            "coords": np.array([[x,y,z], [x,y,z], ...]),
            "residues": { ... }
        }
    }
    '''

    structure = parser.get_structure(protein_id, filepath)

    atom_index = 0
    protein = {}

    for j, chain in enumerate(structure.get_chains()):

        aa = False
        het = False

        aa_resnames = []
        aa_residues_dict = {}

        hetatm_residues_dict = {}
        water_residues_coords = []

        chain_atomcoords = []

        for i, residue in enumerate(chain.get_residues()):
            
            res_id = residue.get_id()
            resname = residue.resname.strip()

            resseq = res_id[1]
            icode = res_id[2] if isinstance(res_id[2], str) else ""
            icode = icode.strip()
            if icode:
                real_resnum = f"{resseq}{icode}"
            else:
                real_resnum = str(resseq)

            if res_id[0].startswith("H") and resname != "HOH":
                het = True

                hetatmnames = []
                hetatm_coords = []

                for atom in residue.get_atoms():
                    hetatmnames.append(atom.get_name())
                    hetatm_coords.append(list(atom.get_vector()))

                hetatm_residues_dict[i] = {
                    'resname': resname, 
                    'atoms': hetatmnames,
                    'hetatmcoords': np.array(hetatm_coords),
                    'resnum': real_resnum
                }

            elif res_id[0].startswith("W") and resname == "HOH":

                for atom in residue.get_atoms():
                    water_residues_coords.append(list(atom.get_vector()))

            else:
                aa = True
                aa_resnames.append(resname)
                aa_residues_dict[i] = {'resname': resname}

                aa_residues_dict[i]['resnum'] = real_resnum

                atoms = []
                atomnames = []
                residue_atomcoords = []

                for atom in residue.get_atoms():
                    atoms.append(atom)

                    atomname = atom.get_name()
                    atomnames.append(atomname)
                    residue_atomcoords.append(list(atom.get_vector()))

                atom_indeces = [ind for ind in range(atom_index, atom_index + len(atoms))]
                atom_index += len(atoms)

                aa_residues_dict[i]['atom_indeces'] = atom_indeces
                aa_residues_dict[i]['atoms'] = atomnames
                aa_residues_dict[i]['coords'] = np.array(residue_atomcoords)

        protein[j] = {'aa_residues': aa_residues_dict}
        protein[j]['chain_id'] = chain.id
        protein[j]['composition'] = [aa, het]
        protein[j]['hetatm_residues'] = hetatm_residues_dict
        protein[j]['water_residues'] = water_residues_coords

        aa_seq = seq1(''.join(aa_resnames))
        protein[j]['aa_seq'] = aa_seq

    return protein

# test
if __name__ == "__main__":
    from Bio.PDB import PDBParser

    pdb_file = "/home/zcheng/workspace/GEMS/GEMS-main/example_dataset/1a1e.pdb"
    protein_id = "test_protein"

    parser = PDBParser(QUIET=True)

    protein = parse_pdb(parser, protein_id, pdb_file)

    import pprint
    pprint.pprint(protein)

    print(protein[1]['aa_seq'])

    if 0 in protein and protein[0]['aa_residues']:
        first_key = sorted(protein[0]['aa_residues'].keys())[0]
        print("resnum:", protein[0]['aa_residues'][first_key].get('resnum', 'N/A'))
