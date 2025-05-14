import os
import json
from Bio import PDB
from Bio.SeqUtils import seq1

def extract_protein_ids(protein_json_file):
    with open(protein_json_file, 'r') as f:
        data = json.load(f)
    return list(data.keys())

def extract_sequence(pdb_filename):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_filename)
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue):
                        sequence += seq1(residue.get_resname())
        return sequence
    except Exception as e:
        print(f"Error processing {pdb_filename}: {e}")
        return ""

def generate_sequences_dict(protein_json_file, pdb_directory, output_file):
    protein_ids = extract_protein_ids(protein_json_file)

    pdb_files = {f.lower(): f for f in os.listdir(pdb_directory)}

    sequences_dict = {}
    for protein_id in protein_ids:
        pdb_filename = f"{protein_id}.pdb"

        if pdb_filename.lower() in pdb_files:
            pdb_path = os.path.join(pdb_directory, pdb_files[pdb_filename.lower()])
            print(f"Processing {pdb_filename}...")
            sequence = extract_sequence(pdb_path)
            sequences_dict[protein_id] = sequence
        else:
            print(f"Warning: {pdb_filename} not found.")
            sequences_dict[protein_id] = ""

    with open(output_file, 'w') as f:
        json.dump(sequences_dict, f, ensure_ascii=False, indent=4)

    print(f"Sequences saved to {output_file}")


protein_json_file = "data/davis/targets.txt"
pdb_directory = "data/davis/target_pdb"
output_file = "data/davis/targets_pdb.txt"

generate_sequences_dict(protein_json_file, pdb_directory, output_file)
