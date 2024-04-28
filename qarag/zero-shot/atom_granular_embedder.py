
import json
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the path to the data directory.')


def main(args):

    model = SentenceTransformer("sentence-transformers/" + args.embedder)

    print("Started embedding atoms.")

    with open(args.data_dir + 'atoms_granular.json', 'r') as f:
            all_chunks_atoms = json.load(f)

    atom_idx_to_chunk_idx = []
    all_atoms = []
    for count, chunk_atoms in enumerate(all_chunks_atoms):
        all_atoms.extend(chunk_atoms)
        curr_idxs = [count] * len(chunk_atoms)
        atom_idx_to_chunk_idx.extend(curr_idxs)

    print("Total number of atoms:", len(all_atoms))
    

    atom_embeddings = np.asarray(model.encode(all_atoms))
    with open(args.data_dir + 'atoms_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, atom_embeddings)

    print("Finished embedding atoms.")

    with open(args.data_dir + 'atoms_granular_mapping'+ '.npy', 'wb') as f:
        np.save(f, np.asarray(atom_idx_to_chunk_idx))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

