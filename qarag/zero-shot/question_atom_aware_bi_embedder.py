
import json
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the path to the data directory.')
parser.add_argument('--qu_count', type=int, default=1, help='Specify the path to the data directory.')


def main(args):

    if 'e5' in args.embedder:
        model = SentenceTransformer("intfloat/" + args.embedder)
    else:
        model = SentenceTransformer("sentence-transformers/" + args.embedder)
    print("Started embedding atom aware questions.")

    if args.qu_count == 1:

        with open(args.data_dir + 'gen_questions_atom_aware_bi.json', 'r') as f:
                all_chunks_atoms_questions = json.load(f)

        all_questions = []
        for chunk_atoms in all_chunks_atoms_questions:
            all_questions.extend(chunk_atoms)

        print("Total number of atom aware questions:", len(all_questions))
        if 'e5' in args.embedder:
            all_questions = ['query: ' + ch for ch in all_questions]
        question_embeddings = np.asarray(model.encode(all_questions))
        with open(args.data_dir + 'questions_atom_aware_bi_' + args.embedder + '.npy', 'wb') as f:
            np.save(f, question_embeddings)

    else:

        with open(args.data_dir + 'gen_questions_atom_aware_bi_' + str(args.qu_count) + '.json', 'r') as f:
                all_chunks_atoms_questions = json.load(f)

        all_questions = []
        for chunk_atoms in all_chunks_atoms_questions:
            all_questions.extend(chunk_atoms)
        if 'e5' in args.embedder:
            all_questions = ['query: ' + ch for ch in all_questions]
        question_embeddings = np.asarray(model.encode(all_questions))
        with open(args.data_dir + 'questions_atom_aware_bi_' + str(args.qu_count) + '_' + args.embedder + '.npy', 'wb') as f:
            np.save(f, question_embeddings)


    print("Finished embedding atom aware questions.")





if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

