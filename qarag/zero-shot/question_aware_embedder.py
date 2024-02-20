
import json
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the path to the data directory.')
parser.add_argument('--qu_count', type=int, default=1, help='The number of questions per chunk.')


def main(args):

    model = SentenceTransformer("sentence-transformers/" + args.embedder)

    print("Started embedding questions.")
    with open(args.data_dir + 'gen_questions_aware.json', 'r') as f:
        questions = json.load(f)
    unwrapped_questions = []
    qu_idx_to_chunk_idx = []
    for count, question_set in enumerate(questions):
        unwrapped_questions.extend(question_set)
        curr_idxs = [count] * len(question_set)
        qu_idx_to_chunk_idx.extend(curr_idxs)

    question_embeddings = np.asarray(model.encode(unwrapped_questions))
    with open(args.data_dir + 'questions_aware_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, question_embeddings)
    print("Finished embedding questions.")

    with open(args.data_dir + 'questions_aware_mapping'+ '.npy', 'wb') as f:
        np.save(f, np.asarray(qu_idx_to_chunk_idx))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

