
import json
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the path to the data directory.')
parser.add_argument('--qu_count', type=int, default=1, help='The number of questions per chunk.')


def main(args):

    if 'e5' in args.embedder:
        model = SentenceTransformer("intfloat/" + args.embedder)
    else:
        model = SentenceTransformer("sentence-transformers/" + args.embedder)

    print("Started embedding questions.")
    with open(args.data_dir + 'expanded_queries.json', 'r') as f:
        questions = json.load(f)
    if 'e5' in args.embedder:
        questions = ['query: ' + qu for qu in questions]
    question_embeddings = np.asarray(model.encode(questions))
    with open(args.data_dir + 'expanded_queries_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, question_embeddings)
    print("Finished embedding expanded queries.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

