
import json
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the path to the data directory.')


def main(args):

    model = SentenceTransformer("sentence-transformers/" + args.embedder)

    print("Started embedding chunks.")
    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)
    chunk_embeddings = model.encode(chunks)
    with open(args.data_dir + 'chunks_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, chunk_embeddings)
    print("Finished embedding chunks.")

    print("Started embedding queries.")
    with open(args.data_dir + 'data.json', 'r') as f:
        data = json.load(f)
    queries = [ex['question'] for ex in data]
    query_embeddings = model.encode(queries)
    with open(args.data_dir + 'queries_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, query_embeddings)
    print("Finished embedding queries.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

