
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
    print("Started embedding sentences.")

    with open(args.data_dir + 'chunks.json', 'r') as f:
            chunks = json.load(f)

    sent_idx_to_chunk_idx = []
    all_sentences = []
    for count, chunk in enumerate(chunks):
        chunk_sentences = chunk.split('.')
        all_sentences.extend(chunk_sentences)
        curr_idxs = [count] * len(chunk_sentences)
        sent_idx_to_chunk_idx.extend(curr_idxs)

    print("Total number of sentences:", len(all_sentences))
    
    if 'e5' in args.embedder:
        all_sentences = ['passage: ' + ch for ch in all_sentences]

    sentence_embeddings = np.asarray(model.encode(all_sentences))
    with open(args.data_dir + 'sentences_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, sentence_embeddings)

    print("Finished embedding sentences.")

    with open(args.data_dir + 'sentences_mapping'+ '.npy', 'wb') as f:
        np.save(f, np.asarray(sent_idx_to_chunk_idx))



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

