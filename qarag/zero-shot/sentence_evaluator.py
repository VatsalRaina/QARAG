import json
import argparse
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--expanded_queries', type=bool, default=False, help='Whether to use expanded queries.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--K', type=int, default=1, help='Recall depth.')

def get_neighbours(Z, B, K):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z = Z.to(device)
    B = B.to(device).transpose(0, 1)

    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True).to(device)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True).to(device)  # Size (1, b).

    # Distance matrix of size (b, n).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).transpose(0, 1)
    cosine_distance = 1 - cosine_similarity

    cosine_distance = cosine_distance.cpu()
    cosine_similarity = cosine_similarity.cpu()

    _, min_indices = torch.topk(cosine_distance, K, 1, False, True)

    return min_indices.numpy()


def main(args):

    with open(args.data_dir + 'data.json', 'r') as f:
        data = json.load(f)
    labels = [ex['context_id'] for ex in data]

    if args.expanded_queries:
        query_embeddings = np.load(args.data_dir + 'expanded_queries_' + args.embedder + '.npy')
    else:
        query_embeddings = np.load(args.data_dir + 'queries_' + args.embedder + '.npy')
    query_embeddings = torch.from_numpy(query_embeddings)
    sentence_embeddings = np.load(args.data_dir + 'sentences_' + args.embedder + '.npy')
    sent_idx_to_chunk_idx = np.load(args.data_dir + 'sentences_mapping'+ '.npy')

    sentence_embeddings = torch.from_numpy(sentence_embeddings)

    # Find closest embeddings for each query (using cosine distance)
    min_indices = get_neighbours(sentence_embeddings, query_embeddings, args.K * 15)
    chunk_indices = sent_idx_to_chunk_idx[min_indices]

    hits = 0
    for count, label in enumerate(labels):
        curr_chunk_idxs = pd.unique(chunk_indices[count])[:args.K]
        if label in curr_chunk_idxs: hits += 1
    print("Recall at ", args.K)
    print(hits/len(labels))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
