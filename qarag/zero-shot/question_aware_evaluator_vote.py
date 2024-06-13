import json
import argparse
import numpy as np
import pandas as pd
import torch
import random
from scipy.stats import mode


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--K', type=int, default=1, help='Recall depth.')


def get_neighbours(Z, B, K):
    B = B.T

    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).

    # Distance matrix of size (b, n).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    cosine_distance = 1 - cosine_similarity

    cosine_distance = cosine_distance.detach().cpu()
    cosine_similarity = cosine_similarity.detach().cpu()

    _, min_indices = torch.topk(cosine_distance, K, 1, False, True)

    return min_indices.numpy()


def main(args):

    with open(args.data_dir + 'data.json', 'r') as f:
        data = json.load(f)
    labels = [ex['context_id'] for ex in data]

    query_embeddings = np.load(args.data_dir + 'queries_' + args.embedder + '.npy')
    query_embeddings = torch.from_numpy(query_embeddings)
    qu_idx_to_chunk_idx = np.load(args.data_dir + 'questions_aware_mapping'+ '.npy')
    if args.qu_count > 1:
        qu_idx_to_chunk_idx = np.tile(qu_idx_to_chunk_idx, args.qu_count)

    all_chunk_indices = []
    MAX = 15
    for count in range(1,MAX+1):
        if count == 1:
            question_embeddings = np.load(args.data_dir + 'questions_aware_' + args.embedder + '.npy')
        else:
            question_embeddings = np.load(args.data_dir + 'questions_aware_' + str(count) + '_' + args.embedder + '.npy')
        question_embeddings = torch.from_numpy(question_embeddings)
        min_indices = get_neighbours(question_embeddings, query_embeddings, 5)
        chunk_indices = qu_idx_to_chunk_idx[min_indices][:,0]
        all_chunk_indices.append(chunk_indices)
    all_chunk_indices = np.asarray(all_chunk_indices)
    modal_values = mode(all_chunk_indices, axis=0).mode[0]

    hits = 0
    for lab, mod in zip(labels, modal_values):
        if lab == mod: hits += 1
    print(hits/len(labels))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
