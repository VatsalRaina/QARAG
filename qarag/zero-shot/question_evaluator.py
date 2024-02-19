import json
import argparse
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--qu_count', type=int, default=1, help='The number of questions per chunk.')
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

    question_embeddings = np.load(args.data_dir + 'questions_2_' + args.embedder + '.npy')
    qu_idx_to_chunk_idx = np.arange(len(labels))

    if args.qu_count > 1:
        for count in range(2, args.qu_count+1):
            print(question_embeddings.shape)
            print(qu_idx_to_chunk_idx.shape)
            curr_qu_embs = torch.from_numpy(np.load(args.data_dir + 'questions_' + str(count) + '_' + args.embedder + '.npy'))
            temp_qu_idx_to_chunk_idx = np.concatenate([qu_idx_to_chunk_idx, np.arange(len(labels))])
            qu_idx_to_chunk_idx = temp_qu_idx_to_chunk_idx
            temp_embeddings = np.concatenate([question_embeddings, curr_qu_embs], axis=0)
            question_embeddings = temp_embeddings
    question_embeddings = torch.from_numpy(question_embeddings)

    # Find closest embeddings for each query (using cosine distance)
    min_indices = get_neighbours(question_embeddings, query_embeddings, args.K * args.qu_count)
    chunk_indices = qu_idx_to_chunk_idx[min_indices]

    hits = 0
    for count, label in enumerate(labels):
        curr_chunk_idxs = pd.unique(chunk_indices[count])[:args.K]
        if label in curr_chunk_idxs: hits += 1
    print("Recall at ", args.K)
    print(hits/len(labels))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
