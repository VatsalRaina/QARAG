import json
import argparse
import numpy as np
import math
import torch

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--expanded_queries', type=bool, default=False, help='Whether to use expanded queries.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--write', type=int, default=0, help='Whether to write retrieved chunk indices.')
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

    # with open(args.data_dir + 'chunks_' + args.embedder + '.npy', 'rb') as f:
    #     chunk_embeddings = np.load(f)
    chunk_embeddings = np.load(args.data_dir + 'chunks_' + args.embedder + '.npy')
    chunk_embeddings = torch.from_numpy(chunk_embeddings)
    # with open(args.data_dir + 'queries_' + args.embedder + '.npy', 'rb') as f:
    #     query_embeddings = np.load(f, allow_pickle=True)
    if args.expanded_queries:
        query_embeddings = np.load(args.data_dir + 'expanded_queries_' + args.embedder + '.npy')
    else:
        query_embeddings = np.load(args.data_dir + 'queries_' + args.embedder + '.npy')
    query_embeddings = torch.from_numpy(query_embeddings)

    # Find closest embeddings for each query (using cosine distance)
    min_indices = get_neighbours(chunk_embeddings, query_embeddings, args.K)
    #print(min_indices.shape)

    unique_min_indices = np.unique( np.concatenate( (np.unique(min_indices), np.asarray(labels)) )) 
    print("Number of unique chunks retrieved:", unique_min_indices.size)

    if args.write == 1:
        np.save(args.data_dir + 'sub_chunk_indices.npy', unique_min_indices)

    hits = 0
    tot_ndcg = 0
    for count, label in enumerate(labels):
        if label in min_indices[count]: 
            hits += 1
            pos = min_indices[count].tolist().index(label) + 1
            tot_ndcg += 1 / math.log2(pos + 1)
    print("Recall at ", args.K)
    print(hits/len(labels))
    print("nDCG at ", args.K)
    print(tot_ndcg/len(labels))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
