import json
import argparse
import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--qu_count', type=int, default=1, help='Specify the path to the data directory.')
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

    query_embeddings = np.load(args.data_dir + 'queries_' + args.embedder + '.npy')
    query_embeddings = torch.from_numpy(query_embeddings)
    question_embeddings = np.load(args.data_dir + 'questions_atom_aware_' + args.embedder + '.npy')
    atom_idx_to_chunk_idx = np.load(args.data_dir + 'atoms_mapping'+ '.npy')
    print("Num mappings:", len(atom_idx_to_chunk_idx))
    print(len(question_embeddings))

    if args.qu_count > 1:
        for count in range(2, args.qu_count+1):
            curr_question_embeddings = np.load(args.data_dir + 'questions_atom_aware_' + str(count) + '_' + args.embedder + '.npy')
            question_embeddings = np.concatenate([question_embeddings, curr_question_embeddings], axis=0)
        atom_idx_to_chunk_idx = np.tile(atom_idx_to_chunk_idx, args.qu_count)

    # #TEMP
    # for count in range(2, args.qu_count+1):
    #     #question_embeddings = torch.from_numpy( np.load(args.data_dir + 'questions_atom_aware_' + str(count) + '_' + args.embedder + '.npy') )
        
    #     min_indices = get_neighbours(question_embeddings, query_embeddings, args.K * 25 * args.qu_count)
    #     chunk_indices = atom_idx_to_chunk_idx[min_indices]
    #     hits = 0
    #     for count, label in enumerate(labels):
    #         curr_chunk_idxs = pd.unique(chunk_indices[count])[:args.K]
    #         if label in curr_chunk_idxs: hits += 1
    #     print(hits/len(labels))
    # ###


    question_embeddings = torch.from_numpy(question_embeddings)

    # Find closest embeddings for each query (using cosine distance)
    min_indices = get_neighbours(question_embeddings, query_embeddings, args.K * 25 * args.qu_count)
    chunk_indices = atom_idx_to_chunk_idx[min_indices]

    hits = 0
    #print(labels)
    #print([ch[0] for ch in chunk_indices])
    for count, label in enumerate(labels):
        curr_chunk_idxs = pd.unique(chunk_indices[count])[:args.K]
        if label in curr_chunk_idxs: hits += 1
        if len(pd.unique(chunk_indices[count])) < args.K: print(count, "Something is wrong")
    print("Recall at ", args.K)
    print(hits/len(labels))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
