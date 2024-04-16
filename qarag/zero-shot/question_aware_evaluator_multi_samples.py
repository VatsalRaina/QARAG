import json
import argparse
import numpy as np
import pandas as pd
import torch
import random

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--qu_count', type=int, default=1, help='Specify the path to the data directory.')
parser.add_argument('--K', type=int, default=1, help='Recall depth.')
parser.add_argument('--save_dir', type=str, default='./out/', help='Directory to save output scores.')


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

    all_question_embeddings = []
    MAX = 15
    for count in range(1,MAX+1):
        if count == 1:
            all_question_embeddings.append( np.load(args.data_dir + 'questions_aware_' + args.embedder + '.npy') )
        else:
            all_question_embeddings.append( np.load(args.data_dir + 'questions_aware_' + str(count) + '_' + args.embedder + '.npy') )

    scores = []
    SAMPLES = 10
    for i in range(SAMPLES):
        print("SAMPLE NO:", i+1)
        positions = random.sample(range(MAX), args.qu_count)
        for pos_count, pos in enumerate(positions):
            if pos_count == 0: question_embeddings = all_question_embeddings[pos]
            else:
                curr_question_embeddings = all_question_embeddings[pos]
                question_embeddings = np.concatenate([question_embeddings, curr_question_embeddings], axis=0)
        question_embeddings = torch.from_numpy(question_embeddings)

        # Find closest embeddings for each query (using cosine distance)
        min_indices = get_neighbours(question_embeddings, query_embeddings, args.K * 15 * args.qu_count)
        chunk_indices = qu_idx_to_chunk_idx[min_indices]

        hits = 0
        for count, label in enumerate(labels):
            curr_chunk_idxs = pd.unique(chunk_indices[count])[:args.K]
            if label in curr_chunk_idxs: hits += 1
        print("Recall at ", args.K)
        print(hits/len(labels))
        print("-----------------------")
        scores.append(hits/len(labels))

    with open(args.save_dir + str(args.qu_count) + "_K" + str(args.K) + ".json", 'w') as f:
        json.dump(scores, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
