import numpy as np
import json
import torch
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--qu_count', type=int, default=1, help='Specify the path to the data directory.')
parser.add_argument('--save_dir', type=str, default='', help='Path to save all cosine distance matrices.')
parser.add_argument('--K', type=int, default=1, help='Recall depth.')



def select_subset(matrix, tau):
    n = len(matrix)
    selected_indices = set(range(n))

    while True:
        # Calculate the number of elements above tau in each row (excluding diagonal)
        above_tau_counts = np.sum(np.triu(matrix > tau, k=1), axis=1)
        
        # Find the index of the question with the maximum count
        max_count_index = np.argmax(above_tau_counts)
        
        # If all counts are zero or less, break the loop
        if above_tau_counts[max_count_index] <= 0:
            break
        
        # Remove the question with the maximum count
        selected_indices.remove(max_count_index)
        
        # Update the matrix by setting the similarities involving the removed question to 0
        matrix[max_count_index, :] = 0
        matrix[:, max_count_index] = 0

    return np.array(list(selected_indices))


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

    all_question_embeddings = []
    questions_per_sentence = 15
    MAX = questions_per_sentence
    for count in range(1,MAX+1):
        if count == 1:
            all_question_embeddings.append( np.load(args.data_dir + 'questions_aware_' + args.embedder + '.npy') )
        else:
            all_question_embeddings.append( np.load(args.data_dir + 'questions_aware_' + str(count) + '_' + args.embedder + '.npy') )

    all_question_embeddings = np.asarray(all_question_embeddings)
    all_question_embeddings = all_question_embeddings.transpose(1, 0, 2)    # Now [num_sentences, 15, emb_dim]

    qu_idx_to_chunk_idx = np.load(args.data_dir + 'questions_aware_mapping'+ '.npy')

    assert len(qu_idx_to_chunk_idx) == len(all_question_embeddings)

    # Cut-off threshold for similarity
    tau = 0.9

    for i in range(len(all_question_embeddings)):
        print(i, len(all_question_embeddings))
        arr = all_question_embeddings[i]
        # Normalize each row
        arr_normalized = arr / np.linalg.norm(arr, axis=1)[:, np.newaxis]
        # Calculate cosine similarity using matrix multiplication
        cosine_similarities = np.dot(arr_normalized, arr_normalized.T)
        idxs = select_subset(cosine_similarities, tau)
        retained_qu_idx_to_chunk_idx = np.asarray( [ qu_idx_to_chunk_idx[i] ] * len(idxs) )
        retained_question_embeddings = arr[idxs]

        if i==0:
            efficient_qu_idx_to_chunk_idx = retained_qu_idx_to_chunk_idx
            efficient_question_embeddings = retained_question_embeddings
        else:
            efficient_qu_idx_to_chunk_idx = np.concatenate([efficient_qu_idx_to_chunk_idx, retained_qu_idx_to_chunk_idx], axis=0)
            efficient_question_embeddings = np.concatenate([efficient_question_embeddings, retained_question_embeddings], axis=0)

    efficient_question_embeddings = torch.from_numpy(efficient_question_embeddings)

    # Find closest embeddings for each query (using cosine distance)
    min_indices = get_neighbours(efficient_question_embeddings, query_embeddings, args.K * 225)
    chunk_indices = efficient_qu_idx_to_chunk_idx[min_indices]

    hits = 0
    for count, label in enumerate(labels):
        curr_chunk_idxs = pd.unique(chunk_indices[count])[:args.K]
        if label in curr_chunk_idxs: hits += 1
    print("Recall at:", args.K)
    print(hits/len(labels))
    print("Efficient questions:", len(efficient_question_embeddings))
    print("Maximum number of questions:", len(all_question_embeddings) * len(all_question_embeddings[0]))
    print("-----------------------")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)






