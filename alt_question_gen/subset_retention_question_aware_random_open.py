import numpy as np
import json
import torch
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--save_dir', type=str, default='', help='Path to save all cosine distance matrices.')
parser.add_argument('--model', type=str, default='flan-t5-small', help='Huggingface model')
parser.add_argument('--K', type=int, default=1, help='Recall depth.')


def get_neighbours(Z, B, K):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z = Z.to(device)
    B = B.to(device).transpose(0, 1)

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
            all_question_embeddings.append( np.load(args.data_dir + 'questions_aware_' + args.model + '_' + args.embedder + '.npy') )
        else:
            all_question_embeddings.append( np.load(args.data_dir + 'questions_aware_' + str(count) + '_' + args.model + '_' + args.embedder + '.npy') )

    all_question_embeddings = np.asarray(all_question_embeddings)
    all_question_embeddings = all_question_embeddings.transpose(1, 0, 2)    # Now [num_atoms, 15, emb_dim]

    qu_idx_to_chunk_idx = np.load(args.data_dir + 'questions_aware_mapping'+ '.npy')

    assert len(qu_idx_to_chunk_idx) == len(all_question_embeddings)

    # restructure
    grouped_atomic_question_embeddings = {}
    for atom_idx, chunk_idx in enumerate(qu_idx_to_chunk_idx):
        if chunk_idx not in grouped_atomic_question_embeddings:
            grouped_atomic_question_embeddings[chunk_idx] = []
        grouped_atomic_question_embeddings[chunk_idx].append(all_question_embeddings[atom_idx])


    out = {
        'num_questions': [],
        'recalls': []
        }

    fracs = [0.05, 0.095, 0.158, 0.217, 0.317, 0.453, 0.75, 1.0]

    for frac in fracs:
        print("Frac:", frac)
        i = 0
        for chunk_idx, chunk_question_embeddings in grouped_atomic_question_embeddings.items():
            arr = np.concatenate( chunk_question_embeddings, axis=0 )
            values = np.arange(len(arr))
            num_samples = int( frac * len(values) )
            idxs = np.random.choice(values, num_samples, replace=False)
            retained_qu_idx_to_chunk_idx = np.asarray( [ chunk_idx ] * len(idxs) )
            retained_question_embeddings = arr[idxs]

            if i==0:
                efficient_qu_idx_to_chunk_idx = retained_qu_idx_to_chunk_idx
                efficient_question_embeddings = retained_question_embeddings
            else:
                efficient_qu_idx_to_chunk_idx = np.concatenate([efficient_qu_idx_to_chunk_idx, retained_qu_idx_to_chunk_idx], axis=0)
                efficient_question_embeddings = np.concatenate([efficient_question_embeddings, retained_question_embeddings], axis=0)
            i+=1

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

        out['num_questions'].append(len(efficient_question_embeddings))
        out['recalls'].append(hits/len(labels))

    with open(args.save_dir + 'K_' + str(args.K) + '.json', 'w') as f:
        json.dump(out, f)






if __name__ == "__main__":
    args = parser.parse_args()
    main(args)






