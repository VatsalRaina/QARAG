import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the model name used to search for correct files.')
parser.add_argument('--qu_count', type=int, default=1, help='Specify the path to the data directory.')
parser.add_argument('--save_dir', type=str, default='', help='Path to save all cosine distance matrices.')


def main(args):

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

    mean_sentence_cosine = []
    all_cosine_distances = []
    all_cosine_matrices = []

    # So for each set of 15 embeddings, I will make 15C2 = 105 pairwise comparisons
    for i in range(len(all_question_embeddings)):
        print(i, len(all_question_embeddings))
        arr = all_question_embeddings[i]
        # Normalize each row
        arr_normalized = arr / np.linalg.norm(arr, axis=1)[:, np.newaxis]
        # Calculate cosine similarity using matrix multiplication
        cosine_distances = 1 - np.dot(arr_normalized, arr_normalized.T)
        # Extract cosine distances from the upper triangle (excluding diagonal)
        upper_triangle_distances = cosine_distances[np.triu_indices(MAX, k=1)]
        all_cosine_distances += upper_triangle_distances.tolist()
        mean_sentence_cosine.append(np.mean(upper_triangle_distances))

        all_cosine_matrices.append( all_cosine_matrices )
    # all_cosine_matrices = np.asarray(all_cosine_matrices)
    # np.save(args.save_path, all_cosine_matrices)

    sns.histplot(all_cosine_distances, kde=False, stat="density")
    plt.xlabel('Pairwise question cosine similarity')
    plt.ylabel('Normalized Frequency')
    plt.savefig(args.save_dir + 'question_similarity.png')

    plt.clf()

    sns.histplot(mean_sentence_cosine, kde=False, stat="density")
    plt.xlabel('Mean pairwise question cosine similarity per sentence')
    plt.ylabel('Normalized Frequency')
    plt.savefig(args.save_dir + 'mean_question_similarity.png')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)