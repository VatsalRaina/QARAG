
import json
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the path to the data directory.')
parser.add_argument('--qu_count', type=int, default=1, help='The number of questions per chunk.')


def main(args):

    model = SentenceTransformer("sentence-transformers/" + args.embedder)

    print("Started embedding questions.")
    with open(args.data_dir + 'gen_questions.json', 'r') as f:
        questions = json.load(f)
    question_embeddings = np.asarray(model.encode(questions))
    with open(args.data_dir + 'questions_' + args.embedder + '.npy', 'wb') as f:
        np.save(f, question_embeddings)
    print("Finished embedding questions.")

    if args.qu_count > 1:
        print("But there's more ...")
        for count in range(2, args.qu_count+1):
            print(count)
            with open(args.data_dir + 'gen_questions_' + str(count) + '.json', 'r') as f:
                questions = json.load(f)
            question_embeddings = np.asarray(model.encode(questions))
            with open(args.data_dir + 'questions_' + str(count) + '_' + args.embedder + '.npy', 'wb') as f:
                np.save(f, question_embeddings)
        print("Finished now!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

