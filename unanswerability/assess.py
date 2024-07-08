import json
import argparse
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--scores_path', type=str, help='Load path to save unanswerability scores')


def main(args):

    with open(args.scores_path) as f:
        scores = json.load(f)

    dataset = load_dataset("rajpurkar/squad_v2")
    dev_split = dataset['validation']

    labels = []
    for ex in dev_split:
        answers = ex['answers']
        if len(answers['text']) == 0:
            labels.append(False)
        else:
            labels.append(True)

    threshold = 0.7
    predictions = [score >= threshold for score in scores]

    # Compute precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    # Print the results
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
