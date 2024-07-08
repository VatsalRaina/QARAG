import json
import argparse
import numpy as np
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

    thresholds = np.arange(0.0, 1.0, 0.01)
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for threshold in thresholds:
        predictions = [score >= threshold for score in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

        print(f"Threshold: {threshold:.2f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    print(f"\nBest Threshold: {best_threshold:.2f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
    print(f"Best F1 Score: {best_f1:.4f}")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
