# THERE IS A BETTER SCRIPT WRITTEN DIRECTLY ON THE HPC IN ./out/

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_path', type=str, default='', help='Specify the path to save the figure.')
parser.add_argument('--out_dir', type=str, default='', help='Path to directory with all saved scores.')


def main(args):

    sns.set_style("darkgrid")

    MAX = 15
    recalls = [1,2,5]
    metrics = ['R@1', 'R@2', 'R@5']

    questions = []
    recall_scores = []
    metric_names = []
    for num_questions in range(1, MAX+1):
        for recall, metric in zip(recalls, metrics):
            with open(args.out_dir + str(args.qu_count) + "_K" + str(recall) + ".json", 'r') as f:
                scores = json.load(f)
            samples = len(scores)
            questions = questions + [num_questions]*samples
            recall_scores = recall_scores + scores
            metric_names = metric_names + [metric]*samples

    data = {
        'Questions': questions,
        'Recall': recall_scores,
        'Metric': metric_names
        }
    df = pd.DataFrame.from_dict(data)

    sns.pointplot(data=df, x="Questions", y="Recall", errorbar="sd", hue="Metric")
    plt.savefig(args.save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
