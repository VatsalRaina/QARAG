import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--save_path', type=str, default='', help='Specify the path to save the figure.')


def main(args):

    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)

    sentence_counts = []
    for chunk in chunks:
        sentences = chunk.split('.')
        sentence_counts.append(len(sentences))
    sentence_counts = np.asarray(sentence_counts)

    sns.boxplot(sentence_counts, whis=(0, 100), orient="h")
    plt.xlabel('Sentences per chunk')
    plt.savefig(args.save_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)