import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_path', type=str, default='', help='Specify the path to save the figure.')

def main(args):

    sns.set_style("darkgrid")

    r1 = [62.9, 66.7, 68.3, 69.4, 70.1, 70.7, 71.2, 71.8, 72.1, 72.5, 72.9, 73.1, 73.3, 73.6, 73.8]
    r2 = [73.5, 76.8, 78.3, 79.3, 80.2, 80.6, 81.1, 81.6, 81.8, 82.2, 82.5, 82.7, 83.0, 83.2, 83.5]
    r5 = [84.2, 86.2, 87.6, 88.3, 89.0, 89.2, 89.5, 90.1, 90.3, 90.5, 90.6, 90.8, 90.9, 91.0, 91.2]

    K = 15
    data = {
        'Questions': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]*3,
        'Recall': r1 + r2 + r5,
        'Metric': ['R@1']*K + ['R@2']*K + ['R@5']*K
        }
    df = pd.DataFrame.from_dict(data)

    sns.lineplot(
        data=df,
        x="Questions", y="Recall", hue="Metric", style="event",
        markers=True, dashes=False
    )
    plt.savefig(args.save_path)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
