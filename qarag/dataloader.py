
from datasets import load_dataset
import json
import argparse

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_dir', type=str, default='', help='Specify the path to save the processed data.')


def process_squad(save_dir):
    dataset = load_dataset("squad")["validation"]
    unique_contexts = []
    for ex in dataset:
        if ex['context'] not in unique_contexts: unique_contexts.append(ex['context'])
    enhanced_dataset = []
    for ex in dataset:
        curr_ex = ex
        curr_ex['context_id'] = unique_contexts.index(ex['context'])
        enhanced_dataset.append(curr_ex)

    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(unique_contexts, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(enhanced_dataset, f)
    
def main(args):

    print("Started processing SQuAD.")
    process_squad(args.save_dir)
    print("Finished processing SQuAD.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
