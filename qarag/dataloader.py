
from datasets import load_dataset
import json
import argparse

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_dir', type=str, default='', help='Specify the path to save the processed data.')
parser.add_argument('--split', type=str, default='validation', help='Split of data.')


def process_squad(save_dir, split):
    dataset = load_dataset("squad")[split]
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
    
def process_hotpotqa(save_dir):
    dataset = load_dataset("hotpot_qa", "fullwiki")['validation']


def process_clapnq(save_dir):
    dataset = []
    with open(save_dir + 'clapnq_dev_answerable.jsonl', 'r') as f:
        for l in f: dataset.append( json.loads(l.strip()) )
    train_dataset = []
    with open(save_dir + 'clapnq_train_answerable.jsonl', 'r') as f:
        for l in f: train_dataset.append( json.loads(l.strip()) )

    unique_contexts = []
    for ex in dataset + train_dataset:
        if ex['passages'][0]['text'] not in unique_contexts: unique_contexts.append( ex['passages'][0]['text'] )

    print("Number of unique chunks:", len(unique_contexts))

    simplified_data = []
    for ex in dataset:
        curr = {'question': ex['input'], 'context_id': unique_contexts.index( ex['passages'][0]['text'] )}
        simplified_data.append(curr)

    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(unique_contexts, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(simplified_data, f)
    

def process_pubmedqaL(save_dir):
    with open(save_dir + 'ori_pqal.json', 'r') as f:
        dataset = json.load(f)
    with open(save_dir + 'ori_pqau.json', 'r') as f:
        dataset_extra = json.load(f)

    unique_contexts = []
    for ex in dataset.keys():
        unique_contexts.append( ' '.join(dataset[ex]['CONTEXTS']) )
    for ex in dataset_extra.keys():
        ctxt = ' '.join(dataset_extra[ex]['CONTEXTS'])
        if ctxt not in unique_contexts: unique_contexts.append(ctxt)

    simplified_data = []
    for ex in dataset.keys():
        curr = {'question': dataset[ex]['QUESTION'], 'context_id': unique_contexts.index( ' '.join(dataset[ex]['CONTEXTS']) ) }
        simplified_data.append(curr)

    print("Number of unique chunks:", len(unique_contexts))

    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(unique_contexts, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(simplified_data, f)    


def main(args):

    # print("Started processing SQuAD.")
    # process_squad(args.save_dir, args.split)
    # print("Finished processing SQuAD.")

    #process_clapnq(args.save_dir)
    process_pubmedqaL(args.save_dir)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
