
from datasets import load_dataset
import json
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--save_dir', type=str, default='', help='Specify the path to save the processed data.')
parser.add_argument('--extra_dir', type=str, default='', help='Specify the path to save the processed data.')
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
    
    with open(save_dir + 'passages.tex', 'r') as f:
        passage_data = f.readlines()

    # Initialize an empty dictionary
    passage_dict = {}
    # Iterate over each element in the list
    for passage in passage_data:
        # Split the string by the tab character
        parts = passage.split('\t')
        if len(parts) >= 2:  # Ensure there are at least two parts after splitting
            passage_id = parts[0]
            context = parts[1]
            # Add the ID and context to the dictionary
            passage_dict[passage_id] = context

    unique_contexts = list(passage_dict.values())
    print("Number of unique chunks:", len(unique_contexts))

    df = pd.read_csv(save_dir + 'question_dev_answerable.tsv', sep='\t')
    questions = df['question'].tolist()
    doc_ids = df['doc-id-list'].tolist()

    simplified_data = []
    for qu, d_id in zip(questions, doc_ids):
        curr = {'question': qu, 'context_id': unique_contexts.index(passage_dict[d_id])}
        simplified_data.append(curr)

    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(unique_contexts, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(simplified_data, f)

def process_sub_clapnq(clapnq_dir, save_dir):
    
    with open(clapnq_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)
    with open(clapnq_dir + 'data.json', 'r') as f:
        data = json.load(f)
    sub_chunk_indices = np.load(clapnq_dir + 'sub_chunk_indices.npy', 'r').tolist()

    sub_chunks = [chunks[idx] for idx in sub_chunk_indices]
    sub_data = []
    for ex in data:
        alt_ex = {'question': ex['question'], 'context_id': sub_chunk_indices.index(ex['context_id'])}
        sub_data.append(alt_ex)
    
    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(sub_chunks, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(sub_data, f)
    

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


def process_sub_pubmedqa(pubmedqa_dir, save_dir):
    
    with open(pubmedqa_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)
    with open(pubmedqa_dir + 'data.json', 'r') as f:
        data = json.load(f)
    sub_chunk_indices = np.load(pubmedqa_dir + 'sub_chunk_indices.npy', 'r').tolist()

    sub_chunks = [chunks[idx] for idx in sub_chunk_indices]
    sub_data = []
    for ex in data:
        alt_ex = {'question': ex['question'], 'context_id': sub_chunk_indices.index(ex['context_id'])}
        sub_data.append(alt_ex)
    
    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(sub_chunks, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(sub_data, f)

def process_bipar(save_dir):

    with open(save_dir + 'bipar_monolingual_test.json', 'r') as f:
        data = json.load(f)['data']

    simplified_data = []
    chunks = []
    for chunk_id, ex in enumerate(data):
        chunk = ex['paragraphs'][0]['context']
        for item in ex['paragraphs'][0]['qas']:
            question = item['question']
            curr = {'question': question, 'context_id': chunk_id }
        chunks.append(chunk)

    print("Total chunks:", len(chunks))
    print("Total queries:", len(simplified_data))

    with open(save_dir + 'chunks.json', 'w') as f:
        json.dump(chunks, f)

    with open(save_dir + 'data.json', 'w') as f:
        json.dump(simplified_data, f) 

def main(args):

    # print("Started processing SQuAD.")
    # process_squad(args.save_dir, args.split)
    # print("Finished processing SQuAD.")

    #process_clapnq(args.save_dir)
    #process_sub_clapnq(args.extra_dir, args.save_dir)
    #process_pubmedqaL(args.save_dir)
    #process_sub_pubmedqa(args.extra_dir, args.save_dir)
    process_bipar(args.save_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
