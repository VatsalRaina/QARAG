import json
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--unans_model', type=str, default='flan-t5-large', help='Huggingface model')
parser.add_argument('--model', type=str, default='flan-t5-small', help='Huggingface model')
parser.add_argument('--save_path', type=str, help='Load path to save unanswerability scores')
parser.add_argument('--qu_count', type=int, default=1, help='Specify the path to the data directory.')



def main(args):

    with open(args.save_path) as f:
        scores = json.load(f)
    start_point = len(scores)

    tokenizer = T5Tokenizer.from_pretrained("google/" + args.unans_model)
    model = T5ForConditionalGeneration.from_pretrained("google/" + args.unans_model, device_map="auto")

    # Token IDs for 'yes' and 'no'
    yes_token_id = tokenizer.convert_tokens_to_ids("▁yes")
    no_token_id = tokenizer.convert_tokens_to_ids("▁no")

    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)

    if args.qu_count == 1:
        with open(args.data_dir + 'gen_questions_aware_' + args.model + '.json', 'r') as f:
            questions = json.load(f)
    else:
        with open(args.data_dir + 'gen_questions_aware_' + str(args.qu_count) + '_' + args.model + '.json', 'r') as f:
            questions = json.load(f)

    assert len(questions) == len(chunks)

    batch_examples = []
    count = 0
    for sub_questions, context in zip(questions, chunks):
        if count < start_point: continue
        print(count, len(chunks))
        count += 1
        curr_scores = []
        for question in sub_questions:
            input_text = "Consider the following reading comprehension question.\n\nQuestion:\n" + question + "\n\nContext:\n" + context + "\n\nIs this question answerable? Reply yes or no only."
            inputs = tokenizer(input_text, return_tensors="pt").to("cuda")        
            with torch.no_grad():
                outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, do_sample=False, max_length=5)
            logits = outputs.scores[0][0]
            yes_no_logits = logits[[yes_token_id, no_token_id]]
            probs = torch.softmax(yes_no_logits, dim=-1)
            yes_prob = probs[0].item()
            curr_scores.append(yes_prob)
        batch_examples.append(curr_scores)
        if len(batch_examples) == 1:
            scores += batch_examples
            batch_examples = []
            with open(args.save_path, 'w') as f:
                json.dump(scores, f)
            print("Saved up to:", count)
            print("----------------------")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
