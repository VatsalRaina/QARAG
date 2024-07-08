import json
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--model', type=str, default='flan-t5-small', help='Huggingface model')
parser.add_argument('--save_path', type=str, help='Load path to save unanswerability scores')


def main(args):

    with open(args.save_path) as f:
        scores = json.load(f)
    start_point = len(scores)

    tokenizer = T5Tokenizer.from_pretrained("google/" + args.model)
    model = T5ForConditionalGeneration.from_pretrained("google/" + args.model, device_map="auto")

    dataset = load_dataset("rajpurkar/squad_v2")
    dev_split = dataset['validation']

    # Token IDs for 'yes' and 'no'
    yes_token_id = tokenizer.convert_tokens_to_ids("▁yes")
    no_token_id = tokenizer.convert_tokens_to_ids("▁no")

    batch_examples = []
    for count, ex in enumerate(dev_split):
        if count < start_point: continue
        print(count, len(dev_split))

        context = ex['context']
        question = ex['question']

        input_text = "Consider the following reading comprehension question.\n\nQuestion:\n" + question + "\n\nContext:\n" + context + "\n\nIs this question unanswerable? Reply yes or no only."
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")        
        with torch.no_grad():
            outputs = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, do_sample=False, max_length=5)
        logits = outputs.scores[0][0]
        yes_no_logits = logits[[yes_token_id, no_token_id]]
        probs = torch.softmax(yes_no_logits, dim=-1)
        yes_prob = probs[0].item()
        batch_examples.append(yes_prob)
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
