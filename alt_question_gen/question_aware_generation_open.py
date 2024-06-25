"""
This file generates synthetic questions for a set of chunks
using each sentence in turn to generate zone aware questions in order to enforce diversity.
"""

import json
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--model', type=str, help='Huggingface model')
parser.add_argument('--save_path', type=str, help='Load path to save results with generated questions')


def main(args):

    with open(args.save_path) as f:
        questions = json.load(f)
    start_point = len(questions)

    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map="auto")

    batch_examples = []
    for count, chunk in enumerate(chunks):
        if count < start_point: continue
        print(count, len(chunks))
        chunk_questions = []
        sentences = chunk.split('.')
        print("Number of sentences:", len(sentences))
        # generate question for each sentence
        for sentence in sentences:
            input_text = "Generate a single closed-answer question using:\n" + chunk + "\nThe answer should be present in the sentence:\n" + sentence
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
            outputs = model.generate(input_ids)
            generated_text = tokenizer.decode(outputs[0])
            print(generated_text)
            chunk_questions.append(generated_text)
        batch_examples.append(chunk_questions)
        if len(batch_examples) == 1:
            questions += batch_examples
            batch_examples = []
            with open(args.save_path, 'w') as f:
                json.dump(questions, f)
            print("Saved up to:", count)
            print("----------------------")



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
