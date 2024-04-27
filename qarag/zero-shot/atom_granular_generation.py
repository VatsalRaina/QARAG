"""
This file breaksdown a chunk of text into a set of atomic facts by looping through each sentence in turn.
"""

import json
import argparse
import openai
import time

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--save_path', type=str, help='Load path to save results with generated atoms')

def main(args):

    openai.api_key = args.api_key

    with open(args.save_path) as f:
        atoms = json.load(f)
    start_point = len(atoms)

    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)

    model = "gpt-3.5-turbo"

    timeout_secs = 10
    for count, chunk in enumerate(chunks):
        if count < start_point: continue
        print(count, len(chunks))
        sentences = chunk.split('.')
        chunk_atoms = []
        for sentence in sentences:
            prompt = "Please breakdown the following sentence into stand-alone atomic facts. Return each fact on a new line.\n" + sentence
            response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], request_timeout=timeout_secs)
            generated_text = response.choices[0].message.content.strip()
            sentence_atoms = generated_text.split('\n')
            chunk_atoms.extend(sentence_atoms)
        atoms.append(chunk_atoms)
        with open(args.save_path, 'w') as f:
            json.dump(atoms, f)
        print("Saved up to:", count)
        print("----------------------")


if __name__ == "__main__":
    args = parser.parse_args()
    for count in range(1,100):
        try:
            main(args)
            time.sleep(0.1)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print("openai.error.ServiceUnavailableError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.APIError:
            print("openai.error.APIError... #{}".format(count))
            print("restart in 20 seconds")
            time.sleep(20)
        except openai.error.Timeout:
            print("openai.error.TimeoutError... #{}".format(count))
            print("restart in 20 seconds")
            time.sleep(20)