"""
This file generates synthetic questions for a set of chunks
using each sentence in turn to generate zone aware questions in order to enforce diversity.
"""

import json
import argparse
import time
import openai

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--save_path', type=str, help='Load path to save results with generated questions')


def main(args):

    openai.api_key = args.api_key

    with open(args.save_path) as f:
        questions = json.load(f)
    start_point = len(questions)

    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)

    model = "gpt-3.5-turbo"

    batch_examples = []
    for count, chunk in enumerate(chunks):
        if count < start_point: continue
        print(count)
        chunk_questions = []
        sentences = chunk.split('.')
        print("Number of sentences:", len(sentences))
        # generate question for each sentence
        for sentence in sentences:
            prompt = "Generate a single closed-answer question using:\n" + chunk + "\nThe answer should be present in the sentence:\n" + sentence
            response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
            generated_text = response.choices[0].message.content.strip()
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