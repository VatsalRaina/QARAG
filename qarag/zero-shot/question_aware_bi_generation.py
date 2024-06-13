import json
import argparse
import time
import random
import openai
import asyncio
import aiohttp

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--save_path', type=str, help='Load path to save results with generated questions')

async def fetch_question(session, model, chunk, sentence, other_sentence, timeout_secs):
    prompt = "Generate a question using:\n" + chunk + "\nThe answer to the question should be based on the sentences:\n" + sentence + "\n" + other_sentence
    # try:
    async with session.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {openai.api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}]
        },
        timeout=aiohttp.ClientTimeout(total=timeout_secs)
    ) as response:
        result = await response.json()
        return result['choices'][0]['message']['content'].strip()
    # except Exception as e:
    #     print(f"Error fetching question: {e}")
    #     return None

async def process_chunk(session, model, chunk, timeout_secs):
    tasks = []
    sentences = chunk.split('.')
    for sentence in sentences:
        if len(sentences) == 1:
            other_sentence = sentence
        else:
            filtered_sentences = [sent for sent in sentences if sent != sentence]
            other_sentence = random.choice(filtered_sentences)
        tasks.append(fetch_question(session, model, chunk, sentence, other_sentence, timeout_secs))
    chunk_questions = await asyncio.gather(*tasks)
    return [q for q in chunk_questions]

async def main(args):
    openai.api_key = args.api_key

    with open(args.save_path) as f:
        questions = json.load(f)
    start_point = len(questions)

    with open(args.data_dir + 'chunks.json', 'r') as f:
        chunks = json.load(f)

    model = "gpt-3.5-turbo"
    timeout_secs = 10

    async with aiohttp.ClientSession() as session:
        batch_examples = []
        for count, chunk in enumerate(chunks):
            if count < start_point:
                continue
            print(count)
            print("Number of sentences:", len(chunk.split('.')))

            chunk_questions = await process_chunk(session, model, chunk, timeout_secs)
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
    for count in range(1, 100):
        try:
            asyncio.run(main(args))
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
        except Exception as e:
            print(e)
            time.sleep(20)


# """
# This file generates synthetic questions for a set of chunks
# using each sentence in turn to generate zone aware questions in order to enforce diversity.
# """

# import json
# import argparse
# import time
# import openai

# parser = argparse.ArgumentParser(description='Get all command line arguments.')
# parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
# parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
# parser.add_argument('--save_path', type=str, help='Load path to save results with generated questions')


# def main(args):

#     openai.api_key = args.api_key

#     with open(args.save_path) as f:
#         questions = json.load(f)
#     start_point = len(questions)

#     with open(args.data_dir + 'chunks.json', 'r') as f:
#         chunks = json.load(f)

#     model = "gpt-3.5-turbo"

#     batch_examples = []
#     timeout_secs = 10
#     for count, chunk in enumerate(chunks):
#         if count < start_point: continue
#         print(count)
#         chunk_questions = []
#         sentences = chunk.split('.')
#         print("Number of sentences:", len(sentences))
#         # generate question for each sentence
#         for sentence in sentences:
#             prompt = "Generate a single closed-answer question using:\n" + chunk + "\nThe answer should be present in the sentence:\n" + sentence
#             response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], request_timeout=timeout_secs)
#             generated_text = response.choices[0].message.content.strip()
#             chunk_questions.append(generated_text)
#         batch_examples.append(chunk_questions)
#         if len(batch_examples) == 1:
#             questions += batch_examples
#             batch_examples = []
#             with open(args.save_path, 'w') as f:
#                 json.dump(questions, f)
#             print("Saved up to:", count)
#             print("----------------------")



# if __name__ == "__main__":
#     args = parser.parse_args()
#     for count in range(1,500):
#         try:
#             main(args)
#             time.sleep(0.1)
#         except openai.error.RateLimitError:
#             print("openai.error.RateLimitError... #{}".format(count))
#             print("restart in 10 seconds")
#             time.sleep(10)
#         except openai.error.ServiceUnavailableError:
#             print("openai.error.ServiceUnavailableError... #{}".format(count))
#             print("restart in 10 seconds")
#             time.sleep(10)
#         except openai.error.APIError:
#             print("openai.error.APIError... #{}".format(count))
#             print("restart in 20 seconds")
#             time.sleep(20)
#         except openai.error.Timeout:
#             print("openai.error.TimeoutError... #{}".format(count))
#             print("restart in 20 seconds")
#             time.sleep(20)