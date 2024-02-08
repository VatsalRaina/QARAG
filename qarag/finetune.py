import json
import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses
import numpy as np
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--n_epochs', type=int, default=2, help='Number of training epochs.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the embedder name.')
parser.add_argument('--is_seen', type=str, default="no", help='Whether the chunks are seen.')
parser.add_argument('--save_path', type=str, default="", help='Save path.')


def main(args):

    # Prepare the data
    train_examples = []
    if args.is_seen == 'no':
        with open(args.data_dir + 'data.json', 'r') as f:
            data = json.load(f)
        for ex in data:
            train_examples.append(InputExample(texts=[ex['question'], ex['context']]))
    else:
        with open(args.data_dir + 'chunks.json', 'r') as f:
            chunks = json.load(f)
        with open(args.data_dir + 'gen_questions.json', 'r') as f:
            questions = json.load(f)
        for chunk, question in zip(chunks, questions):
            train_examples.append(InputExample(texts=[question, chunk]))
        with open(args.data_dir + 'gen_questions_2.json', 'r') as f:
            questions = json.load(f)
        for chunk, question in zip(chunks, questions):
            train_examples.append(InputExample(texts=[question, chunk]))
        with open(args.data_dir + 'gen_questions_3.json', 'r') as f:
            questions = json.load(f)
        for chunk, question in zip(chunks, questions):
            train_examples.append(InputExample(texts=[question, chunk]))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Prepare the model
    model = SentenceTransformer("sentence-transformers/" + args.embedder)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=args.n_epochs) 
    model.save(args.save_path)



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)