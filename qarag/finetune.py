import json
import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses
import numpy as np
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--data_dir', type=str, default='', help='Specify the path to the data directory.')
parser.add_argument('--embedder', type=str, default="sentence-t5-base", help='Specify the embedder name.')
parser.add_argument('--save_path', type=str, default="", help='Save path.')



def main(args):


    # Prepare the data

    with open(args.data_dir + 'data.json', 'r') as f:
        data = json.load(f)
    train_examples = []
    for count, ex in enumerate(data):
        train_examples.append(InputExample(texts=[ex['question'], ex['context']]))
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)



    # Prepare the model
    model = SentenceTransformer("sentence-transformers/" + args.embedder)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2) 
    model.save(args.save_path)





if __name__ == "__main__":
    args = parser.parse_args()
    main(args)