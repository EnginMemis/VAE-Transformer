import torch
from models.VAE_transformer import VariationalTransformer
from utils.model_utils import *
from utils.data_utils import *
from models.masking import *
from transformers import AutoTokenizer
import pandas as pd

import json, os, torch, argparse
from collections import OrderedDict


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('-dn','--data_name', type=str, default='data')
    parser.add_argument('-bin', '--save_model_path', type=str, default='models')
    parser.add_argument('-ckpt', '--checkpoint', type=str)
    parser.add_argument('-ml', '--max_sentence_length', type=int, default=60)
    parser.add_argument('-ra', '--rate', type=float, default=0.5)

    args = parser.parse_args()

    with open(os.path.join(args.save_model_path, "transformer_model_params.json"), "r") as f:
        params = json.load(f)

    load_checkpoint = args.checkpoint

    model = VariationalTransformer(**params)
    model.load_state_dict(torch.load(load_checkpoint))

    model2 = VariationalTransformer(**params)

    if params["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    latent_size = params["latent_size"]

    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    if args.data_name[-3:] == "csv":
        dataset = pd.read_csv(args.data_name)
    elif args.data_name[-4:] == "xlsx":
        dataset = pd.read_excel(args.data_name)
    try:
        dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
    except:
        pass

    generated_sentences = {}
    criterion = nn.CrossEntropyLoss()
    df = pd.DataFrame(columns = ["Sentence", "Label"])

    print("boyut")
    print(len(dataset))
    for i in range(len(dataset)):
        try:
            if i % 100 == 0:
                print(i)

            sentence = dataset.loc[i]["Sentence"]

            seq = tokenizer(sentence).input_ids
            src = seq[1:-1]
            target = seq[1:-1]

            target = torch.tensor([target]).to(device=device)
            src = torch.tensor([src]).to(device=device)
            #prior = torch.ones((1, latent_size)).to(device=device)
            target_seq = target.contiguous().view(-1)
            
            for j in range(1):
                prior = sample_from_prior(1, latent_size, device)
                _, _, logp= model(src, target, prior)

                loss = criterion(logp, target_seq)

                _, sample = torch.topk(logp, 1, dim=-1)
                sample = sample.reshape(-1)[:args.max_sentence_length]

                newSentence = tokenizer.decode(sample, skip_special_tokens=True)

                generated_List = []
                generated_List.append(newSentence)
                generated_List.append(dataset.loc[i]["Label"])

                generated_sentences[loss.item()] = generated_List

        except Exception as e:
            print("Error " + str(e))

    generated_sentences = OrderedDict(sorted(generated_sentences.items()))
    
    keys = list(generated_sentences.keys())

    for i in range(int(len(keys) / 1 * args.rate)):
        dataset.loc[len(dataset)] = generated_sentences[keys[i]]

    print("Yeni Boyut")
    print(len(dataset))

    if args.data_name[-3:] == "csv":
        dataset.to_csv(str(int(args.rate * 100)) + "_100Uretim_" + args.data_name)
    elif args.data_name[-4:] == "xlsx":
        dataset.to_excel(str(int(args.rate * 100)) + "_100Uretim_" + args.data_name)

    print("Sentences have been generated")
