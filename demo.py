import json, os, argparse, torch
from models.VAE_transformer import VariationalTransformer
from transformers import AutoTokenizer
from models.masking import *
from utils.model_utils import *
from utils.data_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('-ckpt', '--checkpoint', type=str)
    parser.add_argument('-dn','--data_name', type=str, default='data')
    parser.add_argument('-bin', '--save_model_path', type=str, default='models')
    parser.add_argument('-noi', '--noise', type=str, default="False")

    args = parser.parse_args()

    if args.noise == "False":
        print("Noise has been set to False.")
    elif args.noise == "True":
        print("Noise has been set to True.")
    else:
        print("Wrong argument! The default setting for noise has been set to False.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.save_model_path, "transformer_model_params.json"), "r") as f:
        params = json.load(f)

    load_checkpoint = args.checkpoint

    model = VariationalTransformer(**params)
    model.load_state_dict(torch.load(load_checkpoint))
    model = model.to(device=device)


    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

    with open(os.path.join("data", args.data_name), "r", encoding="utf-8") as file:
        satirlar = file.readlines()


        modelCumleler = []
        inputCumleler = []
        for satir in satirlar:
            sentence = satir[:-1]
            inputCumleler.append(sentence)

            seq = tokenizer(sentence).input_ids
            src = seq[1:-1]
            target = seq[1:-1]

            target = torch.tensor([target]).to(device=device)
            src = torch.tensor([src]).to(device=device)

            if args.noise == "True":
                prior = sample_from_prior(1, 16, device)
            else:
                prior = torch.ones((1, 16)).to(device=device)
            
            output, output2, logp= model(src, target, prior)


            _, sample = torch.topk(logp, 1, dim=-1)
            sample = sample.reshape(-1)

            newSentence = tokenizer.decode(sample, skip_special_tokens=True)
            modelCumleler.append(newSentence)

        with open(os.path.join("data", str(args.noise) + "_model_" + args.data_name), "w", encoding="utf-8") as file:
            for j in range(len(modelCumleler)):
                file.write("Verilen Cumle: ")
                file.write(inputCumleler[j])
                file.write("\n")
                file.write("Model Ciktisi: ")
                file.write(modelCumleler[j])
                file.write("\n")
                file.write("\n")

        print(f"{len(modelCumleler)} Tane Cumle Uretildi!")