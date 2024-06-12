import math
import os
from tempfile import TemporaryDirectory
import time
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset
from collections import defaultdict
import numpy as np
import pandas as pd
from utils.model_utils import *
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from models.VAE_transformer import *
from utils.data_utils import TextDataLoader
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type=str, default='corpus')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--max_sequence_length', type=int, default=60)
parser.add_argument('--create_data', action='store_true')
parser.add_argument('--min_occ', type=int, default=1)
parser.add_argument('-ep', '--epochs', type=int, default=1)
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-dm', '--d_model', type=int, default=512)
parser.add_argument('-dk', '--d_k', type=int, default=64)
parser.add_argument('-dv', '--d_v', type=int, default=64)
parser.add_argument('-vs', '--vocab_size', type=int, default=52000)
parser.add_argument('-di', '--d_inner', type=int, default=2048)
parser.add_argument('-nl', '--n_layers', type=int, default=6)
parser.add_argument('-nh', '--n_head', type=int, default=8)
parser.add_argument('-dp', '--dropout', type=float, default=0.1)
parser.add_argument('-ls', '--latent_size', type=int, default=16)
parser.add_argument('-np', '--n_position', type=int, default=61)
parser.add_argument('-pi', '--pad_idx', type=int, default=0)
parser.add_argument('-li', '--log_interval', type=int, default=200)
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

splits = ['train', 'valid']

datasets = OrderedDict()

for split in splits:
    datasets[split] = TextDataLoader(
        data_name = args.data_name,
        data_dir = args.data_dir,
        split=split,
        create_data=args.create_data,
        max_sequence_length = args.max_sequence_length,
        min_occ = args.min_occ
    )


params = dict(
        vocab_size=52000,
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_v,
        d_word_vec=args.d_model,
        d_inner=args.d_inner,
        n_layers=args.n_layers,
        n_head=args.n_head,
        dropout=args.dropout,
        latent_size=args.latent_size,
        n_position=args.n_position,
        max_sequence_length=args.max_sequence_length,
        pad_idx=args.pad_idx,
        device=device.type
    )


model = VariationalTransformer(**params).to(device)

#load_checkpoint = "models/best_model_params3_1.pt"
#model.load_state_dict(torch.load(load_checkpoint))

with open(os.path.join("models", 'transformer_model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

criterion = nn.CrossEntropyLoss()
lr = args.learning_rate  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float('inf')

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        for split in splits:
            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                pin_memory=torch.cuda.is_available()   
            )        

            total_val_loss = 0.
            total_loss = 0.
            log_interval = args.log_interval
            start_time = time.time()

            num_batches = len(data_loader.dataset) // args.batch_size
            for iteration, batch in enumerate(data_loader):
                batch_size = batch['input'].size(0)
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)
                

                prior = sample_from_prior(batch_size, args.latent_size, device)
                targets = batch['target']

                if split == 'train':

                    model.train()

                    output, output2, logp= model(batch['input'], targets, prior)
                    targets = targets.to(torch.long)

                    targets = targets.contiguous().view(-1)
                    loss = criterion(logp, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    total_loss += loss.item()
                    if iteration % log_interval == 0 and iteration > 0:
                        lr = scheduler.get_last_lr()[0]
                        ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                        cur_loss = total_loss / log_interval
                        ppl = math.exp(cur_loss)
                        print(f'| epoch {epoch:3d} | {iteration:5d}/{num_batches:5d} batches | '
                            f'lr {lr:02.5f} | ms/batch {ms_per_batch:5.2f} | '
                            f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                        total_loss = 0
                        start_time = time.time()

                
                if split == 'valid':
                    model.eval()  # turn on evaluation mode
                    with torch.no_grad():
                        prior = sample_from_prior(batch_size, args.latent_size, device)
                        targets = batch['target']
                        output, output2, logp= model(batch['input'], targets, prior)

                        targets = targets.contiguous().view(-1)
                        targets = targets.to(torch.long)

                        total_val_loss += criterion(logp, targets).item()
        
        total_val_loss = total_val_loss / (iteration + 1) 
        val_ppl = math.exp(total_val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
            f'valid loss {total_val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        epoch_string = "models/best_model_params" + str(epoch) + ".pt"
        torch.save(model.state_dict(), epoch_string)

        scheduler.step()
    
    stop = time.time()
    print(stop - start)