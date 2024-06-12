import torch
import torch.nn as nn

from models.layers import *
from models.encoder_decoder import *
from models.masking import *
from utils.model_utils import *
import random

class VariationalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_k, d_v, d_word_vec, d_inner, n_layers, n_head, dropout, 
                 latent_size, n_position, max_sequence_length, pad_idx, device):
        super(VariationalTransformer, self).__init__()

        if device=="cuda":
            self.tensor = torch.cuda.FloatTensor
        else:
            self.tensor = torch.Tensor
        

        self.encoder = Encoder(
            n_src_vocab = vocab_size,
            n_position = n_position,
            d_word_vec = d_word_vec,
            d_model = d_model,
            d_inner = d_inner,
            n_layers = n_layers,
            n_head = n_head,
            d_k = d_k,
            d_v = d_k,
            pad_idx = pad_idx,
            dropout = dropout,
        )

        self.decoder = Decoder(
            n_trg_vocab = vocab_size,
            n_position = n_position,
            d_word_vec = d_word_vec,
            d_model = d_model,
            d_inner = d_inner,
            n_layers = n_layers,
            n_head = n_head,
            d_k = d_k,
            d_v = d_v,
            pad_idx = pad_idx,
            dropout = dropout,
        )

        self.trg_word_prj = nn.Linear(d_model, vocab_size, bias=True)
        self.enc_max_seq_len = max_sequence_length

        self.device = device
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.sos_idx = 2
        self.eos_idx = 3

        self.latent_size = latent_size
        self.num_layers = n_layers
        self.word_dropout_rate = dropout

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(p=dropout)

        self.vae_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(),
        )

        self.vae_decoder = nn.Sequential(
            nn.Linear(latent_size, d_model // 2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(d_model // 2, d_model),
            nn.GELU(),
            nn.Dropout(),
        )
        self.outputs2vocab = nn.Linear(d_model, vocab_size)
        self.context2mean = nn.Linear(d_model // 2, latent_size)
        self.context2std = nn.Linear(d_model // 2, latent_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src_seq, trg_seq, prior):
    
    
        src_mask = get_pad_mask(src_seq, self.pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.pad_idx) & get_subsequent_mask(trg_seq)


        enc_output, *_ = self.encoder(src_seq, src_mask)

        hidden = enc_output

        cls_token = enc_output[:,0,:]
    
        latent = self.vae_encoder(cls_token)

        mean = self.context2mean(latent)
        std = self.context2std(latent)

        z = prior * torch.exp(0.5 * std) + mean

        to_dec = self.vae_decoder(z)
    
        to_dec = to_dec.repeat(1,self.enc_max_seq_len)

        to_dec = to_dec.reshape(src_seq.size(0), self.enc_max_seq_len, -1)

        dec_output, *_ = self.decoder(trg_seq, trg_mask, to_dec, src_mask = None) #enc_output

        logp = nn.functional.log_softmax(self.trg_word_prj(dec_output.view(-1, dec_output.size(2))), dim=-1)
        return dec_output, hidden, logp


    
