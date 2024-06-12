import torch

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    len_s = seq.size(1)
    subsequent_mask = (
        1 - torch.triu(
        torch.ones(
            (1, len_s, len_s),
            device=seq.device
        ),
        diagonal=1)
    ).bool()
    return subsequent_mask