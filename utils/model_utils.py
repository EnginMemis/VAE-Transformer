import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class OrderedCounter(Counter, OrderedDict):
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def sample_from_prior(batch_size, latent_dim, device):
    return torch.randn([batch_size, latent_dim]).to(device)