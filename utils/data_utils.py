from collections import defaultdict
from torch.utils.data import Dataset
import os, io, json, torch, re
import numpy as np
from utils.model_utils import OrderedCounter
from transformers import AutoModel, AutoTokenizer

class TextDataLoader(Dataset):
    def __init__(self, data_name, data_dir, split, create_data, **kwargs):
        super(TextDataLoader, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, data_name+'.'+split+'.txt')
        self.data_file = data_name+'.'+split+'.json'
        self.vocab_file = data_name+'.vocab.json'

        if create_data:
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("{} preprocessed file not found at {}. Creating new.".format(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def _load_data(self,vocab=True):
        with open(os.path.join(self.data_dir, self.data_file), 'r', encoding="utf-8") as file:
            self.data = json.load(file)

        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r', encoding="utf-8") as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self,vocab=True):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r', encoding="utf-8") as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

        data = defaultdict(dict)

        with open(self.raw_data_path, 'r', encoding="utf-8") as file:
            for i, line in enumerate(file):
                line = line[:512]
                words = tokenizer.tokenize(line)
                input = ['[SOS]'] + words
                input = input[:self.max_sequence_length]
                target = words[:self.max_sequence_length-1]
                target = target + ['[EOS]']

                assert len(input) == len(target), "%i, %i"%(len(input), len(target))
                length = len(input)

                input.extend(['[PAD]'] * (self.max_sequence_length-length))
                target.extend(['[PAD]'] * (self.max_sequence_length-length))

                input = [self.w2i.get(w, self.w2i['[UNK]']) for w in input]
                target = [self.w2i.get(w, self.w2i['[UNK]']) for w in target]

                id = len(data)
                data[id]['input'] = input
                data[id]['target'] = target
                data[id]['length'] = length

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):
        assert self.split == 'train', "Vocablurary can only be created for training file."

        if not os.path.exists(os.path.join(self.data_dir, self.vocab_file)):
            print("sa")
            tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
            w2c = OrderedCounter()
            t2c = OrderedCounter()
            w2i = dict()
            i2w = dict()

            special_tokens = ['[PAD]', '[UNK]', '[SOS]', '[EOS]']
            for st in special_tokens:
                i2w[len(w2i)] = st
                w2i[st] = len(w2i)

            with open(self.raw_data_path, 'r', encoding="utf-8") as file:
                for i, line in enumerate(file):
                    line = line[:512]
                    words = tokenizer.tokenize(line)

                    tokens = tokenizer(line)
                    tokens = tokens.input_ids[1:-1]
                    
                    w2c.update(words)
                    t2c.update(tokens)

                wordsList = list(w2c.keys())
                tokensList = list(t2c.keys())
                
                assert len(wordsList) == len(tokensList)
                for i in range(len(wordsList)):
                  if wordsList[i] not in special_tokens:
                    w2i[wordsList[i]] = tokensList[i]
                    i2w[tokensList[i]] = wordsList[i]

            assert len(w2i) == len(i2w)
            print("Vocabulary of {} keys created.".format(len(w2i)))

            vocab = dict(w2i=w2i, i2w=i2w)
            with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
                data = json.dumps(vocab, ensure_ascii=False)
                vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

        assert len(self.w2i) == len(self.i2w)
        print("Vocabulary of {} keys created.".format(len(self.w2i)))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['[PAD]']

    @property
    def sos_idx(self):
        return self.w2i['[SOS]']

    @property
    def eos_idx(self):
        return self.w2i['[EOS]']

    @property
    def unk_idx(self):
        return self.w2i['[UNK]']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w
