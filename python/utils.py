import warnings

import pickle
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

from torch.autograd import Variable
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DataBuilder(data.Dataset):
    
    def __init__(self, df, input_col, target_col, idx_dict, seq_length):
        
        self.df = df
        self.df.set_index(np.arange(self.df.shape[0]))
        
        self.input = self.df[input_col].tolist()
        self.target = self.df[target_col].tolist()
        
        self.idx_dict = idx_dict
        self.seq_length = seq_length
        
    def __getitem__(self, index):
        
        #构造idx特征
        input_ = self.input[index]
        target = self.target[index]

        input_idx, mask = self.get_idx(input_, self.idx_dict, self.seq_length)
        input_tensor = torch.LongTensor(input_idx)
        mask = torch.FloatTensor(mask)
        target_tensor = torch.LongTensor([target])

        return input_tensor, mask, target_tensor

    def __len__(self):
        return len(self.target)

    def get_idx(self, sentence, idx_dict, max_length):

        idxs = []
        sentence = list(str(sentence))
        pad_index = idx_dict['PAD']
        
        for word in sentence:
            if word in idx_dict.keys():
                idx = idx_dict[word]
            else:
                idx = idx_dict['UKN']
            idxs.append(idx)

        L = len(idxs)
        if L >= max_length:
            idxs = idxs[:max_length]
            mask = [1] * max_length
        else:
            idxs = idxs + (max_length - L) * [pad_index]
            mask = [1] * L + [0] * (max_length - L)

        return idxs, mask
        
    
def dict_build(cols, use_char=True, save_dir=None):
    if type(cols) != list:
        cols = [cols]

    word = pd.concat(cols).tolist()
    
    if use_char is True:
        word = ['UKN', 'PAD'] + list(''.join(word))
    else:
        word = ['UKN', 'PAD'] + ','.join(word).split(',')
        
    word = list(set(word))
    idx = range(0, len(word))

    items = zip(word, idx)
    word2idx = dict((i, j) for i, j in items)

    if save_dir is not None:
        output = open(save_dir, 'wb')
        pickle.dump(word2idx, output)
        output.close()

    return word2idx


def get_data(data_path, input_col, rebulid=False, train_size=0.7, random_seed=42):

    d = pd.read_csv(data_path)

    if rebulid:
        d_train, d_dev = train_test_split(d, test_size=1 - train_size, random_state=random_seed)
        d_dev, d_test = train_test_split(d_dev, test_size=0.5, random_state=random_seed)

        d_train.to_csv('../data/train.csv', index=None)
        d_dev.to_csv('../data/dev.csv', index=None)
        d_test.to_csv('../data/test.csv', index=None)

        char2idx = dict_build(d_train[input_col], save_dir='../resource/char2idx.pkl')

    else:
        d_train = pd.read_csv('../data/train.csv')
        d_dev = pd.read_csv('../data/dev.csv')
        d_test = pd.read_csv('../data/test.csv')
        char2idx = open('../resource/char2idx.pkl', 'rb')
        char2idx = pickle.load(char2idx)

    
    return d_train, d_dev, d_test, char2idx
    
    
def eval(model, dataloader, lossfunc):

    y_pred = []
    y_true = []
    total_loss = 0
    count = 0
    for (sentence, mask, label) in dataloader:

        sentence = Variable(sentence).to(device)
        label = Variable(label).view(-1).to(device)
        mask = Variable(mask).to(device)
        
        logits = model(sentence, mask)
        predict = logits.max(1)[1]
        loss = lossfunc(logits, label)
        total_loss += loss.item()
        count += 1        

        y_pred += predict.tolist()
        y_true += label.tolist()
    
    return y_true, y_pred, total_loss / count
