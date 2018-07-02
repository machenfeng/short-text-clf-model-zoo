import time
import ast
import argparse

from sklearn.metrics import accuracy_score, classification_report

import torch
from torch import nn
from torch.utils.data import DataLoader

import utils
from utils import DataBuilder

from SWEM_C import Model as swem_c
from SWEM_H import Model as swem_h
from TextCNN import Model as textcnn
from TextRCNN import Model as textrcnn
from Inception import Model as inception


def parse_args():
    
    parser = argparse.ArgumentParser(description='text classification model zoo')

    parser.add_argument('-m',
                        dest='model',
                        help='model name',
                        default='swem_c', type=str)

    parser.add_argument('-data_path',
                        help='location of the data',
                        default='../data/data.csv', type=str)

    parser.add_argument('-rebuild',
                        dest='rebuild',
                        help='whether to rebulid data',
                        default=False, type=ast.literal_eval)

    parser.add_argument('-input_col',
                        help='col name of input data',
                        default='title', type=str)

    parser.add_argument('-target_col',
                        help='col name of target data',
                        default='target', type=str)

    parser.add_argument('-train_size',
                        help='the proportion of data used in model training',
                        default=0.7, type=float)

    parser.add_argument('-random_seed',
                        help='random_seed',
                        default=42, type=int)

    parser.add_argument('-save',
                        help='whether to save model at every epoch',
                        default=False, type=ast.literal_eval)
    
    # model hyper-parameters
    parser.add_argument('-sl',
                        dest='seq_length',
                        help='sequence length, default',
                        default=None,
                        type=int)

    parser.add_argument('-ed',
                        dest='embed_dim',
                        help='dimension of embedding',
                        default=300, type=int)

    parser.add_argument('-ws',
                        dest='window_size',
                        help='window size of hier avg operation',
                        default=5, type=int)

    parser.add_argument('-oc',
                        dest='out_channels',
                        help='out channels of conv layers',
                        default=512, type=int)

    parser.add_argument('-ks',
                        dest='kernel_sizes',
                        help='kernel sizes of conv layers',
                        default=[2, 3, 4, 5, 6, 7], type=ast.literal_eval)

    parser.add_argument('-hs',
                        dest='hidden_state',
                        help='number of units of hidden_state',
                        default=512, type=float)

    parser.add_argument('-lr',
                        dest='learning_rate',
                        help='learning rate',
                        default=1e-3, type=float)

    parser.add_argument('-bs',
                        dest='batch_size',
                        help='size of mini batch',
                        default=512, type=int)

    parser.add_argument('-ep',
                        dest='epoch_num',
                        help='number of epoch',
                        default=5, type=int)

    args = parser.parse_args()
    
    return args


args = parse_args()
torch.manual_seed(args.random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_zoo = {'swem_c': swem_c, 'swem_h': swem_h, 'textcnn': textcnn, 'textrcnn': textrcnn, 'inception': inception}

d_train, d_dev, _, tokenizer = utils.get_data(args.data_path, args.input_col,
                                              args.rebuild, args.train_size, args.random_seed)

train_size = d_train.shape[0]
vocab = tokenizer.keys()
vocab_size = len(vocab) + 1

args.vocab_size = vocab_size
args.num_classes = int(d_train[args.target_col].max() + 1)
if args.seq_length is None:
    args.seq_length = d_train[args.input_col].apply(len).max()

trainbulider = DataBuilder(d_train, args.input_col, args.target_col, tokenizer, args.seq_length)
trainloader = DataLoader(trainbulider, args.batch_size, shuffle=False)

devbulider = DataBuilder(d_dev, args.input_col, args.target_col, tokenizer, args.seq_length)
devloader = DataLoader(devbulider, args.batch_size, shuffle=False)

model = model_zoo[args.model](args).to(device)

lossfunc = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

print("""Device: %s
Model: %s
Train size: %d""" % (device, args.model, train_size))


for i in range(args.epoch_num):
    
    t1 = time.time()
    
    y_pred = []
    y_true = []
    epoch_loss = 0

    model.train()
    for (input_, mask, target) in trainloader:

        input_ = input_.to(device)
        target = target.view(-1).to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(input_, mask)
        loss = lossfunc(logits, target)

        predict = logits.max(1)[1]
        epoch_loss += float(loss.item())

        y_pred += predict.tolist()
        y_true += target.tolist()

        loss.backward()
        optimizer.step()
        
    t2 = time.time()
    
    model.eval()
    dev_true, dev_pred, dev_loss = utils.eval(model, devloader, lossfunc)
    dev_acc = accuracy_score(dev_true, dev_pred)

    epoch = (i + 1)
    time_cost = (t2 - t1)
    train_loss = epoch_loss / train_size
    train_acc = accuracy_score(y_true, y_pred)

    print('''------------------------------------
Epoch %d  Time cost: %.2fs
Train_loss: %.4f  Dev_loss: %.4f
Train_acc: %.4f  Dev_acc: %.4f
------------------------------------''' % (epoch, time_cost, train_loss, dev_loss, train_acc, dev_acc))

if args.save:
    print(args)
    torch.save(model.state_dict(), '../model/%s_bs%d_lr%g_epoch%d.pkl'
            % (args.model, args.batch_size, args.learning_rate, i + 1))

print(classification_report(dev_true, dev_pred))
