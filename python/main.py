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

parser = argparse.ArgumentParser(description='short text classification model zoo')
parser.add_argument('-m', type=str, default='swem_c', dest='model', help='model name')
parser.add_argument('-data_path', type=str, default='../data/data.csv', help='location of the data')
parser.add_argument('-rebuild', type=ast.literal_eval, default=False, help='whether to rebulid data')
parser.add_argument('-input_col', type=str, default='title', help='col name of input data')
parser.add_argument('-target_col', type=str, default='target', help='col name of target data')
parser.add_argument('-train_size', type=float, default=0.7, help='the proportion of data used in model training')
parser.add_argument('-random_seed', type=int, default=42, help='random_seed')
parser.add_argument('-save', type=ast.literal_eval, default=False, help='whether to save model at every epoch')
# model hyper-parameters
parser.add_argument('-sl', type=int, default=None, dest='seq_length', help='sequence length, default')
parser.add_argument('-ed', type=int, default=300, dest='embed_dim', help='dimension of embedding')
parser.add_argument('-ws', type=int, default=5, dest='window_size', help='window size of hier avg operation')
parser.add_argument('-oc', dest='out_channels', help='out channels of conv layers', default=512, type=int)
parser.add_argument('-ks', type=ast.literal_eval, default=[2, 3, 4, 5, 6, 7], dest='kernel_sizes',
                    help='kernel sizes of conv layers')
parser.add_argument('-hs', type=float, default=512, dest='hidden_state', help='number of units of hidden_state')
parser.add_argument('-lr', type=float, default=1e-3, dest='learning_rate', help='learning rate')
parser.add_argument('-bs', type=int, default=512, dest='batch_size', help='size of mini batch')
parser.add_argument('-ep', type=int, default=5, dest='epoch_num', help='number of epoch')
args = parser.parse_args()


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
        torch.save(model.state_dict(), '../model/%s_bs%d_lr%g_epoch%d.pkl'
                % (args.model, args.batch_size, args.learning_rate, i + 1))
    

print(args)
print(classification_report(dev_true, dev_pred))
