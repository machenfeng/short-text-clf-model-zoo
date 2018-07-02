import ast
import numpy as np
import argparse
import matplotlib.pyplot as plt

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
parser.add_argument('-hs', type=int, default=512, dest='hidden_state', help='number of units of hidden_state')
parser.add_argument('-lr', type=float, default=1e-3, dest='learning_rate', help='learning rate')
parser.add_argument('-bs', type=int, default=512, dest='batch_size', help='size of mini batch')
parser.add_argument('-ep', type=int, default=5, dest='epoch_num', help='number of epoch')
# params for learning rate finder
parser.add_argument('-start_lr', type=float, default=1e-5,
                    help='start learning rate in lr finder, also act as the min learning rate')
parser.add_argument('-end_lr', type=float, default=1e-1,
                    help='end learning rate in lr finder, also act as the max learning rate')
parser.add_argument('-step', type=float, default=1e-4,
                    help='step size in every iterations')
parser.add_argument('-num_iters', type=int, default=1000,
                    help='number of iterations in lr finder, also used to calculate the max learning rate')
args = parser.parse_args()


torch.manual_seed(args.random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.switch_backend('agg')
model_zoo = {'swem_c': swem_c, 'swem_h': swem_h, 'textcnn': textcnn, 'textrcnn': textrcnn, 'inception': inception}

d_train, _, __, tokenizer = utils.get_data(args.data_path, args.input_col,
                                              args.rebuild, args.train_size, args.random_seed)

train_rows = d_train.shape[0]
vocab = tokenizer.keys()
vocab_size = len(vocab) + 1

args.vocab_size = vocab_size
args.num_classes = int(d_train[args.target_col].max() + 1)
if args.seq_length is None:
    args.seq_length = d_train[args.input_col].apply(len).max()

trainbulider = DataBuilder(d_train, args.input_col, args.target_col, tokenizer, args.seq_length)
trainloader = DataLoader(trainbulider, args.batch_size, shuffle=True)

model = model_zoo[args.model](args).to(device)
lossfunc = nn.CrossEntropyLoss().to(device)
model.train()
plt.figure(figsize=(16, 9))

print("""Device: %s
Model: %s
Train size: %d""" % (device, args.model, train_rows))

start_lr = args.start_lr
end_lr = args.end_lr

num_imgs = int(np.log10(end_lr / start_lr))

interval_lrs = []
for i in range(num_imgs + 1):
    interval_lrs.append(start_lr * 10 ** i)

for i in range(num_imgs):
    start = interval_lrs[i]
    end = interval_lrs[i + 1]
    step = start / 10

    lrs = np.arange(start, end, step)
    stop = False
    x = []
    y = []
    j = 0

    while not stop:

        for (input_, mask, target) in trainloader:

            if j > len(lrs) - 1:
                stop=True
                break

            x.append(lrs[j])
            optimizer = torch.optim.Adam(model.parameters(), lrs[j])

            input_ = input_.to(device)
            target = target.view(-1).to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(input_, mask)
            loss = lossfunc(logits, target)
            y.append(float(loss.item()))

            if loss.item() / input_.size(0) < 0.0001:
                stop = True
                break

            loss.backward()
            optimizer.step()

            j += 1

    print('start: %g' % start)
    print('end: %g' % end)
    print('step: %g' % step)

    plt.plot(x, y, '--')
    plt.xscale('log', basex=10)
    plt.xlabel("learning rate log scale")
    plt.ylabel("loss")
plt.savefig('../lr_check/%s_bs%d.jpg' % (args.model, args.batch_size))
