import time
import argparse

from sklearn.metrics import accuracy_score, classification_report

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import utils
from utils import DataBuilder

from SWEM import Model as swem
from TextCNN import Model as textcnn
from TextRCNN import Model as textrcnn

def parse_args():
    
    parser = argparse.ArgumentParser(description='text classification model zoo')

    parser.add_argument('--model',
                        help='model name',
                        default='swem', type=str)

    parser.add_argument('--data_path',
                        help='location of the data',
                        default='../data/data.csv', type=str)

    parser.add_argument('--input_col',
                        help='col name of input data',
                        default='title', type=str)

    parser.add_argument('--target_col',
                        help='col name of target data',
                        default='target', type=str)

    parser.add_argument('--train_size',
                        help='train size',
                        default=0.7, type=float)

    parser.add_argument('--random_seed',
                        help='random_seed',
                        default=42, type=int)

    parser.add_argument('--save',
                        help='whether to save model at every epoch',
                        default=True, type=bool)
    
    #model hyper-parameters
    parser.add_argument('--seq_length',
                        help='sequence length',
                        default=36, type=int)

    parser.add_argument('--embed_dim',
                        help='dimension of embedding',
                        default=300, type=int)
    
    parser.add_argument('--out_channels',
                        help='out channels of conv layers',
                        default=256, type=int)

    parser.add_argument('--kernel_sizes',
                        help='kernel sizes of conv layers',
                        default=[2, 3, 4, 5, 6, 7], type=list)

    parser.add_argument('--hidden_state',
                        help='number of units of hidden_state',
                        default=512, type=float)

    parser.add_argument('--learning_rate',
                        help='learning_rate',
                        default=0.0003, type=float)

    parser.add_argument('--batch_size',
                        help='size of mini batch',
                        default=32, type=int)

    parser.add_argument('--epoch_num',
                        help='number of epoch',
                        default=5, type=int)

    args = parser.parse_args()
    
    return args


args = parse_args()
torch.manual_seed(args.random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_zoo = {'swem': swem, 'textcnn': textcnn, 'textrcnn': textrcnn}

d_train, d_dev, _, tokenizer = utils.get_data(args.data_path, args.input_col, args.train_size, args.random_seed)

train_size = d_train.shape[0]
vocab = tokenizer.keys()
vocab_size = len(vocab) + 1

args.vocab_size = vocab_size
args.num_classes = int(d_train[args.target_col].max() + 1)

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

        input_ = Variable(input_).to(device)
        target = Variable(target).view(-1).to(device)
        mask = Variable(mask).to(device)

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
    dev_acc =accuracy_score(dev_true, dev_pred)

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

print(classification_report(dev_true, dev_pred))
