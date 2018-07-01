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
                        default='swem_h', type=str)

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

_, __, d_test, tokenizer = utils.get_data(args.data_path, args.input_col,
                                          args.rebuild, args.train_size, args.random_seed)

test_size = d_test.shape[0]
vocab = tokenizer.keys()
vocab_size = len(vocab) + 1

args.vocab_size = vocab_size
args.num_classes = 14
args.seq_length = 47

testbulider = DataBuilder(d_test, args.input_col, args.target_col, tokenizer, args.seq_length)
testloader = DataLoader(testbulider, args.batch_size, shuffle=False)

model = model_zoo[args.model](args).to(device)
model_dir = '../model/%s.pkl' % args.model
model.load_state_dict(torch.load(model_dir))

lossfunc = nn.CrossEntropyLoss().to(device)

print("""Device: %s
Model: %s
Test size: %d""" % (device, args.model, test_size))


t1 = time.time()

y_pred = []
y_true = []

model.eval()
for (input_, mask, target) in testloader:
    input_ = input_.to(device)
    target = target.view(-1).to(device)
    mask = mask.to(device)

    logits = model(input_, mask)
    predict = logits.max(1)[1]

    y_pred += predict.tolist()
    y_true += target.tolist()

t2 = time.time()

time_cost = (t2 - t1)
test_acc = accuracy_score(y_true, y_pred)

print('''------------------------------------
Time cost: %.2fs
Test_acc: %.4f
------------------------------------''' % (time_cost, test_acc))