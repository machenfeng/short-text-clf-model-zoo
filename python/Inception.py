import torch
import torch.nn as nn


class InceptionBlock(nn.Module):

    def __init__(self, args):
        super(InceptionBlock, self).__init__()

        oc = args.out_channels
        ed = args.embed_dim

        # (bs, 1, sl, ed) -> (bs, oc, sl, 1)
        self.branch1 = nn.Conv2d(1, oc, (1, ed))

        # (bs, 1, sl, ed) -> (bs, oc, sl-3+1, 1)
        self.branch2 = nn.Sequential(nn.Conv2d(1, oc, (1, ed)),
                                     nn.ReLU(),
                                     nn.Conv2d(oc, oc, (3, 1)))

        # (bs, 1, sl, ed) -> (bs, oc, sl-3+1-5+1, 1)
        self.branch3 = nn.Sequential(nn.Conv2d(1, oc, (3, ed)),
                                     nn.ReLU(),
                                     nn.Conv2d(oc, oc, (5, 1)))
        # (bs, 1, sl, ed) -> (bs, oc, sl-2, 1)
        self.branch4 = nn.Conv2d(1, oc, (3, ed))

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(oc)

    def forward(self, x):

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat((b1, b2, b3, b4), 2).squeeze(3)
        out = self.bn(out)
        out = self.relu(out)

        return out


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        oc = args.out_channels
        vs = args.vocab_size
        nc = args.num_classes
        sl = args.seq_length
        hs = args.hidden_state
        ed = args.embed_dim

        self.embed = nn.Embedding(vs, ed)

        self.inception_block = InceptionBlock(args)

        self.fc = nn.Sequential(nn.Linear(oc * (sl * 4 - 10), hs),
                                nn.BatchNorm1d(hs),
                                nn.ReLU(),
                                nn.Linear(hs, nc))

    def forward(self, x, _):

        # (bs, sl) -> (bs, 1， sl， ed)
        out = self.embed(x).unsqueeze(1)

        # (bs, 1， sl， ed) -> (bs, oc, (sl * 4 - 10))
        out = self.inception_block(out)

        # (bs, oc, (sl * 4 - 10)) -> (bs, nc)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)

        return logits
