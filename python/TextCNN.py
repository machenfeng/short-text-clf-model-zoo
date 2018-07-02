import torch
import torch.nn as nn


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        oc = args.out_channels
        ks = args.kernel_sizes
        vs = args.vocab_size
        nc = args.num_classes
        sl = args.seq_length
        hs = args.hidden_state
        self.ed = args.embed_dim

        self.embed = nn.Embedding(vs, self.ed)
        self.convs = nn.ModuleList([self.conv_layer(oc, k, self.ed, sl) for k in ks])
        self.fc = nn.Sequential(nn.Linear(oc * len(ks), hs), 
                                nn.BatchNorm1d(num_features=hs), 
                                nn.ReLU(), 
                                nn.Linear(hs, nc))

    def conv_layer(self, oc, k, ed, sl):
        
        # (bs, 1, sl, ed) -> (bs, oc, sl - k + 1, 1)
        conv = nn.Conv2d(in_channels=1, out_channels=oc, kernel_size=(k, ed))
        
        relu = nn.ReLU()
        
        # (bs, oc, sl - k + 1, 1) -> (bs, oc, 1, 1)
        maxpool = nn.MaxPool2d(kernel_size=(sl - k + 1, 1))
        
        return nn.Sequential(conv, relu, maxpool)
    
    def forward(self, x, _):
        
        # (bs, sl) -> (bs, sl, ed)
        out = self.embed(x)
        
        # (bs, sl, ed) -> (bs, 1, sl, ed)
        out = out.unsqueeze(1)
        # (bs, 1, sl, ed) -> [(bs, oc, 1, 1)] * len(ks)
        out = [conv(out) for conv in self.convs]

        # [(bs, oc, 1, 1)] * len(ks) -> (bs, oc * len(ks))
        out = torch.cat(out, 1)
        out = out.view(out.size(0), -1)

        # (bs, oc * len(ks)) -> (bs, nc)
        logits = self.fc(out)

        return logits
