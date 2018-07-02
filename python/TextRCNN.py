import torch
import torch.nn as nn


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        vs = args.vocab_size
        nc = args.num_classes
        hs = args.hidden_state

        self.ed = args.embed_dim
        self.sl = args.seq_length
        
        self.embed = nn.Embedding(vs, self.ed)
        self.lstm = nn.LSTM(self.ed, hs, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2 * hs + self.ed, hs),
                                nn.BatchNorm1d(hs),
                                nn.ReLU(),
                                nn.Linear(hs, nc))

    def rcnn_main(self, embed, lstm):

        # (bs, sl, ed) -> (bs, sl, 2 * hs)
        out = lstm(embed)[0]

        # (bs, sl, 2 * hs) -> (bs, sl, 2 * hs + ed)
        out = torch.cat((out, embed), 2)
        out = torch.tanh(out)

        # (bs, sl, 2 * hs + ed) -> (bs, 1,  2 * hs + ed)
        out = torch.max(out, dim=1)[0]
        
        return out

    def forward(self, x, mask):

        bs = x.size(0)
        # (bs, sl) -> (bs, sl, ed)
        out = self.embed(x)

        mask4fill = abs(mask - 1).byte()
        mask4fill = mask4fill.unsqueeze(2).expand([bs, self.sl, self.ed])
        out = out.masked_fill_(mask4fill, 0)

        # (bs, sl, ed) -> (bs, 2 * hs + ed)
        out = self.rcnn_main(out, self.lstm)
        out = out.view(out.size(0), -1)

        # (bs, 2 * hs + ed) -> (bs, nc)
        logits = self.fc(out)
        
        return logits
