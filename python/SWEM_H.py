import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self, args):
        super(Model, self).__init__()

        Nc = args.num_classes
        Hs = args.hidden_state
        Vs = args.vocab_size

        self.Ws = args.window_size
        self.Sl = args.seq_length
        self.Ed = args.embed_dim

        self.embed = nn.Embedding(Vs, self.Ed)
        
        self.maxpool = nn.MaxPool2d((self.Sl - self.Ws + 1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=(self.Ws, 1), stride=1)
        
        self.fc = nn.Sequential(nn.Linear(self.Ed, Hs), 
                                nn.BatchNorm1d(num_features=Hs), 
                                nn.Sigmoid(),
                                nn.Linear(Hs, Nc))
        
    
    def hier_avgpool(self, word_vec, mask, window_size, eps=1e-8):
        """
        Because the input was padded, the value of the last kernel would be lower than real one.
            eg: (0.33 + 0.16 + 0) / 3, denominator should be 2 not 3 
        In order to fix the value of average operation, we use mask to count the real denominater in kernels       
        """

        mask = mask.unsqueeze(2)
        hier_sum = self.avgpool(word_vec) * window_size
        hier_count = self.avgpool(mask) * window_size + eps
        hier_avg = torch.div(hier_sum, hier_count)

        return hier_avg
    
    def forward(self, x, mask):

        #（Bs, Sl） -> (Bs, Sl, Ed)
        word_vec = self.embed(x)

        #(Bs, Sl, Ed) -> (Bs, Ed)
        hier_avg = self.hier_avgpool(word_vec, mask, self.Ws, eps=1e-8)
        hier_avg = self.maxpool(hier_avg).squeeze(1)

        #(Bs, Ed * 2) -> (Bs, Nc)
        logits = self.fc(hier_avg)

        return logits
