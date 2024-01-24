import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.helpers import truncated_normal_




class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, device='cpu'):
        super(Generator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.lstm2out = nn.Linear(hidden_dim * 2, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self._init_params()
        
    def forward(self, inp, hidden, need_hidden=False):
        emb = self.embedding(inp)
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)
        
        print("================ start =====================")
        print('emb size', emb.size())
        out, hidden = self.lstm(emb, hidden)
        print('out bef size', out.size())
        out = out.contiguous().view(-1, out.size(-1))
        print('out after size', out.size())
        
        out = self.lstm2out(out)
        pred = self.softmax(out)
        print("pred size ", pred.size())
        
        print("================ end =====================")
        if need_hidden:
            return pred, hidden
        else:
            return pred

    def _init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1. / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)
                    
                    
class Discriminator(nn.Module):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, dropout=0.2, device='cpu'):
        super(Discriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = 1 # sum(num_filters)
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(5, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self._init_params()
    
    def _init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

    def forward(self, inp):
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))
        
        return pred
    
    def get_feature(self, inp):
        print("inp shape ", inp.shape)
        emb = self.embeddings(inp).unsqueeze(1)
        print("emb shape ", emb.shape)
        convs = [F.relu(conv(emb)) for conv in self.convs]
        print('conv 0', convs[0].shape)
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)
        print('pred', pred.shape)
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred
        print(pred)
        return pred

def sample_z(batch_size=1, d_noise=100, device='cpu'):
    return torch.randn(batch_size, d_noise, device=device)