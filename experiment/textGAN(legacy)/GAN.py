import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from utils.helpers import truncated_normal_



class Generator(nn.Module):
    def __init__(self, encoder, decoder, output_size):
        super(Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_size = output_size

    def forward(self, input_seq, mask_indices):
        batch_size = input_seq.size(0)
        max_seq_len = input_seq.size(1)
        
        # 인코더의 초기 은닉 상태
        encoder_hidden = self.encoder.initHidden()
        encoder_hidden = (encoder_hidden[0].repeat(1, batch_size, 1),
                          encoder_hidden[1].repeat(1, batch_size, 1))
        
        # 인코더를 통과
        encoder_outputs, encoder_hidden = self.encoder(input_seq, encoder_hidden)
        
        # 디코더의 입력을 준비 (마스크된 위치에만 MASK_TOKEN_INDEX를 넣는다)
        decoder_input = input_seq.clone()
        decoder_input[mask_indices] = MASK_TOKEN_INDEX
        
        # 디코더의 초기 상태는 인코더의 최종 상태
        decoder_hidden = encoder_hidden
        
        # 마스크된 위치에 대해서만 디코딩
        outputs = torch.zeros(batch_size, max_seq_len, self.output_size)
        for i in range(max_seq_len):
            is_masked = mask_indices[:, i]  # 현재 인덱스가 마스크된 인덱스인지 확인
            if is_masked.any():
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input[:, i], decoder_hidden
                )
                outputs[is_masked, i, :] = decoder_output[is_masked, :]
        
        return outputs
                    
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