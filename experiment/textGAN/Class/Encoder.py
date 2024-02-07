import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, tokenizer, device='cpu'):
        super(Encoder, self).__init__()
        self.device = device
        
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        
        self.tokenizer = tokenizer
        n_vocab = len(tokenizer.dataset.uniq_words)
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        
    def one_step(self, x, prev_state):
        embed = self.embedding(x)
        # print('embed', embed.shape, prev_state[0].shape, prev_state[1].shape)
        output, (state_h, state_c) = self.lstm(embed, prev_state)
        return output, (state_h, state_c)
    
    def forward(self, sentence, mask_array):
        
        # words = sentence.split()
        state_h, state_c = self.init_state(1) # (len(words))
        
        ids_list = self.tokenizer.tokenize_sentence(sentence)
        
        for ids, is_masked in zip(ids_list, mask_array):
            x = torch.tensor([[ids]]).to(self.device)
            # x = torch.tensor([[self.dataset.word_to_index[word]]])
            # print('new', word, x.shape)
            y_pred, (state_h, state_c) = self.one_step(x, (state_h, state_c))
            
        return state_h, state_c
    
    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.device),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.device),
        )