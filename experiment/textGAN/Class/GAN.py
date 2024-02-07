import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
from .Decoder import Decoder
from .Encoder import Encoder
from .Tokenizer import Tokenizer

class Generator(nn.Module):
    def __init__(self, tokenizer):
        super(Generator, self).__init__()
        self.encoder = Encoder(tokenizer)
        self.decoder = Decoder(tokenizer) 
    
    def forward(self, sentence, mask_array):
        state_h, state_c = self.encoder(sentence, mask_array)
        words = self.decoder(sentence, mask_array, state_h, state_c)
        new_sentence = ' '.join(words)
        print(new_sentence)
        return new_sentence
        
class Discriminator(nn.Module):
    def __init__(self, tokenizer):
        super(Discriminator, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        self.num_filters = 20
        self.kernel_size = 5
        
        self.tokenizer = tokenizer
        n_vocab = len(tokenizer.dataset.uniq_words)
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)
        self.conv = nn.Conv1d(self.embedding_dim, 128, self.kernel_size)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters, 
                                kernel_size=self.kernel_size)
        self.lstm = nn.LSTM(input_size=self.num_filters, hidden_size=self.embedding_dim, 
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.embedding_dim, 2)
        
    
    def forward(self, sentence):
        words = sentence.split()
        # inp = torch.tensor([[self.tokenizer.dataset.word_to_index[word] for word in words]])
        
        ids_list = self.tokenizer.tokenize_sentence(sentence)
        ids = torch.tensor([ids_list])
        feature = self._get_feature(ids)
        pred = self.fc(feature)
        return pred
    
    def _get_feature(self, inp):
        # print("inp shape ", inp.shape)
        emb = self.embedding(inp)
        
        emb = emb.permute(0, 2, 1)
        # print("emb shape ", emb.shape)
        feature = self.conv1d(emb)
        
        # print("feature shape 1 (conv)", feature.shape)
        
        feature = feature.permute(0, 2, 1)
        
        prev_states = self._init_state(feature.shape)
        # print('shjape : ',prev_states[0].shape)
        feature, (last_h, last_c) = self.lstm(feature, prev_states)
        
        
        print("feature shape 2 (lstm)", feature.shape)
        
        return feature
    
    def _init_state(self, input_shape):
        return (
            torch.zeros((1, 1, 128)),
            torch.zeros((1, 1, 128)),
        )
    
class TextGAN:
    def __init__(self, dataset):
        tokenizer = Tokenizer(dataset)
        self.generator = Generator(tokenizer)
        self.discriminator = Discriminator(tokenizer)
    def train(self):
        pass
    
    def predict(self):
        pass