import numpy as np
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, tokenizer, device='cpu'):
        super(Decoder, self).__init__()
        self.device = device
        
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        
        self.tokenizer = tokenizer
        
        n_vocab = len(tokenizer.dataset.uniq_words)
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        self.fc = nn.Linear(self.lstm_size, n_vocab)
        
    def one_step(self, x, prev_state):
        embed = self.embedding(x)
        output, (state_h, state_c) = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, (state_h, state_c)
    
    def forward(self, sentence, mask_array, state_h=None, state_c=None):
        ids_list = self.tokenizer.tokenize_sentence(sentence)
        if state_h is None:
            state_h, state_c = self.init_state(1)
        
        new_words = []
        for ids, is_masked in zip(ids_list, mask_array):
            # x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
            
            if is_masked:
                word_index = self.tokenizer.dataset.word_to_index[new_words[-1]]
                x = torch.tensor([[word_index]]).to(self.device)
            else:
                x = torch.tensor([[ids]]).to(self.device)
            y_pred, (state_h, state_c) = self.one_step(x, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            new_words.append(self.tokenizer.dataset.index_to_word[word_index])
        
        return new_words
    
    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.device),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size).to(self.device),
        )