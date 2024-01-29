import numpy as np
import torch
import torch.nn as nn
from collections import Counter
import torch.optim as optim
import torch.nn.functional as F

# LSTM 설정
input_size = 10  # 입력 크기
hidden_size = 5  # 은닉 상태 크기
num_layers = 1  # LSTM 레이어 수

def predict(dataset, model, text, next_words=100):
    model.eval()
    
    words = text.split()
    state_h, state_c = model.init_state(len(words))
    
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        print(x)
        print("x shape :",  x.shape)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words

class Dataset(torch.utils.data.Dataset):
    def __init__(self, sequence_length=4, data_path='data/training_set.csv'):
        # self.args = args/
        self.sequence_length = sequence_length
        self.data_path = data_path
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
    
    def __len__(self):
        return len(self.words_indexes) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.sequence_length]),
            torch.tensor(self.words_indexes[index+1:index+self.sequence_length+1]),
        )
    
    def load_words(self):
        with open(self.data_path, 'r') as f:
            words = [word for line in f.readlines() for word in line.split()]
            # print(words)
            return words
    
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
class Encoder(nn.Module):
    def __init__(self, dataset):
        super(Encoder, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        
        self.dataset = dataset
        n_vocab = len(dataset.uniq_words)
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        
    def one_step(self, x, prev_state):
        embed = self.embedding(x)
        print('embed', embed.shape, prev_state[0].shape, prev_state[1].shape)
        output, (state_h, state_c) = self.lstm(embed, prev_state)
        return output, (state_h, state_c)
    
    def forward(self, text):
        words = text.split()
        state_h, state_c = self.init_state(1) # (len(words))
        
        for i in range(0, len(words)):
            # x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
            print(self.dataset.word_to_index[words[i]])
            x = torch.tensor([[self.dataset.word_to_index[words[i]]]])
            print('new', words[i], x.shape)
            y_pred, (state_h, state_c) = self.one_step(x, (state_h, state_c))
            
        return state_h, state_c
    
    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
        )
    
class Decoder(nn.Module):
    def __init__(self, dataset):
        super(Decoder, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        
        self.dataset = dataset
        n_vocab = len(dataset.uniq_words)
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        self.fc = nn.Linear(self.lstm_size, n_vocab)
        
    def one_step(self, x, prev_state):
        embed = self.embedding(x)
        output, (state_h, state_c) = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, (state_h, state_c)
    
    def forward(self, text, state_h=None, state_c=None):
        words = text.split()
        if state_h is None:
            state_h, state_c = self.init_state(1)
        
        new_words = []
        
        
        for i in range(0, len(words)):
            # x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
            x = torch.tensor([[self.dataset.word_to_index[words[i]]]])
            y_pred, (state_h, state_c) = self.one_step(x, (state_h, state_c))
            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            new_words.append(self.dataset.index_to_word[word_index])
        
        return new_words
    
    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
        )
    

class Ganerator:
    def __init__(self, dataset):
        self.encoder = Encoder(dataset)
        self.decoder = Decoder(dataset)
        
    def predict(self, text):
        state_h, state_c = self.encoder(text)
        words = self.decoder(text, state_h, state_c)
    
        print(words)
        
class Discriminator:
    def __init__(self, dataset):
        pass
    
    def forward(self):
        pass
    
class TextGAN:
    def __init__(self, dataset):
        self.generator = Ganerator(dataset)
        self.discriminator = Discriminator(dataset)
        
    def train(self):
        pass
    
    def predict(self):
        pass

if __name__ == '__main__':
    dataset = Dataset()
    
    textgan = TextGAN(dataset)
    
    g_optimizer = optim.Adam(textgan.generator.parameters(), lr=0.001)
    d_optimizer = optim.Adam(textgan.discriminator.parameters(), lr=0.001)
    
    d_criterion = nn.BCEWithLogitsLoss()
    
    textgan
    # generator.predict('censored censored')
    

    # output_text = predict(dataset, model, text='think think, is this?')
    # print(output_text)