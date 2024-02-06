import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
import torch.optim as optim
import torch.nn.functional as F
import random

# LSTM 설정
input_size = 10  # 입력 크기
hidden_size = 5  # 은닉 상태 크기
num_layers = 1  # LSTM 레이어 수



def run_epoch(generator, discriminator, g_optimizer, d_optimizer, d_criterion, dataset):
    # generator.train()
    # discriminator.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    k = 3
    for text_batch in train_data_loader:
        sentence = text_batch[0]
        # print('tb :', text_batch)
        # sentence = sentence.to(device)
        sentence_length = len(sentence.split())
        # Train Discriminator
        d_optimizer.zero_grad()
        
        p_real = discriminator(sentence)
        
        random_indices = random.sample(range(1, sentence_length), k)
        mask_array = [True if i in random_indices else False for i in range(sentence_length)]
        
        p_fake = discriminator(generator(sentence))
        loss_real = -1 * torch.log(p_real)
        loss_fake = -1 * torch.log(1. - p_fake).mean()
        loss_d = (loss_real + loss_fake).mean()
        
        loss_d.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        p_fake = discriminator(generator(sentence, mask_array))
        
        loss_g = -1 * torch.log(p_fake).mean()
        loss_g.backward()
        g_optimizer.step()

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
    def __init__(self, data_path='data/training_set.csv'):
        # self.args = args/
        self.max_sequence_length = -1
        self.data_path = data_path
        self.sentences = self.load_sentences()
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()
        
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        
    def __len__(self):
        # return len(self.words_indexes) - self.sequence_length
        return len(self.sentences)
    
    def __getitem__(self, index):
        # return self.sentences[index:index+self.sequence_length], self.sentences[index+self.sequence_length]
        return self.sentences[index]
    
    def get_vocab_size(self):
        return len(self.uniq_words)
        
    def load_sentences(self):
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            df.dropna(inplace=True)
            sentences = df['essay'].values
            self.max_sequence_length = max([len(sentence.split()) for sentence in sentences])
            return sentences
        else:
            with open(self.data_path, 'r') as f:
                sentences = [line for line in f.readlines()]
                self.max_sequence_length = max([len(sentence.split()) for sentence in sentences])
                return sentences
    
    def load_words(self):
        words = [word for sentence in self.sentences for word in sentence.split()]
        # print(words)
        return words
    
    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
class Tokenizer:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def tokenize_word(self, word):
        return self.dataset.word_to_index.get(word, 0)
        
    def tokenize_sentence(self, sentence):
        seq = [self.dataset.word_to_index[word] for word in sentence.split()]
        
        # add padding
        if len(seq) < self.dataset.max_sequence_length:
            seq = seq + [0] * (self.dataset.max_sequence_length - len(seq))
        return seq
    
class Encoder(nn.Module):
    def __init__(self, tokenizer):
        super(Encoder, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        
        self.tokenizer = tokenizer
        n_vocab = len(tokenizer.dataset.uniq_words)
        
        self.embedding = nn.Embedding(n_vocab, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        
    def one_step(self, x, prev_state):
        embed = self.embedding(x)
        print('embed', embed.shape, prev_state[0].shape, prev_state[1].shape)
        output, (state_h, state_c) = self.lstm(embed, prev_state)
        return output, (state_h, state_c)
    
    def forward(self, sentence, mask_array) :
        words = sentence.split()
        state_h, state_c = self.init_state(1) # (len(words))
        
        ids_list = self.tokenizer.tokenize_sentence(sentence)
        
        for word, is_masked in zip(words, mask_array):
            x = torch.tensor([[self.dataset.word_to_index[word]]])
            print('new', word, x.shape)
            y_pred, (state_h, state_c) = self.one_step(x, (state_h, state_c))
            
        return state_h, state_c
    
    def init_state(self, sequence_length):
        return (
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
            torch.zeros(self.num_layers, sequence_length, self.lstm_size),
        )
    
class Decoder(nn.Module):
    def __init__(self, tokenizer):
        super(Decoder, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 1
        
        self.dataset = dataset
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
        words = sentence.split()
        if state_h is None:
            state_h, state_c = self.init_state(1)
        
        new_words = []
        
        
        for word, is_masked in enumerate(words, mask_array):
            # x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]])
            if is_masked:
                word_index = self.dataset.word_to_index[new_words[-1]]
                x = torch.tensor([[word_index]])
            else:
                word_index = self.dataset.word_to_index[word]
                x = torch.tensor([[word_index]])
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
    

class Generator(nn.Module):
    def __init__(self, tokenizer):
        super(Generator, self).__init__()
        self.encoder = Encoder(tokenizer)
        self.decoder = Decoder(tokenizer) 
        
    def predict(self, text):
        state_h, state_c = self.encoder(text)
        words = self.decoder(text, state_h, state_c)
    
        print(words)
        
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
        self.conv = nn.Conv1d(self.embedding_dim, 128, 5)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        
        self.embedding = nn.Embedding(num_embeddings=n_vocab, embedding_dim=self.embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=self.embedding_dim, out_channels=self.num_filters, kernel_size=self.kernel_size)
        self.lstm = nn.LSTM(input_size=self.num_filters, hidden_size=self.embedding_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(self.embedding_dim, 2)
        
    
    def forward(self, sentence):
        words = sentence.split()
        print('words lenth :', len(words))
        inp = torch.tensor([[self.tokenizer.dataset.word_to_index[word] for word in words]])
        feature = self._get_feature(inp)
        pred = self.feature2out(feature)
        return pred
    
    def _get_feature(self, inp):
        print("inp shape ", inp.shape)
        emb = self.embedding(inp)
        print("emb shape ", emb.shape)
        feature = self.conv1d(emb)
        print("feature shape ", feature.shape)
        feature = self.lstm(feature)
        
        return feature
    
class TextGAN:
    def __init__(self, dataset):
        tokenizer = Tokenizer(dataset)
        self.generator = Generator(tokenizer)
        self.discriminator = Discriminator(tokenizer)
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
     
    run_epoch(textgan.generator, textgan.discriminator, g_optimizer, d_optimizer, d_criterion, dataset)
    # generator.predict('censored censored')
    
    # output_text = predict(dataset, model, text='think think, is this?')
    # print(output_text)