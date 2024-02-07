import pandas as pd
from collections import Counter
import torch

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
            # preprocess
            sentences = [sentence.lower() for sentence in sentences]
            
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