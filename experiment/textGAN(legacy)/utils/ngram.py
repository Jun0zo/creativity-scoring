import torch
from torch.utils.data import Dataset, DataLoader

def get_quadgrams(sentences):
    quadgrams = []
    for sentence in sentences:
        words = sentence.split()
        # 문장을 4-grams으로 변환
        quadgrams.extend([words[i:i+4] for i in range(len(words)-3)])
    return quadgrams

# 문장을 인덱스 시퀀스로 변환
def sentence_to_indices(sentence, quadgram_to_index):
    words = sentence.split()
    quadgrams = [words[i:i+4] for i in range(len(words)-3)]
    indices = [quadgram_to_index[tuple(q)] for q in quadgrams]
    return indices

class QuadgramDataset(Dataset):
    def __init__(self, sentences, quadgram_to_index):
        self.sentences = sentences
        self.quadgram_to_index = quadgram_to_index
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        indices = sentence_to_indices(sentence, self.quadgram_to_index)
        return torch.tensor(indices, dtype=torch.long)