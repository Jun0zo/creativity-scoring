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