
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchvision.datasets as dsets
import pandas

class Vocab:
    def __init__(self, data_path):
        self.tokenizer = get_tokenizer('basic_english')
        
        datalist = self._get_datalist(data_path)
        
        self.vocab = build_vocab_from_iterator(
                                      self._yield_tokens(datalist),
                                      specials=['<UNK>'],
                                      min_freq=2
                                    )
        self.vocab.set_default_index(self.vocab['<UNK>'])
        
    def split_test_train(self, test_ratio=0.2):
        pass
    
        
    def get_len(self):
        return len(self.vocab)
        
    def _yield_tokens(self, sentences):
        for text in sentences:
            yield self.tokenizer(text)
        
    def text2vec(self, text):
        print("tokenized vec", self.tokenizer(text))
        return [self.vocab[token] for token in self.tokenizer(text)]
            
    def _get_datalist(self, data_path):
        df = pandas.read_csv(data_path)
        return df['essay'].tolist()