import torch
import torch.nn as nn

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.lstm_size = 256
        self.embedding_dim = 128
        self.num_layers = 1 # lstm layer
        
        self.embedding = nn.Embedding(10000, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_size, self.num_layers)
        
    def forward(self, x, lengths):
        embeded = self.embedding(x)
        packed_input = nn.utils.rnn.pack_padded_sequence(embeded, lengths.tolist(), batch_first=True)
        input_batch, (hidden, cell) = self.lstm(packed_input)
        
        print(input_batch)
        output, _ = nn.utils.rnn.pad_packed_sequence(input_batch, batch_first=True)
        
        print('output shape 1' , output.shape)
        output = output[1, :, :]
        
        print("===============")
        print('hidden shape :', hidden.shape)
        print('cell shape :', cell.shape)


        print("===============")
        print('output shape 2' , output.shape)
        return output
    
model = TextLSTM()
x = torch.tensor([[1, 2, 3, 4, 5, 6, 7], 
                  [5, 6, 7, 8,  9, 10 ,11], 
                  [9, 10, 11, 12, 13, 14, 15], 
                  [9, 10, 11, 12, 13, 14, 15],
                  [9, 10, 11, 12, 13, 14, 15]])
lengths = torch.tensor([7, 7, 7, 7, 7])
output = model(x, lengths)
