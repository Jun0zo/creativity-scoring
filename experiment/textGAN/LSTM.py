import torch
import torch.nn as nn

EOS_token_index = 0
MASK_token_index = 1

# 인코더 정의 (BiLSTM)
class EncoderBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, (hidden, cell) = self.bilstm(embedded, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_size), 
                torch.zeros(2, batch_size, self.hidden_size)) # BiLSTM has hidden state and cell state

# 디코더 정의 (BiLSTM)
class DecoderBiLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, max_length):
        super(DecoderBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length  # 최대 생성할 텍스트 길이
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

    def generate(self, sequence, hidden):
        """
        sequence: 인코더로부터의 출력 시퀀스, 마스크된 위치는 특별한 마스크 토큰으로 표시됨
        hidden: 인코더로부터의 마지막 은닉 상태
        """
        inputs = sequence
        outputs = sequence.clone()
        
        # 마스크 토큰의 인덱스를 찾음
        mask_indices = (sequence == MASK_token_index).nonzero(as_tuple=False)
        
        for idx in mask_indices:
            mask_pos = idx.item()
            input_token = sequence[mask_pos].view(1, -1)
            output, hidden = self.forward(input_token, hidden)
            topv, topi = output.topk(1)
            outputs[mask_pos] = topi.item()

        return outputs