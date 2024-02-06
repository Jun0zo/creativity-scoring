from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 사전 훈련된 모델과 토크나이저 로드
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 분석할 텍스트
text = ["I love machine learning", "I want to love machine learning"]

# 텍스트를 토크나이즈하여 PyTorch 텐서로 변환
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# inputs to text
decoded_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)

print(inputs)
print('='*50)
print(inputs['input_ids'])
print(pad_packed_sequence(inputs['input_ids']))
print('='*50)
print(decoded_text)

# 모델 예측
with torch.no_grad():
    outputs = model(**inputs)

# 출력에서 softmax를 사용하여 확률로 변환
probs = softmax(outputs.logits, dim=-1)

# 결과 출력
print(probs)
