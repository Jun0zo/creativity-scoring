import torch
import pickle
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the pre-trained model and tokenizer
model_name = "pretrained"  # Change this to your pre-trained model
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Sample data: You should replace this with your own dataset
df = pd.read_csv("p1_essay.csv")
labels = df["score"].map(lambda x :int(round(x) - 1))
texts = df["essay"].tolist()

# Tokenize the texts
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Perform inference
with torch.no_grad():
    logits = model(**encoded_inputs).logits
    predictions = torch.argmax(logits, dim=1)

# Convert tensors to numpy arrays for calculating confusion matrix
labels = torch.tensor(labels).numpy()
predictions = predictions.numpy()

with open("labels.pkl", "w") as f:
    pickle.dump(labels, f)

with open("predictions.pkl", "w") as f:
    pickle.dump(predictions, f)
