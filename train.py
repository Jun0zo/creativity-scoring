from tqdm import tqdm
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Load and preprocess the data
df = pd.read_csv("p1_essay.csv")
labels = df["score"].map(lambda x :int(round(x) - 1))
texts = df["essay"]

# Initialize the BERT tokenizer and tokenize the data
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
encoded_texts = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt")

# Create the input dataset
input_dataset = TensorDataset(
    encoded_texts["input_ids"],
    encoded_texts["attention_mask"],
    torch.tensor(labels.tolist())
)

# Split the data into train, validation, and test sets
train_size = int(0.8 * len(input_dataset))
val_size = (len(input_dataset) - train_size) // 2
test_size = len(input_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(input_dataset, [train_size, val_size, test_size])

# Create data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Initialize the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=5)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tuning the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device :", device)
model.to(device)

epochs = 100
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()

        # print('ids shape ', input_ids.shape)
        # print('labels shape ', labels.shape)
        # print(labels[:])
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        # outputs = model(input_ids, attention_mask=attention_mask)
        # print(outputs.loss)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

# Evaluation on the validation set
model.eval()
total_val_loss = 0
correct_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_val_loss += loss.item()

        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()

val_accuracy = correct_predictions / len(val_dataset)
avg_val_loss = total_val_loss / len(val_loader)
print(f"Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Evaluation on the test set
model.eval()
correct_predictions = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        _, predicted_labels = torch.max(logits, dim=1)
        correct_predictions += torch.sum(predicted_labels == labels).item()

test_accuracy = correct_predictions / len(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save_pretrained("pretrained")