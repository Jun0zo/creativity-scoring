import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the movie review dataset from the datasets library
dataset = load_dataset("imdb")

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Set padding token (in this case, using EOS token)
tokenizer.pad_token = tokenizer.eos_token

# Define the training and evaluation data
train_data = dataset['train']['text']
train_labels = dataset['train']['label']

eval_data = dataset['test']['text']
eval_labels = dataset['test']['label']

# Tokenize the data
train_encodings = tokenizer(train_data, truncation=True, padding=True, return_tensors='pt')
eval_encodings = tokenizer(eval_data, truncation=True, padding=True, return_tensors='pt')

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
eval_labels = torch.tensor(eval_labels)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir='./results',
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=True,
    overwrite_output_dir=True,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=(train_encodings, train_labels),
    eval_dataset=(eval_encodings, eval_labels),
)

# Fine-tune the model
trainer.train()

# Evaluate the fine-tuned model
results = trainer.evaluate()

print(results)
