import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_dataset, Split
from sklearn.model_selection import train_test_split


# Load the custom dataset
dataset = load_dataset('csv', data_files='p1_essay.csv')
dataset = dataset.remove_columns(column_names=['id'])
dataset = dataset.map(lambda row : {'score': int(round(row['score'])) - 1})

# Split the dataset into train and test sets
split_dataset = dataset['train'].train_test_split(test_size=0.2, shuffle=True)

# Get the train and test sets
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Step 2: Initialize Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 4: Load Pretrained GPT-2 Model for Sequence Classification
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=5)


# Step 5: Fine-tune Model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()