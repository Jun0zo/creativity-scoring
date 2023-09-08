import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


# Load the custom dataset
custom_dataset = load_dataset('p1_essay.csv')

# Access the dataset splits (e.g., train, validation, test)
train_data = custom_dataset['train']
validation_data = custom_dataset['validation']
test_data = custom_dataset['test']

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

sep = int(len(labels) * 0.8)
train_texts = inputs['input_ids'][:sep]
train_labels = labels[:sep]
test_texts = inputs['input_ids'][sep:]
test_labels = labels[sep:]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# Step 6: Evaluate Model
results = trainer.evaluate()
print(results)
