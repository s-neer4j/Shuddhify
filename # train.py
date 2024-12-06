import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load a pretrained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. Load the offensive language dataset
dataset = load_dataset("tweet_eval", "offensive")  # Using Hugging Face's `datasets` library

# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Subset for training
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))

# train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Subset for training
# eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(200))


# 4. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",    
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=500,
    logging_dir="./logs",
)

# 5. Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# 6. Save the model and tokenizer
model.save_pretrained("models/offensive_model")
tokenizer.save_pretrained("models")
