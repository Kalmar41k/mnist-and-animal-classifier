"""
Script for training a Named Entity Recognition (NER) model using DistilBERT.

This script performs the following steps:
1. Loads a dataset of sentences with labeled entities.
2. Tokenizes the sentences and aligns labels using a DistilBERT tokenizer.
3. Defines and trains a DistilBERT-based token classification model.
4. Evaluates the trained model on a test dataset and reports classification metrics.
5. Saves the trained model and tokenizer for later use.

The model predicts whether a word belongs to the 'ANIMAL' entity class or not.
"""

import json
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import torch
from sklearn.metrics import classification_report

# Load the dataset from a JSON file
with open("data/ner_dataset.json", "r") as f:
    data = json.load(f)

# Load the tokenizer for DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Define the unique labels and create mappings between labels and IDs
unique_labels = ['O', 'ANIMAL']
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_align_labels(examples):
    """
    Tokenizes input text and aligns labels with tokenized words.
    
    Parameters:
        examples (dict): A dictionary containing sentences and their corresponding labels.
    
    Returns:
        dict: Tokenized inputs with aligned labels.
    """
    tokenized_inputs = tokenizer(examples['sentence'], truncation=True, padding=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = [-100 if word_id is None else label2id[label[word_id]] for word_id in word_ids]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Initialize the model for token classification
model = DistilBertForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)


# Convert the dataset into a Hugging Face Dataset format
train_data = Dataset.from_dict(data)
train_data = train_data.map(tokenize_and_align_labels, batched=True)

# Split the dataset into training and evaluation sets
train_test_split = train_data.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Define training arguments for the Trainer API
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer instance for training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Perform inference on the evaluation dataset
trainer = Trainer(model=model)
predictions = trainer.predict(eval_dataset)

# Extract logits and convert them to predicted labels
pred_logits = predictions.predictions
predicted_labels = np.argmax(pred_logits, axis=-1)

def convert_labels(labels):
    """
    Converts label IDs to label names, ignoring padding tokens.
    
    Parameters:
        labels (list): A list of label IDs.
    
    Returns:
        list: A list of corresponding label names.
    """
    return [id2label[label_id] if label_id != -100 else "O" for label_id in labels]

# Convert true and predicted label IDs to label names
true_labels = [convert_labels(labels) for labels in eval_dataset['labels']]
pred_labels = [convert_labels(pred) for pred in predicted_labels]

# Flatten the label lists for evaluation
true_labels_flat = [label for sublist in true_labels for label in sublist]
pred_labels_flat = [label for sublist in pred_labels for label in sublist]

# Print the classification report
print(classification_report(true_labels_flat, pred_labels_flat))

# Save the trained model and tokenizer for future use
model.save_pretrained("models/ner_model")
tokenizer.save_pretrained("models/ner_model")