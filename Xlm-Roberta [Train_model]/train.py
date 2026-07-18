import json
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##########################################################
# Configuration
##########################################################

MODEL_NAME = "xlm-roberta-base"

TRAIN_FILE = "data/train_subtask1.json"
DEV_FILE = "data/dev_subtask1.json"

SAVE_MODEL_PATH = "./saved_model"

MAX_LENGTH = 256

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 2e-5

##########################################################
# Label Mapping
##########################################################

label2id = {
    "SUPPORTS":0,
    "REFUTES":1,
    "NOT ENOUGH INFO":2
}

id2label = {
    0:"SUPPORTS",
    1:"REFUTES",
    2:"NOT ENOUGH INFO"
}

##########################################################
# Load JSON
##########################################################

def load_json(path):

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data:

        rows.append({
            "text": item["Text"],
            "evidence": item["Evidence"],
            "label": label2id[item["Label"]]
        })

    return pd.DataFrame(rows)

train_df = load_json(TRAIN_FILE)
dev_df = load_json(DEV_FILE)

print(train_df.head())

##########################################################
# HuggingFace Dataset
##########################################################

train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)

##########################################################
# Tokenizer
##########################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):

    return tokenizer(
        batch["evidence"],
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)

columns = [
    "input_ids",
    "attention_mask",
    "label"
]

train_dataset.set_format(type="torch", columns=columns)
dev_dataset.set_format(type="torch", columns=columns)

##########################################################
# Model
##########################################################

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

##########################################################
# Metrics
##########################################################

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted"
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

##########################################################
# Training Arguments
##########################################################

training_args = TrainingArguments(

    output_dir="./results",

    evaluation_strategy="epoch",

    save_strategy="epoch",

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=False
)

##########################################################
# Trainer
##########################################################

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=dev_dataset,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics
)

##########################################################
# Train
##########################################################

trainer.train()

##########################################################
# Save Model
##########################################################

trainer.save_model(SAVE_MODEL_PATH)

tokenizer.save_pretrained(SAVE_MODEL_PATH)

print("Training Completed.")