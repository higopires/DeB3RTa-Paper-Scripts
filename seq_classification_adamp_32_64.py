import os
import random
from typing import Optional

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, get_cosine_schedule_with_warmup,
                          TrainingArguments, Trainer)
import optuna
from optuna.samplers import BruteForceSampler
from adamp import AdamP
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import math

os.environ["WANDB_DISABLED"] = "true"


def set_seed(seed: Optional[int] = None):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-m", "--model", type=str, help="Model Name")

# Read arguments from command line
args = parser.parse_args()

_model_type = "deberta"
adam_epsilon = 1e-6
NUM_EPOCHS = 4
WARMUP_RATIO = 0.1  # 10% of total steps will be warmup steps

# Step 1: Load and preprocess the dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
eval_df = pd.read_csv("validation.csv")

# Step 2: Load the pre-trained model and tokenizer
model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Tokenize the input data
train_labels = train_df["label"].tolist()
train_texts = train_df["text"].tolist()

eval_labels = eval_df["label"].tolist()
eval_texts = eval_df["text"].tolist()

test_labels = test_df["label"].tolist()
test_texts = test_df["text"].tolist()

train_inputs = tokenizer(train_texts, max_length=128, padding=True, truncation=True)
eval_inputs = tokenizer(eval_texts, max_length=128, padding=True, truncation=True)
test_inputs = tokenizer(test_texts, max_length=128, padding=True, truncation=True)


class customDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = customDataset(train_inputs, train_labels)
eval_dataset = customDataset(eval_inputs, eval_labels)
test_dataset = customDataset(test_inputs, test_labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    f1 = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions, average="macro")
    recall = recall_score(labels, predictions, average="macro")
    pr_auc = average_precision_score(
        np.eye(len(set(labels)))[labels], logits, average="macro"
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pr_auc": pr_auc,
    }

def model_init():
    config = AutoConfig.from_pretrained(model_name)
    config.update({'num_labels':train_df["label"].nunique()})
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df['label'].nunique())
    
    return model

def calculate_training_steps(batch_size, dataset_size, num_epochs):
    """Calculate total training steps and warmup steps based on dataset size and batch size."""
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_training_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    return total_training_steps, warmup_steps

def compute_objective(trial):
    model = model_init()
    lr = trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [32, 64])
    
    # Calculate steps based on dataset size and batch size
    num_training_steps, num_warmup_steps = calculate_training_steps(
        batch_size=batch_size,
        dataset_size=len(train_dataset),
        num_epochs=NUM_EPOCHS
    )
    
    optimizer = AdamP(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    training_args = TrainingArguments(
        output_dir=f'./results_{opt_model_name}_adamp',
        save_total_limit=1,
        report_to=None,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    eval_results = trainer.evaluate()
    
    return eval_results["eval_f1"]

opt_model_name = model_name.replace("/", "_")
opt_model_name = opt_model_name.replace(".", "")

study = optuna.create_study(
    direction="maximize",
    sampler=BruteForceSampler(),
    load_if_exists=True,
    study_name=f'{opt_model_name}-study-adamp',
    storage=f'sqlite:///{opt_model_name}-study-adamp.db'
)
study.optimize(compute_objective, n_trials=10)

best = study.best_params

model = model_init()
optimizer = AdamP(model.parameters(), lr=best['learning_rate'])

# Calculate steps for final training
num_training_steps, num_warmup_steps = calculate_training_steps(
    batch_size=best['per_device_train_batch_size'],
    dataset_size=len(train_dataset),
    num_epochs=NUM_EPOCHS
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

training_args = TrainingArguments(
    output_dir=f'./results_{opt_model_name}_adamp',
    save_total_limit=1,
    report_to=None,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=best['per_device_train_batch_size'],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
)

trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print all metrics
print(f"F1: {round(eval_results['eval_f1'], 4)}")
print(f"Precision: {round(eval_results['eval_precision'], 4)}")
print(f"Recall: {round(eval_results['eval_recall'], 4)}")
print(f"PR-AUC: {round(eval_results['eval_pr_auc'], 4)}")

# Save all metrics to results file
with open("results_seq_classification_adamp.txt", "a") as arquivo:
    arquivo.write(f"eval adamp {opt_model_name}:\n")
    arquivo.write(f"F1 = {round(eval_results['eval_f1'], 4)}\n")
    arquivo.write(f"Precision = {round(eval_results['eval_precision'], 4)}\n")
    arquivo.write(f"Recall = {round(eval_results['eval_recall'], 4)}\n")
    arquivo.write(f"PR-AUC = {round(eval_results['eval_pr_auc'], 4)}\n\n")

trainer.save_model(output_dir=f'./results_{opt_model_name}_adamp')