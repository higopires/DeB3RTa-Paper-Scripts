import os
import random
from typing import Optional
import math

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, get_cosine_schedule_with_warmup,
                          TrainingArguments, Trainer)
import optuna
from optuna.samplers import BruteForceSampler
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

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

# Step 1: Load and preprocess the dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
eval_df = pd.read_csv("validation.csv")

def calculate_training_steps(dataset_size: int, batch_size: int, num_epochs: int = 4) -> tuple[int, int]:
    """
    Calculate the number of training steps and warmup steps based on dataset size and batch size
    """
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    num_training_steps = steps_per_epoch * num_epochs
    num_warmup_steps = num_training_steps // 10  # 10% of training steps for warmup
    return num_training_steps, num_warmup_steps

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

def deberta_AdamW_LLRD(model, init_lr, head_lr, weight_decay, layerwise_learning_rate_decay):
    opt_parameters = []
    named_parameters = list(model.named_parameters()) 
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    lr = init_lr

    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) and not any(nd in n for nd in no_decay)]
    opt_parameters.append({"params": params_0, "lr": head_lr, "weight_decay": 0.0})    
    opt_parameters.append({"params": params_1, "lr": head_lr, "weight_decay": weight_decay})    

    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n and not any(nd in n for nd in no_decay)]
        opt_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0})   
        opt_parameters.append({"params": params_1, "lr": lr, "weight_decay": weight_decay})       
        lr *= layerwise_learning_rate_decay

    params_0 = [p for n,p in named_parameters if "embeddings" in n and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n and not any(nd in n for nd in no_decay)]
    opt_parameters.append({"params": params_0, "lr": lr, "weight_decay": 0.0}) 
    opt_parameters.append({"params": params_1, "lr": lr, "weight_decay": weight_decay})        
    
    return AdamW(opt_parameters, lr=init_lr)

def model_init():
    config = AutoConfig.from_pretrained(model_name)
    config.update({'num_labels':train_df["label"].nunique()})
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df['label'].nunique())

def compute_objective(trial):
    llrd_learning_rate = trial.suggest_categorical("llrd_learning_rate", [1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
    layerwise_learning_rate_decay = trial.suggest_categorical("layerwise_learning_rate_decay", [0.9, 0.95])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2, 1e-1])
    head_multiplier = trial.suggest_categorical("head_multiplier", [1.02, 1.03])
    batch_size = trial.suggest_categorical('per_device_train_batch_size', [32, 64])

    # Create a single model instance to be used for both optimizer and trainer
    model = model_init()

    optimizer = deberta_AdamW_LLRD(
        model=model,  # Use the same model instance
        init_lr=llrd_learning_rate,
        head_lr=llrd_learning_rate * head_multiplier,
        weight_decay=weight_decay,
        layerwise_learning_rate_decay=layerwise_learning_rate_decay,
    )

    num_training_steps, num_warmup_steps = calculate_training_steps(
        dataset_size=len(train_dataset),
        batch_size=batch_size
    )
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    training_args = TrainingArguments(
        output_dir=f'./results_llrd_{opt_model_name}',
        save_total_limit=1,
        report_to=None,
        num_train_epochs=4,
        per_device_train_batch_size=batch_size,
    )

    trainer = Trainer(
        model=model,  # Use the same model instance as used for optimizer
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )
    
    try:
        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_f1"]
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        return float('-inf')

opt_model_name = model_name.replace("/", "_").replace(".", "")
study = optuna.create_study(direction="maximize", sampler=BruteForceSampler(), study_name=f'{opt_model_name}-study-llrd', storage=f'sqlite:///{opt_model_name}-study-llrd.db')
study.optimize(compute_objective, n_trials=160)

best = study.best_params
model = model_init()

optimizer = deberta_AdamW_LLRD(
    model=model,  # Use the same model instance
    init_lr=best['llrd_learning_rate'],
    head_lr=best['llrd_learning_rate'] * best['head_multiplier'],
    weight_decay=best['weight_decay'],
    layerwise_learning_rate_decay=best['layerwise_learning_rate_decay']
)

batch_size = best['per_device_train_batch_size']
num_training_steps, num_warmup_steps = calculate_training_steps(
    dataset_size=len(train_dataset),
    batch_size=batch_size
)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
training_args = TrainingArguments(
    output_dir=f'./results_llrd_{opt_model_name}',
    save_total_limit=1,
    report_to=None,
    num_train_epochs=4,
    per_device_train_batch_size=batch_size,
)

trainer = Trainer(
    model=model,  # Use the same model instance as used for optimizer
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
)
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Save all metrics to results file
with open("results_seq_classification_llrd.txt", "a") as arquivo:
    arquivo.write(f"eval llrd {opt_model_name}:\n")
    arquivo.write(f"F1 = {round(eval_results['eval_f1'], 4)}\n")
    arquivo.write(f"Precision = {round(eval_results['eval_precision'], 4)}\n")
    arquivo.write(f"Recall = {round(eval_results['eval_recall'], 4)}\n")
    arquivo.write(f"PR-AUC = {round(eval_results['eval_pr_auc'], 4)}\n\n")

trainer.save_model(output_dir=f'./results_llrd_{opt_model_name}')