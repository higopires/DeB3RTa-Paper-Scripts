import os
import random
from typing import Optional

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
                          TrainingArguments, Trainer)
import optuna
from optuna.samplers import BruteForceSampler
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

os.environ["WANDB_DISABLED"] = "true"


def set_seed(seed: Optional[int] = None):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
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

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="Model Name")
args = parser.parse_args()

_model_type = "deberta"

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
eval_df = pd.read_csv("validation.csv")

model_name = args.model
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df['label'].nunique())
    config = AutoConfig.from_pretrained(model_name)
    return model


def compute_objective(trial):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df['label'].nunique())
    config = AutoConfig.from_pretrained(model_name)
    
    LAYERS_TO_INITIALIZE = trial.suggest_categorical(
            "LAYERS_TO_INITIALIZE", [1, 2, 3]
        )

    if LAYERS_TO_INITIALIZE > 0:
        print(f"Reinitializing the layer {13 - LAYERS_TO_INITIALIZE} ...")
        encoder_temp = getattr(model, _model_type)
        layer = encoder_temp.encoder.layer[-LAYERS_TO_INITIALIZE]
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        print("Layer reinitialization done.")
    
    training_args = TrainingArguments(
        output_dir=f'./results_layer_reinit_{opt_model_name}',
        save_total_limit=1,
        report_to=None,
        learning_rate=trial.suggest_categorical(
            "learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        ),
        num_train_epochs=4,
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [16, 32]),
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results["eval_f1"]


opt_model_name = model_name.replace("/", "_")
opt_model_name = opt_model_name.replace(".", "")

study = optuna.create_study(
    direction="maximize",
    sampler=BruteForceSampler(), load_if_exists=True,
    study_name=f'{opt_model_name}-study-layer_reinit',
    storage=f'sqlite:///{opt_model_name}-study-layer_reinit.db'
)
study.optimize(compute_objective, n_trials=30)

best = study.best_params

model = model_init()
config = AutoConfig.from_pretrained(model_name)

LAYERS_TO_INITIALIZE = best['LAYERS_TO_INITIALIZE']

if LAYERS_TO_INITIALIZE > 0:
    print(f"Reinitializing the layer {13 - LAYERS_TO_INITIALIZE} ...")
    encoder_temp = getattr(model, _model_type)
    layer = encoder_temp.encoder.layer[-LAYERS_TO_INITIALIZE]
    for module in layer.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    print("Layer reinitialization done.")

training_args = TrainingArguments(
    output_dir=f'./results_layer_reinit_{opt_model_name}',
    report_to=None,
    num_train_epochs=4,
    learning_rate=best['learning_rate'],
    warmup_ratio=0.1,
    per_device_train_batch_size=best['per_device_train_batch_size']
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()

print(f"F1-Score: {round(eval_results['eval_f1'], 4)}")
print(f"Precision: {round(eval_results['eval_precision'], 4)}")
print(f"Recall: {round(eval_results['eval_recall'], 4)}")
print(f"PR-AUC: {round(eval_results['eval_pr_auc'], 4)}")

# Save all metrics to results file
with open("results_seq_classification_layer_reinit.txt", "a") as arquivo:
    arquivo.write(f"eval layer reinit {opt_model_name}:\n")
    arquivo.write(f"F1 = {round(eval_results['eval_f1'], 4)}\n")
    arquivo.write(f"Precision = {round(eval_results['eval_precision'], 4)}\n")
    arquivo.write(f"Recall = {round(eval_results['eval_recall'], 4)}\n")
    arquivo.write(f"PR-AUC = {round(eval_results['eval_pr_auc'], 4)}\n\n")

trainer.save_model(output_dir=f'./results_layer_reinit_{opt_model_name}')