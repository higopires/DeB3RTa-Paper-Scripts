import os
import random
from typing import Optional
import argparse
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import optuna
from optuna.samplers import BruteForceSampler
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import math

os.environ["WANDB_DISABLED"] = "true"

# Constants
NUM_EPOCHS = 4
WARMUP_RATIO = 0.1
MAX_LENGTH = 128

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

def calculate_training_steps(batch_size, dataset_size, num_epochs):
    """Calculate total training steps and warmup steps based on dataset size and batch size."""
    steps_per_epoch = math.ceil(dataset_size / batch_size)
    total_training_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    return total_training_steps, warmup_steps

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

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

def model_init(model_name, num_labels):
    config = AutoConfig.from_pretrained(model_name)
    config.update({'num_labels': num_labels})
    return AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model Name", required=True)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv("train.csv")
    eval_df = pd.read_csv("validation.csv")
    test_df = pd.read_csv("test.csv")
    
    model_name = args.model
    print(f"Using model: {model_name}")
    
    # Prepare tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Tokenizing data...")
    train_inputs = tokenizer(train_df['text'].tolist(), max_length=MAX_LENGTH, padding=True, truncation=True)
    eval_inputs = tokenizer(eval_df['text'].tolist(), max_length=MAX_LENGTH, padding=True, truncation=True)
    test_inputs = tokenizer(test_df['text'].tolist(), max_length=MAX_LENGTH, padding=True, truncation=True)
    
    train_dataset = CustomDataset(train_inputs, train_df['label'].tolist())
    eval_dataset = CustomDataset(eval_inputs, eval_df['label'].tolist())
    test_dataset = CustomDataset(test_inputs, test_df['label'].tolist())
    
    def objective(trial):
        batch_size = trial.suggest_categorical("per_device_train_batch_size", [32, 64])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5])
        swa_lr = trial.suggest_categorical("swa_lr", [1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        
        # Initialize models and optimizer
        base_model = model_init(model_name, train_df['label'].nunique()).to(device)
        swa_model = AveragedModel(base_model).to(device)
        optimizer = AdamW(base_model.parameters(), lr=learning_rate)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        
        # Calculate steps
        num_training_steps, num_warmup_steps = calculate_training_steps(
            batch_size=batch_size,
            dataset_size=len(train_dataset),
            num_epochs=NUM_EPOCHS
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
        
        # Training loop with SWA
        global_step = 0
        swa_start_step = num_training_steps // 2
        
        for epoch in range(NUM_EPOCHS):
            base_model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = base_model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                if global_step >= swa_start_step:
                    swa_model.update_parameters(base_model)
                    swa_scheduler.step()
        
        update_bn(train_loader, swa_model)
        
        # Evaluation
        swa_model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = swa_model(**batch)
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(batch['labels'].cpu().numpy())
        
        metrics = compute_metrics((np.vstack(all_logits), np.concatenate(all_labels)))
        return metrics["f1"]
    
    # Hyperparameter optimization
    study_name = f"{model_name.replace('/', '_')}_study_swa"
    study = optuna.create_study(
        direction="maximize",
        sampler=BruteForceSampler(),
        study_name=study_name
    )
    
    print("\nStarting hyperparameter optimization...")
    study.optimize(objective, n_trials=50)
    
    print("\nBest hyperparameters found:")
    for param, value in study.best_trial.params.items():
        print(f"{param}: {value}")
    print(f"Best validation F1: {study.best_trial.value:.4f}")

    print("\nTraining final model with best hyperparameters on training data...")
    # Initialize final models with best hyperparameters
    base_model = model_init(model_name, train_df['label'].nunique()).to(device)
    swa_model = AveragedModel(base_model).to(device)
    optimizer = AdamW(base_model.parameters(), lr=study.best_trial.params["learning_rate"])
    swa_scheduler = SWALR(optimizer, swa_lr=study.best_trial.params["swa_lr"])
    
    # Setup data loaders for final training
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=study.best_trial.params["per_device_train_batch_size"], 
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=study.best_trial.params["per_device_train_batch_size"]
    )
    
    # Calculate steps for final training
    num_training_steps, _ = calculate_training_steps(
        batch_size=study.best_trial.params["per_device_train_batch_size"],
        dataset_size=len(train_dataset),
        num_epochs=NUM_EPOCHS
    )
    swa_start_step = num_training_steps // 2
    
    # Final training loop
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        base_model.train()
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = base_model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            if global_step >= swa_start_step:
                swa_model.update_parameters(base_model)
                swa_scheduler.step()
    
    update_bn(train_loader, swa_model)
    
    # Final evaluation on test set
    print("\nEvaluating final model on test set...")
    swa_model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = swa_model(**batch)
            all_logits.append(outputs.logits.cpu().numpy())
            all_labels.append(batch['labels'].cpu().numpy())
    
    final_metrics = compute_metrics((np.vstack(all_logits), np.concatenate(all_labels)))
    
    print("\nFinal Test Set Metrics:")
    print(f"F1: {final_metrics['f1']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"PR-AUC: {final_metrics['pr_auc']:.4f}")
    
    # Save results
    result_file = f"results_swa_{model_name.replace('/', '_')}.txt"
    with open(result_file, "w") as f:
        f.write(f"Model: {model_name}\n\n")
        f.write("Best hyperparameters:\n")
        for param, value in study.best_trial.params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nValidation F1: {study.best_trial.value:.4f}\n")
        f.write("\nTest Set Metrics:\n")
        for metric, value in final_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"\nResults saved to {result_file}")

if __name__ == "__main__":
    main()