from typing import Optional
import random
import numpy as np
import torch
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import optuna
from optuna.samplers import BruteForceSampler
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

import os
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
        os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Model Name")
    args = parser.parse_args()

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    eval_df = pd.read_csv('validation.csv')

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(texts):
        return tokenizer(texts, max_length=128, padding=True, truncation=True)

    train_inputs = tokenize_data(train_df['text'].tolist())
    eval_inputs = tokenize_data(eval_df['text'].tolist())
    test_inputs = tokenize_data(test_df['text'].tolist())

    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = NewsDataset(train_inputs, train_df['label'].tolist())
    eval_dataset = NewsDataset(eval_inputs, eval_df['label'].tolist())
    test_dataset = NewsDataset(test_inputs, test_df['label'].tolist())

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=train_df['label'].nunique()
        )

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

    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]),
            "per_device_train_batch_size": trial.suggest_categorical('per_device_train_batch_size', [16, 32]),
        }

    opt_model_name = model_name.replace("/", "_").replace(".", "")
    training_args = TrainingArguments(
        output_dir=f'./results_{opt_model_name}',
        save_total_limit=1,
        report_to=None,
        num_train_epochs=4,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=10,
        study_name=f'{opt_model_name}-study-seq_classification',
        storage=f'sqlite:///{opt_model_name}-study-seq_classification.db',
        sampler=BruteForceSampler(),
        load_if_exists=True,
        compute_objective=lambda metrics: metrics["eval_f1"]
    )

    print("\nBest trial:")
    print(f"Value: {best_trial.objective:.4f}")
    print("Params: ")
    for key, value in best_trial.hyperparameters.items():
        print(f"    {key}: {value}")

    best = best_trial.hyperparameters
    trainer = Trainer(
        model=None,
        args=TrainingArguments(
            output_dir=f'./results_{opt_model_name}',
            save_total_limit=1,
            report_to=None,
            num_train_epochs=4,
            warmup_ratio=0.1,
            **best
        ),
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    
    print(f"\nResults for {opt_model_name}:")
    print("============================")
    metrics = {
        "F1": eval_results['eval_f1'],
        "Precision": eval_results['eval_precision'],
        "Recall": eval_results['eval_recall'],
        "PR-AUC": eval_results['eval_pr_auc']
    }
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {round(value, 4)}")
    print("============================\n")
    
    # Save all metrics to results file
    with open("results_seq_classification.txt", "a") as arquivo:
        arquivo.write(f"eval {opt_model_name}:\n")
        arquivo.write(f"F1 = {round(eval_results['eval_f1'], 4)}\n")
        arquivo.write(f"Precision = {round(eval_results['eval_precision'], 4)}\n")
        arquivo.write(f"Recall = {round(eval_results['eval_recall'], 4)}\n")
        arquivo.write(f"PR-AUC = {round(eval_results['eval_pr_auc'], 4)}\n\n")

    trainer.save_model(output_dir=f'./results_{opt_model_name}')

if __name__ == "__main__":
    main()