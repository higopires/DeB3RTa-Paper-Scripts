import os
import random
from typing import Optional
import math

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd.function import InplaceFunction
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    logging,
    TrainingArguments,
    Trainer,
)
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
parser.add_argument("-m", "--model", type=str, help="Model Name")
args = parser.parse_args()

_model_type = "deberta"

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

class Mixout(InplaceFunction):
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "A mix probability of mixout has to be between 0 and 1,"
                " but got {}".format(p)
            )
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = (
                (1 - ctx.noise) * target + ctx.noise * output - ctx.p * target
            ) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None

def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)

class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(
            input, mixout(self.weight, self.target, self.p, self.training), self.bias
        )

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out",
            self.p,
            self.in_features,
            self.out_features,
            self.bias is not None,
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

def model_init():
    config = AutoConfig.from_pretrained(model_name)
    config.update({"num_labels": train_df["label"].nunique()})
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df['label'].nunique())
    return model

def compute_objective(trial):
    config = AutoConfig.from_pretrained(model_name)
    config.update({"num_labels": train_df["label"].nunique()})
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_df['label'].nunique())

    _mixout = trial.suggest_categorical(
            "_mixout", [0.1, 0.3, 0.5, 0.7, 0.9]
        )

    if _mixout > 0:
        print("Initializing Mixout Regularization")
        for sup_module in model.modules():
            for name, module in sup_module.named_children():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(
                        module.in_features,
                        module.out_features,
                        bias,
                        target_state_dict["weight"],
                        _mixout,
                    )
                    new_module.load_state_dict(target_state_dict)
                    setattr(sup_module, name, new_module)
        print("Done.!")

    training_args = TrainingArguments(
        output_dir=f'./results_{opt_model_name}_mixout',
        save_total_limit=1,
        report_to=None,
        learning_rate=trial.suggest_categorical(
            "learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
        ),
        num_train_epochs=4,
        per_device_train_batch_size=trial.suggest_categorical('per_device_train_batch_size', [32, 64]),
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
    sampler=BruteForceSampler(),
    load_if_exists=True,
    study_name=f"{opt_model_name}-study-mixout",
    storage=f"sqlite:///{opt_model_name}-study-mixout.db",
)
study.optimize(compute_objective, n_trials=50)

best = study.best_params

model = model_init()

_mixout = best['_mixout']

if _mixout > 0:
    print("Initializing Mixout Regularization")
    for sup_module in model.modules():
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.Linear):
                target_state_dict = module.state_dict()
                bias = True if module.bias is not None else False
                new_module = MixLinear(
                    module.in_features,
                    module.out_features,
                    bias,
                    target_state_dict["weight"],
                    _mixout,
                )
                new_module.load_state_dict(target_state_dict)
                setattr(sup_module, name, new_module)
    print("Done.!")

training_args = TrainingArguments(
    output_dir=f'./results_{opt_model_name}_mixout',
    save_total_limit=1,
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

# Evaluate the model with all metrics
eval_results = trainer.evaluate()

# Print all metrics
print(f"F1-Score: {round(eval_results['eval_f1'], 4)}")
print(f"Precision: {round(eval_results['eval_precision'], 4)}")
print(f"Recall: {round(eval_results['eval_recall'], 4)}")
print(f"PR-AUC: {round(eval_results['eval_pr_auc'], 4)}")

# Save all metrics to results file
with open("results_seq_classification_mixout.txt", "a") as arquivo:
    arquivo.write(f"eval mixout {opt_model_name}:\n")
    arquivo.write(f"F1 = {round(eval_results['eval_f1'], 4)}\n")
    arquivo.write(f"Precision = {round(eval_results['eval_precision'], 4)}\n")
    arquivo.write(f"Recall = {round(eval_results['eval_recall'], 4)}\n")
    arquivo.write(f"PR-AUC = {round(eval_results['eval_pr_auc'], 4)}\n\n")

# Save the model
trainer.save_model(output_dir=f'./results_{opt_model_name}_mixout')