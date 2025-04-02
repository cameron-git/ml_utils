# Imports

import torch
from torch.utils.data import DataLoader
import datasets
import transformers  # TODO: remove this dependency
from accelerate import Accelerator
import random
import numpy as np
from tqdm.auto import trange

import ml_utils as mu


# Environment Hyperparameters

project_dir = "."


# Reproducibility

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# Architecture Hyperparameters

vocab_size = 256
hidden_dim = 256
num_layers = 4
num_heads = 8
head_dim = 32
mlp_dim = 1024
seq_len = 128
mask_block_size = 1
metric_names = ["loss", "perplexity"]

# Training Hyperparameters

batch_size = 16
train_steps = 5000
log_interval = 100
val_interval = 100
max_val_steps = 1000
learning_rate = 1e-3
warmup_steps = train_steps // 10
weight_decay = 1e-4
adam_betas = (0.9, 0.95)
mixed_precision = "bf16"
gradient_accumulation_steps = 1  # TODO: Implement gradient accumulation
use_cpu = False


# Data
dataset_name = "karpathy/tiny_shakespeare"
tokenizer_path = "./tokenizers/bytelevel"
dataset = datasets.load_dataset(dataset_name)
tokenizer: transformers.PreTrainedTokenizer = (
    transformers.AutoTokenizer.from_pretrained(tokenizer_path)
)


def tokenize_fn(examples):
    outputs = tokenizer(
        examples["text"],
        max_length=seq_len + mask_block_size,
        stride=mask_block_size,
        truncation=True,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_ids = []
    target_ids = []
    for length, ids in zip(outputs["length"], outputs["input_ids"]):
        if length - mask_block_size == seq_len:
            input_ids.append(ids[:-mask_block_size])
            target_ids.append(ids[mask_block_size:])
    return {"input_ids": input_ids, "target_ids": target_ids}


tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset["train"].column_names,
)
collate_fn = transformers.DefaultDataCollator()
train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    tokenized_dataset["test"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# Model

model = None
model = torch.compile(model)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=adam_betas,
    weight_decay=weight_decay,
    fused=True,
)
lr_scheduler = mu.lr_schedulers.get_lr_scheduler(
    optimizer=optimizer,
    train_steps=train_steps,
    warmup_steps=warmup_steps,
    scheduler_type="cosine",
)

# Train

kwargs = {
    "project_dir": project_dir,
    "vocab_size": vocab_size,
    "hidden_dim": hidden_dim,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "head_dim": head_dim,
    "mlp_dim": mlp_dim,
    "seq_len": seq_len,
    "mask_block_size": mask_block_size,
    "tokenizer_path": tokenizer_path,
    "batch_size": batch_size,
    "train_steps": train_steps,
    "log_interval": log_interval,
    "val_interval": val_interval,
    "max_val_steps": max_val_steps,
    "learning_rate": learning_rate,
    "warmup_steps": warmup_steps,
    "weight_decay": weight_decay,
    "adam_betas": adam_betas,
    "mixed_precision": mixed_precision,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "use_cpu": use_cpu,
    "dataset": dataset_name,
    "metric_names": metric_names,
}

mu.train.train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    lr_scheduler=lr_scheduler,
    **kwargs,
)


# Test

if test_loader is not None:
    mu.eval.test(
        model=model,
        split="test",
        test_loader=test_loader,
        accelerator=mu.train.accelerator,
        progress=None,
        metric_names=metric_names,
        step=train_steps,
        max_test_steps=None,
    )


# Generate

# TODO: Implement generation

# Save model

# TODO: Implement saving

# Wrap up

accelerator.end_training()
