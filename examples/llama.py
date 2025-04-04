# Reference Llama training script
# TODO: Selective Activation Checkpointing https://pytorch.org/blog/activation-checkpointing-techniques/
# TODO: Generation

# Imports

import torch
from torch.utils.data import DataLoader
import datasets
import transformers  # TODO: replace with tokenizers
import accelerate
import random
import numpy as np
import datetime

import ml_utils as mu


# Environment Hyperparameters

project_dir = "./.logs"
run_name = "llama"
wandb_project_name = "ut"


# Reproducibility

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# Architecture Hyperparameters

vocab_size = 256
embed_dim = 256
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
# adam_betas = (0.9, 0.95)
adam_betas = (0.9, 0.999)
mixed_precision = "bf16"
# NOTE: Will need to run more steps to get the same number of updates
gradient_accumulation_steps = 1
max_grad_norm = 1.0
max_grad_value = None
use_cpu = False


# Accelerator

trackers = [
    mu.trackers.AimTracker(run_name=run_name, logging_dir=project_dir),
    mu.trackers.SimpleGeneralTracker(run_name=run_name, logging_dir=project_dir),
    mu.trackers.WandBTracker(project_name=wandb_project_name, run_name=run_name),
]

accelerator = accelerate.Accelerator(
    mixed_precision=mixed_precision,
    log_with=trackers,
    gradient_accumulation_steps=gradient_accumulation_steps,
    cpu=use_cpu,
    # rng_types=None, # TODO: Implement RNG types
    project_dir=project_dir,
)
accelerator.start_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")


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

model = mu.models.Llama(
    vocab_size=vocab_size,
    num_layers=num_layers,
    embed_dim=embed_dim,
    head_dim=head_dim,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    max_seq_len=seq_len,
    mask_block_size=mask_block_size,
)
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


# Kwargs

kwargs = {
    "project_dir": project_dir,
    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
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
    "max_grad_norm": max_grad_norm,
    "max_grad_value": max_grad_value,
    "dataset": dataset_name,
    "metric_names": metric_names,
}
accelerator.init_trackers(
    run_name,
    kwargs,
)


# Train

mu.train.train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    lr_scheduler=lr_scheduler,
    accelerator=accelerator,
    **kwargs,
)


# Test

if test_loader is not None:
    mu.eval.test(
        model=model,
        split="test",
        test_loader=test_loader,
        accelerator=accelerator,
        progress=None,
        metric_names=metric_names,
        step=train_steps,
        max_test_steps=None,
    )


# Generate

# TODO: Implement generation

# Save model

accelerator.save_model(
    model, f"{project_dir}/checkpoints/{accelerator.start_time}/final"
)

# Cleanup

accelerator.end_training()
