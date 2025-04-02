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
tokenizer_path = "./tokenizers/bytelevel"


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
dataset_name = "karpathy/tiny_shakespeare"


# Data

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


# Accelerate

accelerator = Accelerator(
    mixed_precision=mixed_precision,
    log_with=[mu.trackers.AimTracker, mu.trackers.SimpleGeneralTracker],
    gradient_accumulation_steps=gradient_accumulation_steps,
    cpu=use_cpu,
    # rng_types=None, # TODO: Implement RNG types
    project_dir=project_dir,
)
model, optimizer, train_loader = accelerator.prepare(
    model,
    optimizer,
    train_loader,
)
if lr_scheduler is not None:
    lr_scheduler = accelerator.prepare(lr_scheduler)
if val_loader is not None:
    val_loader = accelerator.prepare(val_loader)
if test_loader is not None:
    test_loader = accelerator.prepare(test_loader)
accelerator.init_trackers(
    "llm",
    {
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
    },
)


# Train

train_metrics = {
    "loss": 0,
    "perplexity": 0,
}
model.train()
step = 0
progress = trange(
    train_steps,
    desc="Training",
    disable=not accelerator.is_local_main_process,
)
while step < train_steps:
    for batch in train_loader:
        # Train step
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs["loss"]
        accelerator.backward(loss)
        optimizer.step()

        # Train metrics
        for k, v in train_metrics.items():
            v += outputs[k].item()  # TODO: Might need to use accelerator.gather

        # Train logging
        if step % log_interval == 0:
            train_metrics = {k: v / log_interval for k, v in train_metrics.items()}
            progress.set_postfix(
                train_loss=train_metrics["loss"],
                train_perplexity=train_metrics["perplexity"],
            )
            accelerator.log(
                {
                    "train_loss": train_metrics["loss"],
                    "train_perplexity": train_metrics["perplexity"],
                },
                step=step,
            )
            train_metrics = {k: 0 for k in train_metrics.keys()}

            # Validation
            if val_loader is not None and step % val_interval == 0:
                model.eval()
                val_metrics = {k: 0 for k in train_metrics.keys()}
                with torch.no_grad():
                    for val_step, batch in enumerate(val_loader):
                        outputs = model(**batch)
                        for k, v in val_metrics.items():
                            v += outputs[k].item()
                        if val_step >= max_val_steps:
                            break

                # Validation logging
                val_metrics = {k: v / val_step for k, v in val_metrics.items()}
                progress.set_postfix(
                    val_loss=val_metrics["loss"],
                    val_perplexity=val_metrics["perplexity"],
                )
                accelerator.log(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_perplexity": val_metrics["perplexity"],
                    },
                    step=step,
                )

                # Generate

                # TODO: Implement generation

                # Checkpoint

                # TODO: Implement checkpointing

                model.train()

        # Step
        step += 1
        progress.update(1)
        if train_steps <= step:
            break

progress.close()


# Test

if test_loader is not None:
    model.eval()
    test_metrics = {k: 0 for k in train_metrics.keys()}
    with torch.no_grad():
        for test_step, batch in enumerate(test_loader):
            outputs = model(**batch)
            for k, v in test_metrics.items():
                v += outputs[k].item()

    # Test logging
    test_metrics = {k: v / test_step for k, v in test_metrics.items()}
    accelerator.log(
        {
            "test_loss": test_metrics["loss"],
            "test_perplexity": test_metrics["perplexity"],
        },
        step=step,
    )


# Generate

# TODO: Implement generation

# Save model

# TODO: Implement saving

# Wrap up

accelerator.end_training()
