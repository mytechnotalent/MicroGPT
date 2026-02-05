"""
GPT-2 Training Script.

This script trains a GPT-2 Medium language model (355M params) on OpenWebText.
It demonstrates the complete training pipeline: data loading, tokenization,
batching, learning rate scheduling, and text generation.

Usage:
    python main.py

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import math
import os
import torch
import tiktoken
from datasets import load_dataset
from config import load_config
from device import get_device, print_device_info
from micro_gpt import GPT2, GPT2Config, save_checkpoint, get_batch, _eval_model

# =============================================================================
# Device Configuration
# =============================================================================

# Print device information (CUDA, MPS, or CPU availability)
print_device_info()

# Get the best available compute device (CUDA > MPS > CPU)
device = get_device()


# =============================================================================
# Configuration Loading
# =============================================================================

# Load configuration from config.json
config = load_config("config.json")
print(f"Configuration loaded: block_size={config.block_size}, n_embd={config.n_embd}")
print(f"Model: n_layer={config.n_layer}, n_head={config.n_head}")


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

# Load OpenWebText dataset from HuggingFace
print("\nLoading OpenWebText dataset...")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# Load GPT-2 tokenizer (BPE with 50257 vocab)
print("Loading GPT-2 tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
print(f"Vocabulary size: {vocab_size}")

# Tokenize dataset examples
print("Processing dataset...")
all_tokens = []
total_tokens = 0
for i, example in enumerate(dataset):
    tokens = tokenizer.encode(example["text"])
    all_tokens.extend(tokens)
    total_tokens += len(tokens)
    if i >= config.pretrain_max_examples - 1:
        break
    if (i + 1) % 50000 == 0:
        print(f"  Processed {i + 1:,} examples, {total_tokens:,} tokens...")

print(f"Total tokens in dataset: {total_tokens:,}")

# Convert to tensor
data = torch.tensor(all_tokens, dtype=torch.long).to(device)
print(f"Data tensor shape: {data.shape}")

# Split into train and validation sets (90/10)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")


# =============================================================================
# Model Initialization
# =============================================================================

# Create GPT-2 configuration
model_config = GPT2Config(
    block_size=config.block_size,
    vocab_size=vocab_size,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    dropout=config.dropout,
    bias=config.bias,
)

# Create model and move to device
model = GPT2(model_config).to(device)

# Print model size
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {n_params:,} (~{n_params * 4 / 1e9:.2f} GB)")

# Create AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95),
)


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def _compute_warmup_lr(step: int) -> float:
    """
    Compute learning rate during warmup phase.

    Args:
        step: Current training step.

    Returns:
        Linearly interpolated learning rate.
    """
    return config.lr * (step + 1) / config.warmup_steps


def _compute_decay_lr(step: int) -> float:
    """
    Compute learning rate during cosine decay phase.

    Args:
        step: Current training step.

    Returns:
        Cosine-decayed learning rate.
    """
    decay_ratio = (step - config.warmup_steps) / (config.epochs - config.warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.lr - config.min_lr)


def get_lr(step: int) -> float:
    """
    Compute learning rate for current step using warmup + cosine decay.

    Args:
        step: Current training step number.

    Returns:
        Learning rate value for this step.
    """
    if step < config.warmup_steps:
        return _compute_warmup_lr(step)
    if step > config.epochs:
        return config.min_lr
    return _compute_decay_lr(step)


# =============================================================================
# Training Loop
# =============================================================================

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# Track best validation loss
best_val_loss = float("inf")

print("\nStarting training...")
print("=" * 70)

for step in range(config.epochs):
    # Update learning rate
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # Get training batch
    xb, yb = get_batch(train_data, config.block_size, config.batch_size, device)

    # Forward pass
    logits, loss = model(xb, yb)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    # Evaluate periodically
    if step % config.eval_interval == 0 or step == config.epochs - 1:
        val_loss = _eval_model(
            model,
            val_data,
            config.block_size,
            config.batch_size,
            device,
            config.eval_iters,
        )
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, step, val_loss, "checkpoints/best_val.pt")
            marker = " ← BEST"
        else:
            marker = ""
        print(
            f"Step {step:>6} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}{marker}"
        )

# Save final checkpoint
save_checkpoint(model, optimizer, step, val_loss, "checkpoints/final.pt")
print("=" * 70)
print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


# =============================================================================
# Text Generation Test
# =============================================================================

print("\nTesting generation with top-p sampling...")
test_text = "The quick brown fox"
context_tokens = tokenizer.encode(test_text)
context = torch.tensor([context_tokens], dtype=torch.long).to(device)

# Generate with top-p sampling
model.eval()
with torch.no_grad():
    output = model.generate(context, max_new_tokens=100, temperature=0.8, top_p=0.9)

print("\nGenerated text:")
print("-" * 70)
print(tokenizer.decode(output[0].tolist()))
print("-" * 70)
