"""
MicroGPT Training Script

This script trains a minimal GPT-style language model on a small corpus.
It demonstrates the complete training pipeline: data loading, tokenization,
batching, training loop, and text generation.

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
from micro_gpt import MicroGPT, save_checkpoint


# =============================================================================
# Device Configuration
# =============================================================================

# Print device information (CUDA, MPS, or CPU availability)
print_device_info()

# Get the best available compute device (CUDA > MPS > CPU)
device = get_device()


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

# Load OpenWebText dataset from HuggingFace
print("Loading OpenWebText dataset...")
dataset = load_dataset("openwebtext", split="train", streaming=True)

# Load configuration to get max examples
config = load_config("config.json")

# Load GPT-2 tokenizer (same as fine-tuning)
print("Loading GPT-2 tokenizer...")
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab  # 50257
print(f"Vocabulary size: {vocab_size}")

# Take a subset of examples and tokenize
print("Processing dataset...")
all_tokens = []
total_tokens = 0
for i, example in enumerate(dataset):
    tokens = tokenizer.encode(example["text"])
    all_tokens.extend(tokens)
    total_tokens += len(tokens)
    if i >= config.pretrain_max_examples - 1:  # Use config value
        break
    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1:,} examples, {total_tokens:,} tokens...")

print(f"Total tokens in dataset: {total_tokens:,}")

# Convert to tensor and move to device
data = torch.tensor(all_tokens, dtype=torch.long).to(device)
print(f"Data tensor shape: {data.shape}")


# =============================================================================
# Batch Generation Function
# =============================================================================


def get_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random batch of training examples.

    Samples random starting positions in the data and creates input-target
    pairs where the target is the input shifted by one position.

    Returns:
        Tuple of (inputs, targets), each of shape (batch_size, block_size).
    """
    # Generate random starting indices for batch_size number of sequences
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # Stack input sequences: data[i:i+block_size] for each starting index
    x = torch.stack([data[i : i + config.block_size] for i in ix])

    # Stack target sequences: data[i+1:i+block_size+1] (shifted by 1)
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])

    # Return input and target tensors
    return x, y


# =============================================================================
# Model Initialization
# =============================================================================

# Create MicroGPT model with config hyperparameters and move to device
model = MicroGPT(
    vocab_size=vocab_size,
    embedding_dim=config.embedding_dim,
    block_size=config.block_size,
    n_heads=config.n_heads,
    n_layers=config.n_layers,
    dropout=config.dropout,
).to(device)

# Create AdamW optimizer with weight decay for regularization
optimizer = torch.optim.AdamW(
    # Pass model parameters to optimizer
    model.parameters(),
    # Set initial learning rate (will be overridden by schedule)
    lr=config.lr,
    # Set weight decay for L2 regularization
    weight_decay=config.weight_decay,
    # Set betas for momentum (standard GPT-2 values)
    betas=(0.9, 0.95),
)


# =============================================================================
# Learning Rate Schedule
# =============================================================================


def get_lr(step: int) -> float:
    """
    Compute learning rate for current step using warmup + cosine decay.

    The learning rate linearly increases from 0 to peak during warmup,
    then decays following a cosine curve to min_lr.

    Args:
        step: Current training step number.

    Returns:
        Learning rate value for this step.
    """
    # Check if we're in the warmup phase
    if step < config.warmup_steps:
        # Linear warmup: scale from 0 to lr over warmup_steps
        return config.lr * (step + 1) / config.warmup_steps

    # Check if we've exceeded total training epochs
    if step > config.epochs:
        # Return minimum learning rate after training
        return config.min_lr

    # Calculate decay ratio (0 to 1) for cosine schedule
    decay_ratio = (step - config.warmup_steps) / (config.epochs - config.warmup_steps)

    # Ensure decay ratio stays in valid range [0, 1]
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)

    # Compute cosine decay coefficient (1 at start, 0 at end)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # Return interpolated learning rate between min_lr and lr
    return config.min_lr + coeff * (config.lr - config.min_lr)


# =============================================================================
# Training Loop
# =============================================================================

# Create checkpoints directory if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

# Track best validation loss for checkpoint saving
best_val_loss = float("inf")

# Iterate through training epochs
for step in range(config.epochs):
    # Get current learning rate from schedule
    lr = get_lr(step)

    # Update optimizer learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        # Set learning rate for this parameter group
        param_group["lr"] = lr

    # Get a batch of training data
    xb, yb = get_batch()

    # Forward pass: compute logits and loss
    logits, loss = model(xb, yb)

    # Reset gradients from previous step
    optimizer.zero_grad()

    # Backward pass: compute gradients
    loss.backward()

    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    # Update model parameters using computed gradients
    optimizer.step()

    # Check if current step is a multiple of 300 for progress logging
    if step % 300 == 0:
        # Evaluate on validation batch
        model.eval()
        with torch.no_grad():
            xb_val, yb_val = get_batch()
            _, val_loss = model(xb_val, yb_val)
        model.train()

        # Check if this is the best validation loss so far
        is_best = val_loss.item() < best_val_loss
        if is_best:
            best_val_loss = val_loss.item()
            checkpoint_path = "checkpoints/best_val.pt"
            save_checkpoint(model, optimizer, step, val_loss.item(), checkpoint_path)
            best_marker = " ← BEST"
        else:
            best_marker = ""

        # Print training and validation progress
        print(
            f"Step {step:>5} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f} | LR: {lr:.2e}{best_marker}"
        )


# =============================================================================
# Text Generation
# =============================================================================

# Test generation with a sample prompt
print("\nTesting generation...")
test_text = "The quick brown"
context_tokens = tokenizer.encode(test_text)
context = torch.tensor([context_tokens], dtype=torch.long).to(device)

# Generate new tokens autoregressively
out = model.generate(context, max_new_tokens=50, temperature=1.0)

# Print the generated text by decoding token indices
print("\nGenerated text:\n")
print(tokenizer.decode(out[0].tolist()))
