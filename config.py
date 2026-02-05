"""
Configuration Management for GPT-2 Medium Model.

This module provides configuration loading and management for the GPT-2
language model architecture. It loads hyperparameters from a JSON file
and provides a typed dataclass for easy access.

Usage:
    from config import load_config, GPTConfig

    config = load_config("config.json")
    print(config.n_embd)  # 1024

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class GPTConfig:
    """
    Configuration container for GPT-2 Medium model architecture (355M params).

    This dataclass holds all hyperparameters needed to configure the GPT-2
    model architecture and training process. Follows OpenAI GPT-2 naming
    conventions for compatibility.

    Architecture Parameters:
        block_size: Maximum sequence length (context window). GPT-2 uses 1024.
        vocab_size: Vocabulary size matching tiktoken gpt2/r50k_base (50257).
        n_layer: Number of transformer blocks (depth). GPT-2 Medium uses 24.
        n_head: Number of attention heads (parallel focus). GPT-2 Medium uses 16.
        n_embd: Embedding dimension size (width). GPT-2 Medium uses 1024.
        dropout: Dropout probability for regularization.
        bias: Whether to use bias in Linear layers and LayerNorms.

    Training Parameters:
        lr: Peak learning rate for the optimizer.
        min_lr: Minimum learning rate after cosine decay.
        warmup_steps: Number of steps for linear learning rate warmup.
        batch_size: Number of sequences per training batch.
        grad_clip: Maximum gradient norm for gradient clipping.
        weight_decay: L2 regularization weight decay coefficient.
        epochs: Number of training iterations (steps).
        eval_interval: Steps between validation evaluations.
        eval_iters: Number of batches to average for validation loss.

    Fine-tuning Parameters:
        finetune_lr: Learning rate for fine-tuning (lower than pre-training).
        finetune_epochs: Number of fine-tuning iterations.
        finetune_eval_interval: Steps between fine-tuning evaluations.
        finetune_eval_iters: Batches to average for fine-tuning validation.
        finetune_max_tokens: Maximum tokens to load for fine-tuning dataset.
        finetune_temperature: Sampling temperature for generation.
        finetune_top_p: Nucleus sampling probability threshold.
        finetune_max_new_tokens: Maximum tokens to generate per response.

    Data Parameters:
        pretrain_max_examples: Maximum examples to load from pre-training dataset.

    Example:
        >>> config = GPTConfig(
        ...     block_size=1024, vocab_size=50257, n_layer=24,
        ...     n_head=16, n_embd=1024, dropout=0.1, bias=True,
        ...     lr=3e-4, min_lr=3e-5, warmup_steps=2000,
        ...     batch_size=4, grad_clip=1.0, weight_decay=0.1,
        ...     epochs=300000, eval_interval=500, eval_iters=50,
        ...     finetune_lr=1e-5, finetune_epochs=10000,
        ...     finetune_eval_interval=100, finetune_eval_iters=50,
        ...     finetune_max_tokens=20000000, finetune_temperature=0.7,
        ...     finetune_top_p=0.9, finetune_max_new_tokens=150,
        ...     pretrain_max_examples=20000000
        ... )
        >>> print(config.n_embd)
        1024

    Note:
        GPT-2 Medium: n_layer=24, n_head=16, n_embd=1024 (~355M params)
        GPT-2 Small: n_layer=12, n_head=12, n_embd=768 (~124M params)
        GPT-2 Large: n_layer=36, n_head=20, n_embd=1280 (~774M params)
    """

    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool
    lr: float
    min_lr: float
    warmup_steps: int
    batch_size: int
    grad_clip: float
    weight_decay: float
    epochs: int
    eval_interval: int
    eval_iters: int
    finetune_lr: float
    finetune_epochs: int
    finetune_eval_interval: int
    finetune_eval_iters: int
    finetune_max_tokens: int
    finetune_temperature: float
    finetune_top_p: float
    finetune_max_new_tokens: int
    pretrain_max_examples: int

    def __post_init__(self) -> None:
        """
        Validate configuration parameters after initialization.

        Ensures architectural constraints are satisfied for proper model
        operation, specifically that n_embd is evenly divisible by n_head.

        Raises:
            ValueError: If n_embd is not divisible by n_head.
        """
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


def _read_json_file(path: Union[str, Path]) -> dict:
    """
    Read and parse a JSON file into a dictionary.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Dictionary containing parsed JSON data.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Union[str, Path] = "config.json") -> GPTConfig:
    """
    Load GPT-2 configuration from a JSON file.

    Reads a JSON configuration file and returns a GPTConfig dataclass
    instance with the loaded hyperparameters for GPT-2 architecture.

    Args:
        path: Path to the JSON configuration file. Can be a string or Path
            object. Defaults to "config.json" in the current directory.

    Returns:
        A GPTConfig instance populated with values from the JSON file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If required configuration keys are missing.
        ValueError: If configuration values fail validation.

    Example:
        >>> config = load_config("config.json")
        >>> print(config.block_size)
        1024
        >>> print(config.n_embd)
        1024
    """
    data = _read_json_file(path)
    return GPTConfig(**data)
