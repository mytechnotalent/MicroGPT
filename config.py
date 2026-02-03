"""
Configuration Management for MicroGPT

This module provides configuration loading and management for the MicroGPT
language model. It loads hyperparameters from a JSON file and provides
a typed dataclass for easy access.

Usage:
    from config import load_config, Config

    config = load_config("config.json")
    print(config.embedding_dim)  # 32

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Config:
    """
    Configuration container for MicroGPT hyperparameters.

    This dataclass holds all the hyperparameters needed to configure the
    MicroGPT model architecture and training process. Values are typically
    loaded from a JSON configuration file.

    Args:
        block_size: Maximum sequence length (context window size). Determines
            how many tokens the model can attend to at once.
        embedding_dim: Dimension of token and position embeddings (d_model).
            This is the hidden size throughout the Transformer.
        n_heads: Number of attention heads per Transformer block. Must evenly
            divide embedding_dim.
        n_layers: Number of stacked Transformer blocks (depth of the model).
        dropout: Dropout probability for regularization.
        lr: Peak learning rate for the optimizer.
        min_lr: Minimum learning rate after decay.
        warmup_steps: Number of steps for learning rate warmup.
        batch_size: Number of sequences per training batch.
        grad_clip: Maximum gradient norm for gradient clipping.
        weight_decay: L2 regularization weight decay coefficient.
        epochs: Number of training iterations (steps).

    Attributes:
        block_size: Maximum sequence length.
        embedding_dim: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        dropout: Dropout probability.
        lr: Peak learning rate.
        min_lr: Minimum learning rate.
        warmup_steps: Warmup steps count.
        batch_size: Training batch size.
        grad_clip: Gradient clipping threshold.
        weight_decay: Weight decay coefficient.
        epochs: Number of training epochs.

    Example:
        >>> config = Config(
        ...     block_size=128,
        ...     embedding_dim=64,
        ...     n_heads=4,
        ...     n_layers=3,
        ...     dropout=0.1,
        ...     lr=1e-3,
        ...     epochs=1000
        ... )
        >>> print(config.embedding_dim)
        64

    Note:
        For production use, consider adding validation to ensure n_heads
        evenly divides embedding_dim.
    """

    # Maximum sequence length (context window size)
    block_size: int
    # Dimension of token and position embeddings
    embedding_dim: int
    # Number of attention heads per Transformer block
    n_heads: int
    # Number of stacked Transformer blocks
    n_layers: int
    # Dropout probability for regularization
    dropout: float
    # Peak learning rate for the optimizer
    lr: float
    # Minimum learning rate after decay
    min_lr: float
    # Number of steps for learning rate warmup
    warmup_steps: int
    # Number of sequences per training batch
    batch_size: int
    # Maximum gradient norm for gradient clipping
    grad_clip: float
    # L2 regularization weight decay coefficient
    weight_decay: float
    # Number of training iterations
    epochs: int
    # Fine-tuning learning rate
    finetune_lr: float
    # Fine-tuning epochs
    finetune_epochs: int
    # Fine-tuning evaluation interval
    finetune_eval_interval: int
    # Fine-tuning evaluation iterations
    finetune_eval_iters: int
    # Fine-tuning max tokens to load
    finetune_max_tokens: int
    # Fine-tuning temperature for generation
    finetune_temperature: float
    # Fine-tuning max new tokens to generate
    finetune_max_new_tokens: int
    # Pre-training max examples from dataset
    pretrain_max_examples: int

    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.

        Raises:
            ValueError: If embedding_dim is not divisible by n_heads.
        """
        # Check that embedding_dim is evenly divisible by n_heads for multi-head attention
        if self.embedding_dim % self.n_heads != 0:
            # Raise ValueError with descriptive message if validation fails
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )


def load_config(path: Union[str, Path] = "config.json") -> Config:
    """
    Load configuration from a JSON file.

    Reads a JSON configuration file and returns a Config dataclass instance
    with the loaded hyperparameters.

    Args:
        path: Path to the JSON configuration file. Can be a string or Path
            object. Defaults to "config.json" in the current directory.

    Returns:
        A Config instance populated with values from the JSON file.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If required configuration keys are missing.
        ValueError: If configuration values fail validation.

    Example:
        >>> config = load_config("config.json")
        >>> print(config.block_size)
        6
        >>> print(config.lr)
        0.001

    Note:
        The JSON file must contain all required keys: block_size, embedding_dim,
        n_heads, n_layers, lr, and epochs.
    """
    # Open the JSON file and parse its contents into a dictionary
    with open(path, "r", encoding="utf-8") as f:
        # Parse JSON file contents into Python dictionary
        data = json.load(f)
    # Create and return Config instance with values from the dictionary
    return Config(
        # Set block_size from JSON data
        block_size=data["block_size"],
        # Set embedding_dim from JSON data
        embedding_dim=data["embedding_dim"],
        # Set n_heads from JSON data
        n_heads=data["n_heads"],
        # Set n_layers from JSON data
        n_layers=data["n_layers"],
        # Set dropout from JSON data
        dropout=data["dropout"],
        # Set lr from JSON data
        lr=data["lr"],
        # Set min_lr from JSON data
        min_lr=data["min_lr"],
        # Set warmup_steps from JSON data
        warmup_steps=data["warmup_steps"],
        # Set batch_size from JSON data
        batch_size=data["batch_size"],
        # Set grad_clip from JSON data
        grad_clip=data["grad_clip"],
        # Set weight_decay from JSON data
        weight_decay=data["weight_decay"],
        # Set epochs from JSON data
        epochs=data["epochs"],
        # Set finetune_lr from JSON data
        finetune_lr=data["finetune_lr"],
        # Set finetune_epochs from JSON data
        finetune_epochs=data["finetune_epochs"],
        # Set finetune_eval_interval from JSON data
        finetune_eval_interval=data["finetune_eval_interval"],
        # Set finetune_eval_iters from JSON data
        finetune_eval_iters=data["finetune_eval_iters"],
        # Set finetune_max_tokens from JSON data
        finetune_max_tokens=data["finetune_max_tokens"],
        # Set finetune_temperature from JSON data
        finetune_temperature=data["finetune_temperature"],
        # Set finetune_max_new_tokens from JSON data
        finetune_max_new_tokens=data["finetune_max_new_tokens"],
        # Set pretrain_max_examples from JSON data
        pretrain_max_examples=data["pretrain_max_examples"],
    )
