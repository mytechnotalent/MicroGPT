"""
MicroGPT: A minimal decoder-only transformer language model implementation.

This module provides a complete implementation of a GPT-style language model
from scratch using PyTorch, including:
    - Self-attention mechanisms with causal masking
    - Multi-head attention
    - Feed-forward networks
    - Transformer blocks with residual connections
    - Complete language model with training utilities

Architecture Overview:
    The model follows the GPT architecture with:
    1. Token and position embeddings
    2. Multiple transformer blocks (self-attention + feed-forward)
    3. Layer normalization and residual connections
    4. Output projection to vocabulary

Components:
    - SelfAttentionHead: Single causal self-attention head
    - MultiHeadAttention: Parallel attention heads with projection
    - FeedForward: Position-wise feed-forward network
    - Block: Complete transformer block
    - MicroGPT: Full language model
    - Utility functions: Data loading, training, evaluation, checkpointing

Usage:
    >>> import torch
    >>> from micro_gpt import MicroGPT, get_batch
    >>>
    >>> # Create model
    >>> model = MicroGPT(
    ...     vocab_size=50257,
    ...     embedding_dim=768,
    ...     block_size=256,
    ...     n_heads=12,
    ...     n_layers=12,
    ...     dropout=0.1
    ... )
    >>>
    >>> # Train
    >>> data = torch.randint(0, 50257, (10000,))
    >>> x, y = get_batch(data, block_size=256, batch_size=32, device="cpu")
    >>> logits, loss = model(x, y)
    >>>
    >>> # Generate
    >>> context = torch.randint(0, 50257, (1, 10))
    >>> output = model.generate(context, max_new_tokens=50, temperature=0.8)

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch.optim import Optimizer


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class SelfAttentionHead(nn.Module):
    """
    Single causal self-attention head with scaled dot-product attention.

    This implements one attention head that computes attention weights using
    scaled dot-product attention with causal masking to prevent attending to
    future positions in the sequence.

    Args:
        embedding_dim: Dimension of input embeddings
        block_size: Maximum sequence length for causal mask buffer
        head_size: Output dimension of this attention head
        dropout: Dropout probability for attention weights (default: 0.1)

    Attributes:
        key: Linear layer for computing keys (no bias)
        query: Linear layer for computing queries (no bias)
        value: Linear layer for computing values (no bias)
        dropout: Dropout layer applied to attention weights
        tril: Lower triangular mask buffer for causal attention
    """

    def __init__(
        self, embedding_dim: int, block_size: int, head_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.tril: torch.Tensor

    def _compute_attention_weights(
        self, q: torch.Tensor, k: torch.Tensor, T: int
    ) -> torch.Tensor:
        """
        Compute scaled, masked attention weights using dot-product attention.

        Implements the attention mechanism: softmax((Q @ K^T) / sqrt(d_k))
        with causal masking to prevent attending to future positions.

        Args:
            q: Query tensor of shape (B, T, head_size)
            k: Key tensor of shape (B, T, head_size)
            T: Sequence length for dynamic masking

        Returns:
            Attention weights of shape (B, T, T) after softmax and dropout

        Example:
            >>> head = SelfAttentionHead(embedding_dim=64, block_size=128, head_size=16)
            >>> q = torch.randn(2, 10, 16)  # (batch=2, seq_len=10, head_size=16)
            >>> k = torch.randn(2, 10, 16)
            >>> weights = head._compute_attention_weights(q, k, T=10)
            >>> weights.shape
            torch.Size([2, 10, 10])
        """
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return self.dropout(F.softmax(wei, dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal self-attention to input sequence.

        Args:
            x: Input tensor of shape (B, T, embedding_dim)

        Returns:
            Output tensor of shape (B, T, head_size)

        Example:
            >>> head = SelfAttentionHead(embedding_dim=64, block_size=128, head_size=16)
            >>> x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, emb_dim=64)
            >>> output = head(x)
            >>> output.shape
            torch.Size([2, 10, 16])
        """
        k, q = self.key(x), self.query(x)
        wei = self._compute_attention_weights(q, k, x.size(1))
        return wei @ self.value(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with projection layer.

    Implements parallel attention heads that attend to different representation
    subspaces at different positions, then concatenates and projects the results.

    Args:
        embedding_dim: Dimension of input embeddings
        block_size: Maximum sequence length for causal mask
        num_heads: Number of parallel attention heads
        dropout: Dropout probability (default: 0.1)

    Attributes:
        heads: ModuleList of SelfAttentionHead instances
        proj: Linear projection layer for concatenated head outputs
        dropout: Dropout layer applied after projection
    """

    def __init__(
        self, embedding_dim: int, block_size: int, num_heads: int, dropout: float = 0.1
    ):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(embedding_dim, block_size, head_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention to input.

        Args:
            x: Input tensor of shape (B, T, embedding_dim)

        Returns:
            Output tensor of shape (B, T, embedding_dim)

        Example:
            >>> mha = MultiHeadAttention(embedding_dim=64, block_size=128, num_heads=4)
            >>> x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, emb_dim=64)
            >>> output = mha(x)
            >>> output.shape
            torch.Size([2, 10, 64])
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with expansion and contraction.

    Two-layer MLP with ReLU activation and 4x expansion ratio,
    as described in "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        n_embd: Input and output embedding dimension
        dropout: Dropout probability (default: 0.1)

    Attributes:
        net: Sequential network containing:
            - Expansion linear layer (n_embd -> 4*n_embd)
            - ReLU activation
            - Dropout
            - Contraction linear layer (4*n_embd -> n_embd)
            - Dropout
    """

    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network to input.

        Args:
            x: Input tensor of shape (B, T, n_embd)

        Returns:
            Output tensor of shape (B, T, n_embd)

        Example:
            >>> ffn = FeedForward(n_embd=64)
            >>> x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, n_embd=64)
            >>> output = ffn(x)
            >>> output.shape
            torch.Size([2, 10, 64])
        """
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block with pre-LayerNorm and residual connections.

    Implements a single transformer block consisting of multi-head self-attention
    followed by a feed-forward network. Uses pre-normalization (LayerNorm before
    each sub-layer) and residual connections around both sub-layers.

    Args:
        embedding_dim: Dimension of input embeddings
        block_size: Maximum sequence length for causal mask
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)

    Attributes:
        sa: Multi-head self-attention layer
        ffwd: Feed-forward network
        ln1: Layer normalization before attention
        ln2: Layer normalization before feed-forward
    """

    def __init__(
        self, embedding_dim: int, block_size: int, n_heads: int, dropout: float = 0.1
    ):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads, dropout)
        self.ffwd = FeedForward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with pre-norm and residual connections.

        Computation:
            1. x = x + MultiHeadAttention(LayerNorm(x))
            2. x = x + FeedForward(LayerNorm(x))

        Args:
            x: Input tensor of shape (B, T, embedding_dim)

        Returns:
            Output tensor of shape (B, T, embedding_dim)

        Example:
            >>> block = Block(embedding_dim=64, block_size=128, n_heads=4)
            >>> x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, emb_dim=64)
            >>> output = block(x)
            >>> output.shape
            torch.Size([2, 10, 64])
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    """
    MicroGPT: A minimal decoder-only transformer language model.

    Implements a GPT-style autoregressive language model with causal self-attention.
    The model predicts the next token in a sequence based on all previous tokens.

    Architecture:
        1. Token embedding: Maps vocabulary indices to dense vectors
        2. Position embedding: Adds learnable position information
        3. N transformer blocks: Self-attention + feed-forward layers
        4. Final layer norm: Normalizes before output projection
        5. Language model head: Projects to vocabulary logits

    Args:
        vocab_size: Size of the vocabulary (number of unique tokens)
        embedding_dim: Dimension of token and position embeddings
        block_size: Maximum sequence length (context window)
        n_heads: Number of attention heads per transformer block
        n_layers: Number of stacked transformer blocks (depth)
        dropout: Dropout probability (default: 0.1)

    Attributes:
        block_size: Maximum sequence length stored for generation
        token_embedding: Embedding layer for token indices
        position_embedding: Embedding layer for position indices
        blocks: ModuleList of transformer blocks
        ln_f: Final layer normalization before projection
        head: Linear layer projecting to vocabulary logits
        dropout: Dropout layer applied to combined embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        block_size: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.ModuleList(
            [
                Block(embedding_dim, block_size, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Combine token and position embeddings with dropout.

        Creates embeddings by adding token embeddings (content) and position
        embeddings (sequential information), then applies dropout for regularization.

        Args:
            idx: Token indices tensor of shape (B, T)

        Returns:
            Combined embeddings tensor of shape (B, T, embedding_dim)

        Example:
            >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
            ...                  n_heads=4, n_layers=2)
            >>> idx = torch.randint(0, 100, (2, 10))
            >>> embeddings = model._embed_tokens(idx)
            >>> embeddings.shape
            torch.Size([2, 10, 64])
        """
        T = idx.size(1)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        return self.dropout(tok_emb + pos_emb)

    def _apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transformer blocks sequentially.

        Passes the input through each transformer block in order, with each
        block applying self-attention and feed-forward transformations with
        residual connections.

        Args:
            x: Input tensor of shape (B, T, embedding_dim)

        Returns:
            Transformed tensor of shape (B, T, embedding_dim)

        Example:
            >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
            ...                  n_heads=4, n_layers=2)
            >>> x = torch.randn(2, 10, 64)
            >>> output = model._apply_blocks(x)
            >>> output.shape
            torch.Size([2, 10, 64])
        """
        for block in self.blocks:
            x = block(x)
        return x

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss between predictions and targets.

        Flattens the logits and targets tensors to compute the cross-entropy
        loss across all positions in the batch simultaneously.

        Args:
            logits: Predicted logits of shape (B, T, vocab_size)
            targets: Target token indices of shape (B, T)

        Returns:
            Scalar cross-entropy loss tensor

        Example:
            >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
            ...                  n_heads=4, n_layers=2)
            >>> logits = torch.randn(2, 10, 100)
            >>> targets = torch.randint(0, 100, (2, 10))
            >>> loss = model._compute_loss(logits, targets)
            >>> loss.shape
            torch.Size([])
        """
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: compute logits and optionally compute loss.

        Processes input tokens through embeddings, transformer blocks, and
        output projection. If targets are provided, also computes the
        cross-entropy loss for training.

        Args:
            idx: Input token indices of shape (B, T)
            targets: Optional target token indices of shape (B, T) for loss computation

        Returns:
            Tuple of (logits, loss) where:
                - logits: Predicted token logits of shape (B, T, vocab_size)
                - loss: Scalar loss tensor if targets provided, else None

        Example:
            >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
            ...                  n_heads=4, n_layers=2)
            >>> idx = torch.randint(0, 100, (2, 10))
            >>> # Inference mode (no targets)
            >>> logits, loss = model(idx)
            >>> logits.shape, loss
            (torch.Size([2, 10, 100]), None)
            >>> # Training mode (with targets)
            >>> targets = torch.randint(0, 100, (2, 10))
            >>> logits, loss = model(idx, targets)
            >>> logits.shape, loss.shape
            (torch.Size([2, 10, 100]), torch.Size([]))
        """
        x = self._embed_tokens(idx)
        logits = self.head(self.ln_f(self._apply_blocks(x)))
        loss = self._compute_loss(logits, targets) if targets is not None else None
        return logits, loss

    def _generate_step(self, idx: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Generate one token using temperature-scaled sampling.

        Performs a single generation step by:
        1. Cropping context to block_size if needed
        2. Computing logits for the current context
        3. Scaling logits by temperature
        4. Sampling from the resulting probability distribution
        5. Appending the new token to the sequence

        Args:
            idx: Current token sequence of shape (B, T)
            temperature: Sampling temperature (higher = more random)

        Returns:
            Extended sequence of shape (B, T+1)

        Example:
            >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
            ...                  n_heads=4, n_layers=2)
            >>> idx = torch.randint(0, 100, (1, 5))
            >>> new_idx = model._generate_step(idx, temperature=1.0)
            >>> new_idx.shape
            torch.Size([1, 6])
        """
        idx_cond = idx[:, -self.block_size :]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        next_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return torch.cat((idx, next_idx), dim=1)

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using the model.

        Generates new tokens one at a time by repeatedly calling the model
        and sampling from the predicted distribution. Uses temperature scaling
        to control the randomness of predictions.

        Args:
            idx: Initial context tokens of shape (B, T0)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature where:
                - Low (< 1.0): More deterministic, focused predictions
                - 1.0: Normal sampling from the distribution
                - High (> 1.0): More random, creative predictions

        Returns:
            Extended sequence of shape (B, T0 + max_new_tokens)

        Example:
            >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
            ...                  n_heads=4, n_layers=2)
            >>> context = torch.randint(0, 100, (1, 10))
            >>> generated = model.generate(context, max_new_tokens=20, temperature=0.8)
            >>> generated.shape
            torch.Size([1, 30])
        """
        for _ in range(max_new_tokens):
            idx = self._generate_step(idx, temperature)
        return idx


# =============================================================================
# TRAINING UTILITIES
# =============================================================================


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch for training with input-target pairs.

    Randomly samples batch_size starting positions from the data, then
    extracts sequences of length block_size as inputs (x) and the next
    token at each position as targets (y).

    Args:
        data: Full dataset tensor of token indices, shape (N,)
        block_size: Length of each sequence
        batch_size: Number of sequences to sample
        device: Device to place tensors on ("cpu", "cuda", or "mps")

    Returns:
        Tuple of (x, y) where:
            - x: Input sequences of shape (batch_size, block_size)
            - y: Target sequences of shape (batch_size, block_size)
              where y[i, j] = x[i, j+1] (next token prediction)

    Example:
        >>> data = torch.randint(0, 100, (1000,))
        >>> x, y = get_batch(data, block_size=16, batch_size=4, device="cpu")
        >>> x.shape, y.shape
        (torch.Size([4, 16]), torch.Size([4, 16]))
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def save_checkpoint(
    model: nn.Module, optimizer: Optimizer, step: int, val_loss: float, filepath: str
) -> None:
    """
    Save model checkpoint to disk with training state.

    Saves a checkpoint containing the model state, optimizer state,
    training step number, and validation loss to enable resuming
    training or loading the best model.

    Args:
        model: The model to save (nn.Module)
        optimizer: The optimizer to save (torch.optim.Optimizer)
        step: Current training step number
        val_loss: Current validation loss value
        filepath: Path where checkpoint will be saved

    Returns:
        None

    Example:
        >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
        ...                  n_heads=4, n_layers=2)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> save_checkpoint(model, optimizer, step=1000, val_loss=2.5,
        ...                 filepath="checkpoints/model.pt")
    """
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        filepath,
    )


def _train_step(
    model: nn.Module,
    optimizer: Optimizer,
    train_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> float:
    """
    Perform a single training step with gradient clipping.

    Executes one complete training iteration:
    1. Sample a batch of data
    2. Forward pass to compute loss
    3. Backward pass to compute gradients
    4. Clip gradients to prevent explosion
    5. Optimizer step to update weights

    Args:
        model: The model to train (nn.Module)
        optimizer: The optimizer for weight updates
        train_data: Training dataset tensor
        block_size: Sequence length for batching
        batch_size: Number of sequences per batch
        device: Device to run training on

    Returns:
        Scalar training loss value (float)

    Note:
        Gradients are clipped to max norm of 1.0 to prevent exploding gradients.

    Example:
        >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
        ...                  n_heads=4, n_layers=2)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> train_data = torch.randint(0, 100, (1000,))
        >>> loss = _train_step(model, optimizer, train_data, 16, 4, "cpu")
        >>> isinstance(loss, float)
        True
    """
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def _eval_model(
    model: nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int,
) -> float:
    """
    Evaluate model on validation data without computing gradients.

    Computes average loss over multiple evaluation batches with the
    model in eval mode (dropout disabled) and no gradient computation.
    Restores the model to training mode after evaluation.

    Args:
        model: The model to evaluate (nn.Module)
        data: Validation dataset tensor
        block_size: Sequence length for batching
        batch_size: Number of sequences per batch
        device: Device to run evaluation on
        eval_iters: Number of evaluation batches to average over

    Returns:
        Average validation loss across all evaluation iterations (float)

    Note:
        Model is temporarily set to eval mode and restored to train mode.

    Example:
        >>> model = MicroGPT(vocab_size=100, embedding_dim=64, block_size=128,
        ...                  n_heads=4, n_layers=2)
        >>> val_data = torch.randint(0, 100, (1000,))
        >>> avg_loss = _eval_model(model, val_data, 16, 4, "cpu", 5)
        >>> isinstance(avg_loss, float)
        True
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xv, yv = get_batch(data, block_size, batch_size, device)
            _, loss = model(xv, yv)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def _should_save(val_loss: float, best_loss: float) -> bool:
    """
    Determine if current validation loss is a new best.

    Simple comparison function to check if the current validation loss
    is strictly better (lower) than the previous best loss.

    Args:
        val_loss: Current validation loss value
        best_loss: Best (lowest) validation loss seen so far

    Returns:
        True if val_loss < best_loss, False otherwise

    Example:
        >>> _should_save(val_loss=2.5, best_loss=3.0)
        True
        >>> _should_save(val_loss=3.0, best_loss=2.5)
        False
    """
    return val_loss < best_loss


# =============================================================================
# MAIN EXPORTS
# =============================================================================

__all__ = [
    "SelfAttentionHead",
    "MultiHeadAttention",
    "FeedForward",
    "Block",
    "MicroGPT",
    "get_batch",
    "save_checkpoint",
    "_train_step",
    "_eval_model",
    "_should_save",
]
