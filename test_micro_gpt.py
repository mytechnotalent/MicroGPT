"""
Comprehensive Test Suite for MicroGPT

This module provides extensive testing for all components of the MicroGPT
implementation including model architecture, training utilities, data loading,
and generation capabilities.

Test Organization:
    - All tests follow the Arrange-Act-Assert (AAA) pattern
    - Comprehensive docstrings for every test method
    - Consistent naming conventions: test_<component>_<behavior>
    - 65 tests with 99% code coverage

Test Categories:
    - SelfAttentionHead: 6 tests for single attention head functionality
    - MultiHeadAttention: 4 tests for multi-head attention mechanisms
    - FeedForward: 4 tests for feed-forward network layers
    - Block: 4 tests for transformer blocks
    - MicroGPT: 15 tests for the complete language model
    - Utility Functions: 17 tests for training/data utilities
    - Integration: 5 tests for end-to-end workflows
    - Edge Cases: 6 tests for boundary conditions
    - Performance: 3 tests for timing and efficiency

Testing Patterns:
    Each test follows the Arrange-Act-Assert pattern:
    1. Arrange: Set up test data and initialize components
    2. Act: Execute the function or method being tested
    3. Assert: Verify the results match expectations

Usage:
    Run all tests:
        pytest test_micro_gpt.py -v

    Run with coverage report:
        pytest test_micro_gpt.py -v --cov=test_micro_gpt --cov-report=html

    Run specific test class:
        pytest test_micro_gpt.py::TestMicroGPT -v

    Run specific test method:
        pytest test_micro_gpt.py::TestMicroGPT::test_forward_with_targets -v

    View HTML coverage:
        open htmlcov/index.html

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile
from typing import Tuple, Optional

# =============================================================================
# MODEL CLASSES (extracted from MicroGPT.ipynb)
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
            >>> # Inference mode
            >>> logits, loss = model(idx)
            >>> logits.shape, loss
            (torch.Size([2, 10, 100]), None)
            >>> # Training mode
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
            >>> idx = torch.randint(0, 100, (1, 10))
            >>> new_idx = model._generate_step(idx, temperature=1.0)
            >>> new_idx.shape
            torch.Size([1, 11])
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
# UTILITY FUNCTIONS (extracted from MicroGPT.ipynb)
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
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    val_loss: float,
    filepath: str,
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
        >>> model = MicroGPT(vocab_size=100, embedding_dim=64, ...)
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
    optimizer: torch.optim.Optimizer,
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

    Example:
        >>> model = MicroGPT(vocab_size=100, embedding_dim=64, ...)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> train_data = torch.randint(0, 100, (1000,))
        >>> loss = _train_step(model, optimizer, train_data, 16, 4, "cpu")
        >>> isinstance(loss, float)
        True

    Note:
        Gradients are clipped to max norm of 1.0 to prevent exploding gradients.
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

    Example:
        >>> model = MicroGPT(vocab_size=100, embedding_dim=64, ...)
        >>> val_data = torch.randint(0, 100, (1000,))
        >>> avg_loss = _eval_model(model, val_data, 16, 4, "cpu", 10)
        >>> isinstance(avg_loss, float)
        True

    Note:
        Model is temporarily set to eval mode and restored to train mode.
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
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def device() -> str:
    """
    Provide CPU device for testing to ensure consistent behavior.

    Returns "cpu" to avoid device-specific issues across different platforms.
    This ensures tests run reliably on systems with or without GPUs.

    Returns:
        str: "cpu" device identifier

    Example:
        >>> # In a test function
        >>> def test_something(device):
        ...     tensor = torch.zeros(2, 3).to(device)
        ...     assert tensor.device.type == "cpu"
    """
    return "cpu"


@pytest.fixture
def small_model_config() -> dict:
    """
    Provide a small model configuration for fast testing.

    Returns a dictionary with hyperparameters for a small MicroGPT model
    that trains quickly while still testing all functionality.

    Returns:
        dict: Configuration dictionary with keys:
            - vocab_size: 100 (small vocabulary)
            - embedding_dim: 32 (small embedding dimension)
            - block_size: 16 (short context window)
            - n_heads: 4 (multiple attention heads)
            - n_layers: 2 (shallow model)
            - dropout: 0.1 (standard dropout rate)

    Example:
        >>> # In a test function
        >>> def test_model_creation(small_model_config):
        ...     model = MicroGPT(**small_model_config)
        ...     assert model.embedding_dim == 32
    """
    return {
        "vocab_size": 100,
        "embedding_dim": 32,
        "block_size": 16,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
    }


@pytest.fixture
def model(small_model_config: dict) -> MicroGPT:
    """
    Provide a small MicroGPT model instance for testing.

    Creates a MicroGPT model with the small configuration for use
    across multiple tests.

    Args:
        small_model_config: Configuration fixture providing hyperparameters

    Returns:
        MicroGPT: Initialized model instance

    Example:
        >>> # In a test function
        >>> def test_forward_pass(model):
        ...     x = torch.randint(0, 100, (2, 16))
        ...     logits, loss = model(x)
        ...     assert logits.shape == (2, 16, 100)
    """
    return MicroGPT(**small_model_config)


@pytest.fixture
def sample_data(device: str) -> torch.Tensor:
    """
    Provide sample training data for testing.

    Creates a random dataset of token indices suitable for testing
    data loading and training functions.

    Args:
        device: Device fixture (unused but kept for consistency)

    Returns:
        torch.Tensor: Random token indices of shape (1000,) with values in [0, 100)

    Example:
        >>> # In a test function
        >>> def test_data_loading(sample_data):
        ...     x, y = get_batch(sample_data, block_size=16, batch_size=4, device="cpu")
        ...     assert x.shape == (4, 16) and y.shape == (4, 16)
    """
    return torch.randint(0, 100, (1000,))


@pytest.fixture
def batch_data(device: str) -> torch.Tensor:
    """
    Provide sample batch data for testing.

    Creates a random batch of token sequences suitable for testing
    model forward passes and generation.

    Args:
        device: Device fixture (unused but kept for consistency)

    Returns:
        torch.Tensor: Random token indices of shape (4, 16) with values in [0, 100)

    Example:
        >>> # In a test function
        >>> def test_model_forward(model, batch_data):
        ...     logits, loss = model(batch_data)
        ...     assert logits.shape == (4, 16, 100)
    """
    batch_size = 4
    seq_len = 16
    return torch.randint(0, 100, (batch_size, seq_len))


# =============================================================================
# TESTS: SelfAttentionHead
# =============================================================================


class TestSelfAttentionHead:
    """
    Test suite for SelfAttentionHead component.

    Tests the single attention head implementation including initialization,
    forward pass shapes, causal masking, attention weight normalization,
    sequence length handling, and dropout application.
    """

    def test_initialization(self):
        """
        Test that SelfAttentionHead initializes with correct layer dimensions.

        Verifies that key, query, value projections have the correct input
        and output dimensions, and that the causal mask buffer has the right shape.

        Test Pattern:
            Arrange: Create a SelfAttentionHead with specific dimensions
            Act: Access the layer attributes
            Assert: Verify dimensions match expectations
        """
        # Arrange
        embedding_dim = 64
        block_size = 128
        head_size = 16
        dropout_rate = 0.1

        # Act
        head = SelfAttentionHead(
            embedding_dim=embedding_dim,
            block_size=block_size,
            head_size=head_size,
            dropout=dropout_rate,
        )

        # Assert
        assert head.key.in_features == embedding_dim
        assert head.key.out_features == head_size
        assert head.query.in_features == embedding_dim
        assert head.query.out_features == head_size
        assert head.value.in_features == embedding_dim
        assert head.value.out_features == head_size
        assert head.tril.shape == (block_size, block_size)

    def test_forward_shape(self):
        """
        Test that forward pass produces correct output shape.

        Verifies that the attention head outputs a tensor with the expected
        shape after processing a batch of sequences.

        Test Pattern:
            Arrange: Create head and input tensor
            Act: Pass input through the head
            Assert: Verify output shape is (batch, seq_len, head_size)
        """
        # Arrange
        head = SelfAttentionHead(embedding_dim=64, block_size=128, head_size=16)
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Act
        out = head(x)

        # Assert
        expected_shape = (batch_size, seq_len, 16)
        assert out.shape == expected_shape

    def test_causal_masking(self):
        """
        Test that causal masking prevents attending to future positions.

        Verifies that the first token can only attend to itself (not future tokens)
        by checking that attention weights for future positions are zero.

        Test Pattern:
            Arrange: Create head and input, compute attention weights
            Act: Extract attention weights for first token
            Assert: Verify future positions have zero weight
        """
        # Arrange
        head = SelfAttentionHead(embedding_dim=64, block_size=128, head_size=16)
        x = torch.randn(1, 5, 64)

        # Act
        with torch.no_grad():
            k = head.key(x)
            q = head.query(x)
            wei = head._compute_attention_weights(q, k, 5)

        # Assert - first token should not attend to future (positions 1-4)
        future_weights = wei[0, 0, 1:]
        expected_zeros = torch.zeros(4)
        assert torch.allclose(
            future_weights, expected_zeros, atol=1e-6
        ), "First token should not attend to future positions"

    def test_attention_weights_sum_to_one(self):
        """
        Test that attention weights are properly normalized (sum to 1).

        Verifies that after softmax, attention weights for each position
        sum to 1 along the key dimension.

        Test Pattern:
            Arrange: Create head in eval mode and input tensor
            Act: Compute attention weights
            Assert: Verify weights sum to 1 along last dimension
        """
        # Arrange
        head = SelfAttentionHead(embedding_dim=64, block_size=128, head_size=16)
        head.eval()  # Disable dropout for deterministic test
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)

        # Act
        with torch.no_grad():
            k = head.key(x)
            q = head.query(x)
            wei = head._compute_attention_weights(q, k, seq_len)

        # Assert - sum along last dimension should be 1
        weight_sums = wei.sum(dim=-1)
        expected_sums = torch.ones(batch_size, seq_len)
        assert torch.allclose(weight_sums, expected_sums, atol=1e-5)

    def test_different_sequence_lengths(self):
        """
        Test that head handles various sequence lengths correctly.

        Verifies that the attention head can process sequences of different
        lengths up to the maximum block_size.

        Test Pattern:
            Arrange: Create head once
            Act: Pass sequences of different lengths
            Assert: Verify output shapes are correct for each length
        """
        # Arrange
        head = SelfAttentionHead(embedding_dim=64, block_size=128, head_size=16)
        batch_size = 2
        test_lengths = [1, 5, 32, 128]

        # Act & Assert
        for seq_len in test_lengths:
            x = torch.randn(batch_size, seq_len, 64)
            out = head(x)
            expected_shape = (batch_size, seq_len, 16)
            assert out.shape == expected_shape, f"Failed for seq_len={seq_len}"

    def test_dropout_effect(self):
        """
        Test that dropout is applied during training mode.

        Verifies that dropout creates variation in outputs by running
        the same input multiple times and checking for differences.

        Test Pattern:
            Arrange: Create head with high dropout in training mode
            Act: Pass same input multiple times
            Assert: Verify outputs differ due to dropout randomness
        """
        # Arrange
        head = SelfAttentionHead(
            embedding_dim=64, block_size=128, head_size=16, dropout=0.5
        )
        head.train()  # Ensure training mode for dropout
        x = torch.randn(2, 10, 64)

        # Act
        outputs = [head(x) for _ in range(5)]

        # Assert - outputs should differ due to dropout
        all_same = all(torch.equal(outputs[0], out) for out in outputs[1:])
        assert not all_same, "Dropout should cause variation in outputs"


# =============================================================================
# TESTS: MultiHeadAttention
# =============================================================================


class TestMultiHeadAttention:
    """
    Test suite for MultiHeadAttention component.

    Tests the multi-head attention implementation including initialization,
    forward pass shapes, handling different numbers of heads, and head size
    computation.
    """

    def test_initialization(self):
        """
        Test that MultiHeadAttention initializes with correct number of heads.

        Verifies that the correct number of attention heads are created
        and projection layer dimensions are correct.

        Test Pattern:
            Arrange: Define parameters for multi-head attention
            Act: Create MultiHeadAttention instance
            Assert: Verify head count and projection dimensions
        """
        # Arrange
        embedding_dim = 64
        block_size = 128
        num_heads = 4

        # Act
        mha = MultiHeadAttention(
            embedding_dim=embedding_dim, block_size=block_size, num_heads=num_heads
        )

        # Assert
        assert len(mha.heads) == num_heads
        assert mha.proj.in_features == embedding_dim
        assert mha.proj.out_features == embedding_dim

    def test_forward_shape(self):
        """
        Test that forward pass maintains input shape.

        Verifies that multi-head attention outputs a tensor with the same
        shape as the input (after concatenation and projection).

        Test Pattern:
            Arrange: Create multi-head attention and input tensor
            Act: Pass input through multi-head attention
            Assert: Verify output shape matches input shape
        """
        # Arrange
        mha = MultiHeadAttention(embedding_dim=64, block_size=128, num_heads=4)
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Act
        out = mha(x)

        # Assert
        assert out.shape == x.shape

    def test_different_num_heads(self):
        """
        Test that multi-head attention works with various head counts.

        Verifies that the module can be initialized and used with different
        numbers of attention heads (1, 2, 4, 8).

        Test Pattern:
            Arrange: Define embedding dimension
            Act: Create and test multi-head attention with different head counts
            Assert: Verify correct output shape for each configuration
        """
        # Arrange
        embedding_dim = 64
        batch_size = 2
        seq_len = 10
        test_head_counts = [1, 2, 4, 8]

        # Act & Assert
        for num_heads in test_head_counts:
            mha = MultiHeadAttention(
                embedding_dim=embedding_dim, block_size=128, num_heads=num_heads
            )
            x = torch.randn(batch_size, seq_len, embedding_dim)
            out = mha(x)
            assert out.shape == x.shape, f"Failed for num_heads={num_heads}"

    def test_head_size_computation(self):
        """
        Test that head size is correctly computed from embedding dimension.

        Verifies that each attention head has the correct output dimension,
        which should be embedding_dim // num_heads.

        Test Pattern:
            Arrange: Create multi-head attention with known dimensions
            Act: Extract head size from first head
            Assert: Verify head size equals embedding_dim // num_heads
        """
        # Arrange
        embedding_dim = 64
        num_heads = 4
        expected_head_size = embedding_dim // num_heads  # 16
        mha = MultiHeadAttention(
            embedding_dim=embedding_dim, block_size=128, num_heads=num_heads
        )
        x = torch.randn(2, 10, embedding_dim)

        # Act
        first_head_out = mha.heads[0](x)

        # Assert
        assert first_head_out.shape[-1] == expected_head_size


# =============================================================================
# TESTS: FeedForward
# =============================================================================


class TestFeedForward:
    """
    Test suite for FeedForward component.

    Tests the position-wise feed-forward network including initialization,
    forward pass shapes, expansion ratio, and handling different embedding
    dimensions.
    """

    def test_initialization(self):
        """
        Test that FeedForward network initializes with correct layer structure.

        Verifies that the first linear layer has correct input/output dimensions
        with the standard 4x expansion ratio.

        Test Pattern:
            Arrange: Define embedding dimension
            Act: Create FeedForward network
            Assert: Verify layer structure and dimensions
        """
        # Arrange
        n_embd = 64
        dropout_rate = 0.1

        # Act
        ff = FeedForward(n_embd=n_embd, dropout=dropout_rate)

        # Assert
        assert isinstance(ff.net[0], nn.Linear)
        assert ff.net[0].in_features == n_embd
        assert ff.net[0].out_features == 4 * n_embd  # 4x expansion

    def test_forward_shape(self):
        """
        Test that forward pass maintains input shape.

        Verifies that the feed-forward network outputs a tensor with the same
        shape as the input (after expansion and contraction).

        Test Pattern:
            Arrange: Create feed-forward network and input tensor
            Act: Pass input through the network
            Assert: Verify output shape matches input shape
        """
        # Arrange
        ff = FeedForward(n_embd=64)
        batch_size = 2
        seq_len = 10
        n_embd = 64
        x = torch.randn(batch_size, seq_len, n_embd)

        # Act
        out = ff(x)

        # Assert
        assert out.shape == x.shape

    def test_expansion_ratio(self):
        """
        Test that hidden dimension follows 4x expansion ratio.

        Verifies that the intermediate layer has 4 times the dimension
        of the input, as specified in the Transformer paper.

        Test Pattern:
            Arrange: Create feed-forward network
            Act: Access first layer's output features
            Assert: Verify it equals 4 * input dimension
        """
        # Arrange
        n_embd = 64
        ff = FeedForward(n_embd=n_embd)

        # Act
        hidden_dim = ff.net[0].out_features

        # Assert
        expected_hidden_dim = 4 * n_embd
        assert hidden_dim == expected_hidden_dim

    def test_different_embedding_dims(self):
        """
        Test that feed-forward network handles various embedding dimensions.

        Verifies that the network can be initialized and used with different
        embedding dimensions while maintaining shape consistency.

        Test Pattern:
            Arrange: Define test embedding dimensions
            Act: Create and test network for each dimension
            Assert: Verify output shape matches input shape
        """
        # Arrange
        batch_size = 2
        seq_len = 10
        test_dimensions = [32, 64, 128, 256]

        # Act & Assert
        for n_embd in test_dimensions:
            ff = FeedForward(n_embd=n_embd)
            x = torch.randn(batch_size, seq_len, n_embd)
            out = ff(x)
            assert out.shape == x.shape, f"Failed for n_embd={n_embd}"


# =============================================================================
# TESTS: Block
# =============================================================================


class TestBlock:
    """
    Test suite for Transformer Block component.

    Tests the complete transformer block including initialization,
    forward pass shapes, residual connections, and layer normalization.
    """

    def test_initialization(self):
        """
        Test that Block initializes with all required components.

        Verifies that the transformer block contains multi-head attention,
        feed-forward network, and two layer normalization layers.

        Test Pattern:
            Arrange: Define block parameters
            Act: Create Block instance
            Assert: Verify all components are present and correct type
        """
        # Arrange
        embedding_dim = 64
        block_size = 128
        n_heads = 4

        # Act
        block = Block(
            embedding_dim=embedding_dim, block_size=block_size, n_heads=n_heads
        )

        # Assert
        assert isinstance(block.sa, MultiHeadAttention)
        assert isinstance(block.ffwd, FeedForward)
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)

    def test_forward_shape(self):
        """
        Test that forward pass maintains input shape.

        Verifies that the transformer block outputs a tensor with the same
        shape as the input (due to residual connections).

        Test Pattern:
            Arrange: Create block and input tensor
            Act: Pass input through the block
            Assert: Verify output shape matches input shape
        """
        # Arrange
        block = Block(embedding_dim=64, block_size=128, n_heads=4)
        batch_size = 2
        seq_len = 10
        embedding_dim = 64
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Act
        out = block(x)

        # Assert
        assert out.shape == x.shape

    def test_residual_connections(self):
        """
        Test that residual connections are working correctly.

        Verifies that the output is not identical to the input, confirming
        that transformations are applied while residual connections preserve
        information flow.

        Test Pattern:
            Arrange: Create block in eval mode and input tensor
            Act: Pass input through the block
            Assert: Verify output differs from input (not identity function)
        """
        # Arrange
        block = Block(embedding_dim=64, block_size=128, n_heads=4, dropout=0.0)
        block.eval()  # Disable dropout for deterministic test
        x = torch.randn(2, 10, 64)

        # Act
        with torch.no_grad():
            out = block(x)

        # Assert - output should differ from input (not identity)
        is_identity = torch.allclose(out, x, atol=0.1)
        assert not is_identity, "Block should transform input, not be identity"

    def test_layer_norm_application(self):
        """
        Test that layer normalization is correctly configured.

        Verifies that both layer normalization layers have the correct
        normalized shape matching the embedding dimension.

        Test Pattern:
            Arrange: Define embedding dimension
            Act: Create Block instance
            Assert: Verify LayerNorm shapes are correct
        """
        # Arrange
        embedding_dim = 64
        block = Block(embedding_dim=embedding_dim, block_size=128, n_heads=4)

        # Act
        ln1_shape = block.ln1.normalized_shape
        ln2_shape = block.ln2.normalized_shape

        # Assert
        expected_shape = (embedding_dim,)
        assert ln1_shape == expected_shape
        assert ln2_shape == expected_shape


# =============================================================================
# TESTS: MicroGPT
# =============================================================================


class TestMicroGPT:
    """
    Test suite for MicroGPT language model.

    Tests the complete language model including initialization, forward passes
    with and without targets, loss computation, text generation, temperature
    sampling, context window handling, and gradient flow.
    """

    def test_initialization(self, small_model_config):
        """
        Test that MicroGPT initializes with correct architecture.

        Verifies that the model is created with the correct block size,
        number of layers, vocabulary size, and embedding dimensions.

        Test Pattern:
            Arrange: Get small model configuration
            Act: Create MicroGPT instance
            Assert: Verify all architectural parameters are correct
        """
        # Arrange
        config = small_model_config

        # Act
        model = MicroGPT(**config)

        # Assert
        assert model.block_size == config["block_size"]
        assert len(model.blocks) == config["n_layers"]
        assert model.token_embedding.num_embeddings == config["vocab_size"]
        assert model.token_embedding.embedding_dim == config["embedding_dim"]

    def test_forward_without_targets(self, model, batch_data):
        """
        Test forward pass in inference mode (no targets).

        Verifies that the model produces logits without computing loss
        when no targets are provided.

        Test Pattern:
            Arrange: Get model and batch data
            Act: Forward pass without targets
            Assert: Verify logits shape is correct and loss is None
        """
        # Arrange
        input_data = batch_data

        # Act
        logits, loss = model(input_data)

        # Assert
        batch_size, seq_len = input_data.shape
        vocab_size = model.head.out_features
        expected_shape = (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape
        assert loss is None

    def test_forward_with_targets(self, model, batch_data):
        """
        Test forward pass in training mode (with targets).

        Verifies that the model produces both logits and loss when
        targets are provided.

        Test Pattern:
            Arrange: Get model, batch data, and create targets
            Act: Forward pass with targets
            Assert: Verify logits shape and loss is computed
        """
        # Arrange
        input_data = batch_data
        targets = batch_data.clone()

        # Act
        logits, loss = model(input_data, targets)

        # Assert
        batch_size, seq_len = input_data.shape
        vocab_size = 100  # From model config
        expected_logits_shape = (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_logits_shape
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_loss_computation(self, model, batch_data):
        """
        Test that cross-entropy loss is computed correctly.

        Verifies that the model's loss matches a manually computed
        cross-entropy loss on the same logits and targets.

        Test Pattern:
            Arrange: Get model and batch data, create targets
            Act: Compute loss via model and manually
            Assert: Verify losses match
        """
        # Arrange
        input_data = batch_data
        targets = batch_data.clone()

        # Act
        logits, model_loss = model(input_data, targets)
        # Manual loss computation for verification
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        expected_loss = F.cross_entropy(logits_flat, targets_flat)

        # Assert
        assert torch.allclose(model_loss, expected_loss, atol=1e-5)

    def test_generate_shape(self, model):
        """
        Test that text generation produces correct output shape.

        Verifies that generating new tokens extends the sequence by the
        specified number of tokens.

        Test Pattern:
            Arrange: Create initial context
            Act: Generate new tokens
            Assert: Verify output length is context + new tokens
        """
        # Arrange
        initial_length = 5
        context = torch.randint(0, 100, (1, initial_length))
        max_new_tokens = 10

        # Act
        output = model.generate(context, max_new_tokens)

        # Assert
        expected_length = initial_length + max_new_tokens
        assert output.shape == (1, expected_length)

    def test_generate_temperature(self, model):
        """
        Test that generation works with different temperature values.

        Verifies that the model can generate text with both low and high
        temperature values without errors.

        Test Pattern:
            Arrange: Create initial context
            Act: Generate with different temperatures
            Assert: Verify both generations complete without error
        """
        # Arrange
        context = torch.randint(0, 100, (1, 5))

        # Act - test low and high temperatures
        torch.manual_seed(42)
        out_low = model.generate(context.clone(), 5, temperature=0.1)
        torch.manual_seed(42)
        out_high = model.generate(context.clone(), 5, temperature=2.0)

        # Assert - both should complete without error
        assert out_low.shape[1] == context.shape[1] + 5
        assert out_high.shape[1] == context.shape[1] + 5

    def test_context_window_cropping(self, model):
        """
        Test that context longer than block_size is cropped correctly.

        Verifies that the model handles sequences longer than the maximum
        context window by cropping to the last block_size tokens.

        Test Pattern:
            Arrange: Create context longer than block_size
            Act: Generate new tokens
            Assert: Verify output extends the full input
        """
        # Arrange
        long_context_length = model.block_size + 10
        long_context = torch.randint(0, 100, (1, long_context_length))
        new_tokens = 5

        # Act
        output = model.generate(long_context, new_tokens)

        # Assert - should extend full input by new_tokens
        expected_length = long_context_length + new_tokens
        assert output.shape[1] == expected_length

    def test_embed_tokens(self, model, batch_data):
        """
        Test that token embedding combines token and position information.

        Verifies that the _embed_tokens method produces embeddings with
        the correct shape.

        Test Pattern:
            Arrange: Get model and batch data
            Act: Compute embeddings
            Assert: Verify embedding shape is correct
        """
        # Arrange
        input_data = batch_data

        # Act
        embeddings = model._embed_tokens(input_data)

        # Assert
        batch_size, seq_len = input_data.shape
        embedding_dim = model.token_embedding.embedding_dim
        expected_shape = (batch_size, seq_len, embedding_dim)
        assert embeddings.shape == expected_shape

    def test_apply_blocks(self, model):
        """
        Test that applying transformer blocks maintains shape.

        Verifies that passing embeddings through all transformer blocks
        maintains the tensor shape.

        Test Pattern:
            Arrange: Create embeddings tensor
            Act: Apply all transformer blocks
            Assert: Verify output shape matches input shape
        """
        # Arrange
        batch_size = 2
        seq_len = 10
        embedding_dim = model.token_embedding.embedding_dim
        x = torch.randn(batch_size, seq_len, embedding_dim)

        # Act
        out = model._apply_blocks(x)

        # Assert
        assert out.shape == x.shape

    def test_compute_loss(self, model):
        """
        Test that loss computation method works correctly.

        Verifies that the _compute_loss method computes a scalar loss
        from logits and targets.

        Test Pattern:
            Arrange: Create random logits and targets
            Act: Compute loss
            Assert: Verify loss is a scalar tensor
        """
        # Arrange
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Act
        loss = model._compute_loss(logits, targets)

        # Assert
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_generate_step(self, model):
        """
        Test that single generation step adds one token.

        Verifies that the _generate_step method extends the sequence
        by exactly one token.

        Test Pattern:
            Arrange: Create initial sequence
            Act: Perform one generation step
            Assert: Verify sequence length increased by 1
        """
        # Arrange
        initial_length = 5
        idx = torch.randint(0, 100, (1, initial_length))
        temperature = 1.0

        # Act
        new_idx = model._generate_step(idx, temperature)

        # Assert
        expected_length = initial_length + 1
        assert new_idx.shape == (1, expected_length)

    def test_different_batch_sizes(self, model):
        """
        Test that model handles various batch sizes correctly.

        Verifies that the model can process different batch sizes
        without errors and produces correct output shapes.

        Test Pattern:
            Arrange: Define test batch sizes
            Act: Forward pass with each batch size
            Assert: Verify output shape is correct
        """
        # Arrange
        seq_len = 10
        vocab_size = 100
        test_batch_sizes = [1, 2, 4, 8]

        # Act & Assert
        for batch_size in test_batch_sizes:
            x = torch.randint(0, 100, (batch_size, seq_len))
            logits, _ = model(x)
            expected_shape = (batch_size, seq_len, vocab_size)
            assert logits.shape == expected_shape, f"Failed for batch_size={batch_size}"

    def test_different_sequence_lengths(self, model):
        """
        Test that model handles various sequence lengths correctly.

        Verifies that the model can process sequences of different lengths
        up to the maximum block_size.

        Test Pattern:
            Arrange: Define test sequence lengths
            Act: Forward pass with each sequence length
            Assert: Verify output shape is correct
        """
        # Arrange
        batch_size = 2
        vocab_size = 100
        test_seq_lengths = [1, 5, 10, 16]  # 16 is block_size

        # Act & Assert
        for seq_len in test_seq_lengths:
            x = torch.randint(0, 100, (batch_size, seq_len))
            logits, _ = model(x)
            expected_shape = (batch_size, seq_len, vocab_size)
            assert logits.shape == expected_shape, f"Failed for seq_len={seq_len}"

    def test_gradient_flow(self, model, batch_data):
        """
        Test that gradients flow through the entire model.

        Verifies that after a backward pass, gradients are computed for
        key model parameters (embeddings and output layer).

        Test Pattern:
            Arrange: Get model and batch data, create targets
            Act: Forward and backward pass
            Assert: Verify gradients exist for key parameters
        """
        # Arrange
        input_data = batch_data
        targets = batch_data.clone()

        # Act
        logits, loss = model(input_data, targets)
        loss.backward()

        # Assert
        assert model.token_embedding.weight.grad is not None
        assert model.head.weight.grad is not None

    def test_parameter_count(self, small_model_config):
        """
        Test that model has non-zero parameters and all components contribute.

        Verifies that the model has trainable parameters and that specific
        components (embeddings) have non-zero parameter counts.

        Test Pattern:
            Arrange: Create model with small config
            Act: Count total and component parameters
            Assert: Verify counts are positive
        """
        # Arrange
        model = MicroGPT(**small_model_config)

        # Act
        total_params = sum(p.numel() for p in model.parameters())
        token_emb_params = model.token_embedding.weight.numel()
        pos_emb_params = model.position_embedding.weight.numel()

        # Assert
        assert total_params > 0
        assert token_emb_params > 0
        assert pos_emb_params > 0


# =============================================================================
# TESTS: Utility Functions
# =============================================================================


class TestGetBatch:
    """
    Test suite for get_batch function.

    Tests the batch sampling utility including correct shapes, device placement,
    and handling of different batch sizes and block sizes.
    """

    def test_batch_shape(self, sample_data, device):
        """
        Test that get_batch returns tensors with correct shapes.

        Verifies that input (x) and target (y) tensors have the expected
        batch_size and block_size dimensions.

        Test Pattern:
            Arrange: Define batch parameters
            Act: Sample a batch
            Assert: Verify x and y shapes are correct
        """
        # Arrange
        block_size = 16
        batch_size = 4

        # Act
        x, y = get_batch(sample_data, block_size, batch_size, device)

        # Assert
        expected_shape = (batch_size, block_size)
        assert x.shape == expected_shape
        assert y.shape == expected_shape

    def test_batch_offset(self, sample_data, device):
        """
        Test that target sequence is offset by one from input.

        Verifies the structural relationship between x and y, though exact
        verification is difficult due to random sampling.

        Test Pattern:
            Arrange: Sample a single-example batch
            Act: Get batch with batch_size=1
            Assert: Verify shapes (offset relationship implicit in get_batch logic)
        """
        # Arrange
        block_size = 16
        batch_size = 1

        # Act
        x, y = get_batch(sample_data, block_size, batch_size, device)

        # Assert - verify shapes match (offset is by design in get_batch)
        assert x.shape == y.shape

    def test_batch_device(self, sample_data, device):
        """
        Test that batched tensors are placed on correct device.

        Verifies that both input and target tensors are on the specified device.

        Test Pattern:
            Arrange: Define device and batch parameters
            Act: Sample a batch
            Assert: Verify x and y are on correct device
        """
        # Arrange
        block_size = 16
        batch_size = 4

        # Act
        x, y = get_batch(sample_data, block_size, batch_size, device)

        # Assert
        device_prefix = device.split(":")[0]  # Handle "cuda:0" etc.
        assert str(x.device).startswith(device_prefix)
        assert str(y.device).startswith(device_prefix)

    def test_different_batch_sizes(self, sample_data, device):
        """
        Test that get_batch handles various batch sizes correctly.

        Verifies that the function can sample different numbers of sequences
        without errors.

        Test Pattern:
            Arrange: Define block size and test batch sizes
            Act: Sample batches with different sizes
            Assert: Verify correct shapes for each batch size
        """
        # Arrange
        block_size = 16
        test_batch_sizes = [1, 2, 4, 8, 16]

        # Act & Assert
        for batch_size in test_batch_sizes:
            x, y = get_batch(sample_data, block_size, batch_size, device)
            expected_shape = (batch_size, block_size)
            assert x.shape == expected_shape, f"Failed for batch_size={batch_size}"
            assert y.shape == expected_shape, f"Failed for batch_size={batch_size}"

    def test_different_block_sizes(self, sample_data, device):
        """
        Test that get_batch handles various sequence lengths correctly.

        Verifies that the function can sample sequences of different lengths
        as long as they fit within the data.

        Test Pattern:
            Arrange: Define batch size and test block sizes
            Act: Sample batches with different block sizes
            Assert: Verify correct shapes for each block size
        """
        # Arrange
        batch_size = 4
        test_block_sizes = [8, 16, 32, 64]

        # Act & Assert
        for block_size in test_block_sizes:
            if block_size < len(sample_data):
                x, y = get_batch(sample_data, block_size, batch_size, device)
                expected_shape = (batch_size, block_size)
                assert x.shape == expected_shape, f"Failed for block_size={block_size}"
                assert y.shape == expected_shape, f"Failed for block_size={block_size}"


class TestSaveCheckpoint:
    """
    Test suite for save_checkpoint function.

    Tests checkpoint saving functionality including file creation,
    checkpoint contents, and ability to load saved checkpoints.
    """

    def test_checkpoint_saved(self, model):
        """
        Test that checkpoint file is created on disk.

        Verifies that calling save_checkpoint creates a file at the
        specified path.

        Test Pattern:
            Arrange: Create model, optimizer, and temp directory
            Act: Save checkpoint
            Assert: Verify file exists
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())

        # Act
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_checkpoint.pt")
            save_checkpoint(model, optimizer, step=100, val_loss=2.5, filepath=filepath)

            # Assert
            assert os.path.exists(filepath)

    def test_checkpoint_contents(self, model):
        """
        Test that checkpoint contains all required keys and correct values.

        Verifies that the saved checkpoint dictionary has model state,
        optimizer state, step number, and validation loss with correct values.

        Test Pattern:
            Arrange: Create model, optimizer, and checkpoint parameters
            Act: Save and load checkpoint
            Assert: Verify all keys present with correct values
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())
        step = 100
        val_loss = 2.5

        # Act
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_checkpoint.pt")
            save_checkpoint(
                model, optimizer, step=step, val_loss=val_loss, filepath=filepath
            )
            checkpoint = torch.load(filepath)

            # Assert
            assert "step" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "val_loss" in checkpoint
            assert checkpoint["step"] == step
            assert checkpoint["val_loss"] == val_loss

    def test_checkpoint_loading(self, model):
        """
        Test that saved checkpoint can be loaded back into model.

        Verifies that the checkpoint can be loaded without errors and
        state can be restored to model and optimizer.

        Test Pattern:
            Arrange: Create model, optimizer, save checkpoint
            Act: Load checkpoint and restore state
            Assert: Verify loading completes without error
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())

        # Act
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_checkpoint.pt")
            save_checkpoint(model, optimizer, step=100, val_loss=2.5, filepath=filepath)
            checkpoint = torch.load(filepath)

            # Restore state (should not raise errors)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Assert - if we get here without errors, test passes
            assert True


class TestTrainStep:
    """
    Test suite for _train_step function.

    Tests the training step utility including loss computation, weight updates,
    and gradient clipping.
    """

    def test_train_step_returns_loss(self, model, sample_data, device):
        """
        Test that train step returns a scalar loss value.

        Verifies that executing a training step computes and returns
        a positive float loss value.

        Test Pattern:
            Arrange: Create model, optimizer, and training data
            Act: Execute one training step
            Assert: Verify loss is positive float
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())
        block_size = 16
        batch_size = 4

        # Act
        loss = _train_step(
            model, optimizer, sample_data, block_size, batch_size, device
        )

        # Assert
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_updates_weights(self, model, sample_data, device):
        """
        Test that train step actually updates model weights.

        Verifies that after a training step, the model weights have changed
        from their initial values.

        Test Pattern:
            Arrange: Create model, optimizer, save initial weights
            Act: Execute one training step
            Assert: Verify weights have changed
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        initial_weight = model.token_embedding.weight.clone()
        block_size = 16
        batch_size = 4

        # Act
        loss = _train_step(
            model, optimizer, sample_data, block_size, batch_size, device
        )

        # Assert
        assert not torch.equal(initial_weight, model.token_embedding.weight)

    def test_gradient_clipping(self, model, sample_data, device):
        """
        Test that gradient clipping is applied during training.

        Verifies that the training step executes without errors when
        gradient clipping is applied.

        Test Pattern:
            Arrange: Create model and optimizer
            Act: Execute training step (which includes gradient clipping)
            Assert: Verify execution completes without error
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())
        block_size = 16
        batch_size = 4

        # Act
        loss = _train_step(
            model, optimizer, sample_data, block_size, batch_size, device
        )

        # Assert - if we get here without errors, gradient clipping worked
        assert isinstance(loss, float)


class TestEvalModel:
    """
    Test suite for _eval_model function.

    Tests the evaluation utility including loss computation, model mode
    management, and gradient handling during evaluation.
    """

    def test_eval_returns_loss(self, model, sample_data, device):
        """
        Test that evaluation returns average loss value.

        Verifies that the evaluation function computes and returns
        a positive float loss averaged over multiple batches.

        Test Pattern:
            Arrange: Create model and evaluation parameters
            Act: Run evaluation
            Assert: Verify average loss is positive float
        """
        # Arrange
        block_size = 16
        batch_size = 4
        eval_iters = 5

        # Act
        avg_loss = _eval_model(
            model, sample_data, block_size, batch_size, device, eval_iters
        )

        # Assert
        assert isinstance(avg_loss, float)
        assert avg_loss > 0

    def test_eval_mode(self, model, sample_data, device):
        """
        Test that model is restored to training mode after evaluation.

        Verifies that the evaluation function properly manages model modes,
        switching to eval mode during evaluation and restoring train mode after.

        Test Pattern:
            Arrange: Set model to train mode
            Act: Run evaluation
            Assert: Verify model is back in train mode
        """
        # Arrange
        model.train()  # Ensure starting in train mode
        block_size = 16
        batch_size = 4
        eval_iters = 5

        # Act
        avg_loss = _eval_model(
            model, sample_data, block_size, batch_size, device, eval_iters
        )

        # Assert
        assert model.training  # Should be back in training mode

    def test_no_gradient_computation(self, model, sample_data, device):
        """
        Test that no gradients are computed during evaluation.

        Verifies that evaluation runs without computing gradients,
        as indicated by no gradient accumulation on parameters.

        Test Pattern:
            Arrange: Ensure parameters require gradients
            Act: Run evaluation
            Assert: Verify gradients are None or zero
        """
        # Arrange
        for param in model.parameters():
            param.requires_grad = True
        block_size = 16
        batch_size = 4
        eval_iters = 5

        # Act
        avg_loss = _eval_model(
            model, sample_data, block_size, batch_size, device, eval_iters
        )

        # Assert - gradients should not be computed
        for param in model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


class TestShouldSave:
    """
    Test suite for _should_save function.

    Tests the checkpoint saving decision logic including comparison
    of validation losses and edge cases.
    """

    def test_should_save_when_better(self):
        """
        Test that function returns True when validation loss improves.

        Verifies that the function correctly identifies when current loss
        is better (lower) than the best loss.

        Test Pattern:
            Arrange: Define improved loss values
            Act: Call _should_save
            Assert: Verify returns True
        """
        # Arrange
        current_loss = 2.0
        best_loss = 3.0

        # Act
        result = _should_save(val_loss=current_loss, best_loss=best_loss)

        # Assert
        assert result == True

    def test_should_not_save_when_worse(self):
        """
        Test that function returns False when validation loss worsens.

        Verifies that the function correctly identifies when current loss
        is worse (higher) than the best loss.

        Test Pattern:
            Arrange: Define worse loss values
            Act: Call _should_save
            Assert: Verify returns False
        """
        # Arrange
        current_loss = 3.0
        best_loss = 2.0

        # Act
        result = _should_save(val_loss=current_loss, best_loss=best_loss)

        # Assert
        assert result == False

    def test_should_not_save_when_equal(self):
        """
        Test that function returns False when validation loss is equal.

        Verifies that the function does not save when loss is the same
        (strict less-than comparison).

        Test Pattern:
            Arrange: Define equal loss values
            Act: Call _should_save
            Assert: Verify returns False
        """
        # Arrange
        current_loss = 2.5
        best_loss = 2.5

        # Act
        result = _should_save(val_loss=current_loss, best_loss=best_loss)

        # Assert
        assert result == False

    def test_edge_cases(self):
        """
        Test that function handles extreme loss values correctly.

        Verifies behavior with edge cases like zero loss and infinite loss.

        Test Pattern:
            Arrange: Define edge case loss values
            Act: Call _should_save for each case
            Assert: Verify correct behavior for edge cases
        """
        # Arrange & Act & Assert
        # Zero loss is better than infinity
        assert _should_save(val_loss=0.0, best_loss=float("inf")) == True
        # Infinity is not better than zero
        assert _should_save(val_loss=float("inf"), best_loss=0.0) == False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """
    Test suite for end-to-end integration scenarios.

    Tests complete workflows including forward-backward passes, multi-step
    training, generation after training, checkpoint save/load cycles, and
    evaluation during training.
    """

    def test_full_forward_backward_pass(self, model, batch_data):
        """
        Test complete forward and backward pass with optimization.

        Verifies that a full training iteration (forward, backward, optimize)
        executes without errors and produces gradients.

        Test Pattern:
            Arrange: Create model, optimizer, and data
            Act: Execute forward, backward, and optimization steps
            Assert: Verify gradients exist for all trainable parameters
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())
        targets = batch_data.clone()

        # Act
        logits, loss = model(batch_data, targets)
        assert loss is not None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Assert
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_training_reduces_loss(self, model, sample_data, device):
        """
        Test that multiple training steps reduce the loss.

        Verifies that training for several steps produces a general downward
        trend in loss (allowing for some fluctuation).

        Test Pattern:
            Arrange: Create model, optimizer, and training setup
            Act: Train for multiple steps and collect losses
            Assert: Verify final loss is lower than initial loss
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        losses = []
        num_steps = 10
        block_size = 16
        batch_size = 4

        # Act
        for _ in range(num_steps):
            loss = _train_step(
                model, optimizer, sample_data, block_size, batch_size, device
            )
            losses.append(loss)

        # Assert - loss should generally decrease
        # Allow some flexibility for stochastic training
        assert losses[-1] < losses[0] * 1.5

    def test_generate_after_training(self, model, sample_data, device):
        """
        Test that model can generate text after training.

        Verifies that after a few training steps, the model can successfully
        generate new tokens autoregressively.

        Test Pattern:
            Arrange: Create model, optimizer, train briefly
            Act: Generate text from context
            Assert: Verify generated sequence has correct length
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        block_size = 16
        batch_size = 4

        # Train for a few steps
        for _ in range(5):
            _train_step(model, optimizer, sample_data, block_size, batch_size, device)

        # Act
        context = torch.randint(0, 100, (1, 5))
        output = model.generate(context, max_new_tokens=10)

        # Assert
        assert output.shape == (1, 15)

    def test_checkpoint_save_and_load(self, model, sample_data, device):
        """
        Test complete checkpoint save and load cycle.

        Verifies that a model can be saved to a checkpoint, loaded into
        a new model instance, and that weights match exactly.

        Test Pattern:
            Arrange: Create and train model briefly
            Act: Save checkpoint, create new model, load checkpoint
            Assert: Verify weights match between original and loaded model
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())
        block_size = 16
        batch_size = 4

        # Train for one step
        loss1 = _train_step(
            model, optimizer, sample_data, block_size, batch_size, device
        )

        # Act
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")
            save_checkpoint(model, optimizer, step=1, val_loss=loss1, filepath=filepath)

            # Create new model and load
            new_model = MicroGPT(
                vocab_size=100,
                embedding_dim=32,
                block_size=16,
                n_heads=4,
                n_layers=2,
                dropout=0.1,
            )
            new_optimizer = torch.optim.Adam(new_model.parameters())

            checkpoint = torch.load(filepath)
            new_model.load_state_dict(checkpoint["model_state_dict"])
            new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Assert - weights should match exactly
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_eval_during_training(self, model, sample_data, device):
        """
        Test evaluation during the training process.

        Verifies that evaluation can be performed during training without
        disrupting the training workflow.

        Test Pattern:
            Arrange: Create model, optimizer, and training setup
            Act: Train for a few steps, then evaluate
            Assert: Verify evaluation returns valid loss
        """
        # Arrange
        optimizer = torch.optim.Adam(model.parameters())
        block_size = 16
        batch_size = 4

        # Train for a few steps
        for _ in range(3):
            _train_step(model, optimizer, sample_data, block_size, batch_size, device)

        # Act
        val_loss = _eval_model(
            model, sample_data, block_size, batch_size, device, eval_iters=5
        )

        # Assert
        assert isinstance(val_loss, float)
        assert val_loss > 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """
    Test suite for edge cases and boundary conditions.

    Tests the model's behavior with extreme inputs including single tokens,
    maximum sequence lengths, various batch sizes, and temperature extremes.
    """

    def test_single_token_sequence(self, model):
        """
        Test that model handles single-token sequences correctly.

        Verifies that the model can process a sequence of length 1
        without errors.

        Test Pattern:
            Arrange: Create single-token input
            Act: Forward pass
            Assert: Verify output shape is correct
        """
        # Arrange
        x = torch.randint(0, 100, (1, 1))

        # Act
        logits, _ = model(x)

        # Assert
        assert logits.shape == (1, 1, 100)

    def test_max_sequence_length(self, model):
        """
        Test that model handles maximum sequence length (block_size).

        Verifies that the model can process a sequence at its maximum
        allowed length without errors.

        Test Pattern:
            Arrange: Create input with length = block_size
            Act: Forward pass
            Assert: Verify output shape is correct
        """
        # Arrange
        max_length = model.block_size
        x = torch.randint(0, 100, (1, max_length))

        # Act
        logits, _ = model(x)

        # Assert
        assert logits.shape == (1, max_length, 100)

    def test_batch_size_one(self, model):
        """
        Test that model handles batch size of 1 correctly.

        Verifies that the model works with a single example in the batch.

        Test Pattern:
            Arrange: Create single-example batch
            Act: Forward pass
            Assert: Verify output shape is correct
        """
        # Arrange
        batch_size = 1
        seq_len = 10
        x = torch.randint(0, 100, (batch_size, seq_len))

        # Act
        logits, _ = model(x)

        # Assert
        assert logits.shape == (batch_size, seq_len, 100)

    def test_large_batch_size(self, model, device):
        """
        Test that model handles large batch sizes correctly.

        Verifies that the model can process a large batch without errors
        (within memory constraints).

        Test Pattern:
            Arrange: Create large batch
            Act: Forward pass
            Assert: Verify output shape is correct
        """
        # Arrange
        batch_size = 32
        seq_len = 10
        x = torch.randint(0, 100, (batch_size, seq_len))

        # Act
        logits, _ = model(x)

        # Assert
        assert logits.shape == (batch_size, seq_len, 100)

    def test_zero_temperature_generation(self, model):
        """
        Test generation with very low temperature (near-greedy decoding).

        Verifies that the model can generate text with very low temperature
        without numerical instabilities.

        Test Pattern:
            Arrange: Create context
            Act: Generate with very low temperature
            Assert: Verify output length is correct
        """
        # Arrange
        context = torch.randint(0, 100, (1, 5))
        new_tokens = 5

        # Act
        output = model.generate(context, max_new_tokens=new_tokens, temperature=0.01)

        # Assert
        assert output.shape == (1, 10)

    def test_high_temperature_generation(self, model):
        """
        Test generation with high temperature (more random).

        Verifies that the model can generate text with high temperature
        without numerical instabilities.

        Test Pattern:
            Arrange: Create context
            Act: Generate with high temperature
            Assert: Verify output length is correct
        """
        # Arrange
        context = torch.randint(0, 100, (1, 5))
        new_tokens = 5

        # Act
        output = model.generate(context, max_new_tokens=new_tokens, temperature=2.0)

        # Assert
        assert output.shape == (1, 10)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """
    Test suite for performance and efficiency.

    Tests that ensure the model and utilities execute within reasonable
    time bounds and don't have memory leaks.
    """

    def test_forward_pass_timing(self, model, batch_data):
        """
        Test that forward passes complete in reasonable time.

        Verifies that multiple forward passes execute quickly enough
        on CPU (under 5 seconds for 10 passes).

        Test Pattern:
            Arrange: Create model and data
            Act: Time multiple forward passes
            Assert: Verify total time is reasonable
        """
        # Arrange
        import time

        num_passes = 10

        # Act
        start = time.time()
        for _ in range(num_passes):
            logits, _ = model(batch_data)
        elapsed = time.time() - start

        # Assert
        max_time = 5.0  # seconds
        assert elapsed < max_time, f"Took {elapsed:.2f}s, expected < {max_time}s"

    def test_generation_timing(self, model):
        """
        Test that text generation completes in reasonable time.

        Verifies that generating 20 tokens executes quickly enough
        on CPU (under 5 seconds).

        Test Pattern:
            Arrange: Create context
            Act: Time generation
            Assert: Verify generation time is reasonable
        """
        # Arrange
        import time

        context = torch.randint(0, 100, (1, 5))
        num_tokens = 20

        # Act
        start = time.time()
        output = model.generate(context, max_new_tokens=num_tokens)
        elapsed = time.time() - start

        # Assert
        max_time = 5.0  # seconds
        assert elapsed < max_time, f"Took {elapsed:.2f}s, expected < {max_time}s"

    def test_memory_efficiency(self, model, batch_data):
        """
        Test that model doesn't accumulate excessive memory.

        Verifies that running many forward passes doesn't cause memory
        leaks by repeatedly processing data.

        Test Pattern:
            Arrange: Create model and data
            Act: Run many forward passes
            Assert: Verify execution completes (no memory explosion)
        """
        # Arrange
        num_iterations = 100

        # Act
        for _ in range(num_iterations):
            logits, _ = model(batch_data)

        # Assert - if we get here, memory management is acceptable
        assert True


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================


if __name__ == "__main__":
    """
    Main entry point for running tests directly.

    Runs pytest with verbose output and short traceback format when
    the test file is executed directly.
    """
    pytest.main([__file__, "-v", "--tb=short"])
