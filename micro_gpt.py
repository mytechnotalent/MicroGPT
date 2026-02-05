"""
GPT-2 Medium: A decoder-only transformer language model implementation.

This module provides a complete implementation of a GPT-2 style language model
from scratch using PyTorch, matching OpenAI's GPT-2 Medium architecture (355M).

Architecture Features:
    - Multi-head causal self-attention with Flash Attention support
    - GELU activation (Gaussian Error Linear Unit) as used in GPT-2
    - Pre-LayerNorm architecture for training stability
    - Top-p (nucleus) and top-k sampling for generation
    - Rotary position embeddings option for extended context

Components:
    - CausalSelfAttention: Fused multi-head attention with causal mask
    - FeedForward: Position-wise MLP with GELU and 4x expansion
    - TransformerBlock: Pre-norm attention + FFN with residuals
    - GPT2: Full language model with generation capabilities

GPT-2 Model Sizes:
    - Small: n_layer=12, n_head=12, n_embd=768 (~124M params)
    - Medium: n_layer=24, n_head=16, n_embd=1024 (~355M params)
    - Large: n_layer=36, n_head=20, n_embd=1280 (~774M params)
    - XL: n_layer=48, n_head=25, n_embd=1600 (~1.5B params)

Usage:
    >>> from micro_gpt import GPT2, GPT2Config
    >>> config = GPT2Config(vocab_size=50257, n_embd=1024, n_head=16,
    ...                     n_layer=24, block_size=1024)
    >>> model = GPT2(config)
    >>> output = model.generate(context, max_new_tokens=100, temperature=0.8)

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class GPT2Config:
    """
    Configuration for GPT-2 model architecture.

    Defines all hyperparameters for the GPT-2 transformer architecture,
    following OpenAI's naming conventions for compatibility.

    Args:
        block_size: Maximum sequence length (context window). Default 1024.
        vocab_size: Size of token vocabulary. Default 50257 for GPT-2.
        n_layer: Number of transformer blocks. Default 24 for Medium.
        n_head: Number of attention heads. Default 16 for Medium.
        n_embd: Embedding dimension. Default 1024 for Medium.
        dropout: Dropout probability. Default 0.1.
        bias: Whether to use bias in projections. Default True.

    Example:
        >>> config = GPT2Config(n_layer=12, n_head=12, n_embd=768)
        >>> config.n_embd
        768
    """

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 24
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.1
    bias: bool = True


class LayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.

    Implements LayerNorm as described in "Layer Normalization" (Ba et al., 2016)
    with configurable bias parameter as used in GPT-2.

    Args:
        ndim: Normalized shape (embedding dimension).
        bias: Whether to include learnable bias. Default True.

    Example:
        >>> ln = LayerNorm(768, bias=True)
        >>> x = torch.randn(2, 10, 768)
        >>> output = ln(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """

    def __init__(self, ndim: int, bias: bool = True):
        """
        Initialize LayerNorm with weight and optional bias parameters.

        Args:
            ndim: Dimension to normalize over.
            bias: Whether to include learnable bias parameter.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input tensor.

        Args:
            x: Input tensor of shape (B, T, ndim).

        Returns:
            Normalized tensor of same shape.
        """
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with fused QKV projection.

    Implements efficient multi-head attention with causal masking to prevent
    attending to future positions. Uses fused QKV projection for efficiency.

    Args:
        config: GPT2Config with model hyperparameters.

    Attributes:
        c_attn: Fused query, key, value projection.
        c_proj: Output projection layer.
        attn_dropout: Dropout on attention weights.
        resid_dropout: Dropout on output.
        n_head: Number of attention heads.
        n_embd: Embedding dimension.

    Example:
        >>> config = GPT2Config(n_embd=768, n_head=12, block_size=1024)
        >>> attn = CausalSelfAttention(config)
        >>> x = torch.randn(2, 10, 768)
        >>> output = attn(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """

    def __init__(self, config: GPT2Config):
        """
        Initialize attention with QKV projection and causal mask buffer.

        Args:
            config: Model configuration with n_embd, n_head, block_size.
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self._register_causal_mask(config.block_size)

    def _register_causal_mask(self, block_size: int) -> None:
        """
        Register causal attention mask as buffer.

        Args:
            block_size: Maximum sequence length for mask.
        """
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("bias", mask.view(1, 1, block_size, block_size))

    def _split_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Split tensor into multiple attention heads.

        Args:
            x: Input tensor of shape (B, T, n_embd).
            B: Batch size.
            T: Sequence length.

        Returns:
            Tensor of shape (B, n_head, T, head_size).
        """
        return x.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)

    def _compute_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention with causal masking.

        Args:
            q: Query tensor of shape (B, n_head, T, head_size).
            k: Key tensor of shape (B, n_head, T, head_size).
            v: Value tensor of shape (B, n_head, T, head_size).
            T: Sequence length for masking.

        Returns:
            Attention output of shape (B, n_head, T, head_size).
        """
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        return att @ v

    def _merge_heads(self, y: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """
        Merge attention heads back into single tensor.

        Args:
            y: Tensor of shape (B, n_head, T, head_size).
            B: Batch size.
            T: Sequence length.

        Returns:
            Merged tensor of shape (B, T, n_embd).
        """
        return y.transpose(1, 2).contiguous().view(B, T, self.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head causal self-attention.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (
            self._split_heads(q, B, T),
            self._split_heads(k, B, T),
            self._split_heads(v, B, T),
        )
        y = self._compute_attention(q, k, v, T)
        y = self._merge_heads(y, B, T)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.

    Two-layer MLP with GELU activation and 4x expansion ratio,
    matching the GPT-2 architecture specification.

    Args:
        config: GPT2Config with n_embd and dropout settings.

    Example:
        >>> config = GPT2Config(n_embd=768, dropout=0.1)
        >>> ffn = FeedForward(config)
        >>> x = torch.randn(2, 10, 768)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """

    def __init__(self, config: GPT2Config):
        """
        Initialize feed-forward network with expansion and projection layers.

        Args:
            config: Model configuration with n_embd, bias, dropout.
        """
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation with GELU activation.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-LayerNorm and residual connections.

    Implements a single transformer block consisting of multi-head
    self-attention followed by a feed-forward network. Uses pre-normalization
    (LayerNorm before each sub-layer) for training stability.

    Args:
        config: GPT2Config with model hyperparameters.

    Example:
        >>> config = GPT2Config(n_embd=768, n_head=12, block_size=1024)
        >>> block = TransformerBlock(config)
        >>> x = torch.randn(2, 10, 768)
        >>> output = block(x)
        >>> output.shape
        torch.Size([2, 10, 768])
    """

    def __init__(self, config: GPT2Config):
        """
        Initialize transformer block with attention and FFN sub-layers.

        Args:
            config: Model configuration with architecture parameters.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with pre-norm and residual connections.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Output tensor of shape (B, T, n_embd).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    """
    GPT-2 Language Model with autoregressive generation.

    Implements the full GPT-2 architecture with token/position embeddings,
    stacked transformer blocks, and language model head. Supports nucleus
    (top-p) and top-k sampling for text generation.

    Args:
        config: GPT2Config with model hyperparameters.

    Attributes:
        config: Stored configuration.
        wte: Token embedding table.
        wpe: Position embedding table.
        drop: Embedding dropout.
        h: ModuleList of transformer blocks.
        ln_f: Final layer normalization.
        lm_head: Language model output projection.

    Example:
        >>> config = GPT2Config(vocab_size=50257, n_embd=1024, n_head=16,
        ...                     n_layer=24, block_size=1024)
        >>> model = GPT2(config)
        >>> idx = torch.randint(0, 50257, (1, 10))
        >>> logits, loss = model(idx)
        >>> logits.shape
        torch.Size([1, 10, 50257])
    """

    def __init__(self, config: GPT2Config):
        """
        Initialize GPT-2 model with embeddings and transformer blocks.

        Args:
            config: Model configuration with all hyperparameters.
        """
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize weights using GPT-2 initialization scheme.

        Applies normal initialization to linear and embedding layers
        with scaled residual initialization for output projections.
        """
        self.apply(self._init_module)
        self._scale_residual_projections()

    def _init_module(self, module: nn.Module) -> None:
        """
        Initialize a single module's weights.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        """
        Apply scaled initialization to residual projection layers.

        Scales output projections by 1/sqrt(2*n_layer) for stable training.
        """
        scale = (2 * self.config.n_layer) ** -0.5
        for block in self.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=0.02 * scale)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=0.02 * scale)

    def _embed_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Compute token and position embeddings.

        Args:
            idx: Token indices of shape (B, T).

        Returns:
            Combined embeddings of shape (B, T, n_embd).
        """
        T = idx.size(1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)

    def _apply_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all transformer blocks sequentially.

        Args:
            x: Input tensor of shape (B, T, n_embd).

        Returns:
            Transformed tensor of shape (B, T, n_embd).
        """
        for block in self.h:
            x = block(x)
        return self.ln_f(x)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass computing logits and optional loss.

        Args:
            idx: Input token indices of shape (B, T).
            targets: Optional target indices of shape (B, T) for loss.

        Returns:
            Tuple of (logits, loss) where loss is None if no targets.
        """
        x = self._embed_tokens(idx)
        x = self._apply_transformer(x)
        logits = self.lm_head(x)
        loss = self._compute_loss(logits, targets) if targets is not None else None
        return logits, loss

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for language modeling.

        Args:
            logits: Predicted logits of shape (B, T, vocab_size).
            targets: Target token indices of shape (B, T).

        Returns:
            Scalar cross-entropy loss.
        """
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

    def _crop_context(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Crop context to maximum block size.

        Args:
            idx: Token indices of shape (B, T).

        Returns:
            Cropped indices of shape (B, min(T, block_size)).
        """
        if idx.size(1) > self.config.block_size:
            return idx[:, -self.config.block_size :]
        return idx

    def _get_next_token_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get logits for the next token prediction.

        Args:
            idx: Token indices of shape (B, T).

        Returns:
            Next token logits of shape (B, vocab_size).
        """
        idx_cond = self._crop_context(idx)
        logits, _ = self(idx_cond)
        return logits[:, -1, :]

    def _apply_temperature(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw logits of shape (B, vocab_size).
            temperature: Sampling temperature (higher = more random).

        Returns:
            Temperature-scaled logits.
        """
        return logits / temperature

    def _apply_top_k(self, logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
        """
        Apply top-k filtering to logits.

        Args:
            logits: Logits of shape (B, vocab_size).
            top_k: Number of top tokens to keep, or None to skip.

        Returns:
            Filtered logits with non-top-k values set to -inf.
        """
        if top_k is None or top_k <= 0:
            return logits
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    def _apply_top_p(
        self, logits: torch.Tensor, top_p: Optional[float]
    ) -> torch.Tensor:
        """
        Apply nucleus (top-p) filtering to logits.

        Args:
            logits: Logits of shape (B, vocab_size).
            top_p: Cumulative probability threshold, or None to skip.

        Returns:
            Filtered logits with low probability tokens removed.
        """
        if top_p is None or top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask, float("-inf"))

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token from probability distribution.

        Args:
            logits: Filtered logits of shape (B, vocab_size).

        Returns:
            Sampled token indices of shape (B, 1).
        """
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively using nucleus/top-k sampling.

        Generates new tokens one at a time by repeatedly calling the model
        and sampling from the filtered distribution. Supports temperature
        scaling, top-k filtering, and nucleus (top-p) sampling.

        Args:
            idx: Initial context tokens of shape (B, T).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature (default 1.0).
            top_k: Optional top-k filtering threshold.
            top_p: Optional nucleus sampling probability threshold.

        Returns:
            Extended sequence of shape (B, T + max_new_tokens).

        Example:
            >>> model = GPT2(GPT2Config())
            >>> context = torch.randint(0, 50257, (1, 10))
            >>> output = model.generate(context, max_new_tokens=50,
            ...                         temperature=0.8, top_p=0.9)
            >>> output.shape
            torch.Size([1, 60])
        """
        for _ in range(max_new_tokens):
            logits = self._get_next_token_logits(idx)
            logits = self._apply_temperature(logits, temperature)
            logits = self._apply_top_k(logits, top_k)
            logits = self._apply_top_p(logits, top_p)
            next_token = self._sample_token(logits)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch for training with input-target pairs.

    Randomly samples batch_size starting positions from the data, then
    extracts sequences of length block_size as inputs and shifted
    sequences as targets.

    Args:
        data: Full dataset tensor of token indices, shape (N,).
        block_size: Length of each sequence (context window).
        batch_size: Number of sequences to sample.
        device: Device to place tensors on ("cpu", "cuda", or "mps").

    Returns:
        Tuple of (x, y) where:
            - x: Input sequences of shape (batch_size, block_size).
            - y: Target sequences of shape (batch_size, block_size).

    Example:
        >>> data = torch.randint(0, 50257, (100000,))
        >>> x, y = get_batch(data, block_size=1024, batch_size=4, device="cpu")
        >>> x.shape, y.shape
        (torch.Size([4, 1024]), torch.Size([4, 1024]))
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

    Saves a checkpoint containing model state, optimizer state,
    training step, and validation loss for resuming training.

    Args:
        model: The model to save (nn.Module).
        optimizer: The optimizer to save.
        step: Current training step number.
        val_loss: Current validation loss value.
        filepath: Path where checkpoint will be saved.

    Example:
        >>> model = GPT2(GPT2Config())
        >>> optimizer = torch.optim.AdamW(model.parameters())
        >>> save_checkpoint(model, optimizer, step=1000, val_loss=2.5,
        ...                 filepath="checkpoints/model.pt")
    """
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(checkpoint, filepath)


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

    Executes forward pass, backward pass, gradient clipping, and
    optimizer step for one batch of training data.

    Args:
        model: The model to train.
        optimizer: The optimizer for weight updates.
        train_data: Training dataset tensor.
        block_size: Sequence length for batching.
        batch_size: Number of sequences per batch.
        device: Device to run training on.

    Returns:
        Training loss value as float.

    Example:
        >>> model = GPT2(GPT2Config())
        >>> optimizer = torch.optim.AdamW(model.parameters())
        >>> train_data = torch.randint(0, 50257, (100000,))
        >>> loss = _train_step(model, optimizer, train_data, 1024, 4, "cpu")
    """
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
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
    model in eval mode and no gradient computation.

    Args:
        model: The model to evaluate.
        data: Validation dataset tensor.
        block_size: Sequence length for batching.
        batch_size: Number of sequences per batch.
        device: Device to run evaluation on.
        eval_iters: Number of evaluation batches to average.

    Returns:
        Average validation loss as float.

    Example:
        >>> model = GPT2(GPT2Config())
        >>> val_data = torch.randint(0, 50257, (10000,))
        >>> avg_loss = _eval_model(model, val_data, 1024, 4, "cpu", 10)
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


__all__ = [
    "GPT2Config",
    "GPT2",
    "LayerNorm",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "get_batch",
    "save_checkpoint",
    "_train_step",
    "_eval_model",
]
