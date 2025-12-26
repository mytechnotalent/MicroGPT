# MicroGPT

A minimal GPT implementation from scratch in PyTorch that learns to predict the next word in a sequence using self-attention and transformer blocks.


### [dataset](https://www.kaggle.com/datasets/mytechnotalent/mary-had-a-little-lamb)

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: MIT


## Chain Rule of Probability

The joint probability of variables $x_1,\dots,x_n$ can be decomposed as:

$$
P(x_1,\dots,x_n) = \prod_{i=1}^n P\big(x_i \mid x_1,\dots,x_{i-1}\big).
$$

$$
P(w_1, w_2, \ldots, w_n) = P(w_1)\times P(w_2\mid w_1)\times P(w_3\mid w_1,w_2)\times \cdots \times P(w_n\mid w_1,\ldots,w_{n-1})
$$



### 1st Iteration

Given:

$$
\text{Mary}
$$

Predict:

$$ 
\text{Mary had}
$$


### 2nd Iteration

Given:

$$
\text{Mary had}
$$

Predict:

$$ 
\text{Mary had a}
$$


### 3rd Iteration

Given:

$$
\text{Mary had a}
$$

Predict:

$$ 
\text{Mary had a little}
$$


### 4th Iteration w/ Probs

$$
P(\text{lamb} \mid \text{Mary had a little}) = .8
$$

$$
P(\text{dog} \mid \text{Mary had a little}) = .1
$$

$$
P(\text{cat} \mid \text{Mary had a little}) = .1
$$


## Transformer Blocks




```python
%pip install torch
```

**Output:**
```
Requirement already satisfied: torch in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (2.9.1)
Requirement already satisfied: filelock in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (3.20.1)
Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (4.15.0)
Requirement already satisfied: sympy>=1.13.3 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (3.6.1)
Requirement already satisfied: jinja2 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (2025.10.0)
Requirement already satisfied: setuptools in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from torch) (80.9.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\assem.kevinthomas\onedrive\documents\data-science\gpt\venv\lib\site-packages (from jinja2->torch) (3.0.3)
Note: you may need to restart the kernel to use updated packages.
```
```

[notice] A new release of pip is available: 25.1.1 -> 25.3
[notice] To update, run: python.exe -m pip install --upgrade pip
```


```python
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
class SelfAttentionHead(nn.Module):
    """A single causal self-attention head.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the input embeddings (C).
    block_size : int
        Maximum sequence length supported (used to build a causal mask).
    head_size : int
        Dimensionality of keys/queries/values for this head.

    Shapes
    ------
    - Input: (B, T, C)
    - Output: (B, T, head_size)

    Notes
    -----
    - The module registers a lower-triangular mask so it can apply causal
      attention (tokens cannot attend to future tokens).
    - Scaling uses the key dimensionality for numerical stability.
    """

    def __init__(self, embedding_dim: int, block_size: int, head_size: int):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal self-attention for one head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, head_size) after applying attention to values.
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Scale by sqrt(head_dim) for numerical stability
        wei = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        # Weighted sum of values
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multi-head attention by concatenating multiple `SelfAttentionHead`s and
    projecting back to the model `embedding_dim`.

    Parameters
    ----------
    embedding_dim : int
        Model embedding dimensionality (C).
    block_size : int
        Maximum sequence length supported (used to create causal masks in heads).
    num_heads : int
        Number of attention heads. `embedding_dim` must be divisible by `num_heads`.

    Shapes
    ------
    - Input: (B, T, C)
    - Output: (B, T, C)
    """

    def __init__(self, embedding_dim: int, block_size: int, num_heads: int):
        super().__init__()
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, block_size, head_size)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, C).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Simple two-layer feed-forward network used inside transformer blocks.

    Uses a 4x expansion on the embedding dim by default (following common practice).

    Parameters
    ----------
    n_embd : int
        Embedding dimensionality (C).

    Shapes
    ------
    - Input: (B, T, C)
    - Output: (B, T, C)
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP element-wise across sequence positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, C).
        """
        return self.net(x)


class Block(nn.Module):
    """Transformer block: multi-head self-attention followed by a feed-forward MLP.

    Each sub-layer has a pre-layer-norm + residual connection:

        x = x + self_attention(LayerNorm(x))
        x = x + feed_forward(LayerNorm(x))

    Parameters
    ----------
    embedding_dim : int
        Embedding dimensionality (C).
    block_size : int
        Maximum sequence length supported (passed to attention heads).
    n_heads : int
        Number of attention heads.

    Shapes
    ------
    - Input: (B, T, C)
    - Output: (B, T, C)
    """

    def __init__(self, embedding_dim: int, block_size: int, n_heads: int):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer block on input tensor `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, T, C).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, T, C) after attention + feed-forward with residuals.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    """A small GPT-like model for demonstration and toy datasets.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary (V). Used for the embedding and
        the final linear head.
    embedding_dim : int
        Dimensionality of token and position embeddings (C).
    block_size : int
        Maximum context length (T). The model uses position embeddings up to
        this length and generation is constrained to the last `block_size`
        tokens.
    n_heads : int
        Number of attention heads in each transformer block.
    n_layers : int
        Number of transformer `Block` layers.

    Shapes
    ------
    - Input: (B, T) token indices
    - Output (forward): logits (B, T, V) and (optionally) scalar loss

    Notes
    -----
    - This implements a minimal forward pass: token + position embeddings,
      stacked transformer blocks, final LayerNorm, and linear head producing
      logits for next-token prediction.
    - Generation is done autoregressively using the model's learned
      distribution over the vocabulary.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, block_size: int, n_heads: int, n_layers: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.LongTensor, targets: torch.LongTensor = None):
        """Compute logits and (optionally) cross-entropy loss for next-token prediction.

        Parameters
        ----------
        idx : torch.LongTensor, shape (B, T)
            Input token indices.
        targets : torch.LongTensor, optional, shape (B, T)
            Target token indices to compute cross-entropy loss. If `None`, no
            loss is returned.

        Returns
        -------
        logits : torch.Tensor, shape (B, T, V)
            Unnormalized log-probabilities for each token in the vocabulary.
        loss : torch.Tensor or None
            Cross-entropy loss reduced over the batch and sequence dimensions
            if `targets` is provided; otherwise `None`.
        """
        B, T = idx.shape
        # Token + position embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        # Pass through transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx: torch.LongTensor, max_new_tokens: int):
        """Autoregressively generate new tokens given a starting `idx`.

        Parameters
        ----------
        idx : torch.LongTensor, shape (B, T0)
            Conditioning token indices. Only the last `block_size` tokens are
            used at each generation step.
        max_new_tokens : int
            Number of tokens to generate.

        Returns
        -------
        idx : torch.LongTensor
            The input tensor concatenated with `max_new_tokens` new token indices.
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


def get_batch(data: torch.Tensor, block_size: int, batch_size: int = 16):
    """Return a random batch of (x, y) pairs for training.

    Parameters
    ----------
    data : torch.Tensor
        The full dataset as a 1D tensor of token indices.
    block_size : int
        Context window size (sequence length).
    batch_size : int
        Number of sequences to return in the batch.

    Returns
    -------
    x : torch.LongTensor, shape (B, T)
        Input token indices.
    y : torch.LongTensor, shape (B, T)
        Target token indices (next-token prediction).
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def load_corpus(path: str, end_token: str = "<END>", val_split: float = 0.1):
    """Load a JSON corpus and prepare it for training.

    Parameters
    ----------
    path : str
        Path to the JSON file containing a list of sentences.
    end_token : str
        Token to append to each sentence (default: "<END>").
    val_split : float
        Fraction of data to use for validation (default: 0.1).

    Returns
    -------
    train_data : torch.LongTensor
        1D tensor of token indices for training.
    val_data : torch.LongTensor
        1D tensor of token indices for validation.
    word2idx : dict
        Mapping from words to indices.
    idx2word : dict
        Mapping from indices to words.
    corpus : list
        List of sentences with end tokens appended.
    vocab_size : int
        Number of unique words in the vocabulary.
    """
    with open(path, "r") as f:
        corpus = json.load(f)
    # Add end token to each sentence
    corpus = [s.strip() + " " + end_token for s in corpus]
    # Build vocabulary (all unique words)
    all_text = " ".join(corpus)
    words = list(set(all_text.split()))
    vocab_size = len(words)
    # Create word <-> index mappings
    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for w, i in word2idx.items()}
    # Convert text to token indices
    data = torch.tensor([word2idx[w] for w in all_text.split()], dtype=torch.long)
    # Split into train and val
    n = int(len(data) * (1 - val_split))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, word2idx, idx2word, corpus, vocab_size
```


## Train


```python
# =============================================================================
# HYPERPARAMETERS
# =============================================================================
block_size = 6      # Context window size
embedding_dim = 32  # Size of embeddings
n_heads = 2         # Number of attention heads
n_layers = 2        # Number of transformer blocks
lr = 5e-4           # Learning rate
epochs = 300        # Training steps

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
train_data, val_data, word2idx, idx2word, corpus, vocab_size = load_corpus("corpus.json")
print(f"Loaded {len(corpus)} sentences")
print(f"Vocabulary size: {vocab_size}")
print(f"Train tokens: {len(train_data)} | Val tokens: {len(val_data)}")

# =============================================================================
# CREATE MODEL AND OPTIMIZER
# =============================================================================
model = MicroGPT(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    block_size=block_size,
    n_heads=n_heads,
    n_layers=n_layers
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# =============================================================================
# TRAINING LOOP
# =============================================================================
print("\nTraining...")
for step in range(epochs):
    xb, yb = get_batch(train_data, block_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch(val_data, block_size)
            _, val_loss = model(xv, yv)
        model.train()
        print(f"Step {step:3d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")
print(f"Final val loss: {val_loss.item():.4f}")

# =============================================================================
# GENERATE TEXT
# =============================================================================
print("\n" + "=" * 50)
print("GENERATION")
print("=" * 50)
start_words = corpus[0].split()[:4]
start_idx = torch.tensor([[word2idx[w] for w in start_words]], dtype=torch.long)
model.eval()
output = model.generate(start_idx, max_new_tokens=1)
generated = " ".join(idx2word[int(i)] for i in output[0])
print(f"Starting words: '{' '.join(start_words)}'")
print(f"Generated: {generated}")
```

**Output:**
```
Loaded 16 sentences
Vocabulary size: 35
Train tokens: 95 | Val tokens: 11
Model parameters: 27,747

Training...
Step   0 | Train: 3.5563 | Val: 3.7721
Step  20 | Train: 3.2179 | Val: 3.6243
Step  40 | Train: 2.9745 | Val: 3.7129
Step  60 | Train: 2.5026 | Val: 3.4522
Step  80 | Train: 2.3059 | Val: 3.4478
Step 100 | Train: 1.8125 | Val: 3.3178
Step 120 | Train: 1.4316 | Val: 3.6147
Step 140 | Train: 1.3991 | Val: 3.4502
Step 160 | Train: 1.1840 | Val: 3.1982
Step 180 | Train: 1.0622 | Val: 3.6726
Step 200 | Train: 0.9778 | Val: 3.3387
Step 220 | Train: 0.7781 | Val: 3.9346
Step 240 | Train: 0.7800 | Val: 3.7933
Step 260 | Train: 0.7748 | Val: 4.0200
Step 280 | Train: 0.5451 | Val: 4.0197
Final val loss: 4.0197

==================================================
GENERATION
==================================================
Starting words: 'mary had a little'
Generated: mary had a little lamb
```

