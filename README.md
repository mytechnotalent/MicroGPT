![image](https://github.com/mytechnotalent/MicroGPT/blob/main/MicroGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# MicroGPT

> A production-ready, fully type-annotated GPT implementation from scratch in PyTorch.

MicroGPT is a clean, educational implementation of the GPT (Generative Pre-trained Transformer) architecture built from first principles with detailed explanations and comprehensive testing.

## 🎯 Core Files

This project centers around three essential files:

### 1. **MicroGPT_Tutorial.pdf**
Comprehensive tutorial explaining every aspect of GPT architecture from scratch. Topics covered:
- Tokenization and vocabulary
- Token and positional embeddings  
- Self-attention mechanisms
- Multi-head attention
- Transformer blocks and residual streams
- Training and text generation

### 2. **micro_gpt.py** (750+ lines)
Complete GPT implementation:
- ✅ **100% Type Annotated** - Full type hints for all functions
- ✅ **Comprehensive Docstrings** - NumPy/Google style with Examples
- ✅ **Production Ready** - Clean, maintainable code

**Components:**
- `SelfAttentionHead` - Single causal attention head
- `MultiHeadAttention` - Parallel attention computation
- `FeedForward` - MLP with 4x expansion
- `Block` - Complete transformer block
- `MicroGPT` - Full language model
- Utilities: `get_batch()`, `save_checkpoint()`, training helpers

### 3. **test_micro_gpt.py** (2,669 lines)
Comprehensive test suite:
- ✅ **65 Tests** - Full component coverage
- ✅ **99% Code Coverage** - Verified with pytest-cov
- ✅ **Type Annotated** - Consistent with micro_gpt.py
- ✅ **Documented** - All methods include Examples

## ⚙️ Installation

```bash
# Clone repository
git clone <repository-url>
cd microgpt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

**Required packages (requirements.txt):**
- `torch` - PyTorch framework
- `tiktoken` - OpenAI's tokenizer
- `datasets` - Hugging Face datasets
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `markdown` - Markdown to HTML conversion
- `weasyprint` - HTML to PDF conversion
- `pygments` - Syntax highlighting

## 🚀 Usage

### Basic Example

```python
from micro_gpt import MicroGPT, get_batch
import torch

# Create model
model = MicroGPT(
    vocab_size=50257,
    embedding_dim=768,
    block_size=256,
    n_heads=12,
    n_layers=12,
    dropout=0.1
)

# Generate text
context = torch.tensor([[1, 2, 3]])
output = model.generate(context, max_new_tokens=100, temperature=0.8)
```

### Training Example

```python
from micro_gpt import MicroGPT, get_batch, save_checkpoint
import torch
import torch.optim as optim

# Create model
model = MicroGPT(
    vocab_size=256,
    embedding_dim=128,
    block_size=64,
    n_heads=4,
    n_layers=3,
    dropout=0.1
)

# Setup training
optimizer = optim.Adam(model.parameters(), lr=3e-4)
train_data = torch.randint(0, 256, (10000,))

# Training step
x, y = get_batch(train_data, block_size=64, batch_size=16, device="cpu")
logits, loss = model(x, y)
loss.backward()
optimizer.step()

# Generate text
context = torch.randint(0, 256, (1, 10))
generated = model.generate(context, max_new_tokens=20, temperature=0.8)

# Save checkpoint
save_checkpoint(model, optimizer, step=100, val_loss=2.5, 
                filepath="checkpoints/model.pt")
```

### View Documentation

```python
from micro_gpt import MicroGPT
help(MicroGPT)              # View class documentation
help(MicroGPT.generate)     # View method documentation
```

## 🧪 Testing

### Run All Tests

```bash
pytest test_micro_gpt.py -v
```

Expected output:
```
collected 65 items

test_micro_gpt.py::TestSelfAttentionHead::test_initialization PASSED      [  1%]
test_micro_gpt.py::TestSelfAttentionHead::test_forward_shape PASSED       [  3%]
...
====================================================== 65 passed in 2.86s ========
```

### Run Specific Tests

```bash
# Test only MicroGPT model
pytest test_micro_gpt.py::TestMicroGPT -v

# Test only integration tests
pytest test_micro_gpt.py::TestIntegration -v

# Test specific function
pytest test_micro_gpt.py::TestMicroGPT::test_forward_with_targets -v
```

### Coverage Report

```bash
# Terminal + HTML report
pytest test_micro_gpt.py -v --cov=test_micro_gpt --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html  # macOS
```

Output:
```
Name                Stmts   Miss  Cover   Missing
-------------------------------------------------
test_micro_gpt.py     521      1    99%   1018
-------------------------------------------------
TOTAL                 521      1    99%
```

## 📖 Tutorial

The **MicroGPT_Tutorial.pdf** provides a comprehensive guide to understanding GPT from scratch. It's designed for high school students and beginners, covering:

1. **Introduction** - What is GPT and how language models work
2. **Tokenization** - Breaking text into tokens
3. **Vocabulary** - Building and using a vocabulary
4. **Token Embeddings** - Converting tokens to vectors
5. **Positional Embeddings** - Encoding position information
6. **Residual Stream** - Data flow through the model
7. **Self-Attention** - How attention mechanisms work
8. **Multi-Head Attention** - Parallel attention computation
9. **Feed-Forward Networks** - Processing within positions
10. **Transformer Block** - Combining components
11. **Model Architecture** - Complete GPT structure
12. **Training** - How the model learns
13. **Text Generation** - Producing new text

**Regenerate PDF:**
```bash
python convert_tutorial_to_pdf.py
```

## 🔧 Model Architecture

### Components

| Component            | Purpose                          |
| -------------------- | -------------------------------- |
| `SelfAttentionHead`  | Single causal attention head     |
| `MultiHeadAttention` | Multiple heads in parallel       |
| `FeedForward`        | MLP with 4x expansion            |
| `Block`              | Transformer block with residuals |
| `MicroGPT`           | Complete language model          |

### Hyperparameters

| Parameter       | Description     | Typical Range |
| --------------- | --------------- | ------------- |
| `vocab_size`    | Vocabulary size | 256-50257     |
| `embedding_dim` | Model width     | 128-768       |
| `block_size`    | Context window  | 64-2048       |
| `n_heads`       | Attention heads | 4-12          |
| `n_layers`      | Model depth     | 2-12          |
| `dropout`       | Regularization  | 0.0-0.3       |

### Memory Usage (Approximate)

| Config              | Parameters | GPU Memory |
| ------------------- | ---------- | ---------- |
| Small (d=128, L=2)  | ~1M        | ~2 GB      |
| Medium (d=384, L=6) | ~10M       | ~8 GB      |
| Large (d=768, L=12) | ~100M      | ~20 GB     |

## 📜 License

MIT License

## 👤 Author

**Kevin Thomas**
- Email: ket189@pitt.edu
- GitHub: [@mytechnotalent](https://github.com/mytechnotalent)

## 🙏 Acknowledgments

Built for educational purposes to help students understand transformer architecture from first principles.
