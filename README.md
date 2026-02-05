![image](https://github.com/mytechnotalent/MicroGPT/blob/main/MicroGPT.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Today's Tutorial [February 5, 2026]
## Lesson 101: ARM-32 Course 2 (Part 36 – Debugging SizeOf Operator)
This tutorial will discuss debugging sizeof operator.

-> Click [HERE](https://0xinfection.github.io/reversing) to read the FREE ebook.

<br>

# MicroGPT

> A production-ready, fully type-annotated GPT implementation from scratch in PyTorch.

MicroGPT is a clean, educational implementation of the GPT (Generative Pre-trained Transformer) architecture built from first principles with detailed explanations and comprehensive testing.

## 🎯 Core Files

### Configuration (Single Source of Truth)
- **config.json** - All hyperparameters (architecture, training, fine-tuning)
- **config.py** - Loads config.json into typed Config dataclass

### Model Implementation
- **micro_gpt.py** (750+ lines) - Complete GPT implementation
  - ✅ **100% Type Annotated** - Full type hints
  - ✅ **Production Ready** - Clean, maintainable code
  - Components: `SelfAttentionHead`, `MultiHeadAttention`, `FeedForward`, `Block`, `MicroGPT`

### Training Pipeline
- **main.py** - Pre-training on OpenWebText dataset (GPT-2 tokenizer)
- **fine_tune_micro_gpt.py** - Fine-tuning for professional chatbot (Stanford Human Preferences)
- **inference_micro_gpt.py** - Interactive chat interface
- **device.py** - Device detection (CUDA/MPS/CPU)

### Testing
- **test_micro_gpt.py** (2,669 lines) - 65 tests, 99% coverage
- **test_fine_tune_micro_gpt.py** - 39 tests for fine-tuning
- **test_inference_micro_gpt.py** - 42 tests for inference

### Documentation
- **MicroGPT_Tutorial.pdf** - Complete transformer architecture tutorial
- **README.md** - This file
- **FILES.md** - Complete file inventory

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

### Complete Training Workflow

**1. Pre-training on OpenWebText** (creates base language model)
```bash
python main.py
```
- Loads 800k examples from OpenWebText dataset
- Uses GPT-2 BPE tokenizer (vocab size: 50,257)
- Trains for 150k steps with cosine LR schedule
- Saves checkpoint to `checkpoints/best_val.pt` (overwrites with better loss)

**2. Fine-tuning for Chatbot** (adds conversational abilities)
```bash
python fine_tune_micro_gpt.py
```
- Loads pre-trained checkpoint from step 1
- Fine-tunes on 10M tokens from Stanford Human Preferences dataset
- Adds professional identity training ("I am MicroGPT, created by Kevin Thomas")
- Saves fine-tuned checkpoint to `checkpoints/finetuned_best_val.pt` (overwrites with better loss)

**3. Interactive Chat**
```bash
python inference_micro_gpt.py
```
- Loads fine-tuned checkpoint from step 2
- Provides interactive chat interface
- Professional, consistent responses (temperature=0.2)

### Direct Model Usage

```python
from micro_gpt import MicroGPT
from config import load_config
import torch
import tiktoken

# Load config
cfg = load_config("config.json")

# Create model with config params
model = MicroGPT(
    vocab_size=50257,  # GPT-2 tokenizer
    embedding_dim=cfg.embedding_dim,
    block_size=cfg.block_size,
    n_heads=cfg.n_heads,
    n_layers=cfg.n_layers,
    dropout=cfg.dropout
)

# Generate text
tokenizer = tiktoken.get_encoding("gpt2")
context_tokens = tokenizer.encode("The quick brown")
context = torch.tensor([context_tokens])
output = model.generate(context, max_new_tokens=50, temperature=0.8)
print(tokenizer.decode(output[0].tolist()))
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

### Current Configuration (config.json)

**Architecture:**
- `vocab_size`: 50257 (GPT-2 tokenizer)
- `embedding_dim`: 896
- `block_size`: 256
- `n_heads`: 14
- `n_layers`: 16
- `dropout`: 0.05
- **Parameters:** ~150M (~600MB)

**Pre-training (main.py):**
- Dataset: OpenWebText (800k examples)
- Batch size: 8
- Learning rate: 6e-4 → 6e-5 (cosine decay)
- Warmup steps: 2000
- Training steps: 150,000

**Fine-tuning (fine_tune_micro_gpt.py):**
- Dataset: Stanford Human Preferences (10M tokens)
- Learning rate: 1e-5
- Epochs: 5000
- Temperature: 0.2 (professional responses)
- Max new tokens: 30

### Memory Usage

| Config                | Parameters | Memory |
| --------------------- | ---------- | ------ |
| Current (d=896, L=16) | ~150M      | ~30 GB |
| Small (d=512, L=8)    | ~45M       | ~8 GB  |
| Large (d=1024, L=24)  | ~300M      | ~40 GB |

**Note:** All parameters are configurable in `config.json` - single source of truth for the entire project.

## 📜 License

MIT License

## 👤 Author

**Kevin Thomas**
- Email: ket189@pitt.edu
- GitHub: [@mytechnotalent](https://github.com/mytechnotalent)

## 🙏 Acknowledgments

Built for educational purposes to help students understand transformer architecture from first principles.
