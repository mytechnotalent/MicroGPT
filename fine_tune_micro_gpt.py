"""
Fine-tune GPT-2 on conversational dataset.

This module provides functionality to fine-tune a pre-trained GPT-2 model
on conversational data for chatbot applications.

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import os
from typing import Tuple, List
from datasets import load_dataset
import tiktoken
import torch
from config import load_config
from micro_gpt import (
    GPT2,
    GPT2Config,
    save_checkpoint,
    _train_step,
    _eval_model,
)


def _load_tokenizer():
    """Load GPT-2 BPE tokenizer.

    Returns:
        Tokenizer object with encode/decode methods.

    Example:
        >>> tokenizer = _load_tokenizer()
        >>> tokens = tokenizer.encode("Hello world")
        >>> isinstance(tokens, list)
        True
    """
    return tiktoken.get_encoding("gpt2")


def _format_conversation(example: dict) -> str:
    """Format conversation example into training text.

    Args:
        example: Dictionary containing 'history' and 'human_ref_A' or 'post_text' fields.

    Returns:
        Formatted conversation string with turn markers.

    Example:
        >>> example = {'history': 'Hello', 'human_ref_A': 'Hi there'}
        >>> text = _format_conversation(example)
        >>> 'User:' in text and 'Assistant:' in text
        True
    """
    # Handle Stanford Human Preferences dataset format
    history = example.get("history", "")
    response = example.get("human_ref_A", example.get("human_ref_B", ""))
    if not history or not response:
        return ""
    return f"User: {history}\nAssistant: {response}\n\n"


def _load_conversational_dataset():
    """Load conversational dataset compatible with current datasets library.

    Returns:
        Dataset object with train and validation splits.

    Example:
        >>> dataset = _load_conversational_dataset()
        >>> 'train' in dataset
        True
    """
    return load_dataset("stanfordnlp/shp", split="train", streaming=True)


def _process_conversation(
    example: dict, tokenizer, tokens: List[int], total: int, i: int, max_tokens: int
) -> Tuple[List[int], int, bool]:
    """Process single conversation and update token list.

    Args:
        example: Conversation example dict.
        tokenizer: Tokenizer with encode method.
        tokens: Current token list.
        total: Current token count.
        i: Current example index.
        max_tokens: Maximum tokens to collect.

    Returns:
        Tuple of (tokens, total, should_break).

    Example:
        >>> tokenizer = _load_tokenizer()
        >>> ex = {'dialog': ['Hi', 'Hello']}
        >>> toks, tot, brk = _process_conversation(ex, tokenizer, [], 0, 0, 1000)
        >>> len(toks) > 0
        True
    """
    text = _format_conversation(example)
    if text:
        encoded = tokenizer.encode(text)
        tokens.extend(encoded)
        total += len(encoded)
        if (i + 1) % 100 == 0:
            print(f"  {i+1} conversations, {total:,} tokens")
    return tokens, total, total >= max_tokens


def _add_identity_examples(tokenizer) -> List[int]:
    """Add synthetic identity training examples.

    Args:
        tokenizer: Tokenizer with encode method.

    Returns:
        List of token IDs for identity examples.

    Example:
        >>> tokenizer = _load_tokenizer()
        >>> tokens = _add_identity_examples(tokenizer)
        >>> len(tokens) > 0
        True
    """
    identity_examples = [
        "User: What is your name?\nAssistant: I am MicroGPT, created by Kevin Thomas.\n\n",
        "User: Who are you?\nAssistant: I'm MicroGPT, a professional AI assistant created by Kevin Thomas.\n\n",
        "User: What's your name?\nAssistant: My name is MicroGPT. I was created by Kevin Thomas.\n\n",
        "User: Who created you?\nAssistant: I was created by Kevin Thomas, a cybersecurity expert and AI researcher.\n\n",
        "User: Tell me about yourself\nAssistant: I'm MicroGPT, a professional AI language model created by Kevin Thomas to provide helpful, accurate, and courteous assistance.\n\n",
        "User: Who is your creator?\nAssistant: My creator is Kevin Thomas. I'm MicroGPT, designed to be a helpful professional assistant.\n\n",
        "User: What are you?\nAssistant: I'm MicroGPT, an AI chatbot created by Kevin Thomas to assist with various tasks and answer questions professionally.\n\n",
        "User: Introduce yourself\nAssistant: Hello! I'm MicroGPT, a professional AI assistant created by Kevin Thomas. I'm here to help you with information and tasks.\n\n",
    ]
    tokens = []
    for example in identity_examples * 25:
        tokens.extend(tokenizer.encode(example))
    return tokens


def _tokenize_conversations(
    dataset, tokenizer, max_tokens: int, split: str = "train"
) -> List[int]:
    """Tokenize conversation examples until max_tokens reached.

    Args:
        dataset: HuggingFace dataset object (streaming).
        tokenizer: Tokenizer with encode method.
        max_tokens: Maximum number of tokens to collect.
        split: Dataset split to use (ignored for streaming).

    Returns:
        List of token IDs.

    Example:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset("stanfordnlp/shp", split="train", streaming=True)
        >>> tokenizer = _load_tokenizer()
        >>> tokens = _tokenize_conversations(dataset, tokenizer, 1000, "train")
        >>> len(tokens) > 0
        True
    """
    tokens = _add_identity_examples(tokenizer)
    total = len(tokens)
    print(f"  Identity examples: {total:,} tokens")
    for i, example in enumerate(dataset):
        tokens, total, should_break = _process_conversation(
            example, tokenizer, tokens, total, i, max_tokens
        )
        if should_break:
            break
    return tokens


def _split_data(
    data: torch.Tensor, val_split: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split data into train and validation sets.

    Args:
        data: Tensor of token IDs.
        val_split: Fraction of data to use for validation.

    Returns:
        Tuple of (train_data, val_data) tensors.

    Example:
        >>> data = torch.randint(0, 1000, (10000,))
        >>> train, val = _split_data(data, 0.1)
        >>> len(train) + len(val) == len(data)
        True
    """
    n = int(len(data) * (1 - val_split))
    return data[:n], data[n:]


def load_conversational_data(
    max_tokens: int = 5_000_000, val_split: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, int, object]:
    """Load and tokenize conversational dataset.

    Args:
        max_tokens: Maximum tokens to load (default 5M for fine-tuning).
        val_split: Validation split fraction.

    Returns:
        Tuple of (train_data, val_data, vocab_size, tokenizer).

    Example:
        >>> train, val, vocab_size, tok = load_conversational_data(1000)
        >>> vocab_size == 50257
        True
        >>> len(train) > 0 and len(val) > 0
        True
    """
    print("Loading Stanford Human Preferences conversational dataset...")
    tokenizer, dataset = _load_tokenizer(), _load_conversational_dataset()
    print("Tokenizing conversations...")
    tokens = _tokenize_conversations(dataset, tokenizer, max_tokens, "train")
    data = torch.tensor(tokens, dtype=torch.long)
    train_data, val_data = _split_data(data, val_split)
    print(f"\nTrain: {len(train_data):,}, Val: {len(val_data):,}")
    return train_data, val_data, tokenizer.n_vocab, tokenizer


def _create_model_architecture(
    vocab_size: int,
    n_embd: int,
    block_size: int,
    n_head: int,
    n_layer: int,
    dropout: float,
    device: str,
) -> GPT2:
    """Create GPT-2 model with specified architecture.

    Args:
        vocab_size: Vocabulary size.
        n_embd: Model embedding dimension.
        block_size: Context window size.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout: Dropout rate.
        device: Device to load model on.

    Returns:
        GPT2 model instance.

    Example:
        >>> model = _create_model_architecture(256, 128, 64, 4, 2, 0.1, "cpu")
        >>> isinstance(model, GPT2)
        True
    """
    config = GPT2Config(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=True,
    )
    return GPT2(config).to(device)


def load_pretrained_model(
    checkpoint_path: str,
    vocab_size: int,
    n_embd: int,
    block_size: int,
    n_head: int,
    n_layer: int,
    dropout: float,
    device: str,
) -> Tuple[GPT2, torch.optim.Optimizer]:
    """Load pre-trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        vocab_size: Vocabulary size.
        n_embd: Model embedding dimension.
        block_size: Context window size.
        n_head: Number of attention heads.
        n_layer: Number of transformer blocks.
        dropout: Dropout rate.
        device: Device to load model on.

    Returns:
        Tuple of (model, optimizer).

    Example:
        >>> model, opt = load_pretrained_model(
        ...     "checkpoints/best_val.pt",
        ...     50257, 896, 256, 14, 16, 0.05, "cpu"
        ... )
        >>> isinstance(model, GPT2)
        True
    """
    print(f"Loading pre-trained model from {checkpoint_path}...")
    model = _create_model_architecture(
        vocab_size, n_embd, block_size, n_head, n_layer, dropout, device
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from step {checkpoint['step']}")
    print(f"  Pre-training validation loss: {checkpoint['val_loss']:.4f}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return model, optimizer


def _should_save(val_loss: float, best_loss: float) -> bool:
    """Check if current validation loss is new best.

    Args:
        val_loss: Current validation loss.
        best_loss: Best validation loss so far.

    Returns:
        True if val_loss is lower than best_loss.

    Example:
        >>> _should_save(2.5, 3.0)
        True
        >>> _should_save(3.0, 2.5)
        False
    """
    return val_loss < best_loss


def _handle_checkpoint(
    val_loss: float,
    best_val_loss: float,
    model: GPT2,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> Tuple[float, bool]:
    """Handle checkpoint saving if validation loss improved.

    Args:
        val_loss: Current validation loss.
        best_val_loss: Best validation loss so far.
        model: Model to save.
        optimizer: Optimizer to save.
        step: Current training step.

    Returns:
        Tuple of (updated best_val_loss, is_best).

    Example:
        >>> cfg = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
        >>> model = GPT2(cfg)
        >>> opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        >>> best, is_best = _handle_checkpoint(2.5, 3.0, model, opt, 100)
        >>> is_best
        True
    """
    if _should_save(val_loss, best_val_loss):
        save_checkpoint(
            model,
            optimizer,
            step,
            val_loss,
            "checkpoints/finetuned_best_val.pt",
        )
        return val_loss, True
    return best_val_loss, False


def fine_tune_model(
    model: GPT2,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: dict,
) -> float:
    """Fine-tune model on conversational data with checkpointing.

    Args:
        model: Pre-trained GPT-2 model.
        optimizer: Optimizer for training.
        train_data: Training data tensor.
        val_data: Validation data tensor.
        config: Configuration dictionary with hyperparameters.

    Returns:
        Best validation loss achieved during fine-tuning.

    Example:
        >>> cfg = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
        >>> model = GPT2(cfg)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        >>> train = torch.randint(0, 256, (10000,))
        >>> val = torch.randint(0, 256, (1000,))
        >>> cfg = {'epochs': 10, 'eval_interval': 5, 'eval_iters': 2,
        ...        'block_size': 64, 'batch_size': 4, 'device': 'cpu'}
        >>> loss = fine_tune_model(model, optimizer, train, val, cfg)
        >>> isinstance(loss, float)
        True
    """
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)
    for step in range(config["epochs"]):
        train_loss = _train_step(
            model,
            optimizer,
            train_data,
            config["block_size"],
            config["batch_size"],
            config["device"],
        )
        if step % config["eval_interval"] == 0 or step == config["epochs"] - 1:
            val_loss = _eval_model(
                model,
                val_data,
                config["block_size"],
                config["batch_size"],
                config["device"],
                config["eval_iters"],
            )
            print(
                f"Step {step:5d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}",
                end="",
            )
            best_val_loss, is_best = _handle_checkpoint(
                val_loss, best_val_loss, model, optimizer, step
            )
            if is_best:
                print(" ← BEST", end="")
            print()
            if val_loss < 0.1:
                print(f"🎉 Fine-tuning Target Reached Step {step}!")
                break
    return best_val_loss


def _extract_assistant_response(full_text: str) -> str:
    """Extract assistant response from generated text.

    Args:
        full_text: Full generated text with markers.

    Returns:
        Extracted assistant response.

    Example:
        >>> text = "User: Hi\\nAssistant: Hello there\\nUser: Bye"
        >>> response = _extract_assistant_response(text)
        >>> "Hello there" in response
        True
    """
    if "Assistant:" not in full_text:
        return full_text
    response = full_text.split("Assistant:")[-1].strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()
    return response


def generate_chat_response(
    model: GPT2,
    tokenizer,
    prompt: str,
    max_tokens: int,
    device: str,
    temp: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Generate chatbot response to user prompt.

    Args:
        model: Fine-tuned GPT-2 model.
        tokenizer: Tokenizer with encode/decode methods.
        prompt: User input text.
        max_tokens: Maximum tokens to generate.
        device: Device model is on.
        temp: Sampling temperature.
        top_p: Nucleus sampling probability threshold.

    Returns:
        Generated response text.

    Example:
        >>> cfg = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
        >>> model = GPT2(cfg)
        >>> tokenizer = _load_tokenizer()
        >>> response = generate_chat_response(
        ...     model, tokenizer, "Hello", 10, "cpu", 0.7
        ... )
        >>> isinstance(response, str)
        True
    """
    model.eval()
    formatted_prompt = f"User: {prompt}\nAssistant:"
    tokens = tokenizer.encode(formatted_prompt)
    context = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(
            context, max_new_tokens=max_tokens, temperature=temp, top_p=top_p
        )
    full_text = tokenizer.decode(output[0].tolist())
    return _extract_assistant_response(full_text)


def _find_best_checkpoint() -> str:
    """Find checkpoint with lowest validation loss.

    Returns:
        Path to best checkpoint file.

    Example:
        >>> path = _find_best_checkpoint()
        >>> 'best_val' in path
        True
    """
    import os

    checkpoint_path = "checkpoints/best_val.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No pre-trained checkpoint found")
    return checkpoint_path


def _create_config() -> dict:
    """Create fine-tuning configuration dictionary from config.json.

    Returns:
        Configuration dictionary with hyperparameters.

    Example:
        >>> config = _create_config()
        >>> 'block_size' in config
        True
    """
    cfg = load_config("config.json")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    best_checkpoint = _find_best_checkpoint()
    print(f"Using checkpoint: {best_checkpoint}")
    return {
        "checkpoint_path": best_checkpoint,
        "block_size": cfg.block_size,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_layer": cfg.n_layer,
        "dropout": cfg.dropout,
        "batch_size": cfg.batch_size,
        "lr": cfg.finetune_lr,
        "epochs": cfg.finetune_epochs,
        "eval_interval": cfg.finetune_eval_interval,
        "eval_iters": cfg.finetune_eval_iters,
        "max_tokens": cfg.finetune_max_tokens,
        "finetune_temperature": cfg.finetune_temperature,
        "finetune_max_new_tokens": cfg.finetune_max_new_tokens,
        "finetune_top_p": cfg.finetune_top_p,
        "device": device,
    }


def _run_fine_tuning(
    model: GPT2,
    optimizer: torch.optim.Optimizer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: dict,
) -> float:
    """Execute fine-tuning loop.

    Args:
        model: Model to fine-tune.
        optimizer: Optimizer for training.
        train_data: Training data tensor.
        val_data: Validation data tensor.
        config: Configuration dict.

    Returns:
        Best validation loss achieved.

    Example:
        >>> cfg = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
        >>> model = GPT2(cfg)
        >>> opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        >>> train, val = torch.randint(0, 256, (10000,)), torch.randint(0, 256, (1000,))
        >>> cfg = {'epochs': 10, 'eval_interval': 5, 'eval_iters': 2, 'block_size': 64, 'batch_size': 4, 'device': 'cpu'}
        >>> loss = _run_fine_tuning(model, opt, train, val, cfg)
        >>> isinstance(loss, float)
        True
    """
    print("=" * 70)
    print("FINE-TUNING")
    print("=" * 70)
    best_loss = fine_tune_model(model, optimizer, train_data, val_data, config)
    print(f"\nComplete! Best Val Loss: {best_loss:.4f}")
    return best_loss


def _test_chat_responses(model: GPT2, tokenizer, device: str, config: dict) -> None:
    """Test model with sample chat prompts.

    Args:
        model: Fine-tuned model.
        tokenizer: Tokenizer for encoding/decoding.
        device: Device model is on.
        config: Configuration dict with temperature and max tokens.

    Returns:
        None

    Example:
        >>> cfg = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
        >>> model = GPT2(cfg)
        >>> tokenizer = _load_tokenizer()
        >>> cfg = {'finetune_temperature': 0.2, 'finetune_max_new_tokens': 30, 'finetune_top_p': 0.9}
        >>> _test_chat_responses(model, tokenizer, "cpu", cfg)
    """
    print()
    print("=" * 70)
    print("CHAT RESPONSES")
    print("=" * 70)
    test_prompts = [
        "Hello, how are you?",
        "What's your name?",
        "Who created you?",
        "Can you help me with a question?",
        "Tell me about yourself",
        "What can you do?",
    ]
    for prompt in test_prompts:
        print(f"\n👤 User: {prompt}")
        response = generate_chat_response(
            model,
            tokenizer,
            prompt,
            config["finetune_max_new_tokens"],
            device,
            temp=config["finetune_temperature"],
            top_p=config["finetune_top_p"],
        )
        print(f"🤖 Assistant: {response}")


def main():
    """Main fine-tuning script.

    Returns:
        None

    Example:
        >>> # main()  # Would run full fine-tuning
        >>> pass
    """
    config = _create_config()
    print(f"Device: {config['device']}")
    print()
    train_data, val_data, vocab_size, tokenizer = load_conversational_data(
        config["max_tokens"]
    )
    model, optimizer = load_pretrained_model(
        config["checkpoint_path"],
        vocab_size,
        config["n_embd"],
        config["block_size"],
        config["n_head"],
        config["n_layer"],
        config["dropout"],
        config["device"],
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} (~{params*4/1e6:.1f}MB)")
    print()
    _run_fine_tuning(model, optimizer, train_data, val_data, config)
    _test_chat_responses(model, tokenizer, config["device"], config)


if __name__ == "__main__":
    main()
