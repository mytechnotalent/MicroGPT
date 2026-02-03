"""
Inference script for conversational MicroGPT.

This module provides functionality to run interactive chat sessions with a
fine-tuned MicroGPT model for conversational AI applications.

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import tiktoken
import torch
from typing import List, Tuple
from config import load_config
from micro_gpt import MicroGPT


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


def _get_device() -> str:
    """Detect and return best available device.

    Returns:
        Device string ('cuda', 'mps', or 'cpu').

    Example:
        >>> device = _get_device()
        >>> device in ['cuda', 'mps', 'cpu']
        True
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _create_model(
    vocab_size: int,
    embedding_dim: int,
    block_size: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    device: str,
) -> MicroGPT:
    """Create MicroGPT model architecture.

    Args:
        vocab_size: Vocabulary size.
        embedding_dim: Model embedding dimension.
        block_size: Context window size.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        dropout: Dropout rate.
        device: Device to load model on.

    Returns:
        MicroGPT model instance.

    Example:
        >>> model = _create_model(256, 128, 64, 4, 2, 0.1, "cpu")
        >>> isinstance(model, MicroGPT)
        True
    """
    return MicroGPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        block_size=block_size,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)


def load_finetuned_model(checkpoint_path: str, device: str) -> MicroGPT:
    """Load fine-tuned MicroGPT model from checkpoint.

    Args:
        checkpoint_path: Path to fine-tuned checkpoint file.
        device: Device to load model on.

    Returns:
        Loaded MicroGPT model in eval mode.

    Example:
        >>> model = load_finetuned_model("checkpoints/finetuned_best_val.pt", "cpu")
        >>> isinstance(model, MicroGPT)
        True
    """
    print(f"Loading fine-tuned model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = load_config("config.json")
    model = _create_model(
        50257,
        cfg.embedding_dim,
        cfg.block_size,
        cfg.n_heads,
        cfg.n_layers,
        cfg.dropout,
        device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Model loaded (val loss: {checkpoint['val_loss']:.4f})")
    return model


def _format_prompt(conversation_history: List[Tuple[str, str]], user_input: str) -> str:
    """Format conversation history and new input as prompt.

    Args:
        conversation_history: List of (user, assistant) message tuples.
        user_input: Current user message.

    Returns:
        Formatted prompt string.

    Example:
        >>> history = [("Hi", "Hello")]
        >>> prompt = _format_prompt(history, "How are you?")
        >>> "User:" in prompt and "Assistant:" in prompt
        True
    """
    formatted = []
    for user_msg, asst_msg in conversation_history:
        formatted.append(f"User: {user_msg}\nAssistant: {asst_msg}")
    formatted.append(f"User: {user_input}\nAssistant:")
    return "\n".join(formatted)


def _extract_response(full_text: str) -> str:
    """Extract assistant response from generated text.

    Args:
        full_text: Full generated text with conversation markers.

    Returns:
        Extracted assistant response.

    Example:
        >>> text = "User: Hi\\nAssistant: Hello\\nUser: Bye"
        >>> response = _extract_response(text)
        >>> "Hello" in response
        True
    """
    if "Assistant:" not in full_text:
        return full_text.strip()
    response = full_text.split("Assistant:")[-1].strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()
    return response


def generate_response(
    model: MicroGPT,
    tokenizer,
    conversation_history: List[Tuple[str, str]],
    user_input: str,
    device: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Generate conversational response from user input.

    Args:
        model: Fine-tuned MicroGPT model.
        tokenizer: Tokenizer with encode/decode methods.
        conversation_history: Previous conversation turns.
        user_input: Current user message.
        device: Device model is on.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        Generated assistant response.

    Example:
        >>> model = MicroGPT(256, 128, 64, 4, 2, 0.1)
        >>> tokenizer = _load_tokenizer()
        >>> response = generate_response(model, tokenizer, [], "Hello", "cpu", 10, 0.7)
        >>> isinstance(response, str)
        True
    """
    prompt = _format_prompt(conversation_history, user_input)
    tokens = tokenizer.encode(prompt)
    context = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model.generate(
            context, max_new_tokens=max_tokens, temperature=temperature
        )
    full_text = tokenizer.decode(output[0].tolist())
    return _extract_response(full_text)


def _display_welcome() -> None:
    """Display welcome message for chat interface.

    Returns:
        None

    Example:
        >>> _display_welcome()
    """
    print("\n" + "=" * 70)
    print("MICROGPT CONVERSATIONAL AI")
    print("=" * 70)
    print("Commands: 'quit' to exit, 'clear' to reset conversation")
    print("=" * 70 + "\n")


def _should_exit(user_input: str) -> bool:
    """Check if user wants to exit chat.

    Args:
        user_input: User's input string.

    Returns:
        True if user wants to exit.

    Example:
        >>> _should_exit("quit")
        True
        >>> _should_exit("hello")
        False
    """
    return user_input.lower() in ["quit", "exit", "q"]


def _should_clear(user_input: str) -> bool:
    """Check if user wants to clear conversation history.

    Args:
        user_input: User's input string.

    Returns:
        True if user wants to clear history.

    Example:
        >>> _should_clear("clear")
        True
        >>> _should_clear("hello")
        False
    """
    return user_input.lower() in ["clear", "reset"]


def _get_user_input() -> str:
    """Get user input from console.

    Returns:
        User's input string.

    Example:
        >>> # input = _get_user_input()  # Interactive
        >>> # isinstance(input, str)
        >>> # True
        >>> pass
    """
    try:
        return input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        return "quit"


def _print_response(response: str) -> None:
    """Print assistant response to console.

    Args:
        response: Assistant's response text.

    Returns:
        None

    Example:
        >>> _print_response("Hello there!")
        MicroGPT: Hello there!
    """
    print(f"MicroGPT: {response}\n")


def run_chat_loop(
    model: MicroGPT,
    tokenizer,
    device: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> None:
    """Run interactive chat loop with MicroGPT.

    Args:
        model: Fine-tuned MicroGPT model.
        tokenizer: Tokenizer for encoding/decoding.
        device: Device model is on.
        max_tokens: Maximum tokens per response.
        temperature: Sampling temperature.

    Returns:
        None

    Example:
        >>> model = MicroGPT(256, 128, 64, 4, 2, 0.1)
        >>> tokenizer = _load_tokenizer()
        >>> # run_chat_loop(model, tokenizer, "cpu", 10, 0.7)  # Interactive
        >>> pass
    """
    conversation_history = []
    _display_welcome()
    while True:
        user_input = _get_user_input()
        if not user_input:
            continue
        if _should_exit(user_input):
            print("Goodbye!")
            break
        if _should_clear(user_input):
            conversation_history = []
            print("Conversation cleared.\n")
            continue
        response = generate_response(
            model,
            tokenizer,
            conversation_history,
            user_input,
            device,
            max_tokens,
            temperature,
        )
        _print_response(response)
        conversation_history.append((user_input, response))


def _get_checkpoint_path() -> str:
    """Get path to best fine-tuned checkpoint.

    Returns:
        Checkpoint file path.

    Example:
        >>> path = _get_checkpoint_path()
        >>> "finetuned_best_val" in path
        True
    """
    import os

    checkpoint_path = "checkpoints/finetuned_best_val.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No fine-tuned checkpoint found")
    return checkpoint_path


def main() -> None:
    """Main inference script for conversational MicroGPT.

    Returns:
        None

    Example:
        >>> # main()  # Would run interactive chat
        >>> pass
    """
    device = _get_device()
    print(f"Device: {device}")
    checkpoint_path = _get_checkpoint_path()
    model = load_finetuned_model(checkpoint_path, device)
    tokenizer = _load_tokenizer()
    run_chat_loop(model, tokenizer, device, max_tokens=50, temperature=0.7)


if __name__ == "__main__":
    main()
