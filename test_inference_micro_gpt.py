"""
Comprehensive Test Suite for GPT-2 Inference.

This module provides extensive testing for all components of the inference
functionality including model loading, response generation, prompt formatting,
and chat loop operations.

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from inference_micro_gpt import (
    _load_tokenizer,
    _get_device,
    _create_model,
    _format_prompt,
    _extract_response,
    _display_welcome,
    _should_exit,
    _should_clear,
    _print_response,
    _get_checkpoint_path,
)
from micro_gpt import GPT2


@pytest.fixture
def tokenizer():
    """Provide tokenizer for tests.

    Returns:
        Tokenizer instance.

    Example:
        >>> tok = tokenizer()
        >>> isinstance(tok.encode("test"), list)
        True
    """
    return _load_tokenizer()


@pytest.fixture
def device() -> str:
    """Provide device for tests.

    Returns:
        Device string ('cpu').

    Example:
        >>> dev = device()
        >>> dev == 'cpu'
        True
    """
    return "cpu"


@pytest.fixture
def model(device):
    """Provide test model.

    Args:
        device: Device string.

    Returns:
        GPT2 model instance.

    Example:
        >>> m = model("cpu")
        >>> isinstance(m, GPT2)
        True
    """
    return _create_model(50257, 128, 64, 4, 2, 0.1, device)


@pytest.fixture
def temp_checkpoint(tmp_path, model):
    """Create temporary checkpoint file.

    Args:
        tmp_path: Pytest temporary directory.
        model: Model instance.

    Returns:
        Path to temporary checkpoint.

    Example:
        >>> path = temp_checkpoint(tmp_path, model)
        >>> os.path.exists(path)
        True
    """
    checkpoint_path = tmp_path / "finetuned_best_val.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


class TestTokenizer:
    """Test tokenizer functionality.

    Example:
        >>> pytest.main([__file__, "::TestTokenizer", "-v"])
    """

    def test_load_tokenizer(self, tokenizer) -> None:
        """Test tokenizer loads correctly.

        Example:
            >>> tok = _load_tokenizer()
            >>> tok.n_vocab == 50257
            True
        """
        assert tokenizer is not None
        assert tokenizer.n_vocab == 50257

    def test_tokenizer_encode(self, tokenizer) -> None:
        """Test tokenizer encoding.

        Example:
            >>> tok = _load_tokenizer()
            >>> tokens = tok.encode("Hello")
            >>> len(tokens) > 0
            True
        """
        tokens = tokenizer.encode("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenizer_decode(self, tokenizer) -> None:
        """Test tokenizer decoding.

        Example:
            >>> tok = _load_tokenizer()
            >>> text = tok.decode([15496, 995])
            >>> "Hello" in text
            True
        """
        tokens = tokenizer.encode("Hello world")
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert "Hello" in decoded


class TestDevice:
    """Test device detection functionality.

    Example:
        >>> pytest.main([__file__, "::TestDevice", "-v"])
    """

    def test_get_device(self) -> None:
        """Test device detection.

        Example:
            >>> device = _get_device()
            >>> device in ['cuda', 'mps', 'cpu']
            True
        """
        device = _get_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_device_is_string(self) -> None:
        """Test device is string type.

        Example:
            >>> device = _get_device()
            >>> isinstance(device, str)
            True
        """
        device = _get_device()
        assert isinstance(device, str)


class TestModelCreation:
    """Test model creation functionality.

    Example:
        >>> pytest.main([__file__, "::TestModelCreation", "-v"])
    """

    def test_create_model(self, device) -> None:
        """Test creating model.

        Example:
            >>> model = _create_model(50257, 128, 64, 4, 2, 0.1, "cpu")
            >>> isinstance(model, GPT2)
            True
        """
        model = _create_model(50257, 128, 64, 4, 2, 0.1, device)
        assert isinstance(model, GPT2)
        assert model.config.block_size == 64

    def test_model_parameters(self, device) -> None:
        """Test model has correct parameters.

        Example:
            >>> model = _create_model(50257, 128, 64, 4, 2, 0.1, "cpu")
            >>> sum(p.numel() for p in model.parameters()) > 0
            True
        """
        model = _create_model(50257, 128, 64, 4, 2, 0.1, device)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0

    def test_model_in_eval_mode(self, device) -> None:
        """Test model can be set to eval mode.

        Example:
            >>> model = _create_model(50257, 128, 64, 4, 2, 0.1, "cpu")
            >>> model.eval()
            >>> model.training
            False
        """
        model = _create_model(50257, 128, 64, 4, 2, 0.1, device)
        model.eval()
        assert not model.training


class TestPromptFormatting:
    """Test prompt formatting functions.

    Example:
        >>> pytest.main([__file__, "::TestPromptFormatting", "-v"])
    """

    def test_format_prompt_empty_history(self) -> None:
        """Test formatting prompt with empty history.

        Example:
            >>> prompt = _format_prompt([], "Hello")
            >>> "User: Hello" in prompt
            True
        """
        prompt = _format_prompt([], "Hello")
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt

    def test_format_prompt_with_history(self) -> None:
        """Test formatting prompt with conversation history.

        Example:
            >>> history = [("Hi", "Hello")]
            >>> prompt = _format_prompt(history, "How are you?")
            >>> "User: Hi" in prompt
            True
        """
        history = [("Hi", "Hello"), ("How are you?", "I'm good")]
        prompt = _format_prompt(history, "What's new?")
        assert "User: Hi" in prompt
        assert "Assistant: Hello" in prompt
        assert "User: What's new?" in prompt

    def test_format_prompt_multiple_turns(self) -> None:
        """Test formatting with multiple conversation turns.

        Example:
            >>> history = [("A", "B"), ("C", "D")]
            >>> prompt = _format_prompt(history, "E")
            >>> prompt.count("User:") == 3
            True
        """
        history = [("A", "B"), ("C", "D")]
        prompt = _format_prompt(history, "E")
        assert prompt.count("User:") == 3
        assert prompt.count("Assistant:") == 3


class TestResponseExtraction:
    """Test response extraction functions.

    Example:
        >>> pytest.main([__file__, "::TestResponseExtraction", "-v"])
    """

    def test_extract_response_valid(self) -> None:
        """Test extracting valid response.

        Example:
            >>> text = "User: Hi\\nAssistant: Hello\\nUser: Bye"
            >>> response = _extract_response(text)
            >>> "Hello" in response
            True
        """
        full_text = "User: Hi\nAssistant: Hello there\nUser: Bye"
        response = _extract_response(full_text)
        assert "Hello there" in response
        assert "User:" not in response

    def test_extract_response_no_marker(self) -> None:
        """Test extraction without assistant marker.

        Example:
            >>> text = "Just text"
            >>> response = _extract_response(text)
            >>> response == "Just text"
            True
        """
        full_text = "Just some text"
        response = _extract_response(full_text)
        assert response == full_text

    def test_extract_response_multiple_assistants(self) -> None:
        """Test extraction with multiple assistant responses.

        Example:
            >>> text = "User: A\\nAssistant: B\\nUser: C\\nAssistant: D"
            >>> response = _extract_response(text)
            >>> "D" in response
            True
        """
        full_text = "User: A\nAssistant: B\nUser: C\nAssistant: D"
        response = _extract_response(full_text)
        assert "D" in response


class TestUserInput:
    """Test user input handling functions.

    Example:
        >>> pytest.main([__file__, "::TestUserInput", "-v"])
    """

    def test_should_exit_quit(self) -> None:
        """Test exit detection with 'quit'.

        Example:
            >>> _should_exit("quit")
            True
        """
        assert _should_exit("quit")

    def test_should_exit_exit(self) -> None:
        """Test exit detection with 'exit'.

        Example:
            >>> _should_exit("exit")
            True
        """
        assert _should_exit("exit")

    def test_should_exit_normal_text(self) -> None:
        """Test no exit with normal text.

        Example:
            >>> _should_exit("hello")
            False
        """
        assert not _should_exit("hello")

    def test_should_exit_case_insensitive(self) -> None:
        """Test exit detection is case insensitive.

        Example:
            >>> _should_exit("QUIT")
            True
        """
        assert _should_exit("QUIT")
        assert _should_exit("Exit")

    def test_should_clear(self) -> None:
        """Test clear detection.

        Example:
            >>> _should_clear("clear")
            True
        """
        assert _should_clear("clear")

    def test_should_clear_case_insensitive(self) -> None:
        """Test clear detection is case insensitive.

        Example:
            >>> _should_clear("CLEAR")
            True
        """
        assert _should_clear("CLEAR")

    def test_should_clear_normal_text(self) -> None:
        """Test no clear with normal text.

        Example:
            >>> _should_clear("hello")
            False
        """
        assert not _should_clear("hello")


class TestOutputFormatting:
    """Test output formatting functions.

    Example:
        >>> pytest.main([__file__, "::TestOutputFormatting", "-v"])
    """

    @patch("builtins.print")
    def test_display_welcome(self, mock_print: MagicMock) -> None:
        """Test welcome message display.

        Example:
            >>> _display_welcome()  # Should print welcome message
        """
        _display_welcome()
        assert mock_print.called
        calls = [str(call) for call in mock_print.call_args_list]
        output = "".join(calls)
        assert "MICROGPT" in output or mock_print.call_count > 0

    @patch("builtins.print")
    def test_print_response(self, mock_print: MagicMock) -> None:
        """Test response printing.

        Example:
            >>> _print_response("Hello")  # Should print response
        """
        _print_response("Test response")
        assert mock_print.called


class TestCheckpointPath:
    """Test checkpoint path handling.

    Example:
        >>> pytest.main([__file__, "::TestCheckpointPath", "-v"])
    """

    @patch("os.path.exists")
    def test_get_checkpoint_path_exists(self, mock_exists: MagicMock) -> None:
        """Test getting checkpoint path when file exists.

        Example:
            >>> path = _get_checkpoint_path()
            >>> "finetuned_best_val" in path
            True
        """
        mock_exists.return_value = True
        path = _get_checkpoint_path()
        assert "finetuned_best_val" in path

    @patch("os.path.exists")
    def test_get_checkpoint_path_not_exists(self, mock_exists: MagicMock) -> None:
        """Test getting checkpoint path when file missing.

        Example:
            >>> path = _get_checkpoint_path()
            >>> # Raises FileNotFoundError if missing
        """
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError):
            _get_checkpoint_path()

    @patch("os.path.exists")
    def test_get_checkpoint_path_returns_string(self, mock_exists: MagicMock) -> None:
        """Test checkpoint path is a string.

        Example:
            >>> path = _get_checkpoint_path()
            >>> isinstance(path, str)
            True
        """
        mock_exists.return_value = True
        path = _get_checkpoint_path()
        assert isinstance(path, str)


class TestIntegration:
    """Integration tests for inference workflow.

    Example:
        >>> pytest.main([__file__, "::TestIntegration", "-v"])
    """

    def test_full_prompt_generation_flow(self, tokenizer) -> None:
        """Test full prompt generation workflow.

        Example:
            >>> tok = _load_tokenizer()
            >>> history = [("Hi", "Hello")]
            >>> prompt = _format_prompt(history, "How are you?")
            >>> len(prompt) > 0
            True
        """
        history = [("Hello", "Hi there")]
        prompt = _format_prompt(history, "How are you?")
        tokens = tokenizer.encode(prompt)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert len(decoded) > 0

    def test_model_generation(self, model, tokenizer, device) -> None:
        """Test model generation process.

        Example:
            >>> model = _create_model(50257, 128, 64, 4, 2, 0.1, "cpu")
            >>> tok = _load_tokenizer()
            >>> prompt = "Hello"
            >>> tokens = tok.encode(prompt)
            >>> output = model.generate(torch.tensor([tokens], device="cpu"), 10)
            >>> output.shape[1] > len(tokens)
            True
        """
        prompt = "Hello"
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)
        output = model.generate(input_ids, max_new_tokens=5)
        assert output.shape[0] == 1
        assert output.shape[1] > len(tokens)

    def test_conversation_history_format(self) -> None:
        """Test conversation history formatting.

        Example:
            >>> history = [("A", "B"), ("C", "D")]
            >>> prompt = _format_prompt(history, "E")
            >>> all(turn in prompt for turn in ["A", "B", "C", "D", "E"])
            True
        """
        history = [("Question 1", "Answer 1"), ("Question 2", "Answer 2")]
        prompt = _format_prompt(history, "Question 3")
        assert "Question 1" in prompt
        assert "Answer 1" in prompt
        assert "Question 2" in prompt
        assert "Answer 2" in prompt
        assert "Question 3" in prompt


class TestEdgeCases:
    """Test edge cases and error conditions.

    Example:
        >>> pytest.main([__file__, "::TestEdgeCases", "-v"])
    """

    def test_empty_prompt(self) -> None:
        """Test handling empty prompt.

        Example:
            >>> prompt = _format_prompt([], "")
            >>> "User:" in prompt
            True
        """
        prompt = _format_prompt([], "")
        assert "User:" in prompt
        assert "Assistant:" in prompt

    def test_very_long_history(self) -> None:
        """Test handling very long conversation history.

        Example:
            >>> history = [(f"Q{i}", f"A{i}") for i in range(100)]
            >>> prompt = _format_prompt(history, "Final")
            >>> "Final" in prompt
            True
        """
        history = [(f"Q{i}", f"A{i}") for i in range(100)]
        prompt = _format_prompt(history, "Final question")
        assert "Final question" in prompt

    def test_special_characters_in_input(self) -> None:
        """Test handling special characters.

        Example:
            >>> prompt = _format_prompt([], "Hello @#$%")
            >>> "@#$%" in prompt
            True
        """
        prompt = _format_prompt([], "Hello @#$%^&*()")
        assert "@#$%^&*()" in prompt

    def test_unicode_in_input(self) -> None:
        """Test handling unicode characters.

        Example:
            >>> prompt = _format_prompt([], "Hello 你好")
            >>> "你好" in prompt
            True
        """
        prompt = _format_prompt([], "Hello 你好")
        assert "你好" in prompt

    def test_multiline_input(self) -> None:
        """Test handling multiline input.

        Example:
            >>> prompt = _format_prompt([], "Line1\\nLine2")
            >>> "Line1" in prompt and "Line2" in prompt
            True
        """
        prompt = _format_prompt([], "Line1\nLine2\nLine3")
        assert "Line1" in prompt
        assert "Line2" in prompt
        assert "Line3" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=inference_micro_gpt", "--cov-report=term"])
