"""
Comprehensive Test Suite for Fine-Tuning GPT-2.

This module provides extensive testing for all components of the fine-tuning
functionality including data loading, model loading, checkpoint handling,
and conversational formatting.

Author: Kevin Thomas (ket189@pitt.edu)
License: MIT
"""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from fine_tune_micro_gpt import (
    _load_tokenizer,
    _format_conversation,
    _add_identity_examples,
    _process_conversation,
    _split_data,
    _should_save,
    _create_model_architecture,
    _handle_checkpoint,
    _extract_assistant_response,
    _create_config,
)
from micro_gpt import GPT2, GPT2Config


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
def temp_checkpoint(tmp_path):
    """Create temporary checkpoint file.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path to temporary checkpoint.

    Example:
        >>> path = temp_checkpoint(tmp_path)
        >>> isinstance(path, str)
        True
    """
    checkpoint_path = tmp_path / "test_checkpoint.pt"
    config = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
    model = GPT2(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    torch.save(
        {
            "step": 100,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": 5.0,
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


class TestConversationFormatting:
    """Test conversation formatting functions.

    Example:
        >>> pytest.main([__file__, "::TestConversationFormatting", "-v"])
    """

    def test_format_conversation_valid(self) -> None:
        """Test formatting valid conversation.

        Example:
            >>> example = {'history': 'Hi', 'human_ref_A': 'Hello'}
            >>> text = _format_conversation(example)
            >>> 'User:' in text
            True
        """
        example = {"history": "Hello", "human_ref_A": "Hi there"}
        result = _format_conversation(example)
        assert "User:" in result
        assert "Assistant:" in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_format_conversation_empty(self) -> None:
        """Test formatting empty conversation.

        Example:
            >>> example = {'history': '', 'human_ref_A': ''}
            >>> text = _format_conversation(example)
            >>> text == ''
            True
        """
        example = {"history": "", "human_ref_A": ""}
        result = _format_conversation(example)
        assert result == ""

    def test_format_conversation_missing_response(self) -> None:
        """Test formatting with missing response.

        Example:
            >>> example = {'history': 'Hi'}
            >>> text = _format_conversation(example)
            >>> text == ''
            True
        """
        example = {"history": "Hello"}
        result = _format_conversation(example)
        assert result == ""

    def test_add_identity_examples(self, tokenizer) -> None:
        """Test adding identity examples.

        Example:
            >>> tok = _load_tokenizer()
            >>> tokens = _add_identity_examples(tok)
            >>> len(tokens) > 0
            True
        """
        tokens = _add_identity_examples(tokenizer)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert "MicroGPT" in decoded
        assert "Kevin Thomas" in decoded


class TestDataProcessing:
    """Test data processing functions.

    Example:
        >>> pytest.main([__file__, "::TestDataProcessing", "-v"])
    """

    def test_process_conversation(self, tokenizer) -> None:
        """Test processing single conversation.

        Example:
            >>> tok = _load_tokenizer()
            >>> ex = {'history': 'Hi', 'human_ref_A': 'Hello'}
            >>> toks, tot, brk = _process_conversation(ex, tok, [], 0, 0, 1000)
            >>> len(toks) > 0
            True
        """
        example = {"history": "Hello", "human_ref_A": "Hi"}
        tokens, total, should_break = _process_conversation(
            example, tokenizer, [], 0, 0, 10000
        )
        assert isinstance(tokens, list)
        assert total > 0
        assert not should_break

    def test_process_conversation_max_tokens(self, tokenizer) -> None:
        """Test processing reaches max tokens.

        Example:
            >>> tok = _load_tokenizer()
            >>> ex = {'history': 'Hi', 'human_ref_A': 'Hello'}
            >>> toks, tot, brk = _process_conversation(ex, tok, [], 5, 0, 5)
            >>> brk
            True
        """
        example = {"history": "Hello", "human_ref_A": "Hi"}
        tokens, total, should_break = _process_conversation(
            example, tokenizer, [], 5, 0, 5
        )
        assert should_break

    def test_split_data(self) -> None:
        """Test data splitting.

        Example:
            >>> data = torch.randint(0, 1000, (100,))
            >>> train, val = _split_data(data, 0.1)
            >>> len(train) + len(val) == 100
            True
        """
        data = torch.randint(0, 1000, (1000,))
        train, val = _split_data(data, 0.1)
        assert len(train) == 900
        assert len(val) == 100
        assert len(train) + len(val) == len(data)

    def test_split_data_different_ratios(self) -> None:
        """Test splitting with different validation ratios.

        Example:
            >>> data = torch.randint(0, 1000, (100,))
            >>> train, val = _split_data(data, 0.2)
            >>> len(train) == 80
            True
        """
        data = torch.randint(0, 1000, (1000,))
        train, val = _split_data(data, 0.2)
        assert len(train) == 800
        assert len(val) == 200


class TestModelLoading:
    """Test model loading and creation.

    Example:
        >>> pytest.main([__file__, "::TestModelLoading", "-v"])
    """

    def test_create_model_architecture(self, device) -> None:
        """Test creating model architecture.

        Example:
            >>> model = _create_model_architecture(256, 128, 64, 4, 2, 0.1, "cpu")
            >>> isinstance(model, GPT2)
            True
        """
        model = _create_model_architecture(256, 128, 64, 4, 2, 0.1, device)
        assert isinstance(model, GPT2)
        assert model.config.block_size == 64

    def test_model_parameters(self, device) -> None:
        """Test model has correct parameters.

        Example:
            >>> model = _create_model_architecture(256, 128, 64, 4, 2, 0.1, "cpu")
            >>> sum(p.numel() for p in model.parameters()) > 0
            True
        """
        model = _create_model_architecture(256, 128, 64, 4, 2, 0.1, device)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0


class TestCheckpointing:
    """Test checkpoint handling.

    Example:
        >>> pytest.main([__file__, "::TestCheckpointing", "-v"])
    """

    def test_should_save_better_loss(self) -> None:
        """Test saving when validation loss improves.

        Example:
            >>> _should_save(2.5, 3.0)
            True
        """
        assert _should_save(2.5, 3.0)

    def test_should_save_worse_loss(self) -> None:
        """Test not saving when validation loss worsens.

        Example:
            >>> _should_save(3.0, 2.5)
            False
        """
        assert not _should_save(3.0, 2.5)

    def test_should_save_equal_loss(self) -> None:
        """Test not saving when validation loss unchanged.

        Example:
            >>> _should_save(2.5, 2.5)
            False
        """
        assert not _should_save(2.5, 2.5)

    def test_handle_checkpoint_improvement(self, device, tmp_path) -> None:
        """Test checkpoint handling with improvement.

        Example:
            >>> cfg = GPT2Config(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=128)
            >>> model = GPT2(cfg)
            >>> opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            >>> best, is_best = _handle_checkpoint(2.5, 3.0, model, opt, 100)
            >>> is_best
            True
        """
        os.makedirs(tmp_path / "checkpoints", exist_ok=True)
        os.chdir(tmp_path)
        model = _create_model_architecture(256, 128, 64, 4, 2, 0.1, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        best_loss, is_best = _handle_checkpoint(2.5, 3.0, model, optimizer, 100)
        assert best_loss == 2.5
        assert is_best


class TestResponseExtraction:
    """Test response extraction functions.

    Example:
        >>> pytest.main([__file__, "::TestResponseExtraction", "-v"])
    """

    def test_extract_assistant_response(self) -> None:
        """Test extracting assistant response.

        Example:
            >>> text = "User: Hi\\nAssistant: Hello"
            >>> response = _extract_assistant_response(text)
            >>> "Hello" in response
            True
        """
        full_text = "User: Hi\nAssistant: Hello there\nUser: Bye"
        response = _extract_assistant_response(full_text)
        assert "Hello there" in response
        assert "User:" not in response

    def test_extract_no_assistant_marker(self) -> None:
        """Test extraction without assistant marker.

        Example:
            >>> text = "Just some text"
            >>> response = _extract_assistant_response(text)
            >>> response == "Just some text"
            True
        """
        full_text = "Just some text"
        response = _extract_assistant_response(full_text)
        assert response == full_text


class TestConfiguration:
    """Test configuration creation.

    Example:
        >>> pytest.main([__file__, "::TestConfiguration", "-v"])
    """

    @patch("fine_tune_micro_gpt._find_best_checkpoint")
    @patch("fine_tune_micro_gpt.load_config")
    def test_create_config(
        self, mock_load_config: MagicMock, mock_find_checkpoint: MagicMock
    ) -> None:
        """Test creating configuration dictionary.

        Example:
            >>> config = _create_config()
            >>> config['epochs'] == 10000
            True
        """
        mock_cfg = MagicMock()
        mock_cfg.block_size = 1024
        mock_cfg.n_embd = 1024
        mock_cfg.n_head = 16
        mock_cfg.n_layer = 24
        mock_cfg.dropout = 0.1
        mock_cfg.batch_size = 4
        mock_cfg.finetune_lr = 1e-5
        mock_cfg.finetune_epochs = 10000
        mock_cfg.finetune_eval_interval = 100
        mock_cfg.finetune_eval_iters = 50
        mock_cfg.finetune_max_tokens = 20000000
        mock_cfg.finetune_temperature = 0.7
        mock_cfg.finetune_max_new_tokens = 150
        mock_cfg.finetune_top_p = 0.9
        mock_load_config.return_value = mock_cfg
        mock_find_checkpoint.return_value = "checkpoints/best_val.pt"
        config = _create_config()
        assert isinstance(config, dict)
        assert "epochs" in config
        assert config["epochs"] == 10000
        assert "block_size" in config
        assert "n_embd" in config

    @patch("fine_tune_micro_gpt._find_best_checkpoint")
    @patch("fine_tune_micro_gpt.load_config")
    def test_config_has_required_keys(
        self, mock_load_config: MagicMock, mock_find_checkpoint: MagicMock
    ) -> None:
        """Test config has all required keys.

        Example:
            >>> config = _create_config()
            >>> all(k in config for k in ['epochs', 'lr', 'device'])
            True
        """
        mock_cfg = MagicMock()
        mock_cfg.block_size = 1024
        mock_cfg.n_embd = 1024
        mock_cfg.n_head = 16
        mock_cfg.n_layer = 24
        mock_cfg.dropout = 0.1
        mock_cfg.batch_size = 4
        mock_cfg.finetune_lr = 1e-5
        mock_cfg.finetune_epochs = 10000
        mock_cfg.finetune_eval_interval = 100
        mock_cfg.finetune_eval_iters = 50
        mock_cfg.finetune_max_tokens = 20000000
        mock_cfg.finetune_temperature = 0.7
        mock_cfg.finetune_max_new_tokens = 150
        mock_cfg.finetune_top_p = 0.9
        mock_load_config.return_value = mock_cfg
        mock_find_checkpoint.return_value = "checkpoints/best_val.pt"
        config = _create_config()
        required_keys = [
            "checkpoint_path",
            "block_size",
            "n_embd",
            "n_head",
            "n_layer",
            "dropout",
            "batch_size",
            "lr",
            "epochs",
            "eval_interval",
            "eval_iters",
            "max_tokens",
            "device",
        ]
        assert all(key in config for key in required_keys)


class TestIntegration:
    """Integration tests for fine-tuning workflow.

    Example:
        >>> pytest.main([__file__, "::TestIntegration", "-v"])
    """

    @patch("fine_tune_micro_gpt._find_best_checkpoint")
    @patch("fine_tune_micro_gpt.load_config")
    def test_full_config_creation(
        self, mock_load_config: MagicMock, mock_find_checkpoint: MagicMock
    ) -> None:
        """Test full configuration creation workflow.

        Example:
            >>> config = _create_config()
            >>> isinstance(config['device'], str)
            True
        """
        mock_cfg = MagicMock()
        mock_cfg.block_size = 1024
        mock_cfg.n_embd = 1024
        mock_cfg.n_head = 16
        mock_cfg.n_layer = 24
        mock_cfg.dropout = 0.1
        mock_cfg.batch_size = 4
        mock_cfg.finetune_lr = 1e-5
        mock_cfg.finetune_epochs = 10000
        mock_cfg.finetune_eval_interval = 100
        mock_cfg.finetune_eval_iters = 50
        mock_cfg.finetune_max_tokens = 20000000
        mock_cfg.finetune_temperature = 0.7
        mock_cfg.finetune_max_new_tokens = 150
        mock_cfg.finetune_top_p = 0.9
        mock_load_config.return_value = mock_cfg
        mock_find_checkpoint.return_value = "checkpoints/best_val.pt"
        config = _create_config()
        assert isinstance(config["device"], str)
        assert config["device"] in ["cuda", "mps", "cpu"]

    @patch("fine_tune_micro_gpt._find_best_checkpoint")
    @patch("fine_tune_micro_gpt.load_config")
    def test_model_creation_with_config(
        self,
        mock_load_config: MagicMock,
        mock_find_checkpoint: MagicMock,
        device,
    ) -> None:
        """Test creating model from config.

        Example:
            >>> config = _create_config()
            >>> model = _create_model_architecture(50257, config['n_embd'], config['block_size'], config['n_head'], config['n_layer'], config['dropout'], "cpu")
            >>> isinstance(model, GPT2)
            True
        """
        mock_cfg = MagicMock()
        mock_cfg.block_size = 64
        mock_cfg.n_embd = 128
        mock_cfg.n_head = 4
        mock_cfg.n_layer = 2
        mock_cfg.dropout = 0.1
        mock_cfg.batch_size = 4
        mock_cfg.finetune_lr = 1e-5
        mock_cfg.finetune_epochs = 10000
        mock_cfg.finetune_eval_interval = 100
        mock_cfg.finetune_eval_iters = 50
        mock_cfg.finetune_max_tokens = 20000000
        mock_cfg.finetune_temperature = 0.7
        mock_cfg.finetune_max_new_tokens = 150
        mock_cfg.finetune_top_p = 0.9
        mock_load_config.return_value = mock_cfg
        mock_find_checkpoint.return_value = "checkpoints/best_val.pt"
        config = _create_config()
        model = _create_model_architecture(
            50257,
            config["n_embd"],
            config["block_size"],
            config["n_head"],
            config["n_layer"],
            config["dropout"],
            device,
        )
        assert isinstance(model, GPT2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=fine_tune_micro_gpt", "--cov-report=term"])
