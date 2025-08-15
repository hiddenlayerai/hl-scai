"""Unit tests for configuration management."""

import os
from unittest.mock import patch

import pytest

from hl_scai.config.settings import AgentConfig, HuggingFaceConfig, get_config, set_config


class TestHuggingFaceConfig:
    """Test cases for HuggingFaceConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default initialization."""
        config = HuggingFaceConfig()
        assert config.huggingface_token is None

    @pytest.mark.unit
    def test_initialization_with_token(self):
        """Test initialization with token."""
        config = HuggingFaceConfig(huggingface_token="test_token")
        assert config.huggingface_token == "test_token"


class TestAgentConfig:
    """Test cases for AgentConfig."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test default initialization."""
        config = AgentConfig()
        assert config.hf_config is not None
        assert isinstance(config.hf_config, HuggingFaceConfig)
        assert config.hf_config.huggingface_token is None

    @pytest.mark.unit
    def test_initialization_with_hf_config(self):
        """Test initialization with custom HuggingFace config."""
        hf_config = HuggingFaceConfig(huggingface_token="custom_token")
        config = AgentConfig(hf_config=hf_config)
        assert config.hf_config.huggingface_token == "custom_token"

    @pytest.mark.unit
    @patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "env_token"})
    def test_from_env_with_env_var(self):
        """Test from_env method with environment variable."""
        config = AgentConfig.from_env()
        assert config.hf_config.huggingface_token == "env_token"

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_without_env_var(self):
        """Test from_env method without environment variable."""
        config = AgentConfig.from_env()
        assert config.hf_config.huggingface_token is None

    @pytest.mark.unit
    @patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "env_token"})
    def test_from_env_with_kwargs_override(self):
        """Test from_env method - currently only supports env vars."""
        # Note: from_env doesn't currently support kwargs, it uses os.getenv
        config = AgentConfig.from_env()
        assert config.hf_config.huggingface_token == "env_token"

    @pytest.mark.unit
    @patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "kwargs_token"})
    def test_from_env_with_env_override(self):
        """Test from_env method uses environment variable."""
        config = AgentConfig.from_env()
        assert config.hf_config.huggingface_token == "kwargs_token"


class TestConfigGlobalFunctions:
    """Test cases for global config functions."""

    def setup_method(self):
        """Reset global config before each test."""
        import hl_scai.config.settings

        hl_scai.config.settings._config = None

    @pytest.mark.unit
    def test_get_config_creates_default(self):
        """Test get_config creates default config if none exists."""
        config = get_config()
        assert isinstance(config, AgentConfig)
        assert config.hf_config is not None

        # Should return same instance on subsequent calls
        config2 = get_config()
        assert config is config2

    @pytest.mark.unit
    def test_set_config(self):
        """Test set_config sets global config."""
        custom_config = AgentConfig(hf_config=HuggingFaceConfig(huggingface_token="custom_token"))
        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.hf_config.huggingface_token == "custom_token"

    @pytest.mark.unit
    @patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "env_token"})
    def test_get_config_uses_env(self):
        """Test get_config uses environment variables."""
        config = get_config()
        assert config.hf_config.huggingface_token == "env_token"
