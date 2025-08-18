"""Pytest configuration and shared fixtures."""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file with AI model usage."""
    file_path = Path(temp_dir) / "sample.py"
    content = """
import openai
from transformers import pipeline

# Constants
MODEL_NAME = "gpt-3.5-turbo"
HF_MODEL = "bert-base-uncased"

class AIService:
    DEFAULT_MODEL = "gpt-4"

    def __init__(self, model=None):
        self.model = model or self.DEFAULT_MODEL
        self.client = openai.OpenAI()

    def generate_text(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response

    def analyze_sentiment(self):
        analyzer = pipeline("sentiment-analysis", model=HF_MODEL)
        return analyzer
"""
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def empty_python_file(temp_dir):
    """Create an empty Python file."""
    file_path = Path(temp_dir) / "empty.py"
    file_path.write_text("")
    return str(file_path)


@pytest.fixture
def syntax_error_file(temp_dir):
    """Create a Python file with syntax errors."""
    file_path = Path(temp_dir) / "syntax_error.py"
    content = """
def broken_function(
    # Missing closing parenthesis
    pass
"""
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def mock_hf_client(mocker):
    """Mock HuggingFace client."""
    mock_client = mocker.Mock()
    mock_client.get_model_tree.return_value = [
        {"path": "model.safetensors", "size": 1000000, "oid": "abc123", "lfs": {"oid": "def456"}},
        {"path": "LICENSE", "size": 1234, "oid": "license123"},
    ]
    mock_client.get_model_details.return_value = {
        "pipeline_tag": "text-generation",
        "safetensors": {"total": 7000000000},
        "library_name": "transformers",
        "cardData": {"license": "apache-2.0"},
    }
    return mock_client
