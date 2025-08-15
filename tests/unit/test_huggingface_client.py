"""Unit tests for HuggingFace client."""

from unittest.mock import Mock, patch

import httpx
import pytest

from hl_scai.clients.huggingface import HuggingFaceClient


class TestHuggingFaceClient:
    """Test cases for HuggingFaceClient."""

    @pytest.fixture
    def hf_client(self):
        """Create a HuggingFace client instance."""
        return HuggingFaceClient(hf_token="test_token")

    @pytest.mark.unit
    def test_client_initialization(self):
        """Test client initialization with and without token."""
        # With token
        client = HuggingFaceClient(hf_token="test_token")
        assert hasattr(client, "client")
        assert isinstance(client.client, httpx.Client)
        assert client.timeout.read == 30.0  # httpx.Timeout has read, write, connect, pool attributes
        assert client.max_retries == 3

        # Without token
        client = HuggingFaceClient()
        assert hasattr(client, "client")
        assert isinstance(client.client, httpx.Client)

        # With custom timeout and retries
        client = HuggingFaceClient(timeout=60.0, max_retries=5)
        assert client.timeout.read == 60.0
        assert client.max_retries == 5

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_model_tree_success(self, mock_get, hf_client):
        """Test successful model tree retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"path": "model.safetensors", "size": 1000000, "oid": "abc123", "lfs": {"oid": "def456"}},
            {"path": "config.json", "size": 500, "oid": "config123"},
        ]
        mock_get.return_value = mock_response

        result = hf_client.get_model_tree("bert-base-uncased", "main")

        assert len(result) == 2
        assert result[0]["path"] == "model.safetensors"
        assert result[1]["path"] == "config.json"

        # Verify API call - no follow_redirects since it's set on client
        expected_url = "https://huggingface.co/api/models/bert-base-uncased/tree/main"
        mock_get.assert_called_once_with(expected_url)

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_model_tree_not_found(self, mock_get, hf_client):
        """Test model tree retrieval for non-existent model."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = hf_client.get_model_tree("nonexistent-model", "main")

        assert result == []

    @pytest.mark.unit
    @patch("httpx.Client.get")
    @patch("time.sleep")  # Mock sleep to speed up test
    def test_get_model_tree_error_with_retry(self, mock_sleep, mock_get, hf_client):
        """Test model tree retrieval with API error and retry logic."""
        mock_get.side_effect = httpx.TimeoutException("Connection timeout")

        # Now returns empty list instead of raising
        result = hf_client.get_model_tree("bert-base-uncased", "main")
        assert result == []

        # Verify retries happened (initial + 3 retries = 4 calls)
        assert mock_get.call_count == 4

        # Verify exponential backoff sleep calls
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_call(1)  # 2^0
        mock_sleep.assert_any_call(2)  # 2^1
        mock_sleep.assert_any_call(4)  # 2^2

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_model_details_success(self, mock_get, hf_client):
        """Test successful model details retrieval."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "modelId": "bert-base-uncased",
            "pipeline_tag": "text-classification",
            "library_name": "transformers",
            "safetensors": {"total": 110000000},
            "cardData": {"license": "apache-2.0", "tags": ["transformers", "pytorch"]},
        }
        mock_get.return_value = mock_response

        result = hf_client.get_model_details("bert-base-uncased", version="main")

        assert result["modelId"] == "bert-base-uncased"
        assert result["pipeline_tag"] == "text-classification"
        assert result["cardData"]["license"] == "apache-2.0"

        # Verify API call - no follow_redirects since it's set on client
        expected_url = "https://huggingface.co/api/models/bert-base-uncased/revision/main"
        mock_get.assert_called_once_with(expected_url)

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_get_model_details_not_found(self, mock_get, hf_client):
        """Test model details retrieval for non-existent model."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = hf_client.get_model_details("nonexistent-model")

        assert result == {}  # Returns empty dict, not None

    @pytest.mark.unit
    @patch("httpx.Client.get")
    @patch("time.sleep")  # Mock sleep to speed up test
    def test_get_model_details_error_with_retry(self, mock_sleep, mock_get, hf_client):
        """Test model details retrieval with API error and retry logic."""
        mock_get.side_effect = httpx.NetworkError("Network unreachable")

        # Now returns empty dict instead of raising
        result = hf_client.get_model_details("bert-base-uncased")
        assert result == {}

        # Verify retries happened
        assert mock_get.call_count == 4

    @pytest.mark.unit
    @patch("httpx.Client")
    def test_api_calls_without_token(self, mock_client_class):
        """Test API calls work without authentication token."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        HuggingFaceClient()  # No token

        # Verify headers.update was not called with Authorization
        assert not mock_client.headers.update.called

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_version_defaults(self, mock_get, hf_client):
        """Test that version defaults to 'main' when not specified."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        # Test get_model_tree without version
        hf_client.get_model_tree("bert-base-uncased")
        expected_url = "https://huggingface.co/api/models/bert-base-uncased/tree/main"
        mock_get.assert_called_with(expected_url)

        mock_get.reset_mock()

        # Test get_model_details without version
        hf_client.get_model_details("bert-base-uncased")
        expected_url = "https://huggingface.co/api/models/bert-base-uncased/revision/main"
        mock_get.assert_called_with(expected_url)

    @pytest.mark.unit
    @patch("httpx.Client.get")
    def test_json_decode_error_handling(self, mock_get, hf_client):
        """Test handling of JSON decode errors."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        # Should return empty results on JSON errors
        result = hf_client.get_model_tree("bert-base-uncased")
        assert result == []

        result = hf_client.get_model_details("bert-base-uncased")
        assert result == {}

    @pytest.mark.unit
    @patch("httpx.Client.get")
    @patch("time.sleep")
    def test_partial_retry_success(self, mock_sleep, mock_get, hf_client):
        """Test successful response after some retries."""
        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"modelId": "test"}

        mock_get.side_effect = [
            httpx.TimeoutException("Timeout 1"),
            httpx.TimeoutException("Timeout 2"),
            mock_response,
        ]

        result = hf_client.get_model_details("test-model")
        assert result == {"modelId": "test"}
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
