import logging
import time
from typing import Any, cast

import httpx

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    def __init__(self, hf_token: str | None = None, timeout: float = 30.0, max_retries: int = 3):
        # Configure with longer timeout and connection pooling
        self.timeout = httpx.Timeout(timeout=timeout, connect=10.0)
        self.max_retries = max_retries

        # Use Transport with retry configuration
        transport = httpx.HTTPTransport(
            retries=max_retries, limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        self.client = httpx.Client(timeout=self.timeout, transport=transport, follow_redirects=True)

        if hf_token:
            self.client.headers.update({"Authorization": f"Bearer {hf_token}"})

    def _make_request_with_retry(self, url: str) -> httpx.Response | None:
        """Make HTTP request with exponential backoff retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.get(url)
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as e:
                if attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s
                    sleep_time = 2**attempt
                    logger.warning(
                        f"Request to {url} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {sleep_time}s: {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Request to {url} failed after {self.max_retries + 1} attempts: {e}")

        return None

    def get_model_details(self, model_name: str, version: str = "main") -> dict[Any, Any]:
        """Get model details from HuggingFace API with error handling."""
        url = f"https://huggingface.co/api/models/{model_name}/revision/{version}"

        try:
            response = self._make_request_with_retry(url)
            if response and response.status_code == 200:
                return cast(dict[Any, Any], response.json())
        except Exception as e:
            logger.error(f"Error fetching model details for {model_name}: {e}")

        return {}

    def get_model_tree(self, model_name: str, version: str = "main") -> list[Any]:
        """Get model file tree from HuggingFace API with error handling."""
        url = f"https://huggingface.co/api/models/{model_name}/tree/{version}"

        try:
            response = self._make_request_with_retry(url)
            if response and response.status_code == 200:
                return cast(list[Any], response.json())
        except Exception as e:
            logger.error(f"Error fetching model tree for {model_name}: {e}")

        return []

    def __del__(self) -> None:
        """Ensure client is properly closed."""
        if hasattr(self, "client"):
            self.client.close()
