import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HuggingFaceConfig:
    huggingface_token: str | None = None


@dataclass
class AgentConfig:
    hf_config: HuggingFaceConfig | None = field(default_factory=HuggingFaceConfig)

    @classmethod
    def from_env(cls, *args: Any, **kwargs: Any) -> "AgentConfig":
        config = cls(*args, **kwargs)

        if config.hf_config is not None:
            config.hf_config.huggingface_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or kwargs.get("huggingface_token")

        return config


# Global configuration instance
_config: AgentConfig | None = None


def get_config() -> AgentConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = AgentConfig.from_env()
    return _config


def set_config(config: AgentConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config
