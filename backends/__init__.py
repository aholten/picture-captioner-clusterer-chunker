from __future__ import annotations

from typing import TYPE_CHECKING

from .base import CaptionBackend, CorruptImageError
from .mock import MockBackend

if TYPE_CHECKING:
    from config import Settings

BACKENDS: dict[str, type[CaptionBackend]] = {"mock": MockBackend}


def load_backend(name: str, model: str, settings: Settings) -> CaptionBackend:
    if name == "mock":
        return MockBackend(model=model)
    if name == "local":
        from .local_hf import LocalHFBackend

        return LocalHFBackend(model, settings)
    if name == "openai":
        from .openai_api import OpenAIBackend

        return OpenAIBackend(model, settings)
    if name == "xai":
        from .openai_api import XAIBackend

        return XAIBackend(model, settings)
    if name == "anthropic":
        from .anthropic_api import AnthropicBackend

        return AnthropicBackend(model, settings)
    if name == "gemini":
        from .gemini_api import GeminiBackend

        return GeminiBackend(model, settings)
    raise ValueError(f"Unknown backend: {name!r}. Available: mock, local, openai, xai, anthropic, gemini")
