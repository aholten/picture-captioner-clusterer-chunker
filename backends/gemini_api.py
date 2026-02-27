from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

from .base import CaptionBackend

if TYPE_CHECKING:
    from config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Describe this photo in one or two sentences. "
    "Focus on the main subject, setting, and activity."
)


class GeminiBackend(CaptionBackend):
    def __init__(self, model: str, settings: Settings, prompt: str = DEFAULT_PROMPT) -> None:
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "Gemini backend requires: pip install google-generativeai tenacity\n"
                "Or: uv run --extra api python caption.py run --backend gemini"
            ) from e

        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set in config.env. "
                "Expected: GEMINI_API_KEY=AI..."
            )

        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.genai = genai
        self.model = genai.GenerativeModel(model)
        self.prompt = prompt

    def caption(self, image: Image.Image) -> str:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
        from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

        @retry(
            retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call():
            response = self.model.generate_content([self.prompt, image])
            return response.text.strip()

        return _call()
