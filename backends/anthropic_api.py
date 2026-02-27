from __future__ import annotations

import base64
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


class AnthropicBackend(CaptionBackend):
    def __init__(self, model: str, settings: Settings, prompt: str = DEFAULT_PROMPT) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Anthropic backend requires: pip install anthropic tenacity\n"
                "Or: uv run --extra api python caption.py run --backend anthropic"
            ) from e

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set in config.env. "
                "Expected: ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = model
        self.prompt = prompt

    def caption(self, image: Image.Image) -> str:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
        from anthropic import RateLimitError, APIConnectionError

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call():
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64,
                                },
                            },
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text.strip()

        return _call()
