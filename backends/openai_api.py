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


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class OpenAIBackend(CaptionBackend):
    def __init__(self, model: str, settings: Settings, prompt: str = DEFAULT_PROMPT) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI backend requires: pip install openai tenacity\n"
                "Or: uv run --extra api python caption.py run --backend openai"
            ) from e

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set in config.env. "
                "Expected: OPENAI_API_KEY=sk-..."
            )

        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model
        self.prompt = prompt

    def caption(self, image: Image.Image) -> str:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
        from openai import RateLimitError, APIConnectionError

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call():
            data_url = _image_to_data_url(image)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()

        return _call()


class XAIBackend(CaptionBackend):
    def __init__(self, model: str, settings: Settings, prompt: str = DEFAULT_PROMPT) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "xAI backend requires: pip install openai tenacity\n"
                "Or: uv run --extra api python caption.py run --backend xai"
            ) from e

        if not settings.XAI_API_KEY:
            raise ValueError(
                "XAI_API_KEY is not set in config.env. "
                "Expected: XAI_API_KEY=xai-..."
            )

        self.client = openai.OpenAI(
            api_key=settings.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
        self.model = model
        self.prompt = prompt

    def caption(self, image: Image.Image) -> str:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
        from openai import RateLimitError, APIConnectionError

        @retry(
            retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _call():
            data_url = _image_to_data_url(image)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                            {"type": "text", "text": self.prompt},
                        ],
                    }
                ],
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()

        return _call()
