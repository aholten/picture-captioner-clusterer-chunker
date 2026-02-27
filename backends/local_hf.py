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


class LocalHFBackend(CaptionBackend):
    def __init__(self, model: str, settings: Settings, prompt: str = DEFAULT_PROMPT) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError(
                "Local backend requires: pip install torch transformers bitsandbytes accelerate\n"
                "Or: uv run --extra local python caption.py run --backend local"
            ) from e

        from transformers import BitsAndBytesConfig

        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        logger.info("Loading model %s (4-bit) on %s...", model, self.device)

        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            quantization_config=quantization_config,
            device_map="auto",
        )
        logger.info("Model loaded.")

    def caption(self, image: Image.Image) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=256)

        # Strip the input tokens from the output
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return result.strip()
