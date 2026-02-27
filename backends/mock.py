import random

from PIL import Image

from .base import CaptionBackend, CorruptImageError


class MockBackend(CaptionBackend):
    def __init__(self, model: str = "mock", error_rate: float = 0.0, **kwargs) -> None:
        self.model = model
        self.error_rate = error_rate

    def caption(self, image: Image.Image) -> str:
        if self.error_rate > 0 and random.random() < self.error_rate:
            raise CorruptImageError(
                path="<mock>", cause=Exception("simulated error")
            )
        return f"a mock caption for {self.model}"
