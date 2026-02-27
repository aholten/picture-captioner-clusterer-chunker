from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image


class CorruptImageError(Exception):
    def __init__(self, path: Path, cause: Exception) -> None:
        self.path = path
        self.cause = cause
        super().__init__(f"Cannot open image {path}: {cause}")


class CaptionBackend(ABC):
    @abstractmethod
    def caption(self, image: Image.Image) -> str: ...
