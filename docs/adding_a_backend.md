# Adding a New Backend

## Steps

1. Create `backends/my_backend.py` implementing `CaptionBackend`
2. Register it in `backends/__init__.py` inside `load_backend()`
3. Add any API key field to `config.py` `Settings`
4. Add SDK to `api` optional deps in `pyproject.toml`
5. Add a row to `docs/api_providers.md`

`caption.py` requires zero changes.

## Template

```python
from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

from PIL import Image

from .base import CaptionBackend

if TYPE_CHECKING:
    from config import Settings

DEFAULT_PROMPT = (
    "Describe this photo in one or two sentences. "
    "Focus on the main subject, setting, and activity."
)


class MyBackend(CaptionBackend):
    def __init__(self, model: str, settings: Settings, prompt: str = DEFAULT_PROMPT) -> None:
        import my_sdk  # lazy import — missing dep gives clear error

        if not settings.MY_API_KEY:
            raise ValueError("MY_API_KEY is not set in config.env.")

        self.client = my_sdk.Client(api_key=settings.MY_API_KEY)
        self.model = model
        self.prompt = prompt

    def caption(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return self.client.generate(model=self.model, image_b64=b64, prompt=self.prompt)
```

## Registration

In `backends/__init__.py`, add a branch to `load_backend()`:

```python
if name == "my_backend":
    from .my_backend import MyBackend
    return MyBackend(model, settings)
```

The lazy import inside the `if` block means users who don't use your backend never need its SDK installed.
