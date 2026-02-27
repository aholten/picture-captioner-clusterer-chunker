from pathlib import Path

from PIL import Image

from backends.base import CorruptImageError


def load_image(path: Path) -> Image.Image:
    """Open image, convert HEIC, strip EXIF, return RGB PIL.Image.

    Raises CorruptImageError on any I/O or decode failure.
    """
    try:
        if path.suffix.lower() in (".heic", ".heif"):
            from pillow_heif import register_heif_opener

            register_heif_opener()
        img = Image.open(path)
        img.load()  # force full decode
        img = img.convert("RGB")
        # Strip EXIF by rebuilding image data
        data = list(img.get_flattened_data())
        clean = Image.new(img.mode, img.size)
        clean.putdata(data)
        return clean
    except CorruptImageError:
        raise
    except Exception as e:
        raise CorruptImageError(path, e) from e
