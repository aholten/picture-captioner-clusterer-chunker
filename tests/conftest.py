import pytest
from PIL import Image


@pytest.fixture
def tmp_journal(tmp_path):
    return tmp_path / "captions.jsonl"


@pytest.fixture
def sample_image():
    """A small valid RGB PIL image."""
    return Image.new("RGB", (64, 64), color=(128, 200, 50))


@pytest.fixture
def sample_photos_dir(tmp_path, sample_image):
    """A temp directory with a few test image files."""
    photos = tmp_path / "photos"
    photos.mkdir()
    sub = photos / "2024"
    sub.mkdir()

    # Valid PNG
    img_path = sub / "test.png"
    sample_image.save(img_path)

    # Valid JPEG
    jpg_path = sub / "test.jpg"
    sample_image.save(jpg_path)

    # Corrupt file
    corrupt_path = sub / "corrupt.jpg"
    corrupt_path.write_bytes(b"not an image")

    return photos
