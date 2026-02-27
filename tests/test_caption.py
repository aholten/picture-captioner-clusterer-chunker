import pytest
from PIL import Image

from backends import CorruptImageError, load_backend
from backends.mock import MockBackend
from image_loader import load_image


def test_mock_backend_returns_caption(sample_image):
    backend = MockBackend(model="test-model")
    result = backend.caption(sample_image)
    assert result == "a mock caption for test-model"


def test_mock_backend_error_rate(sample_image):
    backend = MockBackend(model="test", error_rate=1.0)  # 100% error rate
    with pytest.raises(CorruptImageError):
        backend.caption(sample_image)


def test_mock_backend_zero_error_rate(sample_image):
    backend = MockBackend(model="test", error_rate=0.0)
    # Should never raise
    for _ in range(20):
        result = backend.caption(sample_image)
        assert isinstance(result, str)


def test_load_image_valid_png(sample_photos_dir):
    img = load_image(sample_photos_dir / "2024" / "test.png")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_valid_jpg(sample_photos_dir):
    img = load_image(sample_photos_dir / "2024" / "test.jpg")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_corrupt_raises(sample_photos_dir):
    with pytest.raises(CorruptImageError):
        load_image(sample_photos_dir / "2024" / "corrupt.jpg")


def test_load_image_strips_exif(sample_photos_dir):
    img = load_image(sample_photos_dir / "2024" / "test.png")
    # Rebuilt image should have no EXIF info attribute
    assert not img.info


def test_load_backend_mock():
    backend = load_backend("mock", "test-model", None)
    assert isinstance(backend, MockBackend)


def test_load_backend_unknown():
    with pytest.raises(ValueError, match="Unknown backend"):
        load_backend("nonexistent", "model", None)
