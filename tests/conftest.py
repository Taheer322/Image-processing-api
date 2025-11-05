import pytest
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import sys
from pathlib import Path
#importing all the src modules
import src.analyzer
import src.models
import src.utils

# Add parent directory to path so we can import main.py
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def sample_image():
    """Create a sample test image with neutral gray"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    return img

@pytest.fixture
def blurry_image():
    """Create a blurry test image"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    img = cv2.GaussianBlur(img, (21, 21), 0)
    return img

@pytest.fixture
def bright_image():
    """Create a bright test image"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    return img

@pytest.fixture
def dark_image():
    """Create a dark test image"""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 50
    return img

@pytest.fixture
def high_contrast_image():
    """Create high contrast image"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img

@pytest.fixture
def test_image_bytes(sample_image):
    """Convert image to bytes for file upload"""
    _, buffer = cv2.imencode('.jpg', sample_image)
    return buffer.tobytes()

@pytest.fixture
def test_image_file(test_image_bytes):
    """Create test image file object"""
    return BytesIO(test_image_bytes), "test.jpg"

@pytest.fixture
def test_png_file():
    """Create test PNG file"""
    img = Image.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer, "test.png"

@pytest.fixture
def invalid_file():
    """Create invalid file"""
    return BytesIO(b"not an image"), "invalid.jpg"

@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create temporary upload directory"""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    return upload_dir