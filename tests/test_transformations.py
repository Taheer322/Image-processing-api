import pytest
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import ImageAnalyzer

class TestImageTransformations:
    """Test image transformation methods"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        return img
    
    def test_resize_image_fixed(self, sample_image):
        """Test resizing to fixed dimensions"""
        resized = ImageAnalyzer.resize_image(sample_image, 50, 50)
        assert resized.shape == (50, 50, 3), "Should resize to 50x50"
    
    def test_resize_by_percentage(self, sample_image):
        """Test resizing by percentage"""
        resized = ImageAnalyzer.resize_by_percentage(sample_image, 50)
        assert resized.shape == (50, 50, 3), "50% should be 50x50"
    
    def test_resize_invalid_percentage(self, sample_image):
        """Test invalid percentage"""
        with pytest.raises(ValueError):
            ImageAnalyzer.resize_by_percentage(sample_image, 0)
    
    def test_crop_image(self, sample_image):
        """Test cropping image"""
        cropped = ImageAnalyzer.crop_image(sample_image, 10, 10, 50, 50)
        assert cropped.shape == (50, 50, 3), "Should crop to 50x50"
    
    def test_crop_invalid_area(self, sample_image):
        """Test invalid crop area"""
        with pytest.raises(ValueError):
            ImageAnalyzer.crop_image(sample_image, 0, 0, 200, 200)
    
    def test_crop_negative_position(self, sample_image):
        """Test negative crop position"""
        with pytest.raises(ValueError):
            ImageAnalyzer.crop_image(sample_image, -1, 0, 50, 50)
    
    def test_apply_filter_blur(self, sample_image):
        """Test blur filter"""
        filtered = ImageAnalyzer.apply_filter(sample_image, "blur")
        assert filtered.shape == sample_image.shape, "Shape should be preserved"
    
    def test_apply_filter_sharpen(self, sample_image):
        """Test sharpen filter"""
        filtered = ImageAnalyzer.apply_filter(sample_image, "sharpen")
        assert filtered.shape == sample_image.shape, "Shape should be preserved"
    
    def test_apply_filter_grayscale(self, sample_image):
        """Test grayscale filter"""
        filtered = ImageAnalyzer.apply_filter(sample_image, "grayscale")
        assert filtered.shape == sample_image.shape, "Shape should be preserved"
    
    def test_apply_filter_sepia(self, sample_image):
        """Test sepia filter"""
        filtered = ImageAnalyzer.apply_filter(sample_image, "sepia")
        assert filtered.shape == sample_image.shape, "Shape should be preserved"
    
    def test_apply_filter_edge(self, sample_image):
        """Test edge detection filter"""
        filtered = ImageAnalyzer.apply_filter(sample_image, "edge")
        assert filtered.shape == sample_image.shape, "Shape should be preserved"
    
    def test_apply_filter_smooth(self, sample_image):
        """Test smooth filter"""
        filtered = ImageAnalyzer.apply_filter(sample_image, "smooth")
        assert filtered.shape == sample_image.shape, "Shape should be preserved"
    
    def test_apply_filter_invalid(self, sample_image):
        """Test invalid filter"""
        with pytest.raises(ValueError):
            ImageAnalyzer.apply_filter(sample_image, "invalid_filter")
    
    def test_rotate_image(self, sample_image):
        """Test image rotation"""
        rotated = ImageAnalyzer.rotate_image(sample_image, 45)
        assert rotated.shape == sample_image.shape, "Shape should be preserved"
    
    def test_rotate_90_degrees(self, sample_image):
        """Test 90 degree rotation"""
        rotated = ImageAnalyzer.rotate_image(sample_image, 90)
        assert rotated.shape == sample_image.shape, "Shape should be preserved"
    
    def test_flip_horizontal(self, sample_image):
        """Test horizontal flip"""
        flipped = ImageAnalyzer.flip_image(sample_image, "horizontal")
        assert flipped.shape == sample_image.shape, "Shape should be preserved"
    
    def test_flip_vertical(self, sample_image):
        """Test vertical flip"""
        flipped = ImageAnalyzer.flip_image(sample_image, "vertical")
        assert flipped.shape == sample_image.shape, "Shape should be preserved"
    
    def test_flip_invalid_direction(self, sample_image):
        """Test invalid flip direction"""
        with pytest.raises(ValueError):
            ImageAnalyzer.flip_image(sample_image, "diagonal")

class TestTransformationEndpoints:
    """Test transformation API endpoints"""
    
    def test_resize_endpoint(self, test_image_file):
        """Test /resize endpoint"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        img_buffer, filename = test_image_file
        response = client.post(
            "/resize?width=50&height=50",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert "new_size" in data, "Should have new_size"
        assert data["new_size"]["width"] == 50, "Width should be 50"
    
    def test_crop_endpoint(self, test_image_file):
        """Test /crop endpoint"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        img_buffer, filename = test_image_file
        response = client.post(
            "/crop?x=10&y=10&width=50&height=50",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert "crop_region" in data, "Should have crop_region"
    
    def test_filter_endpoint(self, test_image_file):
        """Test /filter endpoint"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        img_buffer, filename = test_image_file
        response = client.post(
            "/filter?filter_type=blur",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert data["filter_applied"] == "blur", "Should apply blur"
    
    def test_rotate_endpoint(self, test_image_file):
        """Test /rotate endpoint"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        img_buffer, filename = test_image_file
        response = client.post(
            "/rotate?angle=45",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert data["rotation_angle"] == 45, "Should rotate 45 degrees"
    
    def test_flip_endpoint(self, test_image_file):
        """Test /flip endpoint"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        img_buffer, filename = test_image_file
        response = client.post(
            "/flip?direction=horizontal",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert data["flip_direction"] == "horizontal", "Should flip horizontally"
    
    def test_filter_all_types(self, test_image_file):
        """Test all available filters"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        filters = ["blur", "sharpen", "edge", "smooth", "grayscale", "sepia"]
        
        for filter_type in filters:
            img_buffer, filename = test_image_file
            response = client.post(
                f"/filter?filter_type={filter_type}",
                files={"file": (filename, img_buffer, "image/jpeg")}
            )
            
            assert response.status_code == 200, f"Filter {filter_type} should work"
    
    def test_resize_invalid_params(self, test_image_file):
        """Test resize with invalid parameters"""
        from fastapi.testclient import TestClient
        from main import app
        client = TestClient(app)
        
        img_buffer, filename = test_image_file
        response = client.post(
            "/resize?width=50",  # Missing height
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 400, "Should reject invalid params"