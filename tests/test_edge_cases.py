import pytest
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import ImageAnalyzer

class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.fixture
    def small_image(self):
        """Create very small image"""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        return img
    
    @pytest.fixture
    def large_image(self):
        """Create large image"""
        img = np.ones((1000, 1000, 3), dtype=np.uint8) * 128
        return img
    
    def test_very_small_image(self, small_image):
        """Test processing very small image"""
        blur = ImageAnalyzer.calculate_blur_score(small_image)
        brightness = ImageAnalyzer.calculate_brightness(small_image)
        contrast = ImageAnalyzer.calculate_contrast(small_image)
        
        assert isinstance(blur, float), "Should return float"
        assert isinstance(brightness, float), "Should return float"
        assert isinstance(contrast, float), "Should return float"
    
    def test_large_image(self, large_image):
        """Test processing large image"""
        blur = ImageAnalyzer.calculate_blur_score(large_image)
        assert isinstance(blur, float), "Should handle large images"
        assert 0 <= blur <= 100, "Should be in valid range"
    
    def test_black_image(self):
        """Test completely black image"""
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        brightness = ImageAnalyzer.calculate_brightness(black_img)
        assert brightness == 0.0, "Black image should have 0 brightness"
    
    def test_white_image(self):
        """Test completely white image"""
        white_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        brightness = ImageAnalyzer.calculate_brightness(white_img)
        assert brightness == 100.0, "White image should have 100 brightness"
    
    def test_gradient_image(self):
        """Test gradient image"""
        gradient = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient[i, :] = int((i / 100) * 255)
        
        contrast = ImageAnalyzer.calculate_contrast(gradient)
        assert contrast > 0, "Gradient should have contrast"