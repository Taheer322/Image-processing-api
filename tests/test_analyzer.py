import pytest
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure main can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import ImageAnalyzer

class TestImageAnalyzer:
    """Test ImageAnalyzer class methods"""
    
    def test_blur_score_calculation(self, sample_image):
        """Test blur detection returns valid score"""
        blur = ImageAnalyzer.calculate_blur_score(sample_image)
        assert isinstance(blur, float), "Blur score should be float"
        assert 0 <= blur <= 100, "Blur score should be between 0-100"
    
    def test_blur_score_with_details(self):
        """Test blur detection with detailed image vs blurry"""
        # Create detailed image with edges
        detailed = np.zeros((100, 100, 3), dtype=np.uint8)
        detailed[20:80, 20:80] = 255  # White square on black background
        
        # Create blurry version
        blurry = cv2.GaussianBlur(detailed, (21, 21), 0)
        
        detailed_blur = ImageAnalyzer.calculate_blur_score(detailed)
        blurry_blur = ImageAnalyzer.calculate_blur_score(blurry)
        
        # Detailed image should have higher blur score (sharper)
        assert detailed_blur > blurry_blur, "Detailed image should have higher blur score than blurry"
    
    def test_brightness_calculation(self, sample_image):
        """Test brightness calculation"""
        brightness = ImageAnalyzer.calculate_brightness(sample_image)
        assert isinstance(brightness, float), "Brightness should be float"
        assert 0 <= brightness <= 100, "Brightness should be 0-100"
    
    def test_brightness_bright_image(self, bright_image):
        """Bright image should have high brightness score"""
        brightness = ImageAnalyzer.calculate_brightness(bright_image)
        assert brightness > 50, "Bright image should have high brightness"
    
    def test_brightness_dark_image(self, dark_image):
        """Dark image should have low brightness score"""
        brightness = ImageAnalyzer.calculate_brightness(dark_image)
        assert brightness < 50, "Dark image should have low brightness"
    
    def test_contrast_calculation(self, sample_image):
        """Test contrast calculation"""
        contrast = ImageAnalyzer.calculate_contrast(sample_image)
        assert isinstance(contrast, float), "Contrast should be float"
        assert contrast >= 0, "Contrast should be non-negative"
    
    def test_contrast_high_contrast_image(self, high_contrast_image):
        """High contrast image should have high contrast score"""
        contrast = ImageAnalyzer.calculate_contrast(high_contrast_image)
        assert contrast > 50, "High contrast image should have high score"
    
    def test_count_objects(self, sample_image):
        """Test object detection"""
        count = ImageAnalyzer.count_objects(sample_image)
        assert isinstance(count, int), "Object count should be int"
        assert count >= 0, "Object count should be non-negative"
    
    def test_enhance_image(self, sample_image):
        """Test image enhancement"""
        enhanced = ImageAnalyzer.enhance_image(sample_image)
        assert enhanced.shape == sample_image.shape, "Shape should be preserved"
        assert enhanced.dtype == sample_image.dtype, "Data type should match"
    
    def test_quality_rating_excellent(self):
        """Test quality rating for excellent image"""
        rating = ImageAnalyzer.get_quality_rating(80, 80, 80)
        assert rating == "Excellent", "Rating should be Excellent"
    
    def test_quality_rating_good(self):
        """Test quality rating for good image"""
        rating = ImageAnalyzer.get_quality_rating(60, 60, 60)
        assert rating == "Good", "Rating should be Good"
    
    def test_quality_rating_fair(self):
        """Test quality rating for fair image"""
        rating = ImageAnalyzer.get_quality_rating(40, 40, 40)
        assert rating == "Fair", "Rating should be Fair"
    
    def test_quality_rating_poor(self):
        """Test quality rating for poor image"""
        rating = ImageAnalyzer.get_quality_rating(20, 20, 20)
        assert rating == "Poor", "Rating should be Poor"
    
    def test_read_image_valid(self, sample_image):
        """Test reading valid image"""
        _, buffer = cv2.imencode('.jpg', sample_image)
        img_bytes = buffer.tobytes()
        
        result = ImageAnalyzer.read_image(img_bytes)
        assert result.shape == sample_image.shape, "Shape should match"
    
    def test_read_image_invalid(self):
        """Test reading invalid image data"""
        invalid_bytes = b"invalid image data"
        
        with pytest.raises(ValueError):
            ImageAnalyzer.read_image(invalid_bytes)