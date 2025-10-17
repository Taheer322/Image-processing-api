import pytest
import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from main import app

# Initialize client after importing app
client = TestClient(app)

class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200, "Root endpoint should return 200"
        data = response.json()
        assert "version" in data, "Response should contain version"
        assert "endpoints" in data, "Response should contain endpoints"
        assert "message" in data, "Response should contain message"
    
    def test_analyze_image_success(self, test_image_file):
        """Test analyze endpoint with valid image"""
        img_buffer, filename = test_image_file
        response = client.post(
            "/analyze",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200 for valid image"
        data = response.json()
        assert "analysis" in data, "Response should contain analysis"
        assert "blur_score" in data["analysis"], "Should have blur_score"
        assert "brightness" in data["analysis"], "Should have brightness"
        assert "contrast" in data["analysis"], "Should have contrast"
        assert "quality_rating" in data["analysis"], "Should have quality_rating"
        assert "recommendations" in data, "Should have recommendations"
    
    def test_analyze_image_png(self, test_png_file):
        """Test analyze with PNG format"""
        img_buffer, filename = test_png_file
        response = client.post(
            "/analyze",
            files={"file": (filename, img_buffer, "image/png")}
        )
        
        assert response.status_code == 200, "PNG should be supported"
        assert "analysis" in response.json(), "Should have analysis"
    
    def test_analyze_invalid_file(self):
        """Test analyze with invalid file"""
        invalid_data = BytesIO(b"not an image")
        response = client.post(
            "/analyze",
            files={"file": ("invalid.jpg", invalid_data, "image/jpeg")}
        )
        
        assert response.status_code == 400, "Invalid image should return 400"
    
    def test_enhance_image_success(self, test_image_file):
        """Test enhance endpoint"""
        img_buffer, filename = test_image_file
        response = client.post(
            "/enhance",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert "original_metrics" in data, "Should have original metrics"
        assert "enhanced_metrics" in data, "Should have enhanced metrics"
        assert "improvements" in data, "Should have improvements"
        assert "download_url" in data, "Should have download URL"
    
    def test_batch_analyze_success(self, test_image_file):
        """Test batch analyze with multiple images"""
        img_buffer1, filename1 = test_image_file
        img_buffer2, filename2 = test_image_file
        
        response = client.post(
            "/batch-analyze",
            files=[
                ("files", (filename1, img_buffer1, "image/jpeg")),
                ("files", (filename2, img_buffer2, "image/jpeg"))
            ]
        )
        
        assert response.status_code == 200, "Should return 200"
        data = response.json()
        assert data["total_processed"] == 2, "Should process 2 images"
        assert len(data["results"]) == 2, "Should have 2 results"
    
    def test_batch_analyze_exceeds_limit(self, test_image_file):
        """Test batch analyze with too many images"""
        img_buffer, filename = test_image_file
        files = [
            ("files", (f"{i}.jpg", img_buffer, "image/jpeg"))
            for i in range(11)
        ]
        
        response = client.post("/batch-analyze", files=files)
        assert response.status_code == 400, "Should reject >10 images"
    
    def test_analyze_metrics_valid_ranges(self, test_image_file):
        """Test that all metrics are in valid ranges"""
        img_buffer, filename = test_image_file
        response = client.post(
            "/analyze",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        
        data = response.json()
        analysis = data["analysis"]
        
        assert 0 <= analysis["blur_score"] <= 100, "Blur should be 0-100"
        assert 0 <= analysis["brightness"] <= 100, "Brightness should be 0-100"
        assert 0 <= analysis["contrast"], "Contrast should be >= 0"
        assert analysis["object_count"] >= 0, "Count should be >= 0"

class TestErrorHandling:
    """Test error handling"""
    
    def test_missing_file_parameter(self):
        """Test missing file parameter"""
        response = client.post("/analyze")
        assert response.status_code == 422, "Should return 422 for missing file"
    
    def test_download_nonexistent_file(self):
        """Test downloading non-existent file"""
        response = client.get("/download/nonexistent_file.jpg")
        assert response.status_code == 404, "Should return 404 for missing file"
