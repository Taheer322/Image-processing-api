"""
System Integration Tests for Image Processing API
Tests complete workflows, file management, and concurrent operations
"""

import pytest
import sys
from pathlib import Path
from io import BytesIO
import numpy as np
import cv2
import time
import concurrent.futures
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from main import app, UPLOAD_DIR

client = TestClient(app)


# ==================== SYSTEM INTEGRATION TEST 1 ====================
class TestCompleteImageProcessingPipeline:
    """
    System Integration Test 1: Complete Image Processing Pipeline
    Tests end-to-end workflows from upload to download including multiple transformations
    """
    
    def test_full_analysis_enhancement_download_workflow(self, test_image_file):
        """Test complete workflow: upload -> analyze -> enhance -> download -> verify"""
        img_buffer, filename = test_image_file
        
        # Step 1: Analyze original image
        analyze_response = client.post(
            "/analyze",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        assert analyze_response.status_code == 200, "Analysis should succeed"
        original_data = analyze_response.json()
        
        # Verify analysis structure
        assert "analysis" in original_data
        assert "blur_score" in original_data["analysis"]
        assert "brightness" in original_data["analysis"]
        assert "contrast" in original_data["analysis"]
        assert "quality_rating" in original_data["analysis"]
        assert "recommendations" in original_data
        
        original_blur = original_data["analysis"]["blur_score"]
        original_brightness = original_data["analysis"]["brightness"]
        original_quality = original_data["analysis"]["quality_rating"]
        
        # Step 2: Enhance the image
        img_buffer.seek(0)
        enhance_response = client.post(
            "/enhance",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        assert enhance_response.status_code == 200, "Enhancement should succeed"
        enhanced_data = enhance_response.json()
        
        # Verify enhancement structure
        assert "original_metrics" in enhanced_data
        assert "enhanced_metrics" in enhanced_data
        assert "improvements" in enhanced_data
        assert "download_url" in enhanced_data
        
        # Step 3: Verify improvements were calculated
        improvements = enhanced_data["improvements"]
        assert "blur_improvement" in improvements
        assert "brightness_improvement" in improvements
        assert "contrast_improvement" in improvements
        
        # Step 4: Download enhanced image
        download_url = enhanced_data["download_url"]
        download_response = client.get(download_url)
        assert download_response.status_code == 200, "Download should succeed"
        assert len(download_response.content) > 0, "Downloaded file should not be empty"
        
        # Step 5: Verify downloaded image is valid
        downloaded_image_bytes = download_response.content
        nparr = np.frombuffer(downloaded_image_bytes, np.uint8)
        downloaded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert downloaded_image is not None, "Downloaded image should be valid"
        assert downloaded_image.shape[0] > 0, "Image should have valid dimensions"
        
        # Step 6: Re-analyze enhanced image to verify improvements
        enhanced_buffer = BytesIO(downloaded_image_bytes)
        reanalyze_response = client.post(
            "/analyze",
            files={"file": ("enhanced.jpg", enhanced_buffer, "image/jpeg")}
        )
        assert reanalyze_response.status_code == 200
        reanalyzed_data = reanalyze_response.json()
        
        # Verify quality was maintained or improved
        quality_order = {"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3}
        assert quality_order[reanalyzed_data["analysis"]["quality_rating"]] >= quality_order[original_quality] - 1
    
    def test_multi_transformation_pipeline(self, test_image_file):
        """Test chaining multiple transformations: resize -> crop -> filter -> rotate -> flip"""
        img_buffer, filename = test_image_file
        
        # Step 1: Resize to 200x200
        resize_response = client.post(
            "/resize?width=200&height=200",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        assert resize_response.status_code == 200
        resize_data = resize_response.json()
        assert resize_data["new_size"]["width"] == 200
        assert resize_data["new_size"]["height"] == 200
        
        # Step 2: Download resized and crop to 100x100
        download_response = client.get(resize_data["download_url"])
        assert download_response.status_code == 200
        
        resized_buffer = BytesIO(download_response.content)
        crop_response = client.post(
            "/crop?x=50&y=50&width=100&height=100",
            files={"file": ("resized.jpg", resized_buffer, "image/jpeg")}
        )
        assert crop_response.status_code == 200
        crop_data = crop_response.json()
        assert crop_data["cropped_size"]["width"] == 100
        assert crop_data["cropped_size"]["height"] == 100
        
        # Step 3: Download cropped and apply grayscale filter
        download_response = client.get(crop_data["download_url"])
        assert download_response.status_code == 200
        
        cropped_buffer = BytesIO(download_response.content)
        filter_response = client.post(
            "/filter?filter_type=grayscale",
            files={"file": ("cropped.jpg", cropped_buffer, "image/jpeg")}
        )
        assert filter_response.status_code == 200
        filter_data = filter_response.json()
        assert filter_data["filter_applied"] == "grayscale"
        
        # Step 4: Download filtered and rotate 90 degrees
        download_response = client.get(filter_data["download_url"])
        assert download_response.status_code == 200
        
        filtered_buffer = BytesIO(download_response.content)
        rotate_response = client.post(
            "/rotate?angle=90",
            files={"file": ("filtered.jpg", filtered_buffer, "image/jpeg")}
        )
        assert rotate_response.status_code == 200
        rotate_data = rotate_response.json()
        assert rotate_data["rotation_angle"] == 90
        
        # Step 5: Download rotated and flip horizontally
        download_response = client.get(rotate_data["download_url"])
        assert download_response.status_code == 200
        
        rotated_buffer = BytesIO(download_response.content)
        flip_response = client.post(
            "/flip?direction=horizontal",
            files={"file": ("rotated.jpg", rotated_buffer, "image/jpeg")}
        )
        assert flip_response.status_code == 200
        flip_data = flip_response.json()
        assert flip_data["flip_direction"] == "horizontal"
        
        # Step 6: Verify final image is valid
        final_download = client.get(flip_data["download_url"])
        assert final_download.status_code == 200
        assert len(final_download.content) > 0
        
        # Verify final image can be decoded
        nparr = np.frombuffer(final_download.content, np.uint8)
        final_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert final_image is not None
    
    def test_batch_processing_workflow_with_quality_analysis(self):
        """Test batch processing with different quality images and verify results"""
        # Create 5 images with different characteristics
        images = []
        
        # Excellent quality: Bright, sharp, high contrast
        excellent_img = np.zeros((150, 150, 3), dtype=np.uint8)
        excellent_img[30:120, 30:120] = 255
        _, buffer = cv2.imencode('.jpg', excellent_img)
        images.append(("files", ("excellent.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Good quality: Moderate brightness and contrast
        good_img = np.ones((150, 150, 3), dtype=np.uint8) * 150
        good_img[40:110, 40:110] = 200
        _, buffer = cv2.imencode('.jpg', good_img)
        images.append(("files", ("good.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Fair quality: Lower contrast
        fair_img = np.ones((150, 150, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', fair_img)
        images.append(("files", ("fair.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Dark image
        dark_img = np.ones((150, 150, 3), dtype=np.uint8) * 40
        _, buffer = cv2.imencode('.jpg', dark_img)
        images.append(("files", ("dark.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Bright image
        bright_img = np.ones((150, 150, 3), dtype=np.uint8) * 220
        _, buffer = cv2.imencode('.jpg', bright_img)
        images.append(("files", ("bright.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Process batch
        response = client.post("/batch-analyze", files=images)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 5, "Should process all 5 images"
        assert len(data["results"]) == 5, "Should have 5 results"
        
        # Verify different quality ratings
        ratings = [r["quality_rating"] for r in data["results"]]
        assert len(set(ratings)) > 1, "Should have different quality ratings"
        
        # Verify all results have required fields
        for result in data["results"]:
            assert "filename" in result
            assert "blur_score" in result
            assert "brightness" in result
            assert "contrast" in result
            assert "quality_rating" in result
            assert result["quality_rating"] in ["Poor", "Fair", "Good", "Excellent"]
    
    def test_error_recovery_in_mixed_batch(self):
        """Test that batch processing handles and reports errors correctly"""
        images = []
        
        # Valid image 1
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img1)
        images.append(("files", ("valid1.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Invalid image (corrupted data)
        images.append(("files", ("invalid.jpg", BytesIO(b"not an image data"), "image/jpeg")))
        
        # Valid image 2
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 200
        _, buffer = cv2.imencode('.jpg', img2)
        images.append(("files", ("valid2.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        # Process batch
        response = client.post("/batch-analyze", files=images)
        
        assert response.status_code == 200, "Batch should complete despite errors"
        data = response.json()
        
        # Should process valid images
        assert data["total_processed"] >= 2, "Should process at least 2 valid images"
        
        # Should report errors
        assert data["total_errors"] >= 1, "Should report at least 1 error"
        assert data["errors"] is not None, "Should contain error details"
        assert len(data["errors"]) >= 1, "Should have error entries"
    
    def test_data_consistency_across_endpoints(self, test_image_file):
        """Test that same image produces consistent results across different endpoints"""
        img_buffer, filename = test_image_file
        
        # Analyze through analyze endpoint
        analyze_response = client.post(
            "/analyze",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        analyze_metrics = analyze_response.json()["analysis"]
        
        # Analyze through enhance endpoint (original_metrics)
        img_buffer.seek(0)
        enhance_response = client.post(
            "/enhance",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        enhance_original_metrics = enhance_response.json()["original_metrics"]
        
        # Analyze through batch endpoint
        img_buffer.seek(0)
        batch_response = client.post(
            "/batch-analyze",
            files=[("files", (filename, img_buffer, "image/jpeg"))]
        )
        batch_metrics = batch_response.json()["results"][0]
        
        # Verify consistency (allow small floating point differences)
        tolerance = 0.5
        
        # Compare analyze vs enhance
        assert abs(analyze_metrics["blur_score"] - enhance_original_metrics["blur_score"]) < tolerance
        assert abs(analyze_metrics["brightness"] - enhance_original_metrics["brightness"]) < tolerance
        assert abs(analyze_metrics["contrast"] - enhance_original_metrics["contrast"]) < tolerance
        
        # Compare analyze vs batch
        assert abs(analyze_metrics["blur_score"] - batch_metrics["blur_score"]) < tolerance
        assert abs(analyze_metrics["brightness"] - batch_metrics["brightness"]) < tolerance
        assert abs(analyze_metrics["contrast"] - batch_metrics["contrast"]) < tolerance
        assert analyze_metrics["quality_rating"] == batch_metrics["quality_rating"]


# ==================== SYSTEM INTEGRATION TEST 2 ====================
class TestFileManagementAndPersistence:
    """
    System Integration Test 2: File Management and Persistence
    Tests file storage, retrieval, format handling, and cleanup
    """
    
    def test_file_persistence_and_retrieval(self, test_image_file):
        """Test that processed files are correctly saved and retrievable"""
        img_buffer, filename = test_image_file
        
        # Enhance image (creates file)
        enhance_response = client.post(
            "/enhance",
            files={"file": (filename, img_buffer, "image/jpeg")}
        )
        assert enhance_response.status_code == 200
        download_url = enhance_response.json()["download_url"]
        
        # Extract file ID from URL
        file_id = download_url.split("/")[-1]
        file_path = UPLOAD_DIR / file_id
        
        # Verify file exists on disk
        assert file_path.exists(), "Enhanced file should be saved to disk"
        assert file_path.is_file(), "Should be a file, not directory"
        assert file_path.stat().st_size > 0, "File should not be empty"
        
        # Verify file can be retrieved via API
        download_response = client.get(download_url)
        assert download_response.status_code == 200
        assert len(download_response.content) > 0
        
        # Verify retrieved file matches disk file
        with open(file_path, 'rb') as f:
            disk_content = f.read()
        assert download_response.content == disk_content, "API response should match disk file"
    
    def test_multiple_transformations_create_unique_files(self, test_image_file):
        """Test that each transformation creates a unique file"""
        img_buffer, filename = test_image_file
        
        operations = [
            ("/resize?width=80&height=80", "resize"),
            ("/crop?x=10&y=10&width=50&height=50", "crop"),
            ("/filter?filter_type=blur", "filter"),
            ("/rotate?angle=45", "rotate"),
            ("/flip?direction=horizontal", "flip"),
            ("/enhance", "enhance"),
        ]
        
        download_urls = []
        file_paths = []
        
        for endpoint, operation_type in operations:
            img_buffer.seek(0)
            response = client.post(
                endpoint,
                files={"file": (filename, img_buffer, "image/jpeg")}
            )
            assert response.status_code == 200, f"{operation_type} should succeed"
            
            url = response.json()["download_url"]
            download_urls.append(url)
            
            file_id = url.split("/")[-1]
            file_path = UPLOAD_DIR / file_id
            file_paths.append(file_path)
        
        # Verify all URLs are unique
        assert len(set(download_urls)) == len(download_urls), "Each operation should create unique URL"
        
        # Verify all files exist and are unique
        for file_path in file_paths:
            assert file_path.exists(), f"File {file_path.name} should exist"
        
        # Verify file contents are different
        file_contents = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                file_contents.append(f.read())
        
        # At least some files should be different (transformations changed the image)
        unique_contents = set(file_contents)
        assert len(unique_contents) > 1, "Different transformations should produce different files"
    
    def test_different_image_formats(self, test_image_file, test_png_file):
        """Test handling different image formats through complete pipeline"""
        # Test JPEG format
        jpg_buffer, jpg_filename = test_image_file
        jpg_response = client.post(
            "/analyze",
            files={"file": (jpg_filename, jpg_buffer, "image/jpeg")}
        )
        assert jpg_response.status_code == 200, "JPEG should be supported"
        
        # Enhance JPEG
        jpg_buffer.seek(0)
        jpg_enhance_response = client.post(
            "/enhance",
            files={"file": (jpg_filename, jpg_buffer, "image/jpeg")}
        )
        assert jpg_enhance_response.status_code == 200
        jpg_url = jpg_enhance_response.json()["download_url"]
        
        # Test PNG format
        png_buffer, png_filename = test_png_file
        png_response = client.post(
            "/analyze",
            files={"file": (png_filename, png_buffer, "image/png")}
        )
        assert png_response.status_code == 200, "PNG should be supported"
        
        # Enhance PNG
        png_buffer.seek(0)
        png_enhance_response = client.post(
            "/enhance",
            files={"file": (png_filename, png_buffer, "image/png")}
        )
        assert png_enhance_response.status_code == 200
        png_url = png_enhance_response.json()["download_url"]
        
        # Download both formats
        jpg_download = client.get(jpg_url)
        png_download = client.get(png_url)
        
        assert jpg_download.status_code == 200
        assert png_download.status_code == 200
        assert len(jpg_download.content) > 0
        assert len(png_download.content) > 0
        
        # Verify both can be decoded
        jpg_img = cv2.imdecode(np.frombuffer(jpg_download.content, np.uint8), cv2.IMREAD_COLOR)
        png_img = cv2.imdecode(np.frombuffer(png_download.content, np.uint8), cv2.IMREAD_COLOR)
        assert jpg_img is not None
        assert png_img is not None
    
    def test_large_image_handling(self):
        """Test system handles large images correctly"""
        # Create a large image (1000x1000)
        large_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', large_img)
        large_buffer = BytesIO(buffer.tobytes())
        
        # Analyze large image
        response = client.post(
            "/analyze",
            files={"file": ("large.jpg", large_buffer, "image/jpeg")}
        )
        assert response.status_code == 200, "Should handle large images"
        
        # Transform large image
        large_buffer = BytesIO(buffer.tobytes())
        resize_response = client.post(
            "/resize?percentage=50",
            files={"file": ("large.jpg", large_buffer, "image/jpeg")}
        )
        assert resize_response.status_code == 200
        
        # Verify resized image
        download_response = client.get(resize_response.json()["download_url"])
        assert download_response.status_code == 200
        
        resized_img = cv2.imdecode(np.frombuffer(download_response.content, np.uint8), cv2.IMREAD_COLOR)
        assert resized_img.shape[0] == 500, "Should be resized to 50%"
        assert resized_img.shape[1] == 500
    
    def test_file_not_found_error(self):
        """Test appropriate error when requesting non-existent file"""
        response = client.get("/download/nonexistent_file_12345.jpg")
        assert response.status_code == 404, "Should return 404 for non-existent file"
    
    def test_upload_directory_integrity(self, test_image_file):
        """Test upload directory maintains integrity across operations"""
        if not UPLOAD_DIR.exists():
            UPLOAD_DIR.mkdir(exist_ok=True)
        
        initial_files = set(UPLOAD_DIR.glob("*"))
        img_buffer, filename = test_image_file
        
        # Perform multiple operations
        operations = [
            "/enhance",
            "/resize?percentage=75",
            "/filter?filter_type=sharpen",
            "/rotate?angle=30",
            "/flip?direction=vertical"
        ]
        
        created_files = []
        for op in operations:
            img_buffer.seek(0)
            response = client.post(op, files={"file": (filename, img_buffer, "image/jpeg")})
            assert response.status_code == 200
            created_files.append(response.json()["download_url"].split("/")[-1])
        
        # Verify new files were created
        final_files = set(UPLOAD_DIR.glob("*"))
        new_files = final_files - initial_files
        
        assert len(new_files) >= len(operations), "Expected number of files should be created"
        
        # Verify all created files exist
        for file_id in created_files:
            file_path = UPLOAD_DIR / file_id
            assert file_path.exists(), f"File {file_id} should exist"





# ==================== SYSTEM INTEGRATION TEST 3 ====================
class TestConcurrentOperationsAndPerformance:
    """
    System Integration Test 3: Concurrent Operations and Performance
    Tests thread safety, concurrent request handling, and system performance
    """
    
    def test_concurrent_analysis_requests(self):
        """Test handling multiple simultaneous analysis requests"""
        # Create test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        def analyze_image(index):
            """Worker function for concurrent analysis"""
            img_buffer = BytesIO(img_bytes)
            response = client.post(
                "/analyze",
                files={"file": (f"test_{index}.jpg", img_buffer, "image/jpeg")}
            )
            return response.status_code, response.json()
        
        # Submit 10 concurrent requests
        num_requests = 10
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(analyze_image, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(status == 200 for status, _ in results), "All concurrent requests should succeed"
        assert len(results) == num_requests, f"Should have {num_requests} results"
        
        # Verify all results have consistent structure
        for status, data in results:
            assert "analysis" in data
            assert "blur_score" in data["analysis"]
            assert "quality_rating" in data["analysis"]
    
    def test_concurrent_different_transformations(self):
        """Test concurrent execution of different transformation types"""
        img = np.ones((150, 150, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        def transform_image(operation):
            """Worker function for concurrent transformations"""
            img_buffer = BytesIO(img_bytes)
            endpoint, params, expected_key = operation
            response = client.post(
                f"{endpoint}{params}",
                files={"file": ("test.jpg", img_buffer, "image/jpeg")}
            )
            return response.status_code, response.json(), expected_key
        
        operations = [
            ("/resize", "?width=50&height=50", "new_size"),
            ("/crop", "?x=20&y=20&width=80&height=80", "crop_region"),
            ("/filter", "?filter_type=blur", "filter_applied"),
            ("/rotate", "?angle=45", "rotation_angle"),
            ("/flip", "?direction=horizontal", "flip_direction"),
            ("/enhance", "", "improvements"),
        ]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(transform_image, op) for op in operations]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(status == 200 for status, _, _ in results), "All transformations should succeed"
        
        # Verify each operation returned expected data
        for status, data, expected_key in results:
            assert expected_key in data, f"Response should contain {expected_key}"
            assert "download_url" in data, "Should have download URL"
    
    def test_concurrent_batch_operations(self):
        """Test multiple batch operations running simultaneously"""
        def create_batch():
            """Create a batch of test images"""
            images = []
            for i in range(3):
                img = np.ones((100, 100, 3), dtype=np.uint8) * (50 + i * 50)
                _, buffer = cv2.imencode('.jpg', img)
                images.append(("files", (f"img_{i}.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
            return images
        
        def process_batch(batch_id):
            """Worker function for concurrent batch processing"""
            files = create_batch()
            response = client.post("/batch-analyze", files=files)
            return response.status_code, response.json(), batch_id
        
        # Run 5 batches concurrently
        num_batches = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_batch, i) for i in range(num_batches)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(status == 200 for status, _, _ in results), "All batch operations should succeed"
        assert len(results) == num_batches, f"Should have {num_batches} results"
        
        # Verify each batch processed 3 images
        for status, data, batch_id in results:
            assert data["total_processed"] == 3, f"Batch {batch_id} should process 3 images"
            assert len(data["results"]) == 3
    
    def test_performance_sequential_operations(self, test_image_file):
        """Test performance of sequential operations meets acceptable thresholds"""
        img_buffer, filename = test_image_file
        
        operations = [
            ("/analyze", {}),
            ("/enhance", {}),
            ("/resize?percentage=75", {}),
            ("/filter?filter_type=sharpen", {}),
            ("/rotate?angle=30", {}),
        ]
        
        start_time = time.time()
        
        for endpoint, _ in operations:
            img_buffer.seek(0)
            response = client.post(endpoint, files={"file": (filename, img_buffer, "image/jpeg")})
            assert response.status_code == 200
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust based on your requirements)
        assert elapsed_time < 5.0, f"Sequential operations took {elapsed_time:.2f}s, expected < 5.0s"
    
    def test_performance_batch_processing(self):
        """Test batch processing performance with maximum allowed size"""
        images = []
        for i in range(10):  # Maximum batch size
            img = np.ones((100, 100, 3), dtype=np.uint8) * (25 * (i % 4) + 50)
            _, buffer = cv2.imencode('.jpg', img)
            images.append(("files", (f"img_{i}.jpg", BytesIO(buffer.tobytes()), "image/jpeg")))
        
        start_time = time.time()
        response = client.post("/batch-analyze", files=images)
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 10, "Should process all 10 images"
        
        # Should complete within reasonable time
        assert elapsed_time < 10.0, f"Batch processing took {elapsed_time:.2f}s, expected < 10.0s"
    
    def test_system_stability_under_load(self):
        """Test system remains stable under sustained load"""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        _, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        
        def make_request(req_id):
            """Make a random request"""
            img_buffer = BytesIO(img_bytes)
            # Alternate between different endpoints
            if req_id % 3 == 0:
                response = client.post("/analyze", files={"file": ("test.jpg", img_buffer, "image/jpeg")})
            elif req_id % 3 == 1:
                response = client.post("/resize?percentage=50", files={"file": ("test.jpg", img_buffer, "image/jpeg")})
            else:
                response = client.post("/filter?filter_type=blur", files={"file": ("test.jpg", img_buffer, "image/jpeg")})
            return response.status_code
        
        # Make 30 requests (mix of concurrent and sequential)
        num_requests = 30
        success_count=0

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            for future in concurrent.futures.as_completed(futures):
                status_code = future.result()
                if status_code == 200:
                    success_count += 1

        # Verify at least 90% of requests succeed
        success_rate = (success_count / num_requests) * 100
        assert success_rate >= 90, f"System stability degraded under load: success rate = {success_rate:.2f}%"

        # Ensure no crashes or long stalls occurred
        print(f"✅ Stability test passed with {success_rate:.2f}% success rate across {num_requests} mixed requests.")