"""
Smart Image Processing API - Enhanced with Transformations
Quality analysis + Image transformations (resize, crop, filters)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import uuid
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Image Processing API",
    description="Analyze, enhance, and transform images with quality metrics",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class ImageAnalyzer:
    """Handles image analysis and processing operations"""
    
    @staticmethod
    def read_image(file_content: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image"""
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image format")
        return img
    
    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """Calculate blur detection score (0-100)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100, (laplacian_var / 500) * 100)
        return round(blur_score, 2)
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate average brightness (0-100)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        return round((brightness / 255) * 100, 2)
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate image contrast (0-100)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        return round((contrast / 128) * 100, 2)
    
    @staticmethod
    def count_objects(image: np.ndarray) -> int:
        """Detect and count objects in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Auto-enhance image quality"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return enhanced
    
    @staticmethod
    def get_quality_rating(blur: float, brightness: float, contrast: float) -> str:
        """Generate quality rating based on metrics"""
        score = (blur + brightness + contrast) / 3
        if score >= 70:
            return "Excellent"
        elif score >= 50:
            return "Good"
        elif score >= 30:
            return "Fair"
        else:
            return "Poor"
    
    # ==================== NEW TRANSFORMATION METHODS ====================
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize image to specified dimensions"""
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        resized = cv2.resize(image, (width, height))
        return resized
    
    @staticmethod
    def resize_by_percentage(image: np.ndarray, percentage: int) -> np.ndarray:
        """Resize image by percentage (50 = 50% of original)"""
        if percentage <= 0 or percentage > 500:
            raise ValueError("Percentage must be between 1 and 500")
        height, width = image.shape[:2]
        new_width = int(width * percentage / 100)
        new_height = int(height * percentage / 100)
        resized = cv2.resize(image, (new_width, new_height))
        return resized
    
    @staticmethod
    def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Crop image from position (x,y) with specified dimensions"""
        img_height, img_width = image.shape[:2]
        
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("Invalid crop parameters")
        
        if x + width > img_width or y + height > img_height:
            raise ValueError("Crop area exceeds image boundaries")
        
        cropped = image[y:y+height, x:x+width]
        return cropped
    
    @staticmethod
    def apply_filter(image: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply various filters to image"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if filter_type == "blur":
            filtered = pil_image.filter(ImageFilter.BLUR)
        elif filter_type == "sharpen":
            filtered = pil_image.filter(ImageFilter.SHARPEN)
        elif filter_type == "edge":
            filtered = pil_image.filter(ImageFilter.FIND_EDGES)
        elif filter_type == "smooth":
            filtered = pil_image.filter(ImageFilter.SMOOTH)
        elif filter_type == "grayscale":
            filtered = pil_image.convert('L').convert('RGB')
        elif filter_type == "sepia":
            filtered = ImageEnhance.Color(pil_image).enhance(0)
            filtered = ImageEnhance.Brightness(filtered).enhance(0.8)
        else:
            raise ValueError(f"Unknown filter: {filter_type}")
        
        result = cv2.cvtColor(np.array(filtered), cv2.COLOR_RGB2BGR)
        return result
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle (degrees)"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height))
        return rotated
    
    @staticmethod
    def flip_image(image: np.ndarray, direction: str) -> np.ndarray:
        """Flip image horizontally or vertically"""
        if direction == "horizontal":
            flipped = cv2.flip(image, 1)
        elif direction == "vertical":
            flipped = cv2.flip(image, 0)
        else:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
        return flipped


def generate_recommendations(blur: float, brightness: float, contrast: float) -> list:
    """Generate recommendations based on metrics"""
    recommendations = []
    
    if blur < 40:
        recommendations.append("Image is too blurry. Consider retaking the photo.")
    if brightness < 30:
        recommendations.append("Image is too dark. Increase lighting.")
    elif brightness > 80:
        recommendations.append("Image is too bright. Reduce exposure.")
    if contrast < 20:
        recommendations.append("Low contrast. Enhance details for better clarity.")
    
    return recommendations if recommendations else ["Image quality is good!"]


@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Smart Image Processing API v2.0",
        "version": "2.0.0",
        "endpoints": {
            "Analysis": [
                "/analyze - Analyze image quality",
                "/enhance - Enhance and analyze image",
                "/batch-analyze - Analyze multiple images"
            ],
            "Transformations": [
                "/resize - Resize image",
                "/crop - Crop image",
                "/filter - Apply filters (blur, sharpen, edge, smooth, grayscale, sepia)",
                "/rotate - Rotate image",
                "/flip - Flip image"
            ]
        }
    }


# ==================== ANALYSIS ENDPOINTS ====================

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image quality without modification"""
    try:
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        
        blur_score = ImageAnalyzer.calculate_blur_score(image)
        brightness = ImageAnalyzer.calculate_brightness(image)
        contrast = ImageAnalyzer.calculate_contrast(image)
        object_count = ImageAnalyzer.count_objects(image)
        rating = ImageAnalyzer.get_quality_rating(blur_score, brightness, contrast)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0],
                "size_kb": len(content) / 1024
            },
            "analysis": {
                "blur_score": blur_score,
                "brightness": brightness,
                "contrast": contrast,
                "object_count": object_count,
                "quality_rating": rating
            },
            "recommendations": generate_recommendations(blur_score, brightness, contrast)
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image analysis failed")


@app.post("/enhance")
async def enhance_image_endpoint(file: UploadFile = File(...)):
    """Enhance image and return analysis"""
    try:
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        
        original_blur = ImageAnalyzer.calculate_blur_score(image)
        original_brightness = ImageAnalyzer.calculate_brightness(image)
        original_contrast = ImageAnalyzer.calculate_contrast(image)
        
        enhanced_image = ImageAnalyzer.enhance_image(image)
        
        enhanced_blur = ImageAnalyzer.calculate_blur_score(enhanced_image)
        enhanced_brightness = ImageAnalyzer.calculate_brightness(enhanced_image)
        enhanced_contrast = ImageAnalyzer.calculate_contrast(enhanced_image)
        
        file_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{file_id}_enhanced.jpg"
        cv2.imwrite(str(output_path), enhanced_image)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "original_metrics": {
                "blur_score": original_blur,
                "brightness": original_brightness,
                "contrast": original_contrast
            },
            "enhanced_metrics": {
                "blur_score": enhanced_blur,
                "brightness": enhanced_brightness,
                "contrast": enhanced_contrast
            },
            "improvements": {
                "blur_improvement": round(enhanced_blur - original_blur, 2),
                "brightness_improvement": round(enhanced_brightness - original_brightness, 2),
                "contrast_improvement": round(enhanced_contrast - original_contrast, 2)
            },
            "download_url": f"/download/{file_id}_enhanced.jpg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Enhancement error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image enhancement failed")


@app.post("/batch-analyze")
async def batch_analyze(files: list[UploadFile] = File(...)):
    """Analyze multiple images in batch"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
    
    results = []
    errors = []
    
    for idx, file in enumerate(files):
        try:
            content = await file.read()
            image = ImageAnalyzer.read_image(content)
            
            blur_score = ImageAnalyzer.calculate_blur_score(image)
            brightness = ImageAnalyzer.calculate_brightness(image)
            contrast = ImageAnalyzer.calculate_contrast(image)
            rating = ImageAnalyzer.get_quality_rating(blur_score, brightness, contrast)
            
            results.append({
                "index": idx,
                "filename": file.filename,
                "blur_score": blur_score,
                "brightness": brightness,
                "contrast": contrast,
                "quality_rating": rating
            })
        
        except Exception as e:
            errors.append({"index": idx, "filename": file.filename, "error": str(e)})
    
    return {
        "total_processed": len(results),
        "total_errors": len(errors),
        "results": results,
        "errors": errors if errors else None
    }


# ==================== TRANSFORMATION ENDPOINTS ====================

@app.post("/resize")
async def resize_image(
    file: UploadFile = File(...),
    width: int = Query(None),
    height: int = Query(None),
    percentage: int = Query(None)
):
    """Resize image - provide either (width, height) or percentage"""
    try:
        if percentage and (width or height):
            raise ValueError("Provide either percentage OR (width, height), not both")
        
        if not percentage and (not width or not height):
            raise ValueError("Provide either percentage OR both width and height")
        
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        
        if percentage:
            resized = ImageAnalyzer.resize_by_percentage(image, percentage)
            new_width, new_height = resized.shape[1], resized.shape[0]
        else:
            resized = ImageAnalyzer.resize_image(image, width, height)
            new_width, new_height = width, height
        
        file_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{file_id}_resized.jpg"
        cv2.imwrite(str(output_path), resized)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "original_size": {"width": image.shape[1], "height": image.shape[0]},
            "new_size": {"width": new_width, "height": new_height},
            "transformation": "resize",
            "download_url": f"/download/{file_id}_resized.jpg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Resize error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image resize failed")


@app.post("/crop")
async def crop_image(
    file: UploadFile = File(...),
    x: int = Query(...),
    y: int = Query(...),
    width: int = Query(...),
    height: int = Query(...)
):
    """Crop image from position (x,y) with specified dimensions"""
    try:
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        cropped = ImageAnalyzer.crop_image(image, x, y, width, height)
        
        file_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{file_id}_cropped.jpg"
        cv2.imwrite(str(output_path), cropped)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "original_size": {"width": image.shape[1], "height": image.shape[0]},
            "crop_region": {"x": x, "y": y, "width": width, "height": height},
            "cropped_size": {"width": cropped.shape[1], "height": cropped.shape[0]},
            "transformation": "crop",
            "download_url": f"/download/{file_id}_cropped.jpg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Crop error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image crop failed")


@app.post("/filter")
async def apply_filter(
    file: UploadFile = File(...),
    filter_type: str = Query(...)
):
    """Apply filter to image: blur, sharpen, edge, smooth, grayscale, sepia"""
    try:
        valid_filters = ["blur", "sharpen", "edge", "smooth", "grayscale", "sepia"]
        if filter_type not in valid_filters:
            raise ValueError(f"Filter must be one of: {', '.join(valid_filters)}")
        
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        filtered = ImageAnalyzer.apply_filter(image, filter_type)
        
        file_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{file_id}_{filter_type}.jpg"
        cv2.imwrite(str(output_path), filtered)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "filter_applied": filter_type,
            "available_filters": ["blur", "sharpen", "edge", "smooth", "grayscale", "sepia"],
            "transformation": "filter",
            "download_url": f"/download/{file_id}_{filter_type}.jpg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Filter error: {str(e)}")
        raise HTTPException(status_code=500, detail="Filter application failed")


@app.post("/rotate")
async def rotate_image(
    file: UploadFile = File(...),
    angle: float = Query(...)
):
    """Rotate image by specified angle (degrees)"""
    try:
        if angle < -360 or angle > 360:
            raise ValueError("Angle must be between -360 and 360")
        
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        rotated = ImageAnalyzer.rotate_image(image, angle)
        
        file_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{file_id}_rotated.jpg"
        cv2.imwrite(str(output_path), rotated)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "rotation_angle": angle,
            "transformation": "rotate",
            "download_url": f"/download/{file_id}_rotated.jpg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Rotate error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image rotation failed")


@app.post("/flip")
async def flip_image(
    file: UploadFile = File(...),
    direction: str = Query(...)
):
    """Flip image horizontally or vertically"""
    try:
        if direction not in ["horizontal", "vertical"]:
            raise ValueError("Direction must be 'horizontal' or 'vertical'")
        
        content = await file.read()
        image = ImageAnalyzer.read_image(content)
        flipped = ImageAnalyzer.flip_image(image, direction)
        
        file_id = str(uuid.uuid4())
        output_path = UPLOAD_DIR / f"{file_id}_flipped_{direction}.jpg"
        cv2.imwrite(str(output_path), flipped)
        
        return {
            "filename": file.filename,
            "timestamp": datetime.now().isoformat(),
            "flip_direction": direction,
            "transformation": "flip",
            "download_url": f"/download/{file_id}_flipped_{direction}.jpg"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Flip error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image flip failed")


@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """Download processed image"""
    file_path = UPLOAD_DIR / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 
