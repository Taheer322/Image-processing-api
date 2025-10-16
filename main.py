"""
Image Processing API - FastAPI Application
Smart image analyzer with quality assessment and enhancement
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image, ImageEnhance
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
    description="Analyze and enhance images with quality metrics",
    version="1.0.0"
)

# Add CORS middleware to allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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
        # Normalize to 0-100 scale
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
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # Enhance sharpness
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
        "message": "Smart Image Processing API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze - Analyze image quality",
            "/enhance - Enhance and analyze image",
            "/batch-analyze - Analyze multiple images"
        ]
    }


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
        
        # Original analysis
        original_blur = ImageAnalyzer.calculate_blur_score(image)
        original_brightness = ImageAnalyzer.calculate_brightness(image)
        original_contrast = ImageAnalyzer.calculate_contrast(image)
        
        # Enhance image
        enhanced_image = ImageAnalyzer.enhance_image(image)
        
        # Enhanced analysis
        enhanced_blur = ImageAnalyzer.calculate_blur_score(enhanced_image)
        enhanced_brightness = ImageAnalyzer.calculate_brightness(enhanced_image)
        enhanced_contrast = ImageAnalyzer.calculate_contrast(enhanced_image)
        
        # Save enhanced image
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