from pydantic import BaseModel
from typing import List, Dict, Any

class ImageAnalysisResult(BaseModel):
    blur_score: float
    brightness: float
    contrast: float
    object_count: int
    quality_rating: str

class AnalysisResponse(BaseModel):
    filename: str
    timestamp: str
    analysis: ImageAnalysisResult
    recommendations: List[str]

class EnhancementResponse(BaseModel):
    filename: str
    original_metrics: Dict[str, float]
    enhanced_metrics: Dict[str, float]
    improvements: Dict[str, float]
    download_url: str

class BatchAnalysisResult(BaseModel):
    total_processed: int
    total_errors: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]] = None