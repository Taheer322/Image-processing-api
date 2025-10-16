import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"

# API Settings
MAX_BATCH_SIZE = 10
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "app.log"

# Image Processing
BLUR_THRESHOLD = 40
BRIGHTNESS_MIN = 30
BRIGHTNESS_MAX = 80
CONTRAST_MIN = 20