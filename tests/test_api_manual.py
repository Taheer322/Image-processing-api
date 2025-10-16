import requests
from pathlib import Path
from PIL import Image
import io

# Start API first: python -m uvicorn main:app --reload

BASE_URL = "http://localhost:8000"

# Test 1: Health Check
print("1. Testing Root Endpoint:")
response = requests.get(f"{BASE_URL}/")
print(response.json())
print()

# Test 2: Create a test image
print("2. Creating test image...")
img = Image.new('RGB', (100, 100), color='blue')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Test 3: Analyze Image
print("3. Testing /analyze endpoint:")
files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
response = requests.post(f"{BASE_URL}/analyze", files=files)
print(response.json())
print()

# Test 4: Enhance Image
print("4. Testing /enhance endpoint:")
img_bytes.seek(0)
files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
response = requests.post(f"{BASE_URL}/enhance", files=files)
print(response.json())