# Smart Image Processing API

A FastAPI-based intelligent image processing system with quality analysis and enhancement capabilities.

## 🎯 Features

- 📊 **Image Quality Analysis** - Analyze blur, brightness, contrast
- 🎨 **Auto Enhancement** - Automatically improve image quality
- 📦 **Batch Processing** - Process up to 10 images simultaneously
- 🔍 **Object Detection** - Count objects in images
- ⭐ **Smart Ratings** - Quality assessment (Poor/Fair/Good/Excellent)
- 🤖 **CI/CD Pipeline** - Automated testing with GitHub Actions

## 📋 Prerequisites

- Python 3.9+
- pip package manager

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/YOUR-USERNAME/image-processing-api.git
cd image-processing-api
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=html

# View coverage report
start htmlcov/index.html
```

### 5. Start API Server
```bash
python -m uvicorn main:app --reload
```

The API will run on: **http://localhost:8000**

## 📚 API Endpoints

### 1. Health Check
```bash
GET /
```
Returns API status and available endpoints.

### 2. Analyze Image
```bash
POST /analyze
```
Analyze image quality without modification.

**Parameters:** image file

**Response:**
```json
{
  "filename": "photo.jpg",
  "analysis": {
    "blur_score": 75.5,
    "brightness": 60.2,
    "contrast": 45.8,
    "object_count": 3,
    "quality_rating": "Good"
  },
  "recommendations": [...]
}
```

### 3. Enhance Image
```bash
POST /enhance
```
Enhance image quality and save result.

**Response:** Original metrics, enhanced metrics, improvements, download URL

### 4. Batch Analyze
```bash
POST /batch-analyze
```
Analyze multiple images (max 10).

**Parameters:** multiple image files

### 5. Download Enhanced
```bash
GET /download/{file_id}
```
Download previously enhanced image.

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_analyzer.py -v
pytest tests/test_endpoints.py -v
pytest tests/test_edge_cases.py -v
```

### Generate Coverage Report
```bash
pytest tests/ -v --cov=. --cov-report=html
```

### Test Statistics
- **Total Tests:** 31
- **Coverage:** 83%
- **Test Files:** 3
  - `test_analyzer.py` - 15 unit tests
  - `test_endpoints.py` - 11 integration tests
  - `test_edge_cases.py` - 5 edge case tests

## 🔄 CI/CD Pipeline

This project uses **GitHub Actions** for automated testing and deployment.

### Workflow Steps
1. **Test Job** - Runs tests on Python 3.9, 3.10, 3.11
2. **Lint Job** - Checks code quality with flake8 and pylint
3. **Build Job** - Verifies build and creates artifacts

### View Pipeline
- Go to GitHub repository
- Click **Actions** tab
- See all workflow runs and results

## 📁 Project Structure

```
image-processing-api/
├── main.py                    # Main FastAPI application
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Pytest configuration
├── README.md                  # This file
├── api_tester.html            # Web UI for testing
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions workflow
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Test fixtures
│   ├── test_analyzer.py       # Unit tests
│   ├── test_endpoints.py      # Integration tests
│   └── test_edge_cases.py     # Edge case tests
│
├── uploads/                   # Generated enhanced images
│   ├── .gitkeep
│   └── .gitignore
│
└── htmlcov/                   # Coverage report (generated)
```

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| FastAPI | Web framework |
| OpenCV | Image processing |
| NumPy | Numerical computations |
| Pillow | Image enhancement |
| Pytest | Testing framework |
| GitHub Actions | CI/CD automation |

## 📊 Code Coverage

```
Total Coverage: 83%
- main.py: 79%
- test_analyzer.py: 92%
- test_endpoints.py: 86%
- test_edge_cases.py: 80%
```

## 🐛 Error Handling

The API gracefully handles:
- Invalid image formats → 400 Bad Request
- Missing files → 404 Not Found
- Batch size exceeded → 400 Bad Request
- Missing parameters → 422 Validation Error

## 📝 Example Usage

### Analyze an Image
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@myimage.jpg"
```

### Enhance an Image
```bash
curl -X POST "http://localhost:8000/enhance" \
  -F "file=@myimage.jpg"
```

### Batch Analyze
```bash
curl -X POST "http://localhost:8000/batch-analyze" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

## 🎓 Learning Outcomes

This project demonstrates:
- RESTful API design
- Image processing techniques
- Comprehensive testing strategies
- CI/CD pipeline implementation
- Python best practices
- Docker and deployment concepts

## 📄 License

This project is for educational purposes.

 

## 🤝 Support

For issues or questions, please open an issue on GitHub.

---

**Status:** ✅ Production Ready | **Tests:** ✅ All Passing | **Coverage:** ✅ 83%