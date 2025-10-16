from setuptools import setup, find_packages

setup(
    name="image-processing-api",
    version="1.0.0",
    description="Smart image processing API with quality analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "opencv-python==4.8.1.78",
        "pillow==10.1.0",
        "numpy==1.24.3",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
            "pytest-asyncio==0.21.1",
            "flake8==6.1.0",
            "pylint==3.0.0",
        ]
    },
)