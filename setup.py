from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="pyimgano",
    version="0.4.0",
    description="Enterprise-Grade Visual Anomaly Detection Toolkit with 37+ algorithms and 80+ operations",
    long_description=README,
    long_description_content_type="text/markdown",
    author="PyImgAno Contributors",
    author_email="pyimgano@example.com",
    url="https://github.com/skygazer42/pyimgano",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.19",
        "scipy>=1.5.1",
        "scikit-learn>=0.22.0",
        "scikit-image>=0.18.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "pyod>=1.1.0,<3.0.0",
        "joblib>=1.0.0",
        "matplotlib>=3.3.0",
        "numba>=0.51",
    ],
    extras_require={
        "diffusion": [
            "diffusers>=0.21.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
        "viz": [
            "seaborn>=0.11.0",
        ],
        "anomalib": [
            "anomalib>=0.10.0",
        ],
        "patchcore_inspection": [
            "patchcore @ git+https://github.com/amazon-science/patchcore-inspection.git",
        ],
        "faiss": [
            "faiss-cpu>=1.7.4",
        ],
        "clip": [
            "open_clip_torch>=2.0.0",
        ],
        "mamba": [
            "mamba-ssm>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-napoleon>=0.7",
        ],
        "all": [
            "pyimgano[backends,diffusion,dev,docs,viz]",
        ],
        "backends": [
            "pyimgano[anomalib,faiss,clip]",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "anomaly-detection",
        "computer-vision",
        "deep-learning",
        "machine-learning",
        "image-processing",
        "industrial-inspection",
        "defect-detection",
        "preprocessing",
        "augmentation",
    ],
    project_urls={
        "Homepage": "https://github.com/skygazer42/pyimgano",
        "Documentation": "https://github.com/skygazer42/pyimgano#readme",
        "Repository": "https://github.com/skygazer42/pyimgano",
        "Bug Tracker": "https://github.com/skygazer42/pyimgano/issues",
        "Changelog": "https://github.com/skygazer42/pyimgano/blob/main/CHANGELOG.md",
    },
    entry_points={
        "console_scripts": [
            "pyimgano-benchmark=pyimgano.cli:main",
            "pyimgano-infer=pyimgano.infer_cli:main",
        ],
    },
)
