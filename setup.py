from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amf-core",
    version="0.2.0",
    author="Jad",
    author_email="jadjbara@live.com",
    description="Atomic Model Fragmentation (AMF) — Molecular Inference Engine for resource-constrained hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jadcrypto/amf-core",
    packages=find_packages(exclude=["tests*", "docs*"]),
    py_modules=["amf", "config", "main"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "gguf>=0.6.0",
        "rich>=13.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "safetensors": [
            "safetensors>=0.4.0",
            "huggingface_hub>=0.20.0",
            "transformers>=4.40.0",
        ],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "flake8", "build", "twine"],
    },
    entry_points={
        "console_scripts": [
            "amf=main:main",
        ],
    },
)
