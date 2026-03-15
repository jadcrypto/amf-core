from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="amf-core",
    version="0.1.0",
    author="Jad",
    description="Atomic Model Fragmentation (AMF) - Universal LLM Decomposition Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/amf-core",
    packages=find_packages(),
    py_modules=["amf"],
    classifiers=[
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
        "safetensors": ["safetensors>=0.4.0", "huggingface_hub>=0.20.0"],
        "dev": ["pytest", "black", "flake8"]
    },
    entry_points={
        "console_scripts": [
            "amf=main:main",
        ],
    },
)
