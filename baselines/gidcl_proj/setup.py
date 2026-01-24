"""
GIDCL安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gidcl",
    version="1.0.0",
    author="GIDCL Team",
    author_email="gidcl@example.com",
    description="Graph-Enhanced Interpretable Data Cleaning with Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SICS-Fundamental-Research-Center/GIDCL",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "networkx>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "viz": [
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
        ],
        "llm": [
            "openai>=1.0.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gidcl=main:main",
        ],
    },
)