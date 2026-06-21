from setuptools import setup, find_packages

setup(
    name="dg-hmcf",
    version="1.0.0",
    description="Dynamic Gated Hierarchical Multi-Scale Cross-Modal Fusion for Depression Detection",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "librosa>=0.10.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
    ],
)
