from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="promptforest",
    version="0.1.1",
    description="Ensemble Prompt Injection Detection",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "sentence-transformers",
        "xgboost",
        "scikit-learn",
        "pyyaml",
        "joblib",
        "protobuf"
    ],
    entry_points={
        "console_scripts": [
            "promptforest=promptforest.cli:main",
        ],
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    python_requires=">=3.8",
)
