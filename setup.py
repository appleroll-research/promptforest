from setuptools import setup, find_packages

setup(
    name="promptforest",
    version="0.1.0",
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
    include_package_data=True,
    python_requires=">=3.8",
)
