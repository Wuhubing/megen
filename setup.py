from setuptools import setup, find_packages

setup(
    name="megen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "numpy",
        "tqdm",
        "scikit-learn"
    ],
) 