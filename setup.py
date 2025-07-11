from setuptools import setup, find_packages

setup(
    name="active-vision-uncertainty",
    version="0.1.0",
    description="Information-theoretic uncertainty reward system for active vision",
    author="Student Project",
    author_email="student@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.3",
        "timm>=0.9",
        "numpy",
        "matplotlib",
        "open3d",
        "pytest",
        "tqdm",
        "jupyter",
        "pandas",
        "scipy",
        "Pillow",
    ],
    extras_require={
        "rlbench": ["rlbench>=1.3"],
        "dev": ["pytest", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 