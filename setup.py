"""
setup.py — Shape-Blend-Splines package setup.
"""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="shape-blend-splines",
    version="0.1.0",
    description=(
        "A Python implementation of the Shape Blend Spline technique: "
        "blending simple parametric shapes into complex geometries using "
        "shape-preserving partition-of-unity basis functions."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Q. Li",
    url="https://github.com/QL-UoHull/Shape-Blend-Splines",
    license="MIT",
    packages=find_packages(exclude=["tests", "notebooks", "examples"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.4",
    ],
    extras_require={
        "notebook": ["ipywidgets>=7.6", "notebook>=6.4", "jupyterlab>=3.0"],
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=(
        "spline shape-blending parametric-shapes geometric-modeling CAD "
        "B-spline partition-of-unity shape-preserving basis-functions "
        "curve-design morphing interpolation"
    ),
)
