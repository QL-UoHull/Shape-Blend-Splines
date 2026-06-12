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
        "Python Shape Blend Splines toolkit for geometric modeling, "
        "shape blending, Jupyter demos, and partition-of-unity curves."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Q. Li",
    url="https://github.com/QL-UoHull/Shape-Blend-Splines",
    project_urls={
        "Source": "https://github.com/QL-UoHull/Shape-Blend-Splines",
        "Issues": "https://github.com/QL-UoHull/Shape-Blend-Splines/issues",
        "Notebook": (
            "https://github.com/QL-UoHull/Shape-Blend-Splines/blob/main/"
            "notebooks/interactive_shape_blend_demo.ipynb"
        ),
    },
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
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    keywords=(
        "shape blend splines spline shape-blending parametric-shapes "
        "geometric-modeling CAD computer-aided-design partition-of-unity "
        "shape-preserving basis-functions curve-design morphing "
        "interpolation jupyter colab"
    ),
)
