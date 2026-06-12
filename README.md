# Shape-Blend-Splines

<p align="center">
  <a href="https://colab.research.google.com/github/QL-UoHull/Shape-Blend-Splines/blob/main/notebooks/interactive_shape_blend_demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"/>
</p>

> **A Python implementation of the Shape Blend Spline (SBS) technique** — a framework for blending simple parametric shapes into complex planar geometries while selectively preserving key features, using shape-preserving partition-of-unity basis functions.

---

## Paper

This repository implements the spline technique described in:

> Q. Li, **"Shape Blend Splines"**,  
> *Computer-Aided Design*, vol. 43, no. 8, pp. 990–1001, 2011.  
> DOI: [10.1016/j.cad.2011.01.006](https://doi.org/10.1016/j.cad.2011.01.006)  
> PII: S0010-4485(11)00008-X

### Research note / transparency

The full text of the paper was not directly accessible during development of this open-source implementation. The framework and equations presented here are based on:

- The repository description and the paper's title and abstract.
- Standard shape-preserving spline / partition-of-unity literature from the same era.
- The raised-cosine bump family of shape-preserving basis functions, which matches the described locality control mechanism.

**Claims about exact reproduction of the paper's formulae are not made.** Specific choices (e.g. the locality kernel form, normalisation scheme, and shape catalogue) are documented assumptions. Corrections and contributions from the author or other domain experts are very welcome.

---

## What is a Shape Blend Spline?

A Shape Blend Spline (SBS) is a planar parametric curve:

$$\mathbf{C}(t) = \sum_{j=0}^{k-1} W_j(t)\, \mathbf{S}_j(t), \quad t \in [0,1]$$

where:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{S}_j(t)$ | The *j*-th constituent **parametric shape** (circle arc, ellipse arc, rectangle, star, …) |
| $W_j(t)$ | **Shape-preserving partition-of-unity** weight; $\sum_j W_j(t) = 1$ for all $t$ |
| $\alpha$ | **Locality parameter** — controls how tightly each weight is concentrated near its centre parameter $t_j$ |

When $\alpha$ is large, $\mathbf{C}(t) \approx \mathbf{S}_j(t)$ near $t = t_j$ (**strong shape preservation**), with smooth blending transitions between shapes.

---

## Repository structure

```
Shape-Blend-Splines/
├── shape_blend_splines/        # Python package
│   ├── __init__.py             # Public API
│   ├── basis.py                # Shape-preserving basis / weight functions
│   ├── shapes.py               # Parametric shape catalogue
│   ├── curve.py                # ShapeBlendSpline & ControlPointSpline
│   └── blend.py                # ShapeBlender, blend helpers, shape_morph
├── notebooks/
│   └── interactive_shape_blend_demo.ipynb  # Jupyter / Colab notebook
├── examples/
│   └── basic_demo.py           # Standalone scripted demo (saves PNG files)
├── tests/
│   └── test_smoke.py           # Pytest smoke tests
├── requirements.txt
├── setup.py
├── LICENSE                     # MIT
└── README.md
```

---

## Installation

### From GitHub (recommended)

```bash
git clone https://github.com/QL-UoHull/Shape-Blend-Splines.git
cd Shape-Blend-Splines
pip install -r requirements.txt
pip install -e .
```

### From PyPI (once published)

```bash
pip install shape-blend-splines
```

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21
- Matplotlib ≥ 3.4
- ipywidgets ≥ 7.6 *(for interactive notebook widgets)*

---

## Quickstart

```python
import numpy as np
import matplotlib.pyplot as plt
from shape_blend_splines import ShapeBlendSpline, blend_two_shapes
from shape_blend_splines.shapes import circle_arc, star_arc, ellipse_arc

# Blend a circle and a star at equal weights (β = 0.5)
blender = blend_two_shapes(circle_arc, star_arc, blend=0.5)
t = np.linspace(0, 1, 500)
pts = blender.evaluate(t)

plt.plot(pts[:, 0], pts[:, 1])
plt.axis('equal')
plt.title('Circle–Star blend  (β = 0.5)')
plt.show()
```

**Five-shape blend with locality control:**

```python
from functools import partial
from shape_blend_splines import ShapeBlendSpline
from shape_blend_splines.shapes import (
    circle_arc, ellipse_arc, superellipse_arc, rectangle_arc, star_arc
)

shapes = [
    circle_arc,
    partial(ellipse_arc, a=1.5, b=0.7),
    partial(superellipse_arc, exponent=4.0),
    rectangle_arc,
    star_arc,
]

sbs = ShapeBlendSpline(shapes, locality=2.5)   # α = 2.5
pts = sbs.evaluate(np.linspace(0, 1, 600))
```

**Free-form curve through control points:**

```python
from shape_blend_splines.curve import ControlPointSpline

ctrl = np.array([[0,0],[1,2],[3,1],[4,3],[6,0]])
sbs  = ControlPointSpline(ctrl, locality=2.0)
pts  = sbs.evaluate(np.linspace(0, 1, 400))
```

---

## Notebook / Colab

An interactive Jupyter notebook is provided in `notebooks/`.

**Open in Google Colab:**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QL-UoHull/Shape-Blend-Splines/blob/main/notebooks/interactive_shape_blend_demo.ipynb)

**Run locally:**

```bash
pip install jupyterlab ipywidgets
jupyter lab notebooks/interactive_shape_blend_demo.ipynb
```

The notebook covers:

1. **Two-shape blend** with interactive β slider
2. **Multi-shape SBS** with per-shape weight sliders
3. **Blend weight visualisation** (partition-of-unity demonstration)
4. **Shape morphing** sequence (β = 0 → 1)
5. **Free-form control-point curve** with interactive point editor
6. **Custom user shapes** (any callable works)

---

## Running the scripted demo

```bash
python examples/basic_demo.py
```

This generates five PNG demonstration figures in the `examples/` directory (no GUI required).

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## API summary

| Class / Function | Description |
|-----------------|-------------|
| `ShapeBlendSpline(shapes, locality=α)` | Main SBS: blend *k* shapes with PU weights |
| `ControlPointSpline(pts, locality=α)` | Smooth curve through control points |
| `ShapeBlender(shapes, weights=[…])` | Uniform weighted blend (no spatial localisation) |
| `blend_two_shapes(S_a, S_b, blend=β)` | Two-shape blend (β=0 → S_a, β=1 → S_b) |
| `blend_shape_series(shapes, locality=α)` | Blend ordered sequence of shapes |
| `shape_morph(S_a, S_b, n_frames=N)` | Compute morphing frames from S_a to S_b |

### Built-in shapes (`shape_blend_splines.shapes`)

| Function | Description |
|----------|-------------|
| `circle_arc(t, center, radius, …)` | Circular arc |
| `ellipse_arc(t, center, a, b, …)` | Elliptic arc |
| `superellipse_arc(t, center, a, b, exponent, …)` | Lamé curve |
| `rectangle_arc(t, center, width, height)` | Closed rectangle |
| `star_arc(t, center, outer_r, inner_r, n_points)` | Regular star polygon |
| `polyline(t, vertices)` | Piecewise-linear path |
| `from_control_points(t, ctrl_pts)` | Catmull–Rom spline through points |

---

## Key parameters

| Parameter | Effect |
|-----------|--------|
| `locality` (α ≥ 0) | **Shape-preservation strength.** α ≈ 0.5 → global blending; α = 1 → smooth raised-cosine; α ≥ 3 → strong local preservation |
| `blend` (β ∈ [0,1]) | **Interpolation** between two shapes |
| `t_centers` | **Centre parameters** of each shape in [0,1] |
| `blend_width` (σ) | **Support width** of each weight function |

---

## Citation

If you use this software in your research, please cite the original paper:

```bibtex
@article{li2011shapeblend,
  title   = {Shape Blend Splines},
  author  = {Li, Q.},
  journal = {Computer-Aided Design},
  volume  = {43},
  number  = {8},
  pages   = {990--1001},
  year    = {2011},
  doi     = {10.1016/j.cad.2011.01.006}
}
```

---

## Contributing

Contributions are welcome! Please open an issue or pull request.  
Areas of particular interest:

- Verification/correction of formulas against the published paper.
- Additional shape types (NURBS patches, B-spline curves, …).
- 3D extension.
- Animation / export utilities.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Keywords

spline, shape blending, parametric shapes, geometric modeling, CAD, B-spline, NURBS,
partition of unity, shape-preserving, basis functions, Jupyter, Google Colab,
computer-aided design, curve design, morphing, interpolation, approximation,
Python, NumPy, matplotlib, interactive, locality parameter
