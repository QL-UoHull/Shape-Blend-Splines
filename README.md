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

> **Shape Blend Splines (SBS)** are non-rational, partition-of-unity spline curves built by blending whole parametric shapes with smooth polynomial basis functions.

This repository provides a Python implementation of the Shape Blend Spline framework from Li (2011), with a focus on the distinctive idea that makes SBS different from classical control-point-only spline workflows:

- the curve is a **non-rational weighted sum**,
- the weights are **piecewise-polynomial smooth-step differences**, and
- the blended object is assembled from **whole parametric shape functions** rather than only from control points.

That makes SBS useful for familiar spline tasks such as open-curve design, closed loops, and control-point-driven modelling, while also exposing a capability that typical NURBS demos do not show directly: **local preservation and transition between entire geometric primitives or edge lines** with tunable locality.

## Reference

> Q. Li, **"Shape Blend Splines"**  
> *Computer-Aided Design*, **43**(8), 990–1001, 2011.  
> DOI: [10.1016/j.cad.2011.01.006](https://doi.org/10.1016/j.cad.2011.01.006)

## Theory

See [`docs/theory.md`](docs/theory.md) for a formal derivation of **how each
B-spline basis function can be expressed as a difference of two smooth step
functions**, the connection to the SBS step-difference basis
$B_{a,b}(t) = S_b(t) - S_a(t)$ in `basis.py`, and references.

## Core formula

For constituent parametric shapes \(\mathbf{S}_j(t)\) and partition-of-unity weights \(W_j(t)\), the blended curve is

$$
\mathbf{C}(t) = \sum_{j=0}^{k-1} W_j(t)\,\mathbf{S}_j(t).
$$

This implementation follows that non-rational form directly.

### Basis construction

For an interval \([a,b]\), SBS uses smooth polynomial step functions and forms

$$
B_{a,b}(t) = S_b(t) - S_a(t).
$$

These basis pieces are then normalised to produce weights satisfying

$$
\sum_j W_j(t)=1.
$$

The locality parameter \(\alpha\) controls how concentrated each weight is around its centre:

- **small \(\alpha\)** → broader, more global mixing,
- **large \(\alpha\)** → narrower support and stronger local shape identity.

## What this repository highlights

### 1. True shape blending
The primary API blends **whole shapes** evaluated at the same global parameter \(t\), not a rational control-point denominator.

### 2. Open and closed curves
The package supports both:
- **open SBS curves** with ordered centres, and
- **closed periodic SBS curves** where the first and last shapes are neighbours.

### 3. Locality-aware design
Unlike a purely global average, the same set of component shapes can produce either a smooth global hybrid or a strongly local patchwork of recognisable shape regions by changing only \(\alpha\).

### 4. Familiar and beyond-familiar workflows
The repository includes:
- `ControlPointSpline` for control-point-driven curve design,
- `ShapeBlendSpline` for open shape sequences,
- `PeriodicShapeBlendSpline` for closed periodic loops,
- `ShapeBlender` for simple global weighted baselines.

This means the package can cover many familiar spline-style modelling tasks while also illustrating SBS-specific capabilities that are awkward to express as standard control-point NURBS examples.

> **Accuracy note:** this repository does **not** claim to replace every strength of NURBS. In particular, rational NURBS remain the classical tool for exact conics. The strength of SBS here is different: locality-aware blending of entire parametric primitives with polynomial partition-of-unity weights.

## Repository structure

```text
Shape-Blend-Splines/
├── shape_blend_splines/
│   ├── __init__.py
│   ├── basis.py
│   ├── blend.py
│   ├── curve.py
│   └── shapes.py
├── notebooks/
│   └── interactive_shape_blend_demo.ipynb
├── docs/
│   └── theory.md
├── examples/
│   └── basic_demo.py
├── tests/
│   └── test_smoke.py
├── pyproject.toml
├── setup.py
└── README.md
```

## Installation

```bash
git clone https://github.com/QL-UoHull/Shape-Blend-Splines.git
cd Shape-Blend-Splines
pip install -r requirements.txt
pip install -e .[dev,notebook]
```

## Quick start

### Closed SBS from 4 control points and 4 edge lines

```python
import numpy as np
from functools import partial
from shape_blend_splines import PeriodicShapeBlendSpline
from shape_blend_splines.shapes import line_segment

# 4 corners of a square (counter-clockwise)
corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

# 4 edge lines as constituent shapes
edges = [
    partial(line_segment, p0=corners[0], p1=corners[1]),  # bottom
    partial(line_segment, p0=corners[1], p1=corners[2]),  # right
    partial(line_segment, p0=corners[2], p1=corners[3]),  # top
    partial(line_segment, p0=corners[3], p1=corners[0]),  # left
]

# Closed periodic SBS — equal weights, varying locality
sbs = PeriodicShapeBlendSpline(edges, locality=2.0)
t = np.linspace(0.0, 1.0, 600, endpoint=False)
pts = sbs.evaluate(t)           # (600, 2) closed curve
weights = sbs.weights_at(t)     # (4, 600) partition-of-unity basis

# Per-edge scalar weights: bias toward the bottom edge
sbs_biased = PeriodicShapeBlendSpline(edges, locality=2.0,
                                       knot_weights=[3.0, 1.0, 1.0, 1.0])
pts_biased = sbs_biased.evaluate(t)
```

### Open blend across a sequence of shapes

```python
import numpy as np
from functools import partial
from shape_blend_splines import ShapeBlendSpline
from shape_blend_splines.shapes import line_segment, circle_arc, from_control_points

shapes = [
    partial(line_segment, p0=(-1.8, -0.5), p1=(1.8, -0.5)),
    partial(circle_arc, center=(0.0, -0.1), radius=1.3,
            theta_start=0.95*np.pi, theta_end=0.05*np.pi),
    partial(from_control_points, control_pts=np.array([
        [-1.7, -0.3],
        [-0.8,  1.2],
        [ 0.6,  0.4],
        [ 1.8,  1.1],
    ])),
]

sbs = ShapeBlendSpline(shapes, locality=2.0)
curve = sbs.evaluate(np.linspace(0.0, 1.0, 600))
```

### Control-point workflow

```python
import numpy as np
from shape_blend_splines import ControlPointSpline

control_pts = np.array([
    [0.0, 0.0],
    [1.0, 1.8],
    [3.0, 0.8],
    [4.0, 2.4],
    [5.5, 0.2],
])

open_curve = ControlPointSpline(control_pts, locality=2.0)
closed_loop = ControlPointSpline(control_pts, locality=2.0, closed=True)
```

## Main APIs

| API | Role |
|-----|------|
| `ShapeBlendSpline` | open non-rational SBS curve through an ordered shape sequence |
| `PeriodicShapeBlendSpline` | closed periodic SBS curve with wrap-around blending |
| `ControlPointSpline` | control-point-driven convenience interface |
| `ShapeBlender` | global weighted blend baseline |
| `apply_knot_weights` | per-knot scalar weights with renormalisation (partition of unity preserved) |
| `blend_shape_series` | helper for open SBS sequences |
| `blend_two_shapes` | simple global interpolation helper |
| `shape_morph` | frame-by-frame global blend sequence |

## Built-in shape primitives

| Function | Description |
|----------|-------------|
| `line_segment` | straight segment |
| `circle_arc` | circular arc or full circle |
| `ellipse_arc` | elliptic arc or full ellipse |
| `superellipse_arc` | Lamé / rounded-rectangle family |
| `rectangle_arc` | closed rectangle parameterised by perimeter |
| `polyline` | piecewise-linear path |
| `star_arc` | regular closed star polygon |
| `from_control_points` | smooth Hermite curve from control points |

## Demos

### Scripted demo

Run:

```bash
python examples/basic_demo.py
```

The script generates figures showing:
- a **closed periodic multi-shape SBS curve**,
- a **locality sweep** for the same shapes,
- an **open SBS design sequence**,
- **periodic weight functions**,
- a **global-vs-local comparison**,
- **open and closed control-point curves**.

### Notebook demo

The notebook at `notebooks/interactive_shape_blend_demo.ipynb` uses the package API directly and walks through:

- **Section 1** — *4 control points / 4 edge lines → closed SBS curve*:
  4 square corners → 4 straight edge lines → closed periodic SBS blend with
  per-corner scalar weights and locality α, showing how different weight/α
  settings produce a family of closed curves from the same geometry.
- **Section 2** — *Open SBS sequence with locality story*:
  an open control-point polygon whose edges are blended by SBS, showing how
  α moves between a smooth global curve and a near-piecewise-linear path.
- **Section 3** — *Interactive per-knot weights + locality*:
  `ipywidgets` sliders (one per edge weight, one for α) that update in real
  time both the SBS basis/weight curves $\hat{W}_j(t)$ and the resulting
  closed curve.  A static fallback grid is also pre-rendered for non-widget
  renderers.
- **Section 4** — *Numerical verification: B-spline as a step-function difference*:
  plots the exact step-difference identity for `sbs_basis` and compares
  peak-normalised cubic B-spline bases against the SBS approximation.
- **Section 5** — *Control-point workflow* (unchanged).

Open it locally with:

```bash
jupyter lab notebooks/interactive_shape_blend_demo.ipynb
```

or in Colab via the badge at the top of this README.

## Testing

```bash
python3 -m pytest tests/ -v
```

## Why SBS is interesting

SBS is most compelling when you want a curve to look locally like different geometric primitives across different regions of the same parameter domain.

Examples from this repository include:
- a closed loop that becomes circle-like, ellipse-like, rounded-rectangle-like, rectangular, and star-like in different regions,
- a locality parameter that continuously moves between global smoothing and strong local identity,
- periodic wrap-around blending for closed curves,
- control-point-driven curves for familiar spline-style modelling.

That combination is the main research story of the repository.

## Citation

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

## License

Released under the **MIT License**. See [LICENSE](LICENSE).
