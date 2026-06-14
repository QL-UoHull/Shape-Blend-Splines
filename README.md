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

> **Shape-Blend-Splines** is a Python research software package for locality-aware blending of parametric shapes using shape-preserving spline basis functions.

This repository presents an open implementation of the **Shape Blend Spline (SBS)** framework for the construction of complex planar curves from simpler constituent parametric shapes. The software is intended for research, teaching, visual experimentation, and computational prototyping in geometric modelling and spline-based shape design.

A central strength of the SBS framework is that it provides a flexible mechanism for constructing smooth blended geometries with strong local shape control **without requiring rational spline basis functions or classical tensor-product spline constructions**.

The framework also emphasises the use of **partition-of-unity basis functions that are piecewise polynomial over general two-dimensional partitions**, rather than being restricted to rectangular parameter domains.

The package supports:
- multi-shape blending with locality control,
- global weighted interpolation between shapes,
- morphing workflows,
- control-point-driven curve construction,
- interactive experimentation in Jupyter and Google Colab.

---

## Scientific context

Shape Blend Splines provide a mechanism for combining multiple constituent shapes into a single smooth parametric curve while selectively preserving local geometric characteristics. The central principle is that each constituent shape contributes through a partition-of-unity weight, and the locality of that contribution can be controlled continuously.

In practical terms, this enables the construction of blended curves that interpolate between geometric identities without reducing the result to a purely uniform global average. An important consequence is that individual features can remain recognisable over subregions of the parameter domain even while the curve as a whole is blended.

More broadly, the partition-of-unity viewpoint is not tied to a single rectangular layout. The basis construction is naturally aligned with **piecewise-polynomial blending over more general two-dimensional partitions**, which is one of the conceptual distinctions highlighted in the original work.

This repository provides a Python implementation suitable for:
- exploratory computational geometry,
- algorithmic prototyping,
- visual demonstrations in teaching,
- reproducible examples for spline-based shape blending.

---

## Reference publication

This repository is based on the following publication:

> Q. Li, **"Shape Blend Splines"**  
> *Computer-Aided Design*, **43**(8), 990–1001, 2011.  
> DOI: [10.1016/j.cad.2011.01.006](https://doi.org/10.1016/j.cad.2011.01.006)

---

## Mathematical formulation

A Shape Blend Spline is a planar parametric curve of the form

$$
\mathbf{C}(t) = \sum_{j=0}^{k-1} W_j(t)\,\mathbf{S}_j(t), \qquad t \in [0,1],
$$

where:

| Symbol | Interpretation |
|--------|----------------|
| $\mathbf{S}_j(t)$ | the *j*-th constituent parametric shape |
| $W_j(t)$ | a partition-of-unity blending weight satisfying $\sum_j W_j(t)=1$ |
| $\alpha$ | a locality parameter controlling the concentration of influence |

As the locality parameter increases, the influence of each constituent shape becomes more concentrated near its associated region of the parameter domain. Consequently, the resulting blended curve can preserve stronger local geometric identity.

From a modelling perspective, the significance of this formulation is that rich blended geometry can be obtained through **shape-preserving piecewise-polynomial basis design**, avoiding the need to encode all detail directly into a single traditional spline representation.

---

## Software scope

The repository currently provides the following capabilities:

- **Locality-aware shape blending** through `ShapeBlendSpline`
- **Ordered multi-shape blending** through `blend_shape_series`
- **Global weighted blending** through `ShapeBlender` and `blend_two_shapes`
- **Frame-based morphing** through `shape_morph`
- **Control-point curve generation** through `ControlPointSpline`
- **Interactive notebook demonstrations** for visual and computational exploration

The implementation is designed to be lightweight, readable, and suitable for extension in research and teaching settings.

---

## Repository structure

```text
Shape-Blend-Splines/
├── shape_blend_splines/        # Core Python package
│   ├── __init__.py             # Public API
│   ├── basis.py                # Basis and weighting functions
│   ├── shapes.py               # Parametric shape definitions
│   ├── curve.py                # ShapeBlendSpline and ControlPointSpline
│   └── blend.py                # Blending and morphing utilities
├── notebooks/
│   └── interactive_shape_blend_demo.ipynb
├── examples/
│   └── basic_demo.py
├── tests/
│   └── test_smoke.py
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Installation

### Install from source

```bash
git clone https://github.com/QL-UoHull/Shape-Blend-Splines.git
cd Shape-Blend-Splines
pip install -r requirements.txt
pip install -e .
```

### Core requirements

- Python 3.8 or later
- NumPy 1.21 or later
- Matplotlib 3.4 or later

### Optional notebook dependencies

```bash
pip install jupyterlab notebook ipywidgets
```

### Planned package installation

If the project is later published to PyPI, installation will be:

```bash
pip install shape-blend-splines
```

---

## Minimal usage examples

### Four-control-point shape family

```python
import numpy as np
from shape_blend_splines.curve import ControlPointSpline


def square_family_control_points(rounding=0.45, n_side=6):
    rounding = float(np.clip(rounding, 0.0, 1.0))
    corners = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0],
    ])
    center = corners.mean(axis=0)

    pts = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        for j in range(n_side):
            s = j / n_side
            edge_pt = (1 - s) * p0 + s * p1
            v = edge_pt - center
            square_pt = v / max(np.max(np.abs(v)), 1e-12)
            circle_pt = v / max(np.linalg.norm(v), 1e-12)
            pts.append((1 - rounding) * square_pt + rounding * circle_pt)
    return np.array(pts)


control_pts = square_family_control_points(rounding=0.45, n_side=6)
sbs = ControlPointSpline(control_pts, locality=2.0)
pts = sbs.evaluate(np.linspace(0, 1, 700))
```

### Locality-aware blending of multiple shapes

```python
import numpy as np
from functools import partial
from shape_blend_splines import ShapeBlendSpline
from shape_blend_splines.shapes import (
    circle_arc,
    ellipse_arc,
    superellipse_arc,
    rectangle_arc,
    star_arc,
)

shapes = [
    circle_arc,
    partial(ellipse_arc, a=1.5, b=0.7),
    partial(superellipse_arc, exponent=4.0),
    rectangle_arc,
    star_arc,
]

sbs = ShapeBlendSpline(shapes, locality=2.5)
pts = sbs.evaluate(np.linspace(0, 1, 600))
```

### Control-point-driven curve construction

```python
import numpy as np
from shape_blend_splines.curve import ControlPointSpline

ctrl = np.array([[0, 0], [1, 2], [3, 1], [4, 3], [6, 0]])
sbs = ControlPointSpline(ctrl, locality=2.0)
pts = sbs.evaluate(np.linspace(0, 1, 400))
```

---

## API overview

| Class / Function | Purpose |
|-----------------|---------|
| `ShapeBlendSpline(shapes, locality=α)` | locality-aware blending of multiple constituent shapes |
| `blend_shape_series(shapes, locality=α)` | ordered multi-shape blending helper |
| `ControlPointSpline(pts, locality=α)` | spline-like curve generation from control points |
| `ShapeBlender(shapes, weights=[...])` | global weighted blending without spatial localisation |
| `blend_two_shapes(S_a, S_b, blend=β)` | interpolation between two shapes |
| `shape_morph(S_a, S_b, n_frames=N)` | morphing sequence generation |

### Built-in shape primitives

| Function | Description |
|----------|-------------|
| `circle_arc(t, center, radius, …)` | circular arc |
| `ellipse_arc(t, center, a, b, …)` | elliptic arc |
| `superellipse_arc(t, center, a, b, exponent, …)` | superellipse / Lamé-type curve |
| `rectangle_arc(t, center, width, height)` | closed rectangular shape |
| `star_arc(t, center, outer_r, inner_r, n_points)` | regular star polygon |
| `polyline(t, vertices)` | piecewise-linear path |
| `from_control_points(t, ctrl_pts)` | control-point-based spline helper |

---

## Choosing between global and locality-aware blending

The package includes both **global weighted blending** and **locality-aware SBS blending**.

| Use case | Recommended API | Interpretation |
|----------|------------------|----------------|
| Interpolate between two shapes with a single blend factor | `blend_two_shapes(...)` | uniform global interpolation |
| Blend several shapes while preserving local character | `ShapeBlendSpline(...)` | partition-of-unity, locality-aware blending |
| Generate a morphing sequence | `shape_morph(...)` | repeated global interpolation |
| Construct a smooth curve through control points | `ControlPointSpline(...)` | SBS-inspired local control |
| Explore a square-to-rounded-square design family | `ControlPointSpline(...)` with a derived control polygon | control-point shape design on a fixed scaffold |

The locality parameter is meaningful for the SBS-based methods and should be interpreted as a control on the spatial concentration of individual shape influence.

---

## Notebook and interactive demonstration

An interactive notebook is provided at:

`notebooks/interactive_shape_blend_demo.ipynb`

### Run in Google Colab

Open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/QL-UoHull/Shape-Blend-Splines/blob/main/notebooks/interactive_shape_blend_demo.ipynb)

Instructions:
1. Open the notebook in Colab using the badge above.
2. If prompted, allow Colab a few moments to install the dependencies in the first code cell.
3. Run the notebook from top to bottom using **Runtime → Run all**.
4. If you reopen the notebook after updates, use **Runtime → Restart session** and rerun all cells so the latest package version is loaded.

### Run in Jupyter Notebook or JupyterLab

After cloning the repository and installing dependencies, start Jupyter locally:

```bash
pip install jupyterlab notebook ipywidgets
jupyter notebook notebooks/interactive_shape_blend_demo.ipynb
```

Or with JupyterLab:

```bash
jupyter lab notebooks/interactive_shape_blend_demo.ipynb
```

Instructions:
1. Clone this repository and install the package with `pip install -e .`.
2. Launch either Jupyter Notebook or JupyterLab using one of the commands above.
3. Open `notebooks/interactive_shape_blend_demo.ipynb`.
4. Run the notebook cells from top to bottom.
5. If you change the package source code while Jupyter is already running, restart the kernel and rerun all cells to pick up the latest local code.

The notebook includes demonstrations of:
1. four-control-point shape design from a square scaffold,
2. locality-aware multi-shape blending,
3. partition-of-unity weight visualisation,
4. morphing sequences,
5. control-point-based curves,
6. custom user-defined parametric shapes.

---

## Example script

A standalone scripted example is provided:

```bash
python examples/basic_demo.py
```

This script generates several figure outputs illustrating representative blending and curve-construction behaviours.

---

## Testing

Basic validation tests are provided in `tests/`.

```bash
pip install pytest
pytest tests/ -v
```

---

## Reproducibility and implementation note

This repository should be understood as an **open research implementation** of the Shape Blend Spline concept.

The current codebase is designed to reflect the mathematical intent and modelling philosophy of the referenced work while also providing a practical, inspectable, and extensible Python implementation.

Accordingly, some implementation details—such as specific locality kernels, normalisation strategies, or shape catalogue design—represent explicit engineering and modelling choices made for clarity, portability, and usability.

Contributions that improve verification against the published method, extend the mathematical framework, or strengthen experimental validation are particularly welcome.

---

## Citation

If you use this repository in research, teaching, or scholarly communication, please cite the original paper:

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

Citation metadata for GitHub is also provided in `CITATION.cff`.

---

## Contributing

Contributions are welcome in areas including:
- verification against the reference publication,
- additional parametric shape families,
- higher-dimensional or surface-based extensions,
- export and visualisation tooling,
- numerical experiments and benchmarks,
- documentation improvements for research and teaching use.

Please open an issue or submit a pull request to discuss proposed changes.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
