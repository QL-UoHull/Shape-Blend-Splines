# Partial Shape-Preserving Splines (PSP Splines)

<p align="center">
  <a href="https://colab.research.google.com/github/QL-UoHull/Shape-Blend-Splines/blob/main/notebooks/interactive_shape_blend_demo.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+"/>
</p>

> **PSP splines are B-spline-like, achieve what NURBS achieves, are *more flexible*
> than NURBS — and are NOT rational.**

This repository provides a faithful Python implementation of the Partial Shape-Preserving (PSP) spline technique from:

> Q. Li, J. Tian, **"Partial shape-preserving splines"**,
> *Computer-Aided Design* **43** (2011) 394–409.

---

## Why PSP splines?

| Property | B-spline | NURBS | **PSP spline** |
|---|---|---|---|
| Polynomial (non-rational) | ✓ | ✗ (rational) | **✓** |
| Partition of unity | ✓ | ✓ | **✓** |
| C^{n-1} smoothness | ✓ | ✓ | **✓** |
| Local control | ✓ | ✓ | **✓** |
| Basis reaches value 1 (flat-top) | ✗ | via rational weights | **✓** |
| Exact primitive reproduction | ✗ | ✓ | **✓** |
| Weights without rational denominator | N/A | ✗ | **✓** (knot spacings) |
| Extra design dimension δ | ✗ | ✗ | **✓** |
| Selective/partial interpolation | ✗ | ✗ | **✓** |

PSP splines keep all the nice B-spline properties, then add:

1. **Flat-top shape preservation** — the basis B^{(n)} equals *exactly 1* on
   [a+δ, b−δ], the *shape-preserving interval*. The corresponding
   control point/primitive is reproduced exactly there. Classical B-splines
   never reach 1; NURBS achieves this only via a rational denominator.

2. **Weights as knot spacings (non-rational)** — weight w_i = a_{i+1} − a_i.
   A larger weight ⇒ wider interval ⇒ stronger pull toward P_i, exactly
   like a NURBS weight but with **no rational denominator** (Eq. 20–21).

3. **Extra design dimension δ** — same control polygon + same weights + different
   δ gives a *different* curve family (Figs. 9b, 11a vs 11b). NURBS has no
   equivalent.

4. **Selective interpolation** — control points whose interval width ≥ 2δ are
   interpolated *exactly*; others are merely approached (Fig. 11). Multiple
   straight segments can be embedded in an otherwise smooth freeform curve.

---

## Core formula (Eq. 17)

**Any PSP basis is the difference of two smooth unit steps:**

$$B^{(n)}_{[a,b],\delta}(x) = H_{n,\delta}(x-a) - H_{n,\delta}(x-b)$$

where H_{n,δ}(x) = H_n(nx/δ) and H_n is a smooth piecewise polynomial
(C^{n-1}) that rises from 0 to 1 over [−δ, δ].

### Flat-top = shape preservation

The basis equals **1 exactly** on [a+δ, b−δ].  On the flat-top, the
corresponding control point (or primitive) is reproduced exactly.
Smaller δ → wider flat-top; larger δ → bump shape (Fig. 5).

### Partition of unity (Eq. 18)

For knots t_0 ≤ … ≤ t_m, the PSP basis matrix satisfies:

$$\sum_i B^{(n)}_{[t_i,t_{i+1}],\delta}(x) = 1 \quad \forall x \in [t_0+\delta,\ t_m-\delta]$$

No rational normalization — ever.

---

## Installation

```bash
git clone https://github.com/QL-UoHull/Shape-Blend-Splines.git
cd Shape-Blend-Splines
pip install -r requirements.txt
pip install -e .[dev,notebook]
```

---

## Quick start

### Weighted control-polygon curve (Fig. 9)

```python
import numpy as np
from shape_blend_splines import WeightedControlPolygonPSPSpline

ctrl = np.array([[0,0], [1,1], [2,0], [3,1], [4,0]], dtype=float)

# Equal weights (uniform PSP)
spl = WeightedControlPolygonPSPSpline(ctrl, n=3, delta=0.4)
t = np.linspace(spl.knots[0], spl.knots[-1], 300)
pts = spl.evaluate(t)

# Unequal weights: P_1 gets a long interval → interpolated exactly
weights = [1, 3, 1, 1, 1]
spl2 = WeightedControlPolygonPSPSpline(ctrl, weights=weights, n=3, delta=0.4)
print("Interpolated:", spl2.interpolated_control_points())  # [1]
```

### Primitive blending (Eq. 22)

```python
from shape_blend_splines import BlendedPrimitivePSPSpline
import numpy as np

def arc(t):
    return np.column_stack([t, np.sin(t)])

def line(t):
    return np.column_stack([t, np.zeros_like(t)])

spl = BlendedPrimitivePSPSpline([arc, line], knots=[0, 3, 6], n=3, delta=0.8)
pts = spl.evaluate(np.linspace(0, 6, 300))
```

### Hermite position + velocity (Eq. 23)

```python
from shape_blend_splines import HermitePSPSpline
import numpy as np

pts = np.array([[0,0],[2,1],[4,0]], dtype=float)
vel = np.array([[1,0],[0,-1],[1,0]], dtype=float)
knots = [0, 2, 4, 6]

herm = HermitePSPSpline(pts, vel, knots=knots, delta=0.6)
curve = herm.evaluate(np.linspace(0, 6, 300))
# P(t_i) = pts[i] and P'(t_i) = vel[i] exactly
```

### Backward compatibility (deprecated aliases)

```python
# Old API still works but emits DeprecationWarning
from shape_blend_splines import PeriodicShapeBlendSpline  # deprecated
from shape_blend_splines.shapes import circle_arc, star_arc
sbs = PeriodicShapeBlendSpline([circle_arc, star_arc], locality=2.0)
```

---

## Main demos

Run these scripts to reproduce the paper's key figures:

```bash
python examples/basic_demo.py             # H_n, PSP basis, partition, B-spline case, Fig. 9
python examples/figure10_nonequal_intervals.py  # Fig. 10: Eq.22 edge-primitive blending (straight edge flat-tops + rounded transitions)
python examples/figure11_selective_interpolation.py  # Fig. 11: selective interpolation
python examples/hermite_motion.py         # Eq. 23: Hermite position+velocity
```

---

## Repository structure

```text
Shape-Blend-Splines/
├── shape_blend_splines/
│   ├── __init__.py          — public API + citations
│   ├── basis.py             — H_n, H_{n,δ}, PSP basis, partition (Eqs. 6,11,17,18)
│   ├── curve.py             — PSPSpline, WeightedControlPolygon, BlendedPrimitive, Hermite
│   ├── blend.py             — global weighted baseline (NOT the paper technique)
│   └── shapes.py            — parametric primitives (line, arc, sine, helix, …)
├── notebooks/
│   └── interactive_shape_blend_demo.ipynb  — interactive PSP explorer
├── docs/
│   └── theory.md            — complete mathematical reference
├── examples/
│   ├── basic_demo.py
│   ├── figure10_nonequal_intervals.py
│   ├── figure11_selective_interpolation.py
│   └── hermite_motion.py
├── tests/
│   └── test_smoke.py        — PSP-faithful regression tests
├── 2011-PSP Splines-final.pdf  — source paper
└── CITATION.cff
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## API summary

### New paper-faithful API

| Class | Eq. | Description |
|---|---|---|
| `PSPSpline` | — | Generic PSP base: control points or callable primitives |
| `WeightedControlPolygonPSPSpline` | 21 | Weights as knot spacings |
| `BlendedPrimitivePSPSpline` | 22 | Blend whole parametric primitives |
| `HermitePSPSpline` | 23 | Interpolate position + velocity |
| `PeriodicPSPSpline` | — | Closed-loop variant |

| Function | Description |
|---|---|
| `basis.smooth_unit_step(x, n)` | H_n(x) (Eq. 6) |
| `basis.smooth_unit_step_delta(x, n, delta)` | H_{n,δ}(x) (Eq. 11) |
| `basis.psp_basis(x, a, b, n, delta)` | B^{(n)}_{[a,b],δ}(x) (Eq. 17) |
| `basis.psp_partition(x, knots, n, delta)` | Basis matrix (Eq. 18) |
| `basis.shape_preserving_interval(a, b, delta)` | [a+δ, b−δ] flat-top |
| `basis.knots_from_weights(weights)` | Knots from interval widths (Eq. 20) |
| `basis.interpolated_indices(knots, delta)` | Which control points are interpolated |

### Deprecated aliases (backward compatible)

The old API still works but emits `DeprecationWarning`:

| Old name | Replacement |
|---|---|
| `ShapeBlendSpline` | `WeightedControlPolygonPSPSpline` / `BlendedPrimitivePSPSpline` |
| `PeriodicShapeBlendSpline` | `PeriodicPSPSpline` |
| `ControlPointSpline` | `WeightedControlPolygonPSPSpline` |
| `ShapeBlender` | (kept; global weighted baseline, not the paper technique) |

---

## Theory

See [`docs/theory.md`](docs/theory.md) for the complete mathematical reference
with all equations (Eqs. 1–23) from Li & Tian (2011).

---

## Reference

> Q. Li, J. Tian, **"Partial shape-preserving splines"**,
> *Computer-Aided Design* **43** (2011) 394–409.
