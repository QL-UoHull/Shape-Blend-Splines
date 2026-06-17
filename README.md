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

> **PSP splines are a natural extension of B-splines, achieve what NURBS achieves,
> are *more flexible* than NURBS — and are NOT rational.**

This repository provides a faithful Python implementation of the Partial Shape-Preserving (PSP) spline technique from:

> Q. Li, J. Tian, **"Partial shape-preserving splines"**,
> *Computer-Aided Design* **43** (2011) 394–409.

---

## What are PSP splines? (the one-paragraph idea)

PSP splines (**PSPS**) are a **natural extension of B-splines**. A B-spline basis
function is built recursively, and that recursion can be written as a **convolution**.
The starting point of the recursion is the **degree-0 top-flat basis** $B^{(0)}(t)$
defined on an interval $[a, b]$ — the box that takes the value **1** when
$t \in [a, b]$ and **0** outside it. The key observation is that this box can be
rewritten as the **difference of two Heaviside unit step functions**, one anchored at
each end of the interval:

$$B^{(0)}_{[a,b]}(t) = H(t - a) - H(t - b).$$

Replacing the hard Heaviside step $H$ by a **smooth** unit step $H_{n,\delta}$
(which rises from 0 to 1 over a small blending range $\delta$) turns this box into a
smooth, **flat-top** basis function while keeping the "difference of two steps"
structure intact (Eq. 17):

$$B^{(n)}_{[a,b],\delta}(t) = H_{n,\delta}(t - a) - H_{n,\delta}(t - b).$$

Because the basis still reaches **exactly 1** on its flat top $[a+\delta,\, b-\delta]$,
the corresponding control point — or whole **control shape** — is reproduced *exactly*
there. This is what upgrades the method from a **control-point–blending** spline
technique (B-spline/NURBS) into a **control-*shape*-blending** design technique that is
**more flexible and versatile than NURBS**, yet remains fully **polynomial
(non-rational)**.

> **GPU-friendly by construction.** Each PSP basis function is defined **locally** as a
> simple difference of two smooth steps. Unlike NURBS, **no global denominator** has to
> be computed to renormalize all the basis functions back to a partition of unity — the
> partition of unity holds *automatically* (the step differences telescope). This makes
> PSPS especially well suited to **GPU mesh-shader and tessellation-shader** pipelines,
> where each shader invocation can evaluate its own basis locally without any global,
> cross-lane normalization pass.

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
| **No global denominator for unity (GPU-friendly)** | ✓ | ✗ | **✓** |
| **Blends control *shapes*, not just points** | ✗ | ✗ | **✓** |
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

5. **Shape blending instead of point blending** — because each flat-top reproduces
   its primitive exactly, whole **parametric shapes** (lines, arcs, helices, …) can
   be blended into one smooth curve while their key features are selectively
   preserved (Eq. 22).

---

## From B-spline recursion to PSP basis (convolution view)

A degree-n B-spline basis is the repeated **convolution** of the degree-0 box with
itself (the classic recursion). PSP splines start from the *same* degree-0 building
block — the **top-flat box** on $[a, b]$ —

$$
B^{(0)}_{[a,b]}(t) =
\begin{cases}
1 & a \le t \le b \\
0 & \text{otherwise}
\end{cases}
\;=\; H(t - a) - H(t - b),
$$

i.e. the **difference of two Heaviside unit step functions** corresponding to the two
ends of the interval. Smoothing each Heaviside step into the C^{n-1} smooth unit step
$H_{n,\delta}$ (Eq. 11) and keeping the same difference-of-steps form yields the PSP
basis (Eq. 17):

$$B^{(n)}_{[a,b],\delta}(t) = H_{n,\delta}(t - a) - H_{n,\delta}(t - b).$$

This recursive-convolution reconstruction is what makes PSPS a **natural extension of
B-splines**: the B-spline is recovered as a special case (uniform knots, δ = n/2; see
[Theory §6](docs/theory.md)), while the flat top — absent from B-splines — emerges
whenever the interval is wider than the blending range (b − a ≥ 2δ).

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

### Partition of unity (Eq. 18) — no global denominator

For knots t_0 ≤ … ≤ t_m, the PSP basis matrix satisfies:

$$\sum_i B^{(n)}_{[t_i,t_{i+1}],\delta}(x) = 1 \quad \forall x \in [t_0+\delta,\ t_m-\delta]$$

This holds because consecutive differences of H_{n,δ} **telescope** — the partition of
unity is built into the construction. **No rational normalization, and no global
denominator, is ever computed.** Each basis function is evaluated purely locally, which
is exactly what makes PSPS friendly to **GPU mesh-shader and tessellation-shader**
pipelines: there is no cross-element/cross-lane normalization pass to synchronize.

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
python examples/figure10_nonequal_intervals.py  # Fig. 10: non-equal intervals + square-spiral
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
