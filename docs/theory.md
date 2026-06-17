# Theory: Partial Shape-Preserving (PSP) Splines

**Reference:** Q. Li, J. Tian, "Partial shape-preserving splines",
*Computer-Aided Design* **43** (2011) 394–409.

---

## 1  Why PSP splines?  B-spline vs NURBS vs PSP

| Property | B-spline | NURBS | PSP spline |
|---|---|---|---|
| Polynomial (non-rational) | ✓ | ✗ (rational) | ✓ |
| Partition of unity | ✓ | ✓ | ✓ |
| C^{n-1} smoothness | ✓ | ✓ | ✓ |
| Local control | ✓ | ✓ | ✓ |
| Basis reaches value 1 (flat-top) | ✗ | via rational weights | ✓ |
| Exact primitive reproduction | ✗ | ✓ | ✓ |
| Weights without rational denominator | N/A | ✗ | ✓ (knot spacings) |
| No global denominator for unity (GPU-friendly) | ✓ | ✗ | ✓ |
| Blends control *shapes*, not just points | ✗ | ✗ | ✓ |
| Extra design dimension δ | ✗ | ✗ | ✓ |
| Selective/partial interpolation | ✗ | ✗ | ✓ |

**The headline:** PSP splines are a **natural extension of B-splines** (polynomial,
C^{n-1}, partition of unity, local control), achieve what NURBS achieves (exact
primitive reproduction), are *more flexible* than NURBS (extra design dimension δ,
selective interpolation, shape blending) — and remain **non-rational**.

### 1.1  The core idea: B-splines reconstructed by convolution

PSP splines are a natural extension of B-splines obtained by **reconstructing each
B-spline basis function recursively in the form of a convolution**. The recursion
starts from the **degree-0 top-flat basis** $B^{(0)}(t)$ defined on an interval
$[a, b]$ — the box that equals **1** for $t \in [a, b]$ and **0** outside it:

$$
B^{(0)}_{[a,b]}(t) =
\begin{cases}
1 & a \le t \le b \\
0 & \text{otherwise}.
\end{cases}
$$

The key step is to rewrite this box as the **difference of two Heaviside unit step
functions**, one anchored at each end of the interval $[a, b]$:

$$
B^{(0)}_{[a,b]}(t) = H_0(t - a) - H_0(t - b).
$$

A degree-$n$ B-spline basis is then the repeated convolution of this box with a kernel;
equivalently, smoothing each hard Heaviside step $H_0$ into the **C^{n-1} smooth unit
step** $H_{n,\delta}$ (§3, Eq. 11) — while keeping the *same* difference-of-two-steps
structure — yields the PSP basis (§4, Eq. 17):

$$
B^{(n)}_{[a,b],\delta}(t) = H_{n,\delta}(t - a) - H_{n,\delta}(t - b).
$$

Because each basis is built from **top-flat** building blocks, the PSP construction
turns a **control-point–blending** spline technique (B-spline / NURBS) into a
**control-*shape*-blending** design technique: on the flat top the corresponding
control point — or whole parametric **shape** — is reproduced *exactly*, while key
features can be **selectively preserved** (§8). This makes PSPS **more flexible and
versatile than NURBS**, yet fully polynomial.

### 1.2  GPU mesh-shader / tessellation-shader friendliness

Each PSP basis function is defined **locally** as a simple difference of two smooth
unit steps. Unlike NURBS, **no global denominator** has to be computed to renormalize
all the basis functions back to a partition of unity — the partition of unity holds
**automatically** because consecutive step differences *telescope* (§5, Eq. 18).

This locality is what makes PSPS friendly to **GPU mesh-shader and tessellation-shader**
pipelines: every shader invocation can evaluate its own basis directly from the two
interval endpoints $a, b$ and the blending range $\delta$, with **no global,
cross-lane/cross-element normalization pass** to synchronize. By contrast, a NURBS
evaluation must form the rational denominator $\sum_j w_j N_j(t)$ across the active
basis functions before any point can be produced.

---

## 2  The smooth unit step H_n(x)  (Section 4)

### Heaviside base (Eq. 1)

$$
H_0(x) = \begin{cases}
  0 & x < 0 \\
  \tfrac{1}{2} & x = 0 \\
  1 & x > 0
\end{cases}
$$

### Recursion (Eq. 2)

$$
H_n(x) = \tfrac{1}{2}\!\left[
  \left(1 + \frac{x}{n}\right) H_{n-1}(x+1)
  + \left(1 - \frac{x}{n}\right) H_{n-1}(x-1)
\right], \quad n \ge 1
$$

### Closed form — preferred for implementation (Eq. 6)

$$
H_n(x) = \frac{1}{n!\,2^n}
\sum_{k=0}^{n} (-1)^k \binom{n}{k}
(x + n - 2k)^n H_0(x + n - 2k)
$$

This is vectorisable and avoids catastrophic cancellation for large |x| because
the early-exit rule H_n = 0 for x ≤ −n and H_n = 1 for x ≥ n is applied first.

### Explicit forms (Eqs. 7–10)

$$
H_1(x) = \tfrac{1}{2}\bigl[(x+1)H_0(x+1) - (x-1)H_0(x-1)\bigr]
$$

$$
H_2(x) = \tfrac{1}{8}\bigl[(x+2)^2 H_0(x+2) - 2x^2 H_0(x) + (x-2)^2 H_0(x-2)\bigr]
$$

$$
H_3(x) = \tfrac{1}{48}\bigl[(x+3)^3 H_0(x+3) - 3(x+1)^3 H_0(x+1)
+ 3(x-1)^3 H_0(x-1) - (x-3)^3 H_0(x-3)\bigr]
$$

Piecewise form of H_3 (page 396) for verification:

$$
H_3(x) = \begin{cases}
  0 & x < -3 \\
  \tfrac{1}{48}(3+x)^3 & -3 \le x < -1 \\
  \tfrac{1}{24}(12 + 9x - x^3) & -1 \le x < 0 \\
  1 - H_3(-x) & x \ge 0
\end{cases}
$$

### Properties (Prop. 4.1)

- **C^{n-1} smooth** for n ≥ 1 (degree-n piecewise polynomial).
- **Monotone increasing**.
- H_n(x) = 1 for x ≥ n;  H_n(x) = 0 for x ≤ −n.
- H_n(0) = 1/2.
- **Antisymmetry:** H_n(−x) = 1 − H_n(x).

### Derivatives (Eqs. 12–13)

$$
H_n^{(i)}(x) = \frac{1}{(n-i)!\,2^n}
\sum_{k=0}^{n} (-1)^k \binom{n}{k}
(x+n-2k)^{n-i} H_0(x+n-2k),
\quad 0 \le i < n
$$

---

## 3  Scaled smooth unit step H_{n,δ}  (Eq. 11)

$$
H_{n,\delta}(x) = H_n\!\left(\frac{n\,x}{\delta}\right), \quad \delta > 0
$$

**δ is the rising/blending range:**

- H_{n,δ}(x) = 1 for x ≥ δ
- H_{n,δ}(x) = 0 for x ≤ −δ
- H_{n,δ}(0) = 1/2

Small δ → narrow transition (steep step); large δ → wide transition (gentle slope).

---

## 4  PSP basis function  (Section 5, Eq. 17)  — THE CORE

Starting from the degree-0 top-flat basis written as a difference of two Heaviside
steps (§1.1), and smoothing each step into H_{n,δ}, the PSP basis on interval [a, b] is:

$$
\boxed{B^{(n)}_{[a,b],\delta}(x) = H_{n,\delta}(x-a) - H_{n,\delta}(x-b)}
$$

**Any PSP basis is the difference of two smooth unit steps.**

### Properties (Section 5)

1. **Non-negative:** 0 ≤ B ≤ 1.
2. **C^{n-1} smooth.**
3. **Flat-top (shape preservation):** B = 1 exactly on [a+δ, b−δ]
   when b−a ≥ 2δ.  This is the *shape-preserving interval*.
4. **Additivity:** B_{[a,c]} + B_{[c,b]} = B_{[a,b]}.
5. **Compact support:** B = 0 outside [a−δ, b+δ].
6. Small δ → wide flat-top (Fig. 5, δ=0.1); large δ → bump shape (δ=1.9).

### Flat-top width

$$
\text{flat-top width} = (b-a) - 2\delta \quad \text{(zero when } b-a < 2\delta\text{)}
$$

This is the key difference from B-splines: a B-spline basis *never* reaches 1 (no flat-top), while a PSP basis equals 1 on a whole interval.  NURBS achieves the same effect but through a rational[...]

### Non-symmetric basis (Eq. 19)

$$
B = H_{n,\delta_a}(x-a) - H_{n,\delta_b}(x-b)
$$

Non-negative when 0 ≤ (b−a−δ_b) ≤ δ_a.

---

## 5  Partition of unity  (Eq. 18)

For knots t_0 ≤ … ≤ t_m (with t_{-1} = −∞, t_{m+1} = +∞):

$$
\sum_{i=0}^{m} B^{(n)}_{[t_{i-1},t_i],\delta}(x) = 1 \quad \text{for all } x
$$

This is a telescoping sum: consecutive differences of H_{n,δ}(x−t_j) cancel, leaving H_{n,δ}(x−t_{-1}) − H_{n,δ}(x−t_{m+1}) = 1 − 0 = 1.

In practice, for a finite design domain [t_0, t_m] with m+1 basis functions:

$$
\sum_{i=0}^{m-1} B^{(n)}_{[t_i,t_{i+1}],\delta}(x)
= H_{n,\delta}(x-t_0) - H_{n,\delta}(x-t_m) = 1
\quad \text{for } x \in [t_0+\delta,\ t_m-\delta]
$$

No rational normalization is ever needed. Because the partition of unity arises purely
from this **telescoping** of local step differences — rather than from a global
denominator $\sum_j w_j N_j(x)$ as in NURBS — each basis function can be evaluated
**independently and locally**. This is the property that makes PSPS well suited to
**GPU mesh-shader and tessellation-shader** evaluation (§1.2): no cross-element
reduction is required to enforce unity.

---

## 6  B-spline as a special case  (page 398)

When knots are **uniformly spaced** with unit spacing, the degree-n B-spline
basis function N_{i,n} equals:

$$
N_{i,n}(t) = B^{(n)}_{[a_i,\,a_i+1],\,\delta}(t)
\quad \text{with } a_i = i + \tfrac{n}{2},\quad \delta = \tfrac{n}{2}
$$

For cubic (n=3): δ=1.5, intervals [1.5,2.5], [2.5,3.5], …

With equal-spaced knots and δ = n/2, the interval width = 1 < 2δ = n, so the
flat-top is *empty* — this is consistent with B-splines never reaching value 1.

For **non-equal knots**, PSP and B-spline differ: B-spline shape depends on the
full knot-span configuration; PSP shape depends only on the two endpoint
smoothing parameters.

This recovers the B-spline directly from the convolution / difference-of-steps
reconstruction of §1.1, confirming that PSPS is a **natural extension** of the
B-spline: the classical basis is the special case in which the flat top degenerates to
a single point.

---

## 7  Curve design  (Section 6)

### 7.1 Weighted control polygon  (Eqs. 20–21; Figs. 9, 10)

Given control points P_0, …, P_N and weights w_i ≥ 0:

**Knots from weights (Eq. 20):**
$$
a_0 = 0, \quad a_{i+1} = a_i + w_i, \quad i = 0,\ldots,N
$$

**Curve (Eq. 21):**
$$
P(t) = \sum_{i=0}^{N} P_i\, B^{(n)}_{[a_i,a_{i+1}],\delta}(t)
$$

- A **larger weight** w_i means a wider interval → wider flat-top → stronger
  pull toward P_i (NURBS-weight effect without rational denominator).
- When w_i ≥ 2δ, P_i is interpolated exactly.
- **Same control polygon + same weights + different δ** → different curve
  family (Fig. 9b) — the extra design dimension absent from NURBS.

### 7.2 Primitive blending  (Eq. 22)

Blend whole parametric primitives P_i(t) (lines, arcs, helix, …):

$$
P(t) = \sum_i P_i(t)\, B^{(n)}_{[t_i,t_{i+1}],\delta}(t)
$$

Each P_i(t) is reproduced exactly on its flat-top.  Primitives may be
different mathematical types. This is the **shape-blending** view: rather than
blending isolated control *points*, whole control *shapes* are blended while each
retains its identity on its flat top — the feature that distinguishes PSPS from
NURBS.

### 7.3 Hermite position + velocity  (Eq. 23)

Using the quadratic (n=2) PSP basis:

$$
P(t) = \sum_{i=0}^{N}
\bigl(P_i + (t - t_i)\,v_i\bigr)
\, B^{(2)}_{[a_i,a_{i+1}],\delta}(t)
$$

When t_i is inside a non-empty flat-top: P(t_i) = P_i, P'(t_i) = v_i.
The straight-line primitive P_i + (t−t_i)v_i is reproduced exactly on the
flat-top → embedded straight segments with smooth joins.

---

## 8  Selective/partial interpolation  (Fig. 11)

A control point P_i (or primitive P_i(t)) is **interpolated exactly** iff the
corresponding interval satisfies:

$$
a_{i+1} - a_i \ge 2\delta
\quad\Longleftrightarrow\quad
\text{flat-top}\ [a_i+\delta,\ a_{i+1}-\delta]\ \text{is non-empty}
$$

With partition of unity, B_i = 1 on the flat-top implies all other B_j = 0
there → P(t) = P_i for t in the flat-top.

**Fig. 11 scenario:**
- P0, P2, P5, P7 have long intervals (w ≥ 2δ) → interpolated.
- P1, P3, P4, P6 have short intervals (w < 2δ) → only approached.
- Fig. 11a (δ=1.0) vs 11b (δ=1.8): *same control polygon*, different δ,
  different curve and different set of interpolated points.

---

## 9  Symbol–function table

| Symbol | Python function | Location |
|---|---|---|
| H_0(x) | `heaviside_step(x)` | `basis.py` |
| H_n(x) | `smooth_unit_step(x, n)` | `basis.py` |
| H_{n,δ}(x) | `smooth_unit_step_delta(x, n, delta)` | `basis.py` |
| H_n^{(i)}(x) | `smooth_unit_step_deriv(x, n, i)` | `basis.py` |
| B^{(n)}_{[a,b],δ}(x) | `psp_basis(x, a, b, n, delta)` | `basis.py` |
| Eq. 17 asymmetric | `psp_basis_asymmetric(...)` | `basis.py` |
| Eq. 18 matrix | `psp_partition(x, knots, n, delta)` | `basis.py` |
| [a+δ, b−δ] | `shape_preserving_interval(a, b, delta)` | `basis.py` |
| Eq. 20 knots | `knots_from_weights(weights)` | `basis.py` |
| Selective interp. | `interpolated_indices(knots, delta)` | `basis.py` |
| Eq. 21 | `WeightedControlPolygonPSPSpline` | `curve.py` |
| Eq. 22 | `BlendedPrimitivePSPSpline` | `curve.py` |
| Eq. 23 | `HermitePSPSpline` | `curve.py` |
| Closed curve | `PeriodicPSPSpline` | `curve.py` |
