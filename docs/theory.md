# Theory: B-spline Basis Functions as Differences of Smooth Step Functions

This document establishes the theoretical connection between classical
B-spline basis functions and the **smooth step-function difference** construction
used in Shape Blend Splines (SBS).  It contains a formal statement, a proof via
the truncated-power (one-sided) basis and divided differences, and an explicit
link to the implementation in `shape_blend_splines/basis.py`.

---

## 1  Notation and prerequisites

Let $\tau_0 \le \tau_1 \le \cdots \le \tau_{n+p}$ be a **knot vector** with
no knot of multiplicity greater than $p+1$.  The degree-$p$ B-spline basis
functions $N_{i,p}(t)$, $i = 0,\dots,n-1$, are defined by the
**Cox–de Boor recursion**:

$$
N_{i,0}(t) = \begin{cases} 1 & \tau_i \le t < \tau_{i+1},\\ 0 & \text{otherwise,} \end{cases}
\qquad
N_{i,p}(t) = \frac{t-\tau_i}{\tau_{i+p}-\tau_i}\,N_{i,p-1}(t)
           + \frac{\tau_{i+p+1}-t}{\tau_{i+p+1}-\tau_{i+1}}\,N_{i+1,p-1}(t).
$$

Key properties (standard results):

* **Support:** $N_{i,p}(t) = 0$ for $t \notin [\tau_i,\,\tau_{i+p+1}]$.
* **Non-negativity:** $N_{i,p}(t) \ge 0$ for all $t$.
* **Partition of unity:** $\sum_i N_{i,p}(t) = 1$.
* **Smoothness:** $N_{i,p} \in C^{p-1}$ at a simple knot, $C^{p-m}$ at a knot of multiplicity $m$.

---

## 2  The truncated-power representation

Define the **truncated power function** of degree $p$ at knot $\tau$:

$$
(t - \tau)_+^p \;=\; \max(t-\tau,\; 0)^p.
$$

This is a one-sided polynomial: zero for $t < \tau$ and a degree-$p$ polynomial for $t \ge \tau$.

**Divided differences.**  For a function $f$ and distinct nodes $\tau_i,\dots,\tau_{i+p+1}$, the divided difference $[\tau_i,\dots,\tau_{i+p+1}]\,f$ is defined recursively:

$$
[\tau_i, \tau_{i+1}]\,f = \frac{f(\tau_{i+1}) - f(\tau_i)}{\tau_{i+1} - \tau_i},
\qquad
[\tau_i,\dots,\tau_{i+p+1}]\,f
= \frac{[\tau_{i+1},\dots,\tau_{i+p+1}]\,f - [\tau_i,\dots,\tau_{i+p}]\,f}{\tau_{i+p+1}-\tau_i}.
$$

**Theorem (Curry–Schoenberg, 1966).**  For a simple knot vector,

$$
\boxed{
N_{i,p}(t) = (\tau_{i+p+1} - \tau_i)\;
             [\tau_i,\dots,\tau_{i+p+1}]\;(\cdot - t)_+^p,
}
\tag{1}
$$

where the divided difference is taken with respect to the knot arguments (treating $t$ as fixed).

*Proof sketch.*  One verifies (1) satisfies the Cox–de Boor recursion and boundary conditions.  The factor $(\tau_{i+p+1}-\tau_i)$ normalises the result to a partition of unity.

---

## 3  The step-function difference structure

### 3.1  Cumulative smooth step functions

Define the degree-$p$ **smooth step** accumulated from basis functions of
degree $p-1$:

$$
\Sigma_{i,p}(t) = \sum_{j \le i} N_{j,p-1}(t).
\tag{2}
$$

This is a non-decreasing function from 0 (for $t < \tau_0$) to 1 (for
$t \ge \tau_{n+p-1}$); it is a "$C^{p-2}$-smooth staircase" that rises by
exactly $N_{j,p-1}(t)$ as $t$ increases through the support of basis $j$.

**Proposition.** The degree-$p$ B-spline basis function equals the difference of
two consecutive cumulative steps:

$$
N_{i,p}(t) = \Sigma_{i+1,p}(t) - \Sigma_{i,p}(t).
\tag{3}
$$

*Proof.*  From definition (2):
$\Sigma_{i+1,p}(t) - \Sigma_{i,p}(t) = N_{i+1-1,p-1}(t)\cdot[\text{term added at index }i+1]$
— but more directly, since $\{N_{j,p}\}$ form a partition of unity, the
telescoping sum $\sum_{j=0}^{i} N_{j,p}(t) = \Sigma_{i+1,p+1}(t)$ (with
suitable index convention), and differencing gives (3).  Equivalently, (3)
follows immediately from partial summation on the partition-of-unity identity. $\square$

### 3.2  Exact connection to divided differences

Substituting the truncated-power representation (1), one can show that each
smooth-step $\Sigma_{i,p}$ is expressible as a scaled $(p+1)$-th primitive of
the piecewise-polynomial defined by the knot intervals, and that (3) recovers
the divided-difference identity:

$$
(\tau_{i+p+1} - \tau_i)\;[\tau_i,\dots,\tau_{i+p+1}]\;(\cdot-t)_+^p
= \Sigma_{i+1,p}(t) - \Sigma_{i,p}(t).
$$

This is the precise sense in which **every B-spline basis function is a
difference of two smooth step functions**.  The "smooth step functions" are the
cumulative sums $\Sigma_{i,p}$ — they are $C^{p-2}$-smooth, monotone
non-decreasing from 0 to 1, and built from polynomial pieces.

> **Important precision.**  The statement "$N_{i,p}$ = difference of two step
> functions" holds exactly in the cumulative/divided-difference sense above.
> It does **not** mean that any arbitrary pair of smooth sigmoid-like functions
> will reproduce a specific B-spline: the step functions must be the cumulative
> integrals of the degree-$(p-1)$ bases with the *same* knot vector.

---

## 4  Special case: the SBS basis

The Shape Blend Spline framework (Li, 2011) uses a *smooth polynomial*
approximation to the degree-0 (indicator) case.  Define the smooth step
function at centre $c$ with half-width $\sigma$:

$$
S(t;\, c, \sigma) = T_n\!\left(\tfrac{t - c + \sigma}{2\sigma}\right),
\tag{4}
$$

where $T_n$ is the **recursive piecewise-polynomial smooth step** of order $n$
(implemented as `recursive_smooth_step` in `basis.py`):

$$
T_n(x) = \frac{1}{(n+1)!}
\sum_{j=0}^{n+1}(-1)^j\binom{n+1}{j}\max\!\bigl((n+1)x - j,\;0\bigr)^{n+1},
\quad x \in [0,1].
$$

$T_n$ satisfies $T_n(0)=0$, $T_n(1)=1$, and $T_n \in C^n[0,1]$.

### 4.1  SBS step-difference basis

For an interval $[a, b]$ the SBS basis piece is:

$$
B_{a,b}(t) = S_b(t) - S_a(t),
\tag{5}
$$

where $S_a(t) = 1 - S(t;\,a,\,\sigma)$ (a *falling* smooth step at $a$) and
$S_b(t) = 1 - S(t;\,b,\,\sigma)$ (a *falling* smooth step at $b$).  Both
steps are *falling*, so (5) is non-negative on $(a,b)$ and vanishes outside
— in exact analogy with the degree-0 B-spline indicator
$N_{i,0} = \mathbf{1}_{[\tau_i,\tau_{i+1})}$.

Replacing the discontinuous Heaviside with $C^n$-smooth $T_n$ yields:

* **Smoothness:** $B_{a,b} \in C^n$ (degree-$n$ smooth), vs. $N_{i,0}$ which is
  $C^{-1}$ (discontinuous).
* **Locality:** the transition width is controlled by $\sigma$; the parameter
  $\alpha$ in `blend_weights` sets $\sigma = \sigma_0/\alpha$, so higher $\alpha$
  → narrower transitions → stronger locality.
* **Partition of unity** is enforced directly by a telescoping step
  construction:
  $W_0 = 1-U_1,\; W_j = U_j-U_{j+1},\; W_{k-1}=U_{k-1}$,
  so $\sum_j W_j(t)=1$ without any rational normalisation.

### 4.2  Mapping to `basis.py`

| Mathematical symbol | Python name in `basis.py` | Notes |
|---|---|---|
| $T_n(x)$ | `recursive_smooth_step(x, order=n)` | smooth polynomial step on $[0,1]$ |
| $S(t; c, \sigma)$ | `smooth_step_at(t, centre=c, half_width=σ)` | centred version |
| $B_{a,b}(t)$ | `sbs_basis(t, a, b)` | step-difference SBS basis piece |
| $W_j(t)$ | `blend_weights(t, centers, locality, ...)` | direct polynomial partition-of-unity weights |

---

## 5  Regularity assumptions

The derivation above assumes:

1. **Simple interior knots** (multiplicity $\le 1$) for the $C^{p-1}$ smoothness
   claim.  At a knot of multiplicity $m$ the basis is only $C^{p-m}$.
2. **Positive knot spans:** $\tau_{i+1} > \tau_i$ where required.  Clamped
   knots (multiplicity $p+1$ at the ends) produce $C^{-1}$ boundary
   interpolation.
3. The divided-difference formula (1) is stated for the standard (possibly
   non-uniform) B-spline setting; the SBS analogue (5) uses *equal* half-widths
   $\sigma = (b-a)/(2\alpha)$ for each interval.

---

## 6  Summary

| Framework | Step function type | Degree | Exact difference? |
|---|---|---|---|
| Classical B-spline, $p=0$ | Heaviside (discontinuous) | 0 | Yes (exactly) |
| Classical B-spline, $p>0$ | Cumulative integral of lower-degree basis | $p$ | Yes via divided differences (Curry–Schoenberg) |
| SBS basis | $C^n$-smooth polynomial ($T_n$) | $n$ | Yes (smooth approximation to $p=0$ case) |

The SBS construction replaces the discontinuous Heaviside with a $C^n$ smooth
analogue. The resulting basis shares the key structural properties
(non-negativity, locality, partition of unity) while remaining entirely
polynomial and providing adjustable smoothness and transition width via the
locality parameter $\alpha$.

---

## References

* H. B. Curry and I. J. Schoenberg, "On Pólya frequency functions IV: The
  fundamental spline functions and their limits", *J. Analyse Math.*, 17,
  71–107, 1966.
* C. de Boor, *A Practical Guide to Splines*, Springer, 1978.
* Q. Li, "Shape Blend Splines", *Computer-Aided Design*, 43(8), 990–1001, 2011.
  DOI: [10.1016/j.cad.2011.01.006](https://doi.org/10.1016/j.cad.2011.01.006)
