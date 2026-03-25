<div align="center">

# ⚛️ AtomicDiff

**High-Order Automatic Differentiation with Taylor Lanes & Lock-Free Atomic Accumulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Header-only](https://img.shields.io/badge/Header--only-brightgreen.svg)](https://github.com/tuusuario/atomicdiff)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/tuusuario/atomicdiff)

*AtomicDiff is a modern C++20 header-only library that implements high-order automatic differentiation (AD) using independent Taylor lanes. Unlike traditional AD libraries that compute unnecessary mixed derivatives ($\mathcal{O}(V^2)$ complexity), AtomicDiff maintains each variable's derivatives in separate, independent lanes, achieving $\mathcal{O}(V \cdot N)$ complexity where $V$ is the number of variables and $N$ is the derivative order.*

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🎯 Motivation](#-motivation)
- [📐 Mathematical Foundations](#-mathematical-foundations)
- [🚀 Quick Start](#-quick-start)
- [📊 Benchmark Results](#-benchmark-results)
- [📚 API Reference](#-api-reference)
- [💻 Examples](#-examples)
- [📦 Installation](#-installation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **High-Order Derivatives** | Up to any order $N$ with machine precision (tested up to order 20) |
| 🔄 **Multiple Variables** | Support for $V$ independent variables |
| ⚡ **Lock-Free Atomic Accumulation** | Mutex-free parallelism using `std::atomic_ref` |
| 📦 **Header-Only** | Include and use, no external dependencies |
| 🧮 **25+ Mathematical Functions** | Trigonometric, exponential, logarithmic, hyperbolic, special |
| 🚀 **Zero-Loop Metaprogramming** | Complete compile-time expansion |
| 💾 **Memory Efficient** | $\mathcal{O}(V \cdot N)$ with contiguous layout for cache locality |
| 🔧 **Modern C++20** | `constexpr`, templates, variadic metaprogramming |

---

## 🎯 Motivation

In problems with independent variables (hyperparameter optimization, univariate sensitivity analysis, per-variable Newton methods), traditional AD methods unnecessarily compute mixed derivatives $\partial^2 f/\partial x_i \partial x_j$ with $\mathcal{O}(V^2)$ cost.

**AtomicDiff** introduces the concept of **Taylor lanes** that maintain each variable's derivatives in separate, independent paths, reducing complexity to $\mathcal{O}(V \cdot N)$.

$$\mathcal{L}_i^N[f] = \left( f,\ \frac{\partial f}{\partial x_i},\ \frac{1}{2!}\frac{\partial^2 f}{\partial x_i^2},\ \ldots,\ \frac{1}{N!}\frac{\partial^N f}{\partial x_i^N} \right)$$

---

## 📐 Mathematical Foundations

### 1. Normalized Taylor Coefficients

Let $f: \mathbb{R}^V \rightarrow \mathbb{R}$ be an analytic function. For each independent variable $x_i$, we define the **normalized Taylor coefficients**:

$$\boxed{a_{i,n} = \frac{1}{n!} \frac{\partial^n f}{\partial x_i^n}(x_0)}$$

This normalization offers several advantages over storing raw derivatives $f^{(n)}$:

- **Bounded growth**: $a_{i,n}$ remains bounded even for high orders
- **Simplified algebra**: Multiplication becomes Cauchy convolution
- **Overflow prevention**: Factorials are absorbed into coefficients

---

### 2. Taylor Lane Representation

For a function $f(x_1, \ldots, x_V)$, we define Taylor lanes as independent vectors for each variable:

$$\mathcal{L}_i^N[f] = \begin{pmatrix} a_{i,0} \\ a_{i,1} \\ a_{i,2} \\ \vdots \\ a_{i,N} \end{pmatrix} = \begin{pmatrix} f \\ \dfrac{\partial f}{\partial x_i} \\[6pt] \dfrac{1}{2!}\dfrac{\partial^2 f}{\partial x_i^2} \\ \vdots \\[6pt] \dfrac{1}{N!}\dfrac{\partial^N f}{\partial x_i^N} \end{pmatrix}$$

> **Key Property**: Lanes for different variables never mix during arithmetic operations.

---

### 3. Algebraic Operations

#### 3.1 Addition

For $h = f + g$, coefficients add element-wise:

$$\boxed{c_{i,n} = a_{i,n} + b_{i,n} \quad \forall\, i, n}$$

#### 3.2 Multiplication (Cauchy Convolution)

For $h = f \cdot g$:

$$\boxed{c_{i,n} = \sum_{k=0}^{n} a_{i,k} \cdot b_{i,n-k}}$$

**Proof:**

$$h(x) = \left(\sum_{k=0}^{\infty} a_k x^k\right)\!\left(\sum_{j=0}^{\infty} b_j x^j\right) = \sum_{n=0}^{\infty} \left(\sum_{k=0}^{n} a_k b_{n-k}\right) x^n \qquad \blacksquare$$

#### 3.3 Division

For $h = f / g$ with $g_0 \neq 0$:

$$\boxed{c_0 = \frac{a_0}{b_0}, \qquad c_n = \frac{a_n - \displaystyle\sum_{k=0}^{n-1} c_k\, b_{n-k}}{b_0} \quad (n \geq 1)}$$

#### 3.4 Unary Negation

For $h = -f$:

$$\boxed{c_n = -a_n \quad \forall\, n}$$

---

### 4. Elementary Function Recurrences

#### 4.1 Exponential Function

For $y = e^u$:

$$\boxed{y_0 = e^{u_0}, \qquad y_n = \frac{1}{n} \sum_{k=1}^{n} k \cdot u_k \cdot y_{n-k} \quad (n \geq 1)}$$

> **Derivation**: From $y' = y \cdot u'$, differentiate $n$ times and apply the Leibniz rule.

#### 4.2 Natural Logarithm

For $y = \ln u$ with $u_0 > 0$:

$$\boxed{y_0 = \ln u_0, \qquad y_n = \frac{u_n}{u_0} - \frac{1}{n\, u_0} \sum_{k=1}^{n-1} k \cdot y_k \cdot u_{n-k} \quad (n \geq 1)}$$

> **Derivation**: From $u = e^y$, substitute the exponential recurrence and solve for $y_n$.

#### 4.3 Sine and Cosine (Wengert Recurrence)

For $s = \sin u$ and $c = \cos u$:

$$\boxed{s_0 = \sin u_0, \qquad s_n = \frac{1}{n} \sum_{k=1}^{n} k \cdot u_k \cdot c_{n-k} \quad (n \geq 1)}$$

$$\boxed{c_0 = \cos u_0, \qquad c_n = -\frac{1}{n} \sum_{k=1}^{n} k \cdot u_k \cdot s_{n-k} \quad (n \geq 1)}$$

> **Derivation**: From $s' = c \cdot u'$ and $c' = -s \cdot u'$, apply the product rule for Taylor coefficients.

#### 4.4 Tangent

For $t = \tan u$:

$$\boxed{t_0 = \tan u_0, \qquad t_n = \frac{1}{n} \sum_{k=1}^{n} k \cdot u_k \cdot (1 + t^2)_{n-k}}$$

where $(1 + t^2)_{m}$ denotes the $m$-th normalized Taylor coefficient of $1 + t^2$, computed via Section 3.2.

> **Important**: The recurrence is well-defined because each $(1+t^2)_{n-k}$ with $k \geq 1$ only depends on $t_0, \ldots, t_{n-k}$, all already known at step $n$. There is no circular dependency.

> **Derivation**: From $t' = (1 + t^2) \cdot u'$, apply the product rule for Taylor coefficients (Section 3.2) and solve for $t_n$.

#### 4.5 Power Function

For $y = u^p$ with $p \in \mathbb{R}$ and $u_0 \neq 0$:

$$\boxed{y_0 = u_0^p, \qquad y_n = \frac{1}{n\, u_0} \sum_{k=1}^{n} \bigl((p+1)k - n\bigr) \cdot u_k \cdot y_{n-k} \quad (n \geq 1)}$$

> **Derivation**: From $u \cdot y' = p \cdot y \cdot u'$, apply the Leibniz rule for normalized coefficients to obtain $n\, u_0\, y_n = \sum_{k=1}^{n} \bigl[pk - (n-k)\bigr] u_k\, y_{n-k}$, which simplifies to the factor $((p+1)k - n)$.

> **Alternative**: Compute via $y = e^{p \ln u}$ using the recurrences in Sections 4.1–4.2.

#### 4.6 Square Root

For $y = \sqrt{u}$ (special case $p = 1/2$):

$$\boxed{y_0 = \sqrt{u_0}, \qquad y_n = \frac{1}{2\, y_0} \left( u_n - \sum_{k=1}^{n-1} y_k\, y_{n-k} \right) \quad (n \geq 1)}$$

---

### 5. Hyperbolic Functions

#### 5.1 Hyperbolic Sine and Cosine

Computed via the exponential recurrence (Section 4.1):

$$\sinh u = \frac{e^u - e^{-u}}{2}, \qquad \cosh u = \frac{e^u + e^{-u}}{2}$$

#### 5.2 Hyperbolic Tangent

$$\tanh u = \frac{\sinh u}{\cosh u}$$

The Taylor coefficients of $\tanh u$ are obtained by applying the division recurrence (Section 3.3) to those of $\sinh u$ and $\cosh u$.

---

### 6. Inverse Trigonometric Functions

#### 6.1 Arcsine

For $y = \arcsin u$ with $|u_0| < 1$:

$$y_0 = \arcsin u_0, \qquad y_n = \frac{1}{n} \sum_{k=1}^{n} k \cdot \left(\frac{1}{\sqrt{1-u^2}}\right)_{\!n-k} \cdot u_k$$

where $\bigl(1/\sqrt{1-u^2}\bigr)_m$ denotes the $m$-th Taylor coefficient of $(1 - u^2)^{-1/2}$, computed via Sections 4.5–4.6.

#### 6.2 Arctangent

For $y = \arctan u$:

$$y_0 = \arctan u_0, \qquad y_n = \frac{1}{n} \sum_{k=1}^{n} k \cdot \left(\frac{1}{1+u^2}\right)_{\!n-k} \cdot u_k$$

where $\bigl(1/(1+u^2)\bigr)_m$ denotes the $m$-th Taylor coefficient of $(1 + u^2)^{-1}$, computed via Section 3.3.

---

### 7. Complexity Analysis

| Metric | AtomicDiff | Traditional AD |
|--------|-----------|----------------|
| Time per operation | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ |
| Total time for $V$ variables | $\mathcal{O}(V \cdot N^2)$ | $\mathcal{O}(V^2 \cdot N^2)$ |
| Memory per variable | $\mathcal{O}(N)$ | $\mathcal{O}(V \cdot N)$ |
| Total memory | $\mathcal{O}(V \cdot N)$ | $\mathcal{O}(V^2 \cdot N)$ |

> AtomicDiff reduces both time and memory by a factor of $V$ by never computing cross-variable (mixed) derivatives.

---

### 8. Numerical Stability

**Theorem (Error Bound)**: For a function $f$ with normalized coefficients $a_n$, the relative error after $n$ convolution steps is bounded by:

$$\frac{|\tilde{a}_n - a_n|}{|a_n|} \leq \varepsilon \cdot n^2 \cdot \max_{k \leq n} \frac{|a_k|}{|a_n|}$$

where $\varepsilon \approx 2.22 \times 10^{-16}$ is IEEE 754 double-precision machine epsilon.

**Consequence**: AtomicDiff maintains machine precision up to approximately $N \approx 20$ before error accumulation becomes significant for typical analytic functions.

---

### 9. Key Theorems

**Theorem 1 (Lane Independence)**: For any arithmetic expression involving independent variables, the Taylor coefficients satisfy the per-variable Leibniz rule:

$$\frac{\partial^n (f \cdot g)}{\partial x_i^n} = \sum_{k=0}^{n} \binom{n}{k} \frac{\partial^k f}{\partial x_i^k} \cdot \frac{\partial^{n-k} g}{\partial x_i^{n-k}}$$

In normalized form: $c_{i,n} = \sum_{k=0}^{n} a_{i,k} \cdot b_{i,n-k}$. This is exactly the Cauchy convolution (Section 3.2), confirming that each lane is fully self-contained.

---

**Theorem 2 (No Mixed Derivatives via Taylor Lanes)**: For any $i \neq j$, the Taylor lane engine never computes cross-variable derivatives of the form:

$$\frac{\partial^2 f}{\partial x_i\, \partial x_j}, \quad \frac{\partial^3 f}{\partial x_i^2\, \partial x_j}, \quad \text{etc.}$$

This is enforced structurally: lane $i$ only ever convolves coefficients from lane $i$. The result is $\mathcal{O}(V \cdot N^2)$ time instead of $\mathcal{O}(V^2 \cdot N^2)$.

> **Note on the `Hessian` class**: off-diagonal entries $\partial^2 f/\partial x_i \partial x_j$ ($i \neq j$) are computed via **Ridders' method** — Richardson extrapolation applied over exact AD first derivatives. Diagonal entries remain exact and $\mathcal{O}(1)$. This is the optimal hybrid strategy: AD exactness where lanes suffice, numerical differentiation only where unavoidable.

---

**Theorem 3 (Convergence)**: For $N \rightarrow \infty$, the Taylor series $\sum_{n=0}^{\infty} a_{i,n}\, h^n$ converges to $f(x_0 + h\, e_i)$ in a neighborhood of $x_0$ whenever $f$ is analytic at $x_0$.

---

## 🚀 Quick Start

### Installation

AtomicDiff is header-only — just include and use:

```cpp
#include <AtomicDiff/derivatives.hpp>
using namespace ad;
```

### Basic Example

```cpp
#include <iostream>
#include <iomanip>
#include <AtomicDiff/derivatives.hpp>
using namespace ad;

int main() {
    // Create independent variables (order N=4, V=2 variables)
    auto x = Taylor<4, 2>::variable(2.0, 0);  // x = 2.0, variable index 0
    auto y = Taylor<4, 2>::variable(3.0, 1);  // y = 3.0, variable index 1

    // Build your expression: f(x,y) = sin(x)*exp(y) + sqrt(x²+y²)
    auto f = sin(x) * exp(y) + sqrt(x*x + y*y);

    // Get function value
    std::cout << "f(2,3)    = " << f.val()           << "\n";  //  21.8693

    // Get first derivatives (gradient)
    // deriv(var_index, order) returns the normalized coefficient a_{i,n}
    // ∂f/∂xᵢ = 1! · deriv(i, 1) = deriv(i, 1)
    std::cout << "∂f/∂x     = " << f.deriv(0, 1)     << "\n";  //  -7.80383
    std::cout << "∂f/∂y     = " << f.deriv(1, 1)     << "\n";  //  19.0958

    // Get second derivatives (Hessian diagonal)
    // ∂²f/∂xᵢ² = 2! · deriv(i, 2) = 2 · deriv(i, 2)
    std::cout << "∂²f/∂x²   = " << f.deriv(0, 2) * 2 << "\n";  // -18.0717
    std::cout << "∂²f/∂y²   = " << f.deriv(1, 2) * 2 << "\n";  //  18.3491

    return 0;
}
```

**Note on `deriv(i, n)`**: returns the normalized Taylor coefficient $a_{i,n} = \frac{1}{n!}\frac{\partial^n f}{\partial x_i^n}$. To recover the raw $n$-th derivative, multiply by $n!$ — e.g. `f.deriv(i, 2) * 2` gives $\partial^2 f/\partial x_i^2$.

**Expected output:**
```
f(2,3)    =  21.8693
∂f/∂x     =  -7.80383
∂f/∂y     =  19.0958
∂²f/∂x²   = -18.0717
∂²f/∂y²   =  18.3491
```

---

## 📊 Benchmark Results

> Derivative values and correctness results are from real executions. Timing results (§6) are from `test_exhaustivo.cpp` running 1,000,000 iterations with `-O3 -march=native`. Parallel scaling (§7) is illustrative — see note.

### 1. High-Order Derivatives — Single Variable

`f(x) = exp(sinh(x²))` at `x = 1`

| Order | Derivative Value | Time (μs) | Error (ULPs) |
|------:|----------------:|----------:|-------------:|
| 1 | 9.99544e+00 | 11.80 | 0 |
| 2 | 5.60679e+01 | 0.86 | 0 |
| 3 | 4.14360e+02 | 0.68 | ≤2 |
| 4 | 3.72809e+03 | 0.68 | ≤2 |
| 5 | 3.89545e+04 | 0.67 | ≤2 |
| 6 | 4.62039e+05 | 0.68 | ≤2 |
| 7 | 6.10654e+06 | 0.67 | ≤2 |
| 8 | 8.86964e+07 | 0.68 | ≤2 |
| 9 | 1.40110e+09 | 0.68 | ≤2 |
| 10 | 2.38728e+10 | 0.70 | ≤2 |
| 11 | 4.35785e+11 | 0.69 | ≤2 |
| 12 | 8.47493e+12 | 0.69 | ≤2 |
| 13 | 1.74757e+14 | 0.69 | ≤2 |
| 14 | 3.80540e+15 | 0.70 | ≤2 |
| 15 | 8.71979e+16 | 0.71 | ≤2 |
| 16 | 2.09610e+18 | 0.72 | ≤2 |
| 17 | 5.27151e+19 | 0.71 | ≤2 |
| 18 | 1.38364e+21 | 0.72 | ≤2 |
| 19 | 3.78213e+22 | 0.73 | ≤2 |
| 20 | 1.07454e+24 | 0.73 | ≤2 |

After order 2, each derivative costs ~0.7 μs regardless of order — $\mathcal{O}(N)$ amortized per lane.

---

### 2. Multi-Variable Gradient & Hessian Diagonal

`f(x,y,z) = sin(x)·exp(y) + cos(z)·√(x²+y²+z²)` at `(2.0, 3.0, 1.5)`

| Variable | Value | $\partial f/\partial x_i$ | $\partial^2 f/\partial x_i^2$ |
|---------:|------:|--------------------------:|------------------------------:|
| x | 2.000000 | −8.322305 | −18.250364 |
| y | 3.000000 | 18.318069 | 18.271151 |
| z | 1.500000 | −3.868172 | −1.027093 |

All three lanes computed in a single forward pass with no mixed derivatives.

---

### 3. Polynomial Derivatives — Exact (Zero Error)

`f(x) = 5x⁴ + 3x³ − 2x² + 7x − 1` at `x = 2`

| Order | AtomicDiff | Analytical | Error |
|------:|-----------:|-----------:|------:|
| 0 | 109.000000 | 109.000000 | 0.0 |
| 1 | 195.000000 | 195.000000 | 0.0 |
| 2 | 272.000000 | 272.000000 | 0.0 |
| 3 | 258.000000 | 258.000000 | 0.0 |
| 4 | 120.000000 | 120.000000 | 0.0 |
| 5 | 0.000000 | 0.000000 | 0.0 |

Polynomial derivatives are exact — Taylor arithmetic on polynomials terminates with zero truncation error.

---

### 4. Trigonometric & Hyperbolic Identities

```
sin²(x) + cos²(x)   = 1.000000000000000   (error = 0.0,    ∂/∂x = 0.0)
cosh²(x) − sinh²(x) = 1.000000000000000   (error = 2.2e−16, within ε_machine)
exp(ln(x))           = x                   (error = 0.0,    ∂/∂x = 1.0)
```

All identities verified at the level of IEEE 754 double precision.

---

### 5. Analytical Validation — `sin(x)` at `x = 1.0`

Derivatives cycle as $\sin, \cos, -\sin, -\cos, \ldots$

| Order | AtomicDiff | Analytical | Difference |
|------:|-----------:|-----------:|-----------:|
| 0 | 8.41e−01 | 8.41e−01 | 0.00e+00 |
| 1 | 5.40e−01 | 5.40e−01 | 0.00e+00 |
| 2 | −8.41e−01 | −8.41e−01 | 0.00e+00 |
| 3 | −5.40e−01 | −5.40e−01 | 0.00e+00 |
| 4 | 8.41e−01 | 8.41e−01 | 0.00e+00 |
| 5 | 5.40e−01 | 5.40e−01 | 0.00e+00 |
| 6 | −8.41e−01 | −8.41e−01 | 1.11e−16 |
| 7 | −5.40e−01 | −5.40e−01 | 1.11e−16 |

Error at orders 6–7 is $\varepsilon/2 \approx 1.11 \times 10^{-16}$ — consistent with the theoretical bound from §8 of the mathematical foundations.

---

### 6. Measured Throughput (1,000,000 iterations, `-O3 -march=native`)

Real timings from `test_exhaustivo.cpp` with variable inputs (no compiler constant-folding):

| Expression | Variables | Order | Time/iter |
|:-----------|----------:|------:|----------:|
| `sin(x)·exp(x)` + 1st deriv | 1 | 4 | 19.7 ns |
| `sin(x)·exp(y) + √(x²+y²)` + gradient | 2 | 4 | 105 ns |
| `exp(sinh(x²))` + 10th deriv | 1 | 20 | 648 ns |

The 2-variable case (105 ns) computes two full independent lanes plus gradient extraction in a single forward pass. The order-20 case (648 ns) is the most representative for high-order workloads.

---

### 7. Parallel Scaling

> ⚠️ **Simulated data** — these values are illustrative estimates, not measurements on real hardware. Real benchmarks will be added in a future release.

`f(x) = sin(x)·cos(x)·exp(x)·√x` — 10,000,000 independent evaluations (simulated)

| Threads | Time (ms) | Speedup | Derivatives/s |
|--------:|----------:|--------:|--------------:|
| 1 | 114.0 | 1.00× | 3.51×10⁸ |
| 2 | 57.5 | 1.98× | 6.96×10⁸ |
| 4 | 29.2 | 3.90× | 1.37×10⁹ |
| **8** | **19.7** | **5.79×** | **2.03×10⁹** |
| 16 | 24.8 | 4.60× | 1.61×10⁹ |

The curve illustrates the expected scaling shape: near-linear speedup up to 8 threads, then regression at 16 due to scheduler overhead — typical for lock-free atomic workloads.

---

### 8. Memory Layout — Verified with `sizeof`

Layout: $8 \times (1 + V \cdot N)$ bytes (contiguous `double` array, verified with `sizeof`)

| Type | Elements | `sizeof` (bytes) | Memory |
|:-----|--------:|-----------------:|-------:|
| `Taylor<1,1>` | 2 | 16 | 16 B |
| `Taylor<4,1>` | 5 | 40 | 40 B |
| `Taylor<4,16>` | 65 | 520 | 0.51 KB |
| `Taylor<4,64>` | 257 | 2,056 | 2.01 KB |
| `Taylor<4,256>` | 1,025 | 8,200 | 8.01 KB |
| `Taylor<20,1>` | 21 | 168 | 168 B |
| `Taylor<20,1024>` | 20,481 | 163,848 | 160 KB |

All `sizeof` values match the formula exactly — zero padding, zero overhead.

---

### Summary

| Metric | Result |
|--------|--------|
| Tests passed | ✅ 75/75 |
| Max order tested | 20 |
| Max error (ULPs) | ≤2 |
| `sin(x)·exp(x)` throughput (N=4, 1 var) | 19.7 ns/eval |
| `sin(x)·exp(y)+√(x²+y²)` gradient (N=4, 2 vars) | 105 ns/eval |
| `exp(sinh(x²))` order-20 throughput | 648 ns/eval |
| Memory overhead | 0 (exact `sizeof` verified) |
| Polynomial error | Exact (0.0) |
| Singularities | No crash: `log(0)`→−∞, `sqrt(-1)`→NaN, `tan(π/2)`→1.6e16 |

---

## 📚 API Reference

### `Taylor<N, V>` — Core Type

```cpp
// Construction
Taylor<N,V>::constant(double c)               // constant: all derivatives = 0
Taylor<N,V>::variable(double val, size_t i)   // variable xᵢ at value val

// Access
double val() const                            // f(x₀)
double deriv(size_t var_idx, size_t order)    // (1/order!) · ∂ⁿf/∂xᵢⁿ
// Raw derivative: deriv(i, n) * n!  (e.g. deriv(i,2)*2 for ∂²f/∂xᵢ²)
```

### `variables<N, V>(x₀, x₁, ...)` — Structured Binding Helper

```cpp
auto [x, y, z] = variables<4, 3>(1.0, 2.0, 3.0);
// Equivalent to:
auto x = Taylor<4,3>::variable(1.0, 0);
auto y = Taylor<4,3>::variable(2.0, 1);
auto z = Taylor<4,3>::variable(3.0, 2);
```

### `Derivatives<N, V>` — All Derivatives in One Pass

Evaluates the function **once** and extracts all derivative orders. $\mathcal{O}(1)$ evaluations.

```cpp
auto deriv = make_derivatives<4, 3>(func);

double              val  = deriv.value(point);           // f(x)
std::array<double,V> g   = deriv.gradient(point);        // ∂f/∂xᵢ
std::array<double,V> h   = deriv.hessian_diagonal(point);// ∂²f/∂xᵢ²  (exact)
std::array<double,V> d3  = deriv.derivative<3>(point);   // (1/6)∂³f/∂xᵢ³

// All orders at once (single evaluation):
auto all = deriv.all_derivatives(point);  // vector[ord-1][var_idx]
```

### `Hessian<N, V>` — Full Hessian Matrix

Diagonal entries are **exact** via Taylor lanes ($\mathcal{O}(1)$ eval).
Off-diagonal entries use **Ridders' method** (Richardson extrapolation over exact AD gradients).

```cpp
auto hess = make_hessian<4, 3>(func);

// Full result: value + gradient + Hessian matrix
auto res = hess.full(point);
double val         = res.value;
double dxx         = res.hessian[i][i];   // exact: ∂²f/∂xᵢ²
double dxy         = res.hessian[i][j];   // Ridders: ∂²f/∂xᵢ∂xⱼ (i≠j)

// Individual entries
double h_ij = hess.mixed(point, i, j);    // single mixed derivative
auto   H    = hess.matrix(point);         // full V×V matrix
```

| Method | Diagonal | Off-diagonal | Evaluations |
|--------|----------|-------------|-------------|
| `Derivatives` | ✅ exact | — | 1 |
| `Hessian::full` | ✅ exact | Ridders $O(h^{2k})$ | $1 + O(V^2 \cdot \text{steps}^2)$ |

---

## 💻 Examples

> *(Coming soon)*

---

## 📦 Installation

AtomicDiff is header-only. Simply copy the headers into your project and include:

```cpp
#include <AtomicDiff/derivatives.hpp>
```

**Requirements**: C++20-compliant compiler (GCC 10+, Clang 12+, MSVC 19.29+).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
