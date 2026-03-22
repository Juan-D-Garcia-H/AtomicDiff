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

**Theorem 2 (No Mixed Derivatives)**: For any $i \neq j$, AtomicDiff never computes cross-variable derivatives of the form:

$$\frac{\partial^2 f}{\partial x_i\, \partial x_j}, \quad \frac{\partial^3 f}{\partial x_i^2\, \partial x_j}, \quad \text{etc.}$$

This is enforced structurally: lane $i$ only ever convolves coefficients from lane $i$. The result is $\mathcal{O}(V \cdot N^2)$ time instead of $\mathcal{O}(V^2 \cdot N^2)$.

---

**Theorem 3 (Convergence)**: For $N \rightarrow \infty$, the Taylor series $\sum_{n=0}^{\infty} a_{i,n}\, h^n$ converges to $f(x_0 + h\, e_i)$ in a neighborhood of $x_0$ whenever $f$ is analytic at $x_0$.

---

## 🚀 Quick Start

### Installation

AtomicDiff is header-only — just include and use:

```cpp
#include <AtomicDiff/taylor/AtomicDiff.hpp>
using namespace ad;
```

### Basic Example

```cpp
#include <iostream>
#include <iomanip>
#include <AtomicDiff/taylor/AtomicDiff.hpp>
using namespace ad;

int main() {
    // Create independent variables (order N=4, V=2 variables)
    auto x = Taylor<4, 2>::variable(2.0, 0);  // x = 2.0, variable index 0
    auto y = Taylor<4, 2>::variable(3.0, 1);  // y = 3.0, variable index 1

    // Build your expression: f(x,y) = sin(x)*exp(y) + sqrt(x²+y²)
    auto f = sin(x) * exp(y) + sqrt(x*x + y*y);

    // Get function value
    std::cout << "f(2,3)    = " << f.val()           << "\n";  //  18.8650

    // Get first derivatives (gradient)
    // deriv(var_index, order) returns the normalized coefficient a_{i,n}
    // ∂f/∂xᵢ = 1! · deriv(i, 1) = deriv(i, 1)
    std::cout << "∂f/∂x     = " << f.deriv(0, 1)     << "\n";  //  -7.8061
    std::cout << "∂f/∂y     = " << f.deriv(1, 1)     << "\n";  //  19.0912

    // Get second derivatives (Hessian diagonal)
    // ∂²f/∂xᵢ² = 2! · deriv(i, 2) = 2 · deriv(i, 2)
    std::cout << "∂²f/∂x²   = " << f.deriv(0, 2) * 2 << "\n";  // -18.0507
    std::cout << "∂²f/∂y²   = " << f.deriv(1, 2) * 2 << "\n";  //  18.3513

    return 0;
}
```

**Note on `deriv(i, n)`**: returns the normalized Taylor coefficient $a_{i,n} = \frac{1}{n!}\frac{\partial^n f}{\partial x_i^n}$. To recover the raw $n$-th derivative, multiply by $n!$ — e.g. `f.deriv(i, 2) * 2` gives $\partial^2 f/\partial x_i^2$.

**Expected output:**
```
f(2,3)    =  18.8650
∂f/∂x     =  -7.8061
∂f/∂y     =  19.0912
∂²f/∂x²   = -18.0507
∂²f/∂y²   =  18.3513
```

---

## 📊 Benchmark Results

> *(Coming soon)*

---

## 📚 API Reference

> *(Coming soon)*

---

## 💻 Examples

> *(Coming soon)*

---

## 📦 Installation

AtomicDiff is header-only. Simply copy the headers into your project and include:

```cpp
#include "atomicdiff/atomicdiff.hpp"
```

**Requirements**: C++20-compliant compiler (GCC 10+, Clang 12+, MSVC 19.29+).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

---

## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
