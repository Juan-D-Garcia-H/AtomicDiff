<div align="center">

# ⚛️ AtomicDiff

**High-Order Automatic Differentiation with Taylor Lanes & Lock-Free Atomic Accumulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Header-only](https://img.shields.io/badge/Header--only-brightgreen.svg)](https://github.com/tuusuario/atomicdiff)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/Juan-D-Garcia-H/AtomicDiff)

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

#### 3.3 Division

For $h = f / g$ with $g_0 \neq 0$:

$$\boxed{c_0 = \frac{a_0}{b_0}, \qquad c_n = \frac{a_n - \displaystyle\sum_{k=0}^{n-1} c_k\, b_{n-k}}{b_0} \quad (n \geq 1)}$$

#### 3.4 Unary Negation

For $h = -f$:

$$\boxed{c_n = -a_n \quad \forall\, n}$$

---

### 4. Elementary Function Recurrences

| Function | Recurrence Formula |
|----------|-------------------|
| $\exp(u)$ | $y_n = \frac{1}{n}\sum_{k=1}^{n} k \cdot u_k \cdot y_{n-k}$ |
| $\ln(u)$ | $y_n = \frac{u_n}{u_0} - \frac{1}{n u_0}\sum_{k=1}^{n-1} k \cdot y_k \cdot u_{n-k}$ |
| $\sin(u)$ | $s_n = \frac{1}{n}\sum_{k=1}^{n} k \cdot u_k \cdot c_{n-k}$ |
| $\cos(u)$ | $c_n = -\frac{1}{n}\sum_{k=1}^{n} k \cdot u_k \cdot s_{n-k}$ |
| $\tan(u)$ | $t_n = \frac{1}{n}\sum_{k=1}^{n} k \cdot u_k \cdot (1 + t^2)_{n-k}$ |
| $u^p$ | $y_n = \frac{1}{n u_0}\sum_{k=1}^{n} ((p+1)k - n) \cdot u_k \cdot y_{n-k}$ |
| $\sqrt{u}$ | $y_n = \frac{1}{2y_0}\left( u_n - \sum_{k=1}^{n-1} y_k y_{n-k} \right)$ |

---

### 5. Complexity Analysis

| Metric | AtomicDiff | Traditional AD |
|--------|-----------|----------------|
| **Time** | $\mathcal{O}(V \cdot N^2)$ | $\mathcal{O}(V^2 \cdot N^2)$ |
| **Memory** | $\mathcal{O}(V \cdot N)$ | $\mathcal{O}(V^2 \cdot N)$ |

> AtomicDiff reduces both time and memory by a factor of $V$ by never computing cross-variable (mixed) derivatives.

---

### 6. Numerical Stability

**Theorem (Error Bound)**: For a function $f$ with normalized coefficients $a_n$, the relative error after $n$ convolution steps is bounded by:

$$\frac{|\tilde{a}_n - a_n|}{|a_n|} \leq \varepsilon \cdot n^2 \cdot \max_{k \leq n} \frac{|a_k|}{|a_n|}$$

where $\varepsilon \approx 2.22 \times 10^{-16}$ is machine epsilon.

---

### 7. Key Theorems

**Theorem 1 (Lane Independence)**: For any arithmetic expression involving independent variables, the Taylor coefficients satisfy:

$$\frac{\partial^n (f \cdot g)}{\partial x_i^n} = \sum_{k=0}^{n} \binom{n}{k} \frac{\partial^k f}{\partial x_i^k} \cdot \frac{\partial^{n-k} g}{\partial x_i^{n-k}}$$

In normalized form: $c_{i,n} = \sum_{k=0}^{n} a_{i,k} \cdot b_{i,n-k}$.

**Theorem 2 (No Mixed Derivatives)**: For any $i \neq j$, AtomicDiff never computes cross-variable derivatives, resulting in $\mathcal{O}(V \cdot N^2)$ time instead of $\mathcal{O}(V^2 \cdot N^2)$.

---

## 🚀 Quick Start

### Installation

AtomicDiff is header-only — just include and use:

```cpp
#include <AtomicDiff/derivatives.hpp>
using namespace AtomicDiff;
