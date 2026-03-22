<a href="https://github.com/tuusuario/atomicdiff" target="_blank">
    <img src="art/atomicdiff-header.svg" width="100%">
</a>

![Linux build status](https://github.com/tuusuario/atomicdiff/workflows/linux/badge.svg?branch=main)
![macOS build status](https://github.com/tuusuario/atomicdiff/workflows/osx/badge.svg?branch=main)
![Windows build status](https://github.com/tuusuario/atomicdiff/workflows/windows/badge.svg?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Header-only](https://img.shields.io/badge/Header--only-brightgreen.svg)](https://github.com/tuusuario/atomicdiff)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/tuusuario/atomicdiff)
[![Code Size](https://img.shields.io/github/languages/code-size/tuusuario/atomicdiff)](https://github.com/tuusuario/atomicdiff)
[![Stars](https://img.shields.io/github/stars/tuusuario/atomicdiff)](https://github.com/tuusuario/atomicdiff)

# AtomicDiff

**High-Order Automatic Differentiation with Taylor Lanes & Lock-Free Atomic Accumulation**

AtomicDiff is a modern C++20 header-only library that implements high-order automatic differentiation (AD) using independent Taylor lanes. Unlike traditional AD libraries that compute unnecessary mixed derivatives ($\mathcal{O}(V^2)$ complexity), AtomicDiff maintains each variable's derivatives in separate, independent lanes, achieving $\mathcal{O}(V \cdot N)$ complexity where $V$ is the number of variables and $N$ is the derivative order.

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

```math
\mathcal{L}_i^N[f] = \left( f, \frac{\partial f}{\partial x_i}, \frac{1}{2!}\frac{\partial^2 f}{\partial x_i^2}, \ldots, \frac{1}{N!}\frac{\partial^N f}{\partial x_i^N} \right)
