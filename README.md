# RegOT-CUDA

**RegOT-CUDA (cuRegOT)** is a CUDA-accelerated library for optimal transport computation, providing high-performance implementations of regularized optimal transport algorithms.

As a complement to this library, the [RegOT-Python](https://github.com/yixuan/regot-python) repository provides efficient CPU-based solvers for regularized optimal transport.

## Problem Formulation

Currently RegOT-CUDA solves the entropic-regularized optimal transport problem:

```math
\begin{align*}
\min_{T\in\mathbb{R}^{n\times m}}\quad & \langle T,M\rangle-\eta\cdot h(T),\\
\text{subject to}\quad & T\mathbf{1}_{m}=a,T^{T}\mathbf{1}_{n}=b,T\ge0,
\end{align*}
```

where $a\in\mathbb{R}^n$ and $b\in\mathbb{R}^m$ are two given
probability vectors with $a_i>0$, $b_j>0$, $\sum_{i=1}^n a_i=\sum_{j=1}^m b_j=1$,
and $M\in\mathbb{R}^{n\times m}$ is a given cost matrix.
The function $h(T)=\sum_{i=1}^{n}\sum_{j=1}^{m}T_{ij}(1-\log T_{ij})$ is the entropy term,
and $\eta>0$ is a regularization parameter.

## Work in Progress

RegOT-CUDA is a work in progress. Currently we have implemented the block coordinate descent algorithm (BCD, equilavent to the well-known Sinkhorn algorithm) and the sparse-plus-low-rank quasi-Newton method (SPLR) for entropic-regularized optimal transport. More state-of-the-art solvers are under development, and a list of candidate algorithms can be found in the [RegOT-Python](https://github.com/yixuan/regot-python) package.

## Requirements

- Python >= 3.11
- NumPy >= 1.23.0
- PyTorch >= 2.10 (optional, for PyTorch interface)
- CUDA Toolkit >= 13.0
- The [cuDSS](https://developer.nvidia.com/cudss) library
- Compatible NVIDIA GPU
- C++ compiler compatible with the installed PyTorch/CUDA toolchain (for source builds)

## Installation

The differentiable API below requires this checkout's Python files **and** its
matching native extensions. An index release named `curegot` is not evidence
that it contains the Stage 5/6A API.

Source builds require CUDA Toolkit (including `nvcc` on `PATH`), cuDSS headers
and libraries, and a compatible C++ toolchain. Set `CUDA_HOME` to the toolkit
prefix; the build searches its `include`, `lib64`, and `lib` directories for
CUDA/cuDSS. Runtime pip dependencies alone do not provide a complete development
toolchain. The torch build uses PyTorch's C++ extension toolchain (C++17 or
newer as required by the installed PyTorch).

In an already provisioned CUDA/cuDSS development environment, install the
build requirements and build from this repository:

```bash
python -m pip install 'setuptools>=80.0' 'pybind11~=3.0.0' 'requests~=2.32.0'
# Install CUDA-enabled PyTorch >=2.10 compatible with your toolkit first.
# CUDA_HOME must point to your CUDA development prefix, with cuDSS available.
python -m pip install --no-build-isolation -e .
```

`-e` uses the source Python package for editable development; omit `-e` for a
regular installation. Both source installation forms can compile native code.
CCCL 3.3.3 is downloaded by setup unless `CCCL_INCLUDE_DIR` points to its
existing `include` directory (or `cccl-v3.3.3/include` exists in the checkout).

PyTorch is **optional at runtime** for the NumPy interface. A NumPy-only source
build is possible without PyTorch using `--no-build-isolation`; adding PyTorch
later does not create the missing `_internal_torch` extension. The default
isolated build declares `torch>=2.10` as a build dependency even for NumPy users.
The differentiable interface needs both CUDA-enabled PyTorch and a successfully
built `_internal_torch`; verify the API explicitly because optional imports can
suppress extension loading errors.

See [the clean-install acceptance procedure](STAGE6A_CLEAN_INSTALL.md) for exact
commands to reuse this machine's dependencies, build and inspect a wheel, and
validate outside the repository. Installing an existing wheel does not compile
CUDA; building a wheel from source does. No native rebuild is needed merely to
use these documentation and validation changes with the existing editable install.

```bash
python -c "import curegot; print(curegot.__file__, curegot.__version__); print(curegot.torch.splr_ot)"
python validate_stage6a_release.py
```

## Differentiable PyTorch quick start

```python
import torch
import curegot

n, m, d = 64, 80, 8
g = torch.Generator(device="cpu").manual_seed(0)
X = torch.rand(n, d, generator=g, dtype=torch.float64).cuda()
Y = torch.rand(m, d, generator=g, dtype=torch.float64).cuda()
# Mean squared distance keeps the cost scale well-conditioned.
M = ((X[:, None, :] - Y[None, :, :]) ** 2).mean(-1).requires_grad_(True)
a = torch.full((n,), 1.0 / n, device="cuda", dtype=torch.float64)
b = torch.full((m,), 1.0 / m, device="cuda", dtype=torch.float64)

T = curegot.torch.splr_ot(M, a, b, eta=0.05, tol=1e-8)
loss = (T ** 2).sum()
loss.backward()
print(M.grad.norm())
```

For an `nn.Module` in a training pipeline:

```python
ot = curegot.torch.SPLROT(eta=0.05, tol=1e-8, backward_preconditioner="diag")
T = ot(M, a, b)
(T ** 2).sum().backward()
```

`SPLROT` holds configuration, with no parameters or buffers and an empty
`state_dict()`. Save its configuration separately. `train()`/`eval()` do not
change solver behavior; move inputs to CUDA yourself. Gradients can propagate
through the cost matrix to upstream embeddings or network parameters. See the
[complete training example](examples/differentiable_splr_ot.py):

```bash
python examples/differentiable_splr_ot.py
```

### Public API and support contract

All names below are under `curegot.torch`; existing APIs remain available.

| Level | API | Purpose |
|---|---|---|
| Recommended | `splr_ot` | Differentiable transport plan |
| Recommended | `SPLROT` | Module wrapper |
| Recommended | `splr_ot_with_info` | Detached plan, dual, iteration count and convergence flag; **not differentiable** |
| Recommended | `splr_ot_batch` | Sequential Python batching of `[B,n,m]` costs and `[B,n]` / `[B,m]` marginals |
| Advanced | `solve_backward`, `sinkhorn_splr_bwd`, `SPLRBackwardContext` | Manual implicit backward and native context |
| Legacy native | `sinkhorn_splr`, `sinkhorn_bcd` | Native forward solvers |

| Gradient | Supported |
|---|---|
| `dL/dM` | Yes |
| `dL/da` | No |
| `dL/db` | No |
| `dL/deta` | No; pass eta as a constant Python scalar |

The differentiable API requires **CUDA float64 only**, with `M`, `a`, and `b`
on the same device. Marginals must be positive probability vectors with equal
unit mass; `eta` must be positive. CPU/float32 inputs raise `RuntimeError`,
`a`/`b` requiring gradients raise `NotImplementedError`, and invalid shapes or
scalar settings raise `ValueError`. Forward or backward non-convergence raises
`RuntimeError` in `splr_ot`; debug/manual APIs expose convergence flags that
callers must check.

Current limitations: one native problem per call; `splr_ot_batch` is sequential
Python batching; no AMP/autocast; no double backward; `torch.compile` is not
guaranteed and may graph-break; TorchScript is unsupported. These restrictions
apply to the differentiable API; legacy native conversion behavior is unchanged.

## Usage

NumPy interface:

```python
import numpy as np
import curegot

# Create data
np.random.seed(123)
n, m = 100, 80
M = np.random.rand(n, m)  # Cost matrix
a = np.random.rand(n)     # Source distribution
a = a / np.sum(a)         # Normalize
b = np.random.rand(m)     # Target distribution
b = b / np.sum(b)         # Normalize
reg = 0.1                 # Regularization parameter

# Call algorithm
result1 = curegot.numpy.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan1 = result1["plan"]
print(plan1[:3, :3])

result2 = curegot.numpy.sinkhorn_splr(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan2 = result2["plan"]
print(plan2[:3, :3])
```

PyTorch interface:

```python
import torch
import curegot

# Create data
torch.manual_seed(123)
n, m = 100, 80
device = "cuda" if torch.cuda.is_available() else "cpu"
M = torch.rand(n, m, device=device)  # Cost matrix
a = torch.rand(n, device=device)     # Source distribution
a = a / torch.sum(a)                 # Normalize
b = torch.rand(m, device=device)     # Target distribution
b = b / torch.sum(b)                 # Normalize
reg = 0.1                            # Regularization parameter

# Call algorithm
result1 = curegot.torch.sinkhorn_bcd(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan1 = result1["plan"]
print(plan1[:3, :3])

result2 = curegot.torch.sinkhorn_splr(M, a, b, reg, tol=1e-6, max_iter=1000, verbose=0)
plan2 = result2["plan"]
print(plan2[:3, :3])
```

## Tests

```bash
cd regot-cuda/test
pip install regot
python test_sinkhorn_bcd.py
python test_sinkhorn_splr.py
```

## Benchmark

The plot below shows the benchmark result for some popular GPU-based solvers. The horizontal axis represents the elapsed wall time, and the vertical axis is the optimization error on a logarithmic scale. Lower curves indicate better performance, achieving lower errors in less time.

![](figs/benchmark.png)

More details on the benchmark can be found in [the paper](https://arxiv.org/pdf/2605.08793).

## Bibliography

Please consider to cite our work if you find our algorithms or
software useful in your research and applications.

```bibtex
@inproceedings{qiu2026curegot,
  title={{{cuRegOT}}: A {{GPU}}-Accelerated Solver for Entropic-Regularized Optimal Transport},
  author={Qiu, Yixuan},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
