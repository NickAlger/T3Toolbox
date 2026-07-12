WORK IN PROGRESS DO NOT USE

# T3Toolbox

A pure-Python (NumPy + optional JAX) library for **Tucker tensor trains (T3)** — a Tucker
decomposition whose central core is stored as a tensor train. When the ranks are moderate, a T3
breaks the curse of dimensionality: storing a dense tensor costs O(N^d) memory, while the T3
representing it costs O(dnr^2 + dnN).

Tensor network diagram for a Tucker tensor train:

        r0        r1        r2       r(d-1)          rd
    1 ------ G0 ------ G1 ------ ... ------ G(d-1) ------ 1
             |         |                    |
             | n0      | n1                 | nd
             |         |                    |
             B0        B1                   Bd
             |         |                    |
             | N0      | N1                 | Nd
             |         |                    |

Here the Gi (TT cores) and Bi (Tucker cores) are small tensors contracted along the edges to form
a large dense N0 x ... x N(d-1) tensor. Unless stated otherwise, operations in this package are
defined with respect to the dense tensor that the T3 *represents*, even though that dense tensor
is never formed.

## Installation

The package is pure Python. Dependencies: [`numpy`](https://numpy.org/install/) (required),
[`jax`](https://docs.jax.dev/en/latest/installation.html) (optional).

Install from source:

	git clone https://github.com/NickAlger/T3Toolbox.git
	cd T3Toolbox
	pip install .

## Quickstart

Create T3s and operate on the tensors they represent (arithmetic is dense-tensor arithmetic —
adding two T3s concatenates ranks; T3-SVD reduces back to minimal ranks):

```python
import numpy as np
import t3toolbox as t3t

np.random.seed(0)
x = t3t.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
y = t3t.TuckerTensorTrain.randn((10, 11, 12), (2, 2, 2), (1, 2, 2, 1))

z = x + y
print(z.tucker_ranks, z.tt_ranks)      # (5, 6, 5) (2, 4, 4, 2)   <- ranks add ...
np.linalg.norm(z.to_dense() - (x.to_dense() + y.to_dense()))  # ... tensors add: 0.0

z2, ss_tucker, ss_tt = z.t3svd()       # reduce to minimal ranks (lossless)
print(z2.tucker_ranks, z2.tt_ranks)    # (4, 6, 4) (1, 4, 4, 1)
```

Sample the represented tensor without forming it (`entries` / `apply` / `probe`, each also
available for tangent vectors and with symmetric-derivative generalizations):

```python
ww = [np.random.randn(N) for N in x.shape]
zz = x.probe(ww)                       # d vectors, one per mode (all but one mode contracted)
a  = x.apply(ww)                       # a scalar (all modes contracted)
e  = x.entries(np.array([3, 1, 2]))    # one entry
```

Fit a fixed-rank T3 to sampled measurements by Riemannian optimization (a zero start on the
manifold; unit-norm probe rows keep the least-squares well-conditioned):

```python
A  = t3t.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))     # the unknown target
ww = [np.random.randn(120, N) for N in A.shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]          # unit-norm rows
b  = A.apply(ww)                                                         # 120 measurements

x0 = t3t.TuckerTensorTrain.zeros((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))
x_fit, stats = t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)
# relative error of the recovery: < 1e-6
```

Everything runs on NumPy or JAX — dispatch is inferred from the input array types, frontend
objects are jax pytrees (`jax.jit` applies to them directly), and passing a
`UniformTuckerTensorTrain` start runs the same fitting calls fully packed and jit-compile-once on
the uniform layer. See [Getting started](https://nickalger.github.io/T3Toolbox/getting_started.html)
for the full tour with verified outputs.

## Included functionality

- The **T3 format**: arithmetic with dense-tensor semantics, orthogonalization, minimal ranks,
  **T3-SVD** (truncation / minimal-rank reduction), save/load, batching (`stack_shape`) on every
  operation.
- The three **sampling operations** — `entries`, `apply`, `probe` — evaluating the represented
  tensor without forming it, plus their **symmetric directional derivatives** (jets) and the
  ambient/corewise/tangent **transposes**.
- The **fixed-rank T3 manifold**: orthogonal frame + gauged variations (`T3Frame`,
  `T3Variations`, `T3Tangent`), gauge projections, retraction, and two geometries — the
  Hilbert-Schmidt `MANIFOLD` and the Euclidean-coordinate `COREWISE`.
- **Least-squares fitting** from any of the sampling operations or their derivatives
  (Gauss-Newton models) with four optimizers: `gradient_descent`, `mc_sgd`, `adam`, `newton_cg`.
- The **uniform layer**: zero-padded supercores + boolean rank masks mirroring the whole stack —
  same results, uniform shapes — for `jax.lax.scan` vectorization, GPU efficiency, and
  compile-once `jit` (optimizers included).
- **NumPy / JAX** backends with dispatch inferred from the input arrays; frontend classes are
  registered jax pytrees.
- **Safe mode**: numerical preconditions (same tangent space, orthogonal frame, gauged
  variations) checked by default, skippable for speed (`t3toolbox.unsafe()`); structural
  problems always error.

## Documentation

https://nickalger.github.io/T3Toolbox/ — [getting started](https://nickalger.github.io/T3Toolbox/getting_started.html),
[user guide](https://nickalger.github.io/T3Toolbox/user_guide.html),
[design notes](https://nickalger.github.io/T3Toolbox/design_notes.html), and the full
[API reference](https://nickalger.github.io/T3Toolbox/api_reference.html) (frontend **and**
backend — the backend is a first-class, fully documented surface). Worked end-to-end fitting
examples live in [`examples/`](examples/).

## Authors

* Nick Alger (nalger225@gmail.com)
* Blake Christierson (bechristierson@utexas.edu)

MIT License. The algorithms are described in *Alger, Christierson, Chen & Ghattas (2026), "Tucker
Tensor Train Taylor Series"*, [arXiv:2603.21141](https://arxiv.org/abs/2603.21141).
