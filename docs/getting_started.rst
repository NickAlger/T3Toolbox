Getting started
===============

This page is a tour of the frontend in five short examples. Everything here is runnable as shown
(the outputs are real); the :doc:`user_guide` explains the concepts, and the :doc:`api_reference`
maps the full surface.

Create and manipulate
---------------------

A :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` is defined by its tensor shape,
Tucker ranks, and TT ranks. Arithmetic is defined with respect to the **represented dense tensor**
(adding two T3s concatenates ranks; it does not add cores), and ``t3svd`` reduces a representation
back to minimal ranks::

	>>> import numpy as np
	>>> import t3toolbox as t3t
	>>> np.random.seed(0)
	>>> x = t3t.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
	>>> print(x.shape, x.tucker_ranks, x.tt_ranks)
	(10, 11, 12) (3, 4, 3) (1, 2, 2, 1)
	>>> y = t3t.TuckerTensorTrain.randn((10, 11, 12), (2, 2, 2), (1, 2, 2, 1))
	>>> z = x + y
	>>> print(z.tucker_ranks, z.tt_ranks)                    # ranks add ...
	(5, 6, 5) (2, 4, 4, 2)
	>>> print(float(np.linalg.norm(z.to_dense() - (x.to_dense() + y.to_dense()))))  # ... tensors add
	0.0
	>>> z2, ss_tucker, ss_tt = z.t3svd()                     # reduce to minimal ranks (lossless)
	>>> print(z2.tucker_ranks, z2.tt_ranks)
	(4, 6, 4) (1, 4, 4, 1)
	>>> print(bool(np.allclose(z2.to_dense(), z.to_dense())))
	True

Sample without densifying
-------------------------

The three sampling operations -- ``entries`` / ``apply`` / ``probe`` -- evaluate the represented
tensor without forming it (see the user guide). ``probe`` contracts with vectors in all but one
mode and returns ``d`` vectors; ``apply`` contracts every mode to a scalar; ``entries`` evaluates
individual entries::

	>>> import numpy as np
	>>> import t3toolbox as t3t
	>>> np.random.seed(0)
	>>> x = t3t.TuckerTensorTrain.randn((10, 11, 12), (3, 4, 3), (1, 2, 2, 1))
	>>> ww = [np.random.randn(N) for N in x.shape]
	>>> zz = x.probe(ww)                                     # d vectors, one per mode
	>>> print([z.shape for z in zz])
	[(10,), (11,), (12,)]
	>>> x_dense = x.to_dense()                               # dense ground truth, for checking only
	>>> z0 = np.einsum('ijk,j,k->i', x_dense, ww[1], ww[2])
	>>> print(bool(np.linalg.norm(zz[0] - z0) < 1e-10))
	True
	>>> a = x.apply(ww)                                      # all modes contracted: a scalar
	>>> print(bool(np.isclose(a, np.einsum('ijk,i,j,k->', x_dense, *ww))))
	True
	>>> e = x.entries(np.array([3, 1, 2]))                   # one entry
	>>> print(bool(np.isclose(e, x_dense[3, 1, 2])))
	True

Tangents and retraction
-----------------------

At a minimal-rank point, the fixed-rank T3s form a manifold.
:py:func:`~t3toolbox.frame_variations_format.t3_orthogonal_representations` factors a point into an
orthogonal frame (``T3Frame``) plus gauged variations; a ``T3Tangent`` is one (frame, variations)
pair, and the ``MANIFOLD`` geometry provides the Hilbert-Schmidt metric, projections, and
retraction::

	>>> import numpy as np
	>>> import t3toolbox as t3t
	>>> np.random.seed(0)
	>>> x = t3t.TuckerTensorTrain.randn((10, 11, 12), (3, 6, 2), (1, 3, 2, 1))
	>>> print(x.has_minimal_ranks)                           # a manifold point
	True
	>>> frame, variations = t3t.t3_orthogonal_representations(x)
	>>> print(type(frame).__name__, type(variations).__name__)
	T3Frame T3Variations
	>>> v = t3t.MANIFOLD.randn(frame)                        # a random tangent vector at x
	>>> print(type(v).__name__, bool(t3t.MANIFOLD.norm(v) > 0))
	T3Tangent True
	>>> x_new = t3t.MANIFOLD.retract(0.1 * v)                # step along v, back onto the manifold
	>>> print(type(x_new).__name__)
	TuckerTensorTrain
	>>> print(x_new.tucker_ranks == x.tucker_ranks, x_new.tt_ranks == x.tt_ranks)
	True True

Fit a tensor from measurements
------------------------------

The fitting layer recovers a T3 from sampled measurements. Here we observe 120 noiseless
``apply`` samples of a low-rank target and recover it by Riemannian Newton-CG -- a zero start is
fine on the manifold, and unit-norm probe rows keep the least-squares well-conditioned::

	>>> import numpy as np
	>>> import t3toolbox as t3t
	>>> np.random.seed(0)
	>>> A = t3t.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))   # the unknown target
	>>> ww = [np.random.randn(120, N) for N in A.shape]                       # 120 probe rows
	>>> ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]       # unit-norm rows
	>>> b = A.apply(ww)                                                       # the measurements
	>>> print(b.shape)
	(120,)
	>>> x0 = t3t.TuckerTensorTrain.zeros((6, 7, 8), (2, 2, 2), (1, 2, 2, 1))  # zero start
	>>> x_fit, stats = t3t.newton_cg(t3t.MANIFOLD, 'apply', ww, b, x0, max_newton=30)
	>>> rel_err = np.linalg.norm(x_fit.to_dense() - A.to_dense()) / np.linalg.norm(A.to_dense())
	>>> print(bool(rel_err < 1e-6))
	True

The same call runs **fully packed and jit-compiled on the uniform layer** if you pass a
``UniformTuckerTensorTrain`` start and the ``UNIFORM_MANIFOLD`` geometry -- ragged vs uniform is
inferred from ``x0``. End-to-end examples (noise, validation, rank continuation) are in the
repository's `examples/ <https://github.com/NickAlger/T3Toolbox/tree/main/examples>`_ directory.

NumPy or JAX
------------

Dispatch is inferred from the input array types -- the same functions run on either backend, and
only no-input constructors take a ``use_jax`` flag. Frontend objects are jax pytrees, so
``jax.jit`` applies to them directly (requires ``jax`` installed)::

	>>> import numpy as np
	>>> import jax
	>>> import jax.numpy as jnp
	>>> import t3toolbox as t3t
	>>> from t3toolbox.backend.common import is_jax_ndarray
	>>> np.random.seed(0)
	>>> x_np  = t3t.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1))
	>>> x_jax = t3t.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 2), (1, 2, 2, 1), use_jax=True)
	>>> print(x_np.contains_jax, x_jax.contains_jax)
	False True
	>>> ww = [np.random.randn(N) for N in (5, 6, 7)]
	>>> print(is_jax_ndarray(x_np.probe(ww)[0]), is_jax_ndarray(x_jax.probe(ww)[0]))
	False True
	>>> apply_jit = jax.jit(lambda t, w: t.apply(w))         # a frontend object as a jit argument
	>>> ww_jax = [jnp.array(w) for w in ww]
	>>> print(bool(jnp.allclose(apply_jit(x_jax, ww_jax), x_jax.apply(ww_jax))))
	True

See the user guide's "NumPy and JAX" section for the design (pytree leaves, compile-once masks,
the SVD autodiff caveat).
