User guide
==========

Tucker tensor trains
--------------------

Tucker tensor trains are the composition of a `Tucker decomposition <https://en.wikipedia.org/wiki/Tucker_decomposition>`_
with a `tensor train <https://en.wikipedia.org/wiki/Matrix_product_state>`_ (also called matrix product states)
representation of the central Tucker core. (They are also known as **extended tensor trains**, ETT.)

Tensor network diagram for a Tucker tensor train::

        r0        r1        r2       r(d-1)          rd
    1 ------ G0 ------ G1 ------ ... ------ G(d-1) ------ 1
             |         |                    |
             | n0      | n1                 | nd
             |         |                    |
             B0        B1                   Bd
             |         |                    |
             | N0      | N1                 | Nd
             |         |                    |

Here:

- Gi and Bi are *cores*, which are small tensors that are being contracted with each other to form a large dense N0 x ... x N(d-1) tensor.
- Edges in the network indicate contraction of adjacent cores.
- Natural numbers Ni, ni, ri, written next to edges, indicate the size of the edge (its "bandwidth", you might say).

The components of a dth order Tucker tensor train are:

- Tucker cores: (B0, B1, ..., B(d-1)) with shapes (ni, Ni).
- TT cores: (G0, G1, ..., G(d-1)) with shapes (ri, ni, r(i+1)).

The structure of a Tucker tensor train is defined by:

- Tensor shape: (N0, N1, ..., N(d-1))
- Tucker ranks: (n0, r1, ..., n(d-1))
- TT ranks: (r0, r1, ..., rd)

Typically, the first and last TT-ranks satisfy r0=rd=1, and "1" in the diagram
is the number 1. However, it is allowed for these ranks to not be 1, in which case
the "1"s in the diagram are vectors of ones.

When the ranks of a Tucker tensor train are moderate, they can break the curse of dimensionality.
Whereas the memory required to store a dense tensor is O(N^d), the memory required to store a
Tucker tensor train is O(dnr^2 + dnN).

Unless specified otherwise, operations in this package are defined with respect
to the dense N0 x ... x N(d-1) tensors that are *represented* by the Tucker tensor train,
even though these dense tensors are not formed during computations.


Tucker tensor trains represent dense tensors
--------------------------------------------

Although the Tucker tensor train is defined by its cores, we always keep in mind that it *represents* a
dense N0 x ... x N(d-1) tensor. Tucker tensor train representations are not unique. In particular, you can
always insert a matrix times its inverse in the middle of an edge, then absorb the matrix into one of the
cores on the edge and absorb the inverse into the other. This changes the T3 representation, but does not change
the dense tensor being represented.

When we perform operations with Tucker tensor trains, like adding them, scaling them, taking inner products,
etc, we typically are simulating these operations on the represented dense tensors, using the cores merely as
a computational device. We are not performing the operations "corewise".

For example, consider the Tucker tensor trains:

	* x = ((A0,...,A(d-1)), (F0,...,F(d-1)))
	* y = ((B0,...,B(d-1)), (G0,...,G(d-1)))
	* z = ((C0,...,C(d-1)), (H0,...,H(d-1)))

We want::

	z "=" x "+" y

to mean the following: if you added the N0 x ... x N(d-1) tensor represented by x to the N0 x ... x N(d-1)
tensor represented by y, then the resulting N0 x ... x N(d-1) tensor can be represented by the Tucker tensor
train z. I.e.,::

	t3_to_dense(z) = t3_to_dense(x) + t3_to_dense(y).

We do not mean "add the cores". Generally,
Ci =/= Ai + Bi and Hi =/= Fi + Gi.

The operations that *are* corewise -- treating the cores as raw coordinates, with no dense-tensor
semantics -- are named with the ``corewise`` marker throughout the library (``corewise_dot``,
``sum_stack_corewise``, the ``COREWISE`` geometry, ...); see :doc:`naming_conventions`.


Minimal rank conditions
-----------------------

Tucker tensor trains are said to have *minimal ranks* if they satisfy:

- Left TT core unfoldings are full rank: r(i+1) <= (ri*ni)
- Right TT core unfoldings are full rank: ri <= (ni*r(i+1))
- Outer TT core unfoldings are full rank: ni <= (ri*r(i+1))
- Tucker cores have full row rank: ni <= Ni

Minimal rank properties:

- Minimal ranks always exist and are unique.
- Minimal Tucker ranks ni are equal to the ranks of Ni x (N1*...*N(i-1)*N(i+1)*...*N(d-1)) matricizations.
- Minimal TT ranks ri are equal to the ranks of (N*...*Ni) x (N(i+1)*...*N(d-1)) matrix unfoldings.
- Minimal rank representations of any T3 may be constructed with T3-SVD.

In this package, minimal ranks are typically defined with respect to a
generic Tucker tensor train of the given structure.
We do not account for possible additional rank deficiency due to
the numerical values within the cores.

Tucker tensor trains that do not have minimal ranks are degenerate, as they can always be reduced to
minimal rank Tucker tensor trains (without changing the represented tensor) using T3-SVD. This
degeneracy is analogous to a low rank matrix approximation A = B C, where A is NxM, B is Nxk,
C is kxM, and k>min(N,M).


T3 manifold
-----------

Under the minimal rank conditions, the set of Tucker tensor trains with fixed ranks forms an embedded
submanifold in R^(N1 x ... x Nd). In this case, any tangent vector may be represented as a sum which
looks like this::

	      1--H0--R1--R2--1   1--L0--H1--R2--1   1--L0--L1--H2--1
	         |   |   |          |   |   |          |   |   |
	v  =     U0  U1  U2    +    U0  U1  U2    +    U0  U1  U2
	         |   |   |          |   |   |          |   |   |

	      1--O0--R1--R2--1   1--L0--O1--R2--1   1--L0--L1--O2--1
	         |   |   |          |   |   |          |   |   |
	   +     V0  U1  U2    +    U0  V1  U2    +    U0  U1  V2
	         |   |   |          |   |   |          |   |   |

- The following "frame" cores are orthogonal representations of the base point where the tangent space
  is attached. When performing computations, these are computed once per tangent space, then fixed for
  all tangent vectors in the space:

  - tucker_cores      = (U0,...,Ud), orthogonal
  - left_tt_cores     = (L0,...Ld), left-orthogonal
  - right_tt_cores    = (R0,...,Rd), right-orthogonal
  - down_tt_cores     = (O0,...,Od), outer-orthogonal

- The following "variation" cores define the tangent vector w.r.t. the frame cores:

  - tucker_variations = (V0,...,Vd)
  - tt_variations     = (H0,...,Hd)

Under certain *gauge conditions*, the representation of a tangent vector by its variation is unique,
and performing linear algebra with the variation is equivalent to performing the corresponding linear
algebra operations with the dense tensor.

In the frontend these are the ``T3Frame`` / ``T3Variations`` classes
(:py:func:`~t3toolbox.frame_variations_format.t3_orthogonal_representations` builds them from a
``TuckerTensorTrain``), and a ``T3Tangent`` bundles the pair. The Hilbert-Schmidt metric lives on
the ``MANIFOLD`` geometry (``MANIFOLD.inner`` / ``norm`` / ``project`` / ``retract`` / ``transport``);
the Euclidean coordinate metric on the same tangent coordinates is the ``COREWISE`` geometry.
To optimize with the Tucker factors constrained equal within groups of modes, wrap either geometry
with ``shared(geometry, sharing)`` -- the shared-factor (SF-T3) submanifold, :doc:`sharing`.


The three sampling operations: entries, apply, probe
----------------------------------------------------

The library's three sampling operations evaluate the dense tensor without ever forming it. They
differ in how many modes stay free and in the test vectors used:

- ``entries`` -- contract with one-hot vectors in all modes: individual tensor entries.
- ``apply`` -- contract with general vectors in all modes: one scalar per sample.
- ``probe`` -- contract with vectors in all but one mode: ``d`` vectors, one per mode.

Containment: ``probe`` ⊃ ``apply`` ⊃ ``entries``. All three exist for tensors and for tangent
vectors (the Riemannian Jacobian 𝒥 and its transpose 𝒥ᵀ), and all three have symmetric
directional-derivative ("jet") generalizations with a leading order axis. What differs, what each
costs, and how each fits a Riemannian least-squares solve: :doc:`entries_apply_probe`.

Probing a N1 x N2 x N3 Tucker tensor with vectors u1, u2, u3 (lengths N1, N2, N3, respectively)
looks like this::

	                 1--G0--G1--G2--1
	                    |   |   |
	first probe:  =     B0  B1  B2
	(len=N1)            |   |   |
	                        u2  u3

	                 1--G0--G1--G2--1
	                    |   |   |
	second probe: =     B0  B1  B2
	(len=N2)            |   |   |
	                    u1      u3

	                 1--G0--G1--G2--1
	                    |   |   |
	third probe:  =     B0  B1  B2
	(len=N3)            |   |   |
	                    u1  u2


Batching and stacking
---------------------

A single ``TuckerTensorTrain`` / ``T3Frame`` / ``T3Variations`` can hold a whole **batch** of objects
at once. Every core is stored as ``core.shape == stack_shape + (tensor/rank axes)``, so operations
vectorize over the leading ``stack_shape``.

There are **three kinds of batch** ("blocks"), which batch different things on different arrays:

- ``C`` -- the **frame/core stack**: a batch of T3s / base points, on every core (this *is* ``stack_shape``).
- ``W`` -- the **probe stack**: a batch of probe-vector sets, on the probe vectors ``ww`` only.
- ``K`` -- the **tangent stack**: a batch of tangent vectors at one base point, on the variation cores only.

Axes are ordered **frame-inner**: ``W + K + C + (tensor axes)``. For example, with 2 base points
(``C``), 3 tangent vectors at each (``K``), probed by 4 probe-sets (``W``), every array's shape is::

    frame Tucker core  U_i   (T3Frame)        C            (2,)      + (n_i, N_i)
    variation core    dU_i  (T3Variations)   K + C        (3, 2)    + (n_i, N_i)
    probe vector      w_i   (ww)             W            (4,)      + (N_i,)
    forward probe     z_i   (tangent.probe)  W + K + C     (4, 3, 2) + (N_i,)

The frame stack ``C`` is *shared* across the ``K`` tangent vectors at it (never copied -- frame-inner
broadcasting handles that for free).

The **transposes** (``probe_transpose``, ``apply_transpose``, ``entries_transpose``) take a
``sum_over_probes`` flag: ``False`` (default) keeps ``W`` as an output stack -- one tangent/tensor per
probe -- while ``True`` sums ``W`` to give the Gauss-Newton back-projection ``Jᵀr`` used in
optimization. The two agree up to that sum (``True`` is ``Σ_W`` of ``False``).

Contractions over these blocks are written in one grouped-einsum dialect --
``contract('WCa,Caib,WCi->WCb', *operands)``, where an uppercase letter stands for a *group* of zero or
more axes whose size is solved from the operand shapes. It is the sole public entry point of
``backend.contractions`` and the machinery behind every sampling op; see :doc:`grouped_contractions`.

For the full design -- why frame-inner, the two contraction machineries, how the ``K``/``C`` split is
recovered, ``vmap``/``jit`` with the frame as a pytree leaf -- see :doc:`batching_and_stacking`
(start with its "Start here" on-ramp).


Uniform Tucker tensor trains
----------------------------

For computational efficiency, it can be helpful to pad the cores of a Tucker tensor train with zeros,
so that it has uniform ranks. Then the computational operations become uniform, which improves
pipelining efficiency and GPU performance -- in particular, everything can run under ``jax.lax.scan``
and compile once.

In this case:

- The Tucker cores B0, ..., Bd all have the same shape (n,N) and can be stacked into a *Tucker
  supercore* with shape (d,n,N).
- The TT-cores G0, ..., Gd all have the same shape (r,n,r), and can be stacked into a *TT supercore*
  with shape (d,r,n,r).

We keep track of which parts of the cores are supposed to be zero (to prevent filling in these parts
during computations) with *edge masks*. For each edge, the mask is a boolean vector of the form
(1,...,1,0,...,0).

- The masks for the edges between Tucker cores and TT-cores have length n. For the ith edge, the
  first ni entries are 1, and the remaining entries are 0. These *Tucker edge masks* are collected
  into a boolean array with shape (d, n).
- The masks for the edges between adjacent TT-cores have length r. For the ith edge, the first ri
  entries are 1, and the remaining entries are 0. These *TT edge masks* are collected into a boolean
  array with shape (d+1, r).

The governing principle is the **equivalence contract**: the uniform layer is a faster ragged
layer -- ``to_uniform -> op -> to_ragged`` equals the ragged op on the real (masked) content, and
the padding is don't-care garbage (:doc:`uniform_equivalence_contract`). The masks are boolean
prefix vectors rather than integer ranks so that rank metadata stays closed under addition and
multiplication (:doc:`uniform_masks_vs_ranks`), and they are deliberately **host numpy** (static
pytree structure) while the supercores flow through numpy or jax (:doc:`contributor/uniform_pytree_composition`).

.. note::
	The uniform **fitting** path requires a **minimal-rank** initial guess, and says so with an error
	if it does not get one. The reason is specific to this layer: the rank masks are loop-invariant,
	while a retraction from a non-minimal point drops the unrealizable rank -- which would desync the
	masks from the actual ranks mid-optimization. The frontend handles it transparently (it reduces
	``x0`` with ``uniform_minimal``); when driving the uniform backend yourself, reduce first via
	``ut3svd`` / ``uniform_minimal``. Any T3 can be reduced to minimal ranks with T3-SVD.

	This is a **representation** constraint of the packed layer, not a mathematical one. Nothing on
	the ragged layer requires minimal ranks: an orthonormal frame carries excess rank as slack and
	every tangent operation on it is exact (:doc:`numerical_contracts`, :doc:`frame_variations`).


NumPy and JAX
-------------

The library has **one** codebase and two array backends. There is no separate jax namespace and no
``use_jax`` flag on operations: **dispatch is inferred from the input array types** at the lowest
level. Pass numpy arrays and you get numpy computations and numpy outputs; pass jax arrays and the
same functions run on jax, with jax outputs. The inference propagates through computed
intermediates, so mixed pipelines behave predictably.

The one exception is **pure constructors with no array inputs** -- ``TuckerTensorTrain.randn`` /
``zeros`` / ``ones`` / ``load`` and their backend twins -- which take a ``use_jax`` flag, because
there is nothing to infer from. Factories that do take an existing object infer from it.

The frontend classes (``TuckerTensorTrain``, ``T3Frame``, ``T3Variations``, ``T3Tangent``, the
Gauss-Newton models) are registered jax **pytrees**, so ``jax.jit`` / ``grad`` / ``vmap`` apply to
frontend methods directly. Two design points make jit practical:

- A ``T3Tangent`` carries its frame as a pytree **leaf** -- the same-frame guard is a *numerical*
  check (``safety.frames_equal``), not object identity -- so the frame flows as traced data and
  changing base points does **not** trigger recompilation.
- A uniform object's masks are static ``aux_data`` with **value-based** hash/equality, so a
  rebuilt-but-identical mask holder is the *same* jit cache key: re-orthogonalizing the frame every
  optimization step does not recompile.

Safe-mode checks (below) are skipped automatically under a jax trace -- you cannot branch on a
tracer -- so jitted code needs no special handling.

.. warning::
	We do not recommend automatically differentiating *through* functions that involve singular
	value decompositions (SVDs), because support for `singular value sensitivity
	<https://en.wikipedia.org/wiki/Eigenvalue_perturbation>`_ in jax is questionable. This includes
	T3-SVD, orthogonalization (which uses SVDs for stability and robustness), and retraction. The
	intended pattern -- fixed frame, optimize over variations, retract *between* differentiated
	steps -- avoids this entirely.


Safe and unsafe mode
--------------------

The library separates two kinds of problem:

- **Structural problems always error**, in both modes: wrong shapes, inconsistent ranks or lengths,
  mismatched stack shapes. The frontend classes validate on construction.
- **Numerical preconditions are checked in safe mode** (the default) and skipped in unsafe mode or
  under a jax trace. An operation with a genuine numerical precondition -- its result is *wrong*
  without it -- checks it and raises; ``with t3toolbox.unsafe():`` turns the checks off for speed
  in a trusted inner loop. This is correctness-neutral (the ``assert`` precedent): the numbers are
  identical either way, only error-catching differs.

The enforced preconditions are concentrated on the manifold surface: **same tangent space** (a
numerical frames-equal test, for tangent arithmetic), **orthogonal frame** (projections, retraction,
transport), and **gauged variations** (the Hilbert-Schmidt ``MANIFOLD.inner``/``norm``). Plain
``TuckerTensorTrain`` arithmetic and the sampling operations are precondition-free (exact for any
cores). A property that only governs what a result *means* -- a caveat, not a precondition -- is
never enforced. The full statement of which operation requires what: :doc:`numerical_contracts`.


Fitting and optimization
------------------------

The fitting layer solves least-squares problems over the fixed-rank T3 manifold from sampled data:
given samples of an unknown tensor -- entries, applies, probes, or their symmetric derivatives --
find a T3 of the chosen ranks that matches them.

- ``GaussNewtonModel`` is the geometry-generic local model: gradient, Gauss-Newton
  Hessian-vector products, and retraction at a base point, over either the ``MANIFOLD`` or
  ``COREWISE`` geometry. One class serves both representations -- hand it a ragged
  ``TuckerTensorTrain`` and its gradient is a ``T3Tangent``, hand it a
  ``UniformTuckerTensorTrain`` and you get a ``UT3Tangent``.
- The six factories -- ``entries_model`` / ``apply_model`` / ``probe_model`` and their
  ``*_derivatives_model`` twins -- assemble the model from sample data.
- The four optimizers -- ``gradient_descent``, ``mc_sgd``, ``adam``, ``newton_cg`` -- drive it.

Ragged vs uniform is **inferred from the initial guess** ``x0``: pass a ``TuckerTensorTrain`` and
everything runs ragged; pass a ``UniformTuckerTensorTrain`` (with a uniform geometry singleton) and
the same calls run fully packed and jit-compile-once on the uniform layer. Worked end-to-end
examples live in ``examples/`` in the repository (``fit_hilbert_*.py``); the design rationale is
:doc:`fitting_and_optimization`.


Frontend and backend
--------------------

The library is a **thin object-oriented frontend over a pure-functional backend**:

- The **frontend** (``import t3toolbox``) is frozen dataclasses that hold only the cores;
  every operation is a method or cached property that delegates to backend functions and re-wraps
  the result. Start here.
- The **backend** (``from t3toolbox.backend import probing, tv_operations, ...``) is stateless
  functions operating on raw ``.data`` tuples. Everything the frontend does is available here, and
  an important minority of users work at this level exclusively -- the backend is a first-class,
  fully documented surface, organized by family-prefix grammar (:doc:`naming_conventions`).

Backend modules are imported explicitly by submodule; the package root deliberately re-exports only
the frontend. See the :doc:`api_reference` for the complete map of both layers.


.. _relevant-literature:

Relevant literature
-------------------

* Most of the Tucker tensor train algorithms are described in Appendix A of our paper [1].

* The probing algorithms are described in Section 5 of our paper [1].

* Under the name **extended tensor train**, the smooth manifold structure of this format is
  studied by Molozhavenko and Rakhuba [9] — the most direct published treatment of the Tucker
  tensor train manifold (there with shared Tucker factors). The shared-factor idea itself comes
  from the SF-Tucker decomposition of Peshekhonov, Arzhantsev, and Rakhuba [37]; this library
  generalizes it from one trailing shared block to an arbitrary partition of the modes
  (:doc:`sharing`).

* The algorithms here are extensions of standard tensor train algorithms (no Tucker):

  * For an introduction to tensor train methods, we highly recommend two extremely well written
    Ph.D. theses: Chapter 1 of Willem Hendrik Voorhaar's Ph.D. thesis [2], and Chapter 3 of
    Patrick Gelß's Ph.D. thesis [3].
  * The foundations behind these algorithms were established over the last 20 years:

    * Oseledets [4] defines the basics of tensor trains and TT-SVD.
    * In [5], Holtz, Rohwedder, and Schneider detail the manifold of fixed rank tensor trains and
      its tangent space and present gauged representations and oblique gauge projection, but only
      used left-orthogonal representations for tangent vector bases.
    * Using both left- and right-orthogonal representations was proposed in Khoromskij, Oseledets,
      and Schneider [6].
    * The modern manifold methods are given in Steinlechner [7] (developed at length in his
      Ph.D. thesis [10]).
    * Some of these methods were known earlier in the physics community, where tensor trains are
      known as "matrix product states" [11]; for introductions from that side see Orús [12] and
      Bridgeman and Chubb [13].

* The basics of the Tucker decomposition are described well in Kolda's review paper [8]; the
  original three-mode analysis is Tucker [14], and the higher-order SVD (the Tucker analog of
  TT-SVD) is De Lathauwer, De Moor, and Vandewalle [15].

* Riemannian optimization background: the standard text is Absil, Mahony, and Sepulchre [16],
  the modern one Boumal [17]; Uschmajew and Vandereycken [18] survey geometric methods on
  low-rank matrix and tensor manifolds — the setting of this library's manifold layer.

* Closely related formats and constructions: the two-level QTT-Tucker format (Dolgov and
  Khoromskij [19]); the hierarchical Tucker format (Hackbusch and Kühn [20]; Grasedyck [21]) and
  its geometry (Uschmajew and Vandereycken [22]); cross/sampling-based tensor train construction
  (Oseledets and Tyrtyshnikov [23]); and tensor train construction from tensor *actions* (our
  earlier work [24]) — the direct ancestor of the probing approach used here.

* Surveys of low-rank tensor methods: Bachmayr [25] (recent), Grasedyck, Kressner, and Tobler
  [26], and Cichocki et al. [27].

* Riemannian methods on low-rank manifolds, more specifically: the matrix case [28]; Riemannian
  tensor completion [29] and preconditioned Riemannian solvers [30]; automatic differentiation
  for Riemannian tensor-train optimization [31]; convergence theory for Riemannian Gauss-Newton
  [32]; second-order tensor-train optimization [33]; dynamical low-rank approximation [34], [35];
  and the alternating linear scheme [36], the classic core-by-core alternative to the
  whole-tensor Riemannian steps used here.

[1] Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026). "Tucker Tensor Train Taylor Series." arXiv preprint arXiv:2603.21141. `https://arxiv.org/abs/2603.21141 <https://arxiv.org/abs/2603.21141>`_

[2] Voorhaar, Willem Hendrik. "Tensor train approximations: Riemannian methods, randomized linear algebra and applications to machine learning." Ph.D. dissertation, Section de Mathématiques, Univ. Geneva, Geneva, Switzerland, 2022.

[3] Gelß, Patrick. "The tensor-train format and its applications: Modeling and analysis of chemical reaction networks, catalytic processes, fluid flows, and Brownian dynamics." Diss. 2017.

[4] Oseledets, Ivan V. "Tensor-train decomposition." SIAM Journal on Scientific Computing 33.5 (2011): 2295-2317.

[5] Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "On manifolds of tensors of fixed TT-rank." Numerische Mathematik 120.4 (2012): 701-731.

[6] Khoromskij, Boris N., Ivan V. Oseledets, and Reinhold Schneider. "Efficient time-stepping scheme for dynamics on TT-manifolds." (2012).

[7] Steinlechner, Michael. "Riemannian optimization for high-dimensional tensor completion." SIAM Journal on Scientific Computing 38.5 (2016): S461-S484. `https://epubs.siam.org/doi/10.1137/15M1010506 <https://epubs.siam.org/doi/10.1137/15M1010506>`_

[8] Kolda, Tamara G., and Brett W. Bader. "Tensor decompositions and applications." SIAM Review 51.3 (2009): 455-500. `https://epubs.siam.org/doi/10.1137/07070111X <https://epubs.siam.org/doi/10.1137/07070111X>`_

[9] Molozhavenko, A. and Rakhuba, M. "Optimization on the extended tensor-train manifold with shared factors." Computational and Applied Mathematics 45(6):221, 2026. `https://doi.org/10.1007/s40314-025-03605-0 <https://doi.org/10.1007/s40314-025-03605-0>`_

[10] Steinlechner, M. M. "Riemannian Optimization for Solving High-Dimensional Problems with Low-Rank Tensor Structure." PhD thesis, École polytechnique fédérale de Lausanne, 2016.

[11] Fannes, M., Nachtergaele, B., and Werner, R. F. "Finitely correlated states on quantum spin chains." Communications in Mathematical Physics 144(3):443-490, 1992.

[12] Orús, R. "A practical introduction to tensor networks: Matrix product states and projected entangled pair states." Annals of Physics 349:117-158, 2014.

[13] Bridgeman, J. C. and Chubb, C. T. "Hand-waving and interpretive dance: an introductory course on tensor networks." Journal of Physics A: Mathematical and Theoretical 50(22):223001, 2017.

[14] Tucker, L. R. "Some mathematical notes on three-mode factor analysis." Psychometrika 31(3):279-311, 1966.

[15] De Lathauwer, L., De Moor, B., and Vandewalle, J. "A multilinear singular value decomposition." SIAM Journal on Matrix Analysis and Applications 21(4):1253-1278, 2000.

[16] Absil, P.-A., Mahony, R., and Sepulchre, R. "Optimization Algorithms on Matrix Manifolds." Princeton University Press, 2008.

[17] Boumal, N. "An Introduction to Optimization on Smooth Manifolds." Cambridge University Press, 2023.

[18] Uschmajew, A. and Vandereycken, B. "Geometric methods on low-rank matrix and tensor manifolds." In Handbook of Variational Methods for Nonlinear Geometric Data, pages 261-313. Springer, 2020.

[19] Dolgov, S. and Khoromskij, B. "Two-level QTT-Tucker format for optimized tensor calculus." SIAM Journal on Matrix Analysis and Applications 34(2):593-623, 2013.

[20] Hackbusch, W. and Kühn, S. "A new scheme for the tensor representation." Journal of Fourier Analysis and Applications 15(5):706-722, 2009.

[21] Grasedyck, L. "Hierarchical singular value decomposition of tensors." SIAM Journal on Matrix Analysis and Applications 31(4):2029-2054, 2010.

[22] Uschmajew, A. and Vandereycken, B. "The geometry of algorithms using hierarchical tensors." Linear Algebra and its Applications 439(1):133-166, 2013.

[23] Oseledets, I. V. and Tyrtyshnikov, E. E. "Breaking the curse of dimensionality, or how to use SVD in many dimensions." SIAM Journal on Scientific Computing 31(5):3744-3759, 2009.

[24] Alger, N., Chen, P., and Ghattas, O. "Tensor Train Construction From Tensor Actions, With Application to Compression of Large High Order Derivative Tensors." SIAM Journal on Scientific Computing 42(5):A3516-A3539, 2020. `https://doi.org/10.1137/20M131936X <https://doi.org/10.1137/20M131936X>`_

[25] Bachmayr, M. "Low-rank tensor methods for partial differential equations." Acta Numerica 32:1-121, 2023.

[26] Grasedyck, L., Kressner, D., and Tobler, C. "A literature survey of low-rank tensor approximation techniques." GAMM-Mitteilungen 36(1):53-78, 2013.

[27] Cichocki, A., Lee, N., Oseledets, I., Phan, A.-H., Zhao, Q., Mandic, D. P., et al. "Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions." Foundations and Trends in Machine Learning 9(4-5):249-429, 2016.

[28] Vandereycken, B. "Low-rank matrix completion by Riemannian optimization." SIAM Journal on Optimization 23(2):1214-1236, 2013.

[29] Kressner, D., Steinlechner, M., and Vandereycken, B. "Low-rank tensor completion by Riemannian optimization." BIT Numerical Mathematics 54(2):447-468, 2014.

[30] Kressner, D., Steinlechner, M., and Vandereycken, B. "Preconditioned low-rank Riemannian optimization for linear systems with tensor product structure." SIAM Journal on Scientific Computing 38(4):A2018-A2044, 2016.

[31] Novikov, A., Rakhuba, M., and Oseledets, I. "Automatic differentiation for Riemannian optimization on low-rank matrix and tensor-train manifolds." SIAM Journal on Scientific Computing 44(2):A843-A869, 2022.

[32] Luo, Y. and Zhang, A. R. "Low-rank tensor estimation via Riemannian Gauss-Newton: Statistical optimality and second-order convergence." Journal of Machine Learning Research 24(381):1-48, 2023.

[33] Psenka, M. and Boumal, N. "Second-order optimization for tensors with fixed tensor-train rank." arXiv preprint arXiv:2011.13395, 2020.

[34] Koch, O. and Lubich, C. "Dynamical tensor approximation." SIAM Journal on Matrix Analysis and Applications 31(5):2360-2375, 2010.

[35] Lubich, C., Rohwedder, T., Schneider, R., and Vandereycken, B. "Dynamical approximation by hierarchical Tucker and tensor-train tensors." SIAM Journal on Matrix Analysis and Applications 34(2):470-494, 2013.

[36] Holtz, S., Rohwedder, T., and Schneider, R. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34(2):A683-A713, 2012.

[37] Peshekhonov, I., Arzhantsev, A., and Rakhuba, M. "Training a Tucker Model With Shared Factors: a Riemannian Optimization Approach." Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 238, 2024. `https://proceedings.mlr.press/v238/peshekhonov24a.html <https://proceedings.mlr.press/v238/peshekhonov24a.html>`_


Related software
----------------

Neighboring tensor libraries, covering the adjacent formats (TT, Tucker, hierarchical Tucker,
CP). None of them centers on the Tucker tensor train / extended tensor train format, though some
may offer partial functionality for it alongside their core format:

* `TT-Toolbox <https://github.com/oseledets/TT-Toolbox>`_ — the classic MATLAB tensor-train
  toolbox (Oseledets et al.): TT arithmetic, rounding, and cross approximation.
* `ttpy <https://github.com/oseledets/ttpy>`_ — its Python counterpart: the TT format with
  interpolation, linear solvers, and eigenproblems.
* `t3f <https://github.com/Bihaqo/t3f>`_ — tensor trains on TensorFlow, with extensive support
  for Riemannian optimization, GPU execution, and automatic differentiation.
* `tntorch <https://github.com/rballester/tntorch>`_ — PyTorch tensor-network learning: CP,
  Tucker, and TT decompositions behind one shared interface.
* `TensorLy <https://tensorly.org/stable/index.html>`_ — pure-Python tensor methods (CP, Tucker,
  TT, and more) with a swappable backend (NumPy, PyTorch, JAX, ...).
* `htucker <https://www.epfl.ch/labs/anchp/index-html/software/htucker/>`_ — the MATLAB toolbox
  for construction and manipulation of tensors in the hierarchical Tucker format (Kressner and
  Tobler).
