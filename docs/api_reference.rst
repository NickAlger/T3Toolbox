API reference
=============

The library has two documented layers (see :doc:`user_guide` "Frontend and backend"):

- the **frontend** -- classes re-exported at the package root (``import t3toolbox``);
- the **backend** -- stateless functions on raw ``.data`` tuples, imported explicitly by submodule
  (``from t3toolbox.backend import probing``). The backend is a first-class surface: everything the
  frontend does is available here.

The frontend surface
--------------------

Everything below is importable directly from ``t3toolbox``.

**Tensors**

- :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` -- the keystone class (ragged);
  ``.data = (tucker_cores, tt_cores)``.
- :py:class:`~t3toolbox.uniform_tucker_tensor_train.UniformTuckerTensorTrain` -- the uniform
  (supercores + masks) mirror; its masks holder :py:class:`~t3toolbox.uniform_tucker_tensor_train.UT3Masks`
  is module-level (not at the root).

**Frames, variations, tangents**

- :py:class:`~t3toolbox.frame_variations_format.T3Frame`,
  :py:class:`~t3toolbox.frame_variations_format.T3Variations`,
  :py:func:`~t3toolbox.frame_variations_format.t3_orthogonal_representations` -- the orthogonal
  frame + gauged variations of a tangent direction;
  :py:func:`~t3toolbox.frame_variations_format.t3svd_orthogonal_representations` -- the same frame in
  the T3-SVD gauge together with its singular values (what a singular-value metric pairs with).
- :py:class:`~t3toolbox.manifold.T3Tangent` -- a tangent vector: one (frame, variations) pair.
- :py:class:`~t3toolbox.uniform_frame_variations_format.UT3Frame`,
  :py:class:`~t3toolbox.uniform_frame_variations_format.UT3Variations`,
  :py:func:`~t3toolbox.uniform_frame_variations_format.ut3_orthogonal_representations`,
  :py:class:`~t3toolbox.uniform_manifold.UT3Tangent` -- the uniform mirrors.
- Module-level (not at the root): the pair guards ``check_fv_pair`` / ``check_ufv_pair`` /
  ``check_fw_pair`` / ``check_ufw_pair``, and
  :py:func:`~t3toolbox.frame_variations_format.fv_to_t3` (substitute one variation core into the
  frame -- the single-term tangent word).

**Geometries**

- ``MANIFOLD`` / ``COREWISE`` (in :py:mod:`~t3toolbox.manifold`) -- the Hilbert-Schmidt manifold
  geometry and the Euclidean coordinate geometry.
- ``UNIFORM_MANIFOLD`` / ``UNIFORM_COREWISE`` (in :py:mod:`~t3toolbox.uniform_manifold`) -- their
  uniform twins.
- :py:func:`~t3toolbox.shared_geometry.shared` with the shorthands
  :py:func:`~t3toolbox.shared_geometry.shared_manifold` /
  :py:func:`~t3toolbox.shared_geometry.shared_corewise` -- wrap any of the four geometries to
  constrain the Tucker factors equal within groups of modes (:doc:`sharing`).
- Module-level (not at the root): the classes behind the singletons and ``shared(...)`` --
  :py:class:`~t3toolbox.manifold.ManifoldGeometry` / :py:class:`~t3toolbox.manifold.CorewiseGeometry`,
  :py:class:`~t3toolbox.uniform_manifold.UniformManifoldGeometry` /
  :py:class:`~t3toolbox.uniform_manifold.UniformCorewiseGeometry`,
  :py:class:`~t3toolbox.shared_geometry.SharedGeometry` -- and
  :py:func:`~t3toolbox.manifold.manifold_dim` (the tangent-space dimension).

**Weights**

- :py:class:`~t3toolbox.tucker_tensor_train.T3Weights` /
  :py:class:`~t3toolbox.uniform_tucker_tensor_train.UT3Weights` -- diagonal edge weights on a T3 *as a
  tensor*; :py:class:`~t3toolbox.frame_variations_format.T3FrameWeights` /
  :py:class:`~t3toolbox.uniform_frame_variations_format.UT3FrameWeights` -- a metric on a tangent's
  coordinates (:doc:`weighting`).
- The free functions ``t3_absorb_weights`` / ``ut3_absorb_weights`` / ``fv_absorb_weights`` /
  ``ufv_absorb_weights``, and ``t3_weighted_norm`` / ``t3_weighted_inner`` (+ the ``ut3_`` twins).

**Fitting and optimization**

- :py:class:`~t3toolbox.fitting.GaussNewtonModel` -- the geometry-generic local model
  (one class for both the ragged and the uniform representation).
- The six model factories (in :py:mod:`~t3toolbox.fitting`):
  :py:func:`~t3toolbox.fitting.entries_model`, :py:func:`~t3toolbox.fitting.apply_model`,
  :py:func:`~t3toolbox.fitting.probe_model`,
  :py:func:`~t3toolbox.fitting.entries_derivatives_model`,
  :py:func:`~t3toolbox.fitting.apply_derivatives_model`,
  :py:func:`~t3toolbox.fitting.probe_derivatives_model`.
- The four optimizers (in :py:mod:`~t3toolbox.optimizers`):
  :py:func:`~t3toolbox.optimizers.gradient_descent`, :py:func:`~t3toolbox.optimizers.mc_sgd`,
  :py:func:`~t3toolbox.optimizers.adam`, :py:func:`~t3toolbox.optimizers.newton_cg`.
  Ragged vs uniform is inferred from the initial guess ``x0``.
- ``Regularizer`` / ``IdentityRegularizer`` -- the optional objective term ``ρ(x)``
  (:doc:`fitting_and_optimization` §4.9). These are the one exception to the rule above: they are
  **not** re-exported at the package root, so import them from the module --
  ``import t3toolbox.optimizers as topt; topt.IdentityRegularizer(1e-3)`` (defined in
  :py:mod:`~t3toolbox.backend.regularization`).

**Safety**

- :py:mod:`~t3toolbox.safety` with the :py:func:`~t3toolbox.safety.safe` /
  :py:func:`~t3toolbox.safety.unsafe` context managers -- the ambient numerical-precondition mode.
  Module-level helpers: ``SafetyTolerances``, ``set_default_safety``, ``current_safety``,
  ``effective_rtol``, ``comparison_rtol``, ``checks_active``, ``require``, ``frames_equal``,
  ``frames_equal_or_skip``, ``is_tracing``, and the ``DEFAULT_RTOL_*`` constants.

The backend surface
-------------------

Backend modules follow the family-prefix grammar (``t3_`` ragged tensor, ``ut3_`` uniform tensor,
``tv_``/``utv_`` tangent-variations, ``fv_``/``ufv_`` frame-variations, ``tt_`` bare TT chains,
``dense_`` dense reference) -- the complete grammar, module map, and deliberate exceptions are in
:doc:`naming_conventions`. By area:

- **Core families**: :py:mod:`~t3toolbox.backend.t3_constructors`,
  :py:mod:`~t3toolbox.backend.t3_conversions`, :py:mod:`~t3toolbox.backend.t3_operations`,
  :py:mod:`~t3toolbox.backend.t3_linalg`, :py:mod:`~t3toolbox.backend.t3_orthogonalization`,
  :py:mod:`~t3toolbox.backend.t3_svd`, and the uniform mirrors
  (:py:mod:`~t3toolbox.backend.ut3_constructors`, :py:mod:`~t3toolbox.backend.ut3_conversions`,
  :py:mod:`~t3toolbox.backend.ut3_operations`, :py:mod:`~t3toolbox.backend.ut3_linalg`,
  :py:mod:`~t3toolbox.backend.ut3_orthogonalization`, :py:mod:`~t3toolbox.backend.ut3_svd`,
  :py:mod:`~t3toolbox.backend.ut3_masking`), plus :py:mod:`~t3toolbox.backend.tt_operations` and
  :py:mod:`~t3toolbox.backend.tt_orthogonalization` for bare TT chains.
- **Frames, variations, tangents**: :py:mod:`~t3toolbox.backend.fv_conversions`,
  :py:mod:`~t3toolbox.backend.fv_operations`, :py:mod:`~t3toolbox.backend.tv_operations`, and the
  uniform mirrors (:py:mod:`~t3toolbox.backend.ufv_conversions`,
  :py:mod:`~t3toolbox.backend.ufv_masking`, :py:mod:`~t3toolbox.backend.ufv_operations`,
  :py:mod:`~t3toolbox.backend.utv_operations`).
- **Sampling** (grouped by sampling type, the deliberate exception):
  :py:mod:`~t3toolbox.backend.probing`, :py:mod:`~t3toolbox.backend.apply`,
  :py:mod:`~t3toolbox.backend.entries`, :py:mod:`~t3toolbox.backend.sampling_derivatives`, and the
  uniform object-type wrappers :py:mod:`~t3toolbox.backend.ut3_sampling`,
  :py:mod:`~t3toolbox.backend.utv_sampling`.
- **Fitting and optimization**: :py:mod:`~t3toolbox.backend.fitting`,
  :py:mod:`~t3toolbox.backend.optimizers`, :py:mod:`~t3toolbox.backend.geometry` (the value-typed
  geometries the optimizers consume), :py:mod:`~t3toolbox.backend.uniform_fitting`,
  :py:mod:`~t3toolbox.backend.regularization` (the ``ρ(x)`` objective term) and
  :py:mod:`~t3toolbox.backend.optimizer_display` (the Newton-CG diagnostic display, so a backend
  user gets the identical output).
- **Shared Tucker factors**: :py:mod:`~t3toolbox.backend.sharing` -- partition validation, the
  tied-factor checkers, the shared-frame companion and the tied post-passes (:doc:`sharing`); the
  frontend geometry wrapper lives in :py:mod:`~t3toolbox.shared_geometry`.
- **Infrastructure**: :py:mod:`~t3toolbox.backend.common`, :py:mod:`~t3toolbox.corewise` (the
  layer-agnostic tree arithmetic on core tuples -- a backend-style module that lives at the top level),
  :py:func:`~t3toolbox.backend.contractions.contract` (the grouped-einsum interpreter --
  :doc:`grouped_contractions`), :py:mod:`~t3toolbox.backend.stacking`,
  :py:mod:`~t3toolbox.backend.linalg`, :py:mod:`~t3toolbox.backend.ranks`.

.. note::
	The rendered signatures on the module pages are regenerated from the parsed source, so each
	function page also shows the **verbatim source signature** -- including the trailing
	shape-contract comments, which are the real type contract (:doc:`contributor/signature_style`).

Complete module pages
---------------------

.. toctree::
   :maxdepth: 2

   autoapi/t3toolbox/index
