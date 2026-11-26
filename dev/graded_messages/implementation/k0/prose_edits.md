# K0: the library prose that is rewritten, before and after

Implementer, 2026-11-25, for `K0_DESIGN_NOTE.md` §1.3 (review r1, S2) and §2.2 (ruling on M2). Line numbers are at
`b5125f76`. The **deletions whole** stay as line ranges in the note (§1.3); here are the texts K0 **rewrites**, each
describing the code as it is after K0, nothing about K1 to K7. Each "before" is quoted from the file; signature
comments and code are not touched by any of these edits.

Reviewed and left as they are (true of the code after K0): `ut3_sampling.py:68-69` (the `.data` layout comment);
the docstrings of the kept `ut3_*` and `utv_*` functions (they describe one engine call each, and the masking the
engine does); `ut3_entries_corewise_hessian`'s "the one-hot vectors are built packed inside the kernel" (the engine
builds them, `w_one_hot_step`; "kernel" is the engine here); `uniform_fitting.py:366-374` (the packing of a batch);
`fitting.py:751-755` (`_KINDS`, `_HOLD_LIST`, `_SCALED` and their comments); `probing.py:77-80` (the note on
weighting). **`probing`'s "Reference module for the signature style" stays** (review S2): its kept functions carry
the house form (`t3_probe`, `tv_probe_transpose`, `dense_probe`), and `docs/contributor/signature_style.md` names the
module; the pages' quotations of `compute_mu` are P19's (note §2.2).

## 1. Module docstrings

### `t3toolbox/backend/probing.py:5-13`

Before:

    """The ``probe`` sampling type AND the shared sampling machinery it exemplifies.

    Probing contracts a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6). This
    module holds the general machinery -- the ``xi/mu/nu/eta`` helper chain, ``assemble_z``, the tv
    probe + frame sweeps + transposes, and the corewise transpose's derivative in the cores (the exact Hessian's
    two-hole) -- which ``apply``/``entries`` specialize (probe ⊃ apply ⊃
    entries; they import from here, never the reverse). Reference module for the signature style
    (``docs/contributor/signature_style.md``).
    """

After:

    """The ``probe`` sampling type: contract a T3 with vectors in all but one mode, giving ``d`` vectors (T4S §6).

    The probe of a point and of a tangent, their transposes (ambient, corewise, tangent), the corewise forward
    derivative and the corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call
    of the engine :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`; and the dense reference
    :py:func:`dense_probe`, whose dense jets ``apply`` builds on (probe ⊃ apply ⊃ entries). Reference module
    for the signature style (``docs/contributor/signature_style.md``).
    """

### `t3toolbox/backend/apply.py:5-12`

Before:

    """The ``apply`` sampling type: contract a T3 with vectors in ALL modes -> one scalar per sample.

    Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise transpose's derivative in the
    cores (the exact Hessian's sampling term), and the frame sweeps for applies.
    Specializes the general probing machinery (containment probe ⊃ apply ⊃ entries; this module
    imports from ``probing``, never the reverse). Costs and the role in Riemannian least-squares:
    ``docs/entries_apply_probe.md``.
    """

After:

    """The ``apply`` sampling type: contract a T3 with vectors in ALL modes -> one scalar per sample.

    Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise forward derivative and the
    corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call of the engine,
    and the dense reference :py:func:`dense_apply` (containment probe ⊃ apply ⊃ entries; this module imports from
    ``probing``, never the reverse). Costs and the role in Riemannian least-squares:
    ``docs/entries_apply_probe.md``.
    """

### `t3toolbox/backend/entries.py:5-15`

Before (the first paragraph; the second, on index semantics, stays):

    """The ``entries`` sampling type: evaluate individual T3 entries (one-hot probes) -> scalars.

    Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise transpose's derivative in the
    cores (the exact Hessian's sampling term), and the frame sweeps for entry sampling. The most-special case of
    probe ⊃ apply ⊃ entries; imports the general machinery from ``probing``, never the reverse. See
    ``docs/entries_apply_probe.md``.

After (first paragraph):

    """The ``entries`` sampling type: evaluate individual T3 entries (one-hot probes) -> scalars.

    Holds the t3 + tv ops, the ambient/tangent/corewise transposes, the corewise forward derivative and the
    corewise Hessian (the exact Hessian's sampling term), each but the ambient transpose one call of the engine,
    and the dense reference :py:func:`dense_entries`. The most-special case of probe ⊃ apply ⊃ entries; imports
    from ``apply`` (its ambient transpose), never the reverse. See ``docs/entries_apply_probe.md``.

### `t3toolbox/backend/ut3_sampling.py:5-19`

Before:

    """Uniform plain-layer sampling wrappers: probe/apply/entries on ut3 data (+ transposes, jets).

    Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules: each
    function here is a thin wrapper -- mask once, pack, delegate to the shared polymorphic ragged
    machinery (the corewise transposes and the two-holes then re-mask their output) -- so it
    lives with the object it wraps. The algorithm story is in
    ``probing``/``apply``/``entries``/``sampling_derivatives`` (``docs/naming_conventions.md``).

    The twins of the ragged sampling functions (``ut3_probe``, ``ut3_apply``, ``ut3_entries`` and their
    ``*_corewise_tangent``, ``*_corewise_transpose``, ``*_corewise_hessian``; graded messages P12) take the ragged
    functions' signatures, with ``x`` the uniform point's ``.data``, and are one call of the engine
    (:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`), which packs a ragged sample, masks every
    operand on entry and every output on exit. A probe output mirrors the sample's form: ragged in, ragged out; packed
    in, packed out. The jet wrappers (``*_derivatives*``) still share ``sampling_derivatives``.
    """

After:

    """Uniform plain-layer sampling: probe/apply/entries on ut3 data, their corewise derivative, transpose and
    Hessian, and the full sum.

    Grouped by OBJECT type, deliberately asymmetric with the ragged by-sampling-type modules
    (``docs/naming_conventions.md``). The twins of the ragged sampling functions (``ut3_probe``, ``ut3_apply``,
    ``ut3_entries`` and their ``*_corewise_tangent``, ``*_corewise_transpose``, ``*_corewise_hessian``) take the
    ragged functions' signatures, with ``x`` the uniform point's ``.data``, and are one call of the engine
    (:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`), which packs a ragged sample, masks every
    operand on entry and every output on exit. A probe output mirrors the sample's form: ragged in, ragged out;
    packed in, packed out.
    """

### `t3toolbox/backend/utv_sampling.py:5-28`

Before:

    """Uniform tangent sampling (UT3Tangent probe / apply / entries).

    The uniform mirror of the ragged ``T3Tangent`` sampling frontend. **Since graded messages P12a** the plain
    (jet-free) functions ``utv_{probe,apply,entries}`` and their transposes are one call of the engine
    (:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`) on the frame's six elements: the engine masks
    the frame and **the variations with the frame's masks** (the variations' own masks are not read; a uniform tangent's
    variation masks are the frame's gauge masks), packs the sample, and masks every output. They take the ragged
    ``tv_*`` signatures (P12b), and ``utv_probe`` mirrors the sample's form. What follows describes the jet functions and the ``*_from_sweep`` seam, which still share the ragged
    machinery. The boundary work is the same as the plain layer's
    ``ut3_sampling`` (the precedent), lifted to the frame-variations pair:

    1. **mask-once** the frame (4) and variation (2) supercores -- so the padded rank slots are zero and the
       core algorithm runs mask-free (the padded bonds contract to zero);
    2. **pack** the probe vectors into one supercore-shaped tensor (``apply``/``probe``; ``entries`` slices
       fibers, so its integer ``index`` needs no packing);
    3. call the polymorphic backend (``is_uniform`` is inferred from the masked supercores being bare ndarrays);
    4. **unpack** the ``probe`` output back to ragged-width vectors (``apply``/``entries`` return a scalar --
       nothing to unpack).

    These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``); the transpose ``𝒥ᵀ`` is the last section of this module.
    The transposes **re-mask their output** with the variation masks they attach, so the padded slots come back as
    exact zeros (one multiply by a host mask; the real block is unchanged bit for bit). The ``*_transpose_from_sweep``
    seam halves stay bare: their callers, the uniform fitting kinds, re-mask.
    """

After:

    """Uniform tangent sampling (UT3Tangent probe / apply / entries).

    The uniform mirror of the ragged ``T3Tangent`` sampling frontend. The functions ``utv_{probe,apply,entries}`` and
    their transposes take the ragged ``tv_*`` signatures and are one call of the engine
    (:py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`) on the frame's six elements: the engine masks
    the frame and **the variations with the frame's masks** (the variations' own masks are not read; a uniform
    tangent's variation masks are the frame's gauge masks), packs the sample, and masks every output; ``utv_probe``
    mirrors the sample's form.

    These are the bare Jacobian ``𝒥`` (no gauge projector ``Π``) and its transpose ``𝒥ᵀ``. The transposes attach the
    result tangent's variation masks (the frame's gauge masks over the new tangent stack) and **re-mask their
    output** with them, so the padded slots come back as exact zeros (one multiply by a host mask; the real block is
    unchanged bit for bit).
    """

### `t3toolbox/backend/fitting.py:1-23`

Before:

    '''Sampling-kind primitives for the geometry-generic Gauss-Newton fitting model.

    The Gauss-Newton model ``m(p) = c + gᵀp + ½ pᵀ(JᵀJ)p`` of a least-squares objective factors over two
    *independent* choices, and the fitting layer is generic in both:

      * the sampling **kind** (apply / entries / probe) -- the bare single-sample Jacobian ``𝒥`` / its
        transpose ``𝒥ᵀ`` (from :py:mod:`t3toolbox.backend.probing`), plus the kind-specific ``‖·‖²``
        reduction over the probe stack; and
      * the **geometry** (manifold / corewise) -- the gauge projection ``Π`` (``geometry.project``), with
        the Riemannian forward ``J = 𝒥∘Π`` and gradient ``Jᵀr = Π∘𝒥ᵀr``.

    This module supplies the **kind** half: a :py:class:`SamplingKind` descriptor bundling the bare probing
    primitives + the reduction per kind (:py:data:`APPLY` / :py:data:`ENTRIES` / :py:data:`PROBE`). The
    **geometry** half lives in the geometry (:py:data:`t3toolbox.manifold.MANIFOLD` /
    :py:data:`~t3toolbox.manifold.COREWISE`); the frontend :py:class:`t3toolbox.fitting.LocalModel`
    composes the two. The §6.3 corewise substitution ``(D,U,L,R) -> (B,G,G,G)`` is now nothing but
    ``CorewiseGeometry.frame`` -- there is no separate corewise backend.

    ``probe`` is vector-valued (one free mode per probe), so its residual / forward output is a sequence of
    ``d`` arrays and its reduction sums the free mode too; apply / entries are the scalar all-modes special
    case. The objective constant ``c = ½‖r‖²`` is just ``½ · sumsq(r)`` (the same reduction as the model's
    quadratic term ``½‖𝒥Πp‖²``), so the kind needs no separate objective function.
    '''

After:

    '''The least-squares oracle and its batch: what the optimizers ask of the data.

    A fit is ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` the residual of a sampling kind (probe, apply or entries) at
    a jet shape ``order``, and ``M`` the batch's data metric. :py:class:`LeastSquaresOracle` (the kind and the order,
    a frozen description with no arrays) answers what an optimizer asks at a frame or a corewise point (``f``,
    ``g``, ``Jp``, ``JtJp``, ``Kp``, the block sums, ...) through one call of the engine
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`, and checks, draws from and counts a batch.
    :py:class:`LeastSquaresBatch` holds the data, arrays only: the sample, ``y``, the weight, the minibatch scale and
    the drawn indices. What the oracle returns is bare, before any projection: ``Jp = 𝒥p``, ``g = 𝒥ᵀ(M r)`` and
    ``JtJp = 𝒥ᵀ M 𝒥p``, with ``𝒥`` the Jacobian of ``z`` at the frame and ``M`` the batch's data metric (its minibatch
    scale inside). The geometry (manifold or corewise, :py:mod:`t3toolbox.backend.geometry`) is the other half of a
    fit: it supplies the projection ``Π`` (``geometry.project``: the gauge projection on the manifold; on the
    corewise geometry the identity, or the per-group mean of tied factors), so that the forward is ``J = 𝒥∘Π`` and
    the gradient ``Jᵀr = Π∘𝒥ᵀr``; :py:mod:`t3toolbox.backend.optimizers` composes the two.
    '''

### `t3toolbox/backend/uniform_fitting.py:4-25`

Before:

    """Uniform-layer fitting: packing a least-squares batch for the uniform layout, the minimal-rank reduction of a
    uniform start, and the old packed ``SamplingKind`` classes.

    :py:func:`pack_batch` (with :py:func:`pack_sample` and :py:func:`pack_data`) packs the loop-invariant sample and data
    of a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` ONCE, so the optimizers run fully packed -- no
    per-matvec pack/unpack: ``newton_cg(geom, oracle, pack_batch(kind, order, batch, N), (x_sc[0], x_sc[1]))`` with a
    uniform geometry (:py:mod:`t3toolbox.backend.geometry`, ``from_point``). :py:func:`uniform_minimal` reduces a
    uniform start to minimal ranks, which the uniform manifold geometry requires.

    The uniform twins of :py:mod:`t3toolbox.backend.fitting`'s old kinds -- subclasses that override the
    layer-specific operations and, for the probe kinds, where ``omega``'s axes sit in the PACKED output -- are no longer
    called by the optimizers (graded messages, P16) and go with the old kinds (P20).
    Each carries the fixed rank it was built at (``shape`` + the plain-UT3 ``masks``) as FIELDS, so a
    rebuilt kind of the same rank is the same jax cache key
    (:py:mod:`t3toolbox.backend.geometry` follows the same rule; the reasoning is
    ``docs/contributor/parameters_not_closures.md``).

    **Protocol note (ruled: keep and document):** the kinds' ``forward`` / ``transpose`` keep
    the ragged Kind protocol's ``(v, sample, frame_data, sweep)`` signature even though the packed
    ``sweep`` already carries the sample and the frame -- the extra arguments are deliberately unread.
    Protocol uniformity is the point: generic code (the shared ``LocalModel``) calls every kind the same
    way, ragged or uniform.
    """

After:

    """Uniform-layer fitting: packing a least-squares batch for the uniform layout, and the minimal-rank reduction of
    a uniform start.

    :py:func:`pack_batch` (with :py:func:`pack_sample` and :py:func:`pack_data`) packs the loop-invariant sample and data
    of a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` ONCE, so the optimizers run fully packed -- no
    per-matvec pack/unpack: ``newton_cg(geom, oracle, pack_batch(kind, order, batch, N), (x_sc[0], x_sc[1]))`` with a
    uniform geometry (:py:mod:`t3toolbox.backend.geometry`, ``from_point``). :py:func:`uniform_minimal` reduces a
    uniform start to minimal ranks, which the uniform manifold geometry requires.
    """

(The first paragraph is today's, less "and the old packed ``SamplingKind`` classes".)

## 2. Comments and docstrings cut or corrected

### Section headers of kept code (review r2, S2)

Each of these blocks sits above a deleted statement today, but after K0 the next statement below it is kept code
that it heads.

- **`probing.py:274-276`**, the banner "Probing a tangent vector": after K0 it heads `tv_probe` (the deleted
  `compute_dxi` ... `tv_probe_jacobian_from_sweep` sit between today). **Kept as it is.**
- **`probing.py:672-674`**, the banner "Transpose of tangent to probes map": after K0 it heads `tv_probe_transpose`,
  `t3_probe_ambient_transpose`, `t3_probe_corewise_tangent` and `t3_probe_corewise_transpose`. After:

      ###############################################################
      ######    Transposes of the probe; the corewise tangent    ######
      ###############################################################

- **`probing.py:1236-1266`**, the banner "Corewise two-hole: the probe Hessian" and its 28-line comment on the
  deleted kernel: after K0 the banner heads `t3_probe_corewise_hessian`. **The banner (`:1236-1238`) stays as it is;
  the comment below it (`:1239-1266`) is deleted** (it describes the dotted kernels, which go; the function's
  docstring says what it computes).
- **`ut3_sampling.py:124-133`**, the header of the corewise functions: after K0 it heads
  `ut3_apply_corewise_tangent` ... `ut3_probe_corewise_transpose`. Before: the ten lines quoted in the review
  ("Gradient of a sampling op w.r.t. the frame's own supercores (the Section 6.3 (U,L,R)->G substitution, via the
  now-polymorphic probing.*_corewise_transpose ...) ... Mask-once + pack at the boundary; return the RAW gradient
  supercores (dB, dG) ... RE-MASKED on exit with the same point masks that masked the input. The raw padding is not
  clean (the boundary-bond squash is a sum over the whole bond, so its gradient broadcasts into the padded
  boundary-bond slots too); the re-mask zeroes it, so a consumer gets exact zeros there. The equivalence contract
  still declares the padding don't-care (every consumer masks on entry -- ut3_squash_tails included, since
  2026-08-22), and the masks, not the values, govern rank."). After:

      # ----------------------------------------------------------------- corewise (non-manifold) tangents and transposes
      # The forward derivative of a sampling op along raw directions in the point's own supercores, and its gradient
      # w.r.t. them (for a uniform core-wise optimizer: Adam, L-BFGS). The transposes return the RAW gradient
      # supercores (dB, dG) -- the uniform mirror of the ragged TuckerTensorTrain.*_corewise_transpose raw-tuple
      # return -- with their padding exactly zero: the engine masks every input on entry and every output on exit.
      # The equivalence contract still declares the padding don't-care (every consumer masks on entry --
      # ut3_squash_tails included, since 2026-08-22), and the masks, not the values, govern rank.

  The sentence on the boundary-bond squash is not kept: it is no longer true of the code. Measured (E1,
  `experiments/e1_e2.py`): with the engine's exit mask removed, the corewise transposes' padding is already exactly
  zero, because every operand is masked on entry and the engine starts its sweeps with ones on the real slots only.
- **`ut3_sampling.py:469-473`**, the header of the plain kinds' Hessians: after K0 it heads
  `ut3_probe_corewise_hessian`, `ut3_apply_corewise_hessian`, `ut3_entries_corewise_hessian`. Before: "corewise
  two-holes: the plain kinds' Hessians / The same wrapper, for the plain probe / apply / entries two-holes (the
  derivatives of ut3_*_corewise_transpose along raw core directions): theta masked with its masks, theta_hat with
  the SAME masks, the residual and the vectors packed (entries: the integer index is not packed -- the one-hots are
  built packed inside the kernel), the polymorphic kernel run on the supercores, the output re-masked." After:

      # ----------------------------------------------------------- corewise two-holes: the plain kinds' Hessians
      # The derivatives of ut3_*_corewise_transpose along raw core directions, one engine call each (Kp): theta and
      # theta_hat masked with theta's masks on entry, the output masked on exit.

- **`utv_sampling.py:444`** (review r2, M1, taken): "the transpose 𝒥ᵀ (probe)" already heads the three kinds'
  transposes today. After: `# ----------------------------------------------------------------- the transpose 𝒥ᵀ (probe, apply, entries)`.

### `t3toolbox/backend/ut3_sampling.py:63-66` (the comment before `UT3Data`)

Before:

    # All re-mask first (so the padded "garbage" contributes nothing), then call the SHARED, already-
    # polymorphic sampling backends (entries/apply/probing) on the masked supercores. Physical vectors are
    # zero-padded to N (pack) and probe results sliced back to the real shape (unpack); index entries need
    # no packing (the real block is the prefix, so an index in [0,Ni) hits the right slot).

After: deleted (the engine docstring and the module docstring say how the masking and the packing are done; this
comment describes the pre-engine wrappers). Lines 68-69 (`.data` layout) stay.

### `t3toolbox/backend/fitting.py:154-160` (before `_flat_gather`)

Before:

    # --------------------------------------------------------------------------------------------------
    # Default-draw layout helpers (the minimal layout the kind exposes so the optimizer can build its
    # DEFAULT flat minibatch draw -- a random subset across the whole probe stack W. A USER-supplied draw
    # bypasses these entirely.) `n_measurements` = the flat W size; `take` gathers a flat W subset out of
    # (sample, data) -- both kind-specific, since W sits in a different place per kind (and the derivative
    # data has a leading order axis).
    # --------------------------------------------------------------------------------------------------

After:

    # --------------------------------------------------------------------------------------------------
    # The flat draw over the sample stack W: flatten W's axes into one and gather the drawn positions. The
    # oracle's `take` uses it for the sample (and, through `_take_W`, for y and the weight), wherever W sits in
    # the array. A USER-supplied draw bypasses it entirely.
    # --------------------------------------------------------------------------------------------------

### `t3toolbox/backend/fitting.py:745-749` (the header of the oracle's section)

Before:

    #####################################################################
    #########    The least-squares oracle and its batch (P14)    ########
    #####################################################################
    # The fitting layer's new extension point (graded messages, Part 1): an oracle answers a query at a frame through
    # the engine, its batch holds the data. Beside the old kinds above, which the optimizers use until P16.

After:

    #####################################################################
    #########    The least-squares oracle and its batch (P14)    ########
    #####################################################################
    # The fitting layer's extension point (graded messages, Part 1): an oracle answers a query at a frame through the
    # engine, its batch holds the data.

### `t3toolbox/backend/entries.py:113-116` (in `t3_entries_ambient_transpose`)

Before:

    # one-hot CP factors e_{index_i}, elm_shape = W + (Ni,). eye(N)[index] follows numpy index semantics
    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes
    # (_onehot_vectors); the former `arange(N) == index` scatter matched nothing for a negative index and
    # silently returned zero factors.

After:

    # one-hot CP factors e_{index_i}, elm_shape = W + (Ni,). eye(N)[index] follows numpy index semantics
    # (a negative index wraps), exactly like the forward t3_entries and the tangent/corewise transposes; the
    # former `arange(N) == index` scatter matched nothing for a negative index and silently returned zero factors.

### `t3toolbox/backend/common.py:524-526` (`readonly_mask_copies`)

Before:

    """Defensive READ-ONLY numpy copies of a mask tuple, for storage in value-hashed aux objects:
    an aliased writeable caller array mutated in place would leave a stale cached jit key
    (the ``_weight_matrix`` precedent)."""

After:

    """Defensive READ-ONLY numpy copies of a mask tuple, for storage in value-hashed aux objects:
    an aliased writeable caller array mutated in place would leave a stale cached jit key."""

### `t3toolbox/backend/utv_sampling.py:449-453` (`_gauge_masks_over_Knew`)

Before:

    """The result tangent's variation masks: the frame's gauge-shifted variation masks
    ``(down, up, frame_left[:-1], frame_right[1:])`` broadcast (constant) over the new tangent stack
    ``K_new`` -- the leading stack of the transpose output minus the frame stack ``C``. ``sum_over_probes``
    determines ``K_new`` (``W+K`` kept / ``K`` summed); we read it off the output supercore rather than
    re-deriving it. Masks are host numpy (static aux), so this stays on ``np``."""

After (one word): ``sum_over_probes`` → ``sum_over_samples``.

### `t3toolbox/backend/probing.py:43-52` (comments inside `__all__`)

The section comments `# Apply -- frame-sweep reuse split ...`, `# Entries -- frame-sweep reuse split ...`,
`# Probe -- frame-sweep reuse split ...`, `# Corewise (non-manifold) transpose -- ...`, `# Transpose of map from
tangent vector to probes`, `# Corewise two-hole: ...` go with the entries they head; the list after K0 is grouped by
four comments, all true of what follows them: `# Probe a Tucker tensor train` (`t3_probe`), `# Probe a tangent
vector, and its transpose` (`tv_probe`, `tv_probe_transpose`), `# Ambient / corewise probe transposes, the
corewise tangent and Hessian` (`t3_probe_ambient_transpose`, `t3_probe_corewise_tangent`, `_transpose`,
`_hessian`), `# Probe a dense tensor` (`dense_probe`).

## 3. `docs/api_reference.rst` (the ruling on review M2: names of what no longer exists, removed where the text
stays correct)

### `:75-80`, the six model factories (deleted in P16)

Before:

    - The six model factories (in :py:mod:`~t3toolbox.fitting`):
      :py:func:`~t3toolbox.fitting.entries_model`, :py:func:`~t3toolbox.fitting.apply_model`,
      :py:func:`~t3toolbox.fitting.probe_model`,
      :py:func:`~t3toolbox.fitting.entries_derivatives_model`,
      :py:func:`~t3toolbox.fitting.apply_derivatives_model`,
      :py:func:`~t3toolbox.fitting.probe_derivatives_model`.

After: the bullet is removed. The bullets before (`LocalModel`) and after (the four optimizers) stay correct.

### `:87-90`, `least_squares_problem`, `Problem`, `LeastSquaresSetup` (deleted in P16)

Before:

    - Module-level (not at the root): :py:func:`~t3toolbox.optimizers.least_squares_problem` -- the
      constructor the four optimizers use, public for scripts that drive the backend directly: it builds the
      backend :py:class:`~t3toolbox.backend.optimizers.Problem` from the frontend vocabulary and returns a
      :py:class:`~t3toolbox.optimizers.LeastSquaresSetup` (``.problem``, ``.x0_cores``, ``.rewrap``).

After: the bullet is removed. The next bullet's "the one exception to the rule above" refers to the section's
"Everything below is importable directly from ``t3toolbox``" (`:14`), which stays true.

### `:128-131`, the sampling modules

Before:

    - **Sampling** (grouped by sampling type, the deliberate exception):
      :py:mod:`~t3toolbox.backend.probing`, :py:mod:`~t3toolbox.backend.apply`,
      :py:mod:`~t3toolbox.backend.entries`, :py:mod:`~t3toolbox.backend.sampling_derivatives`, and the
      uniform object-type wrappers :py:mod:`~t3toolbox.backend.ut3_sampling`,
      :py:mod:`~t3toolbox.backend.utv_sampling`.

After:

    - **Sampling** (grouped by sampling type, the deliberate exception):
      :py:mod:`~t3toolbox.backend.probing`, :py:mod:`~t3toolbox.backend.apply`,
      :py:mod:`~t3toolbox.backend.entries`, and the
      uniform object-type wrappers :py:mod:`~t3toolbox.backend.ut3_sampling`,
      :py:mod:`~t3toolbox.backend.utv_sampling`.

A scan of every `:py:` role of the page against the library (an import and `getattr` per target) finds no other
missing target: these nine names (six factories, `least_squares_problem`, `Problem`, `LeastSquaresSetup`) and, after
K0, `sampling_derivatives`. The new names (the oracle, the batch, the engine's module) are P19's to add.

## 4. Test prose (for completeness)

- `tests/test_jets.py:896-897`, `test_the_validators_of_pp_are_gone`'s docstring. Before: "There is no ``pp``: the
  two validators P3 copied from ``sampling_derivatives`` are not in ``jets`` (the originals stay in
  ``sampling_derivatives``, called by the old frontend methods, until P20)." After: "There is no ``pp``: the two
  validators of the old ``sampling_derivatives`` (deleted in K0) are not in ``jets``."
- `tests/test_contractions_interpreter.py:237`, `:241`, `:299` (review M3): **all three left as they are**. They
  annotate frozen rows of `HISTORICAL`, which records the strings the library has held; history, as the table is.
- The module docstrings of `tests/backend/test_curvature.py` and `test_uniform_curvature.py`: the bullets of the
  deleted classes go whole (T5 probe derivatives, T5 apply / entries derivatives, and the uniform two-holes of the
  derivative kinds); the T5 plain-kinds bullet loses "(and their ``_from_sweep`` forms)", "sweep reuse bitwise;" and
  "each probe piece against a finite difference of the GN kernel it differentiates; the factored TT assemblies
  against their unfactored product rule"; `_curvature_oracles.py`'s "reference twin" paragraph goes whole.
- `TestTransposeOutputHygiene`'s class docstring: clause (b) reads "the real block is the ragged face's, per stack
  element, to 1e-12", and the non-vacuity sentence reads "Without the masks (the engine's, and the tangent faces'
  re-mask), (c) fails: the negative control."
