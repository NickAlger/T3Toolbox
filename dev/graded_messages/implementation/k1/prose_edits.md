# K1: the library prose that is rewritten, before and after

Implementer (the second), 2026-11-25, for `K1_DESIGN_NOTE.md` §10. Line numbers are in the working tree with K0 built.
Each text describes the code **after K1** (nothing about K2 to K7). Where part 0 builds a transitional text (the batch
before its two weights), it is the final text with today's sentences on the one weight kept; the part's handover
shows it. Doctest outputs marked `<to be pasted>` are filled from a run during the build, never by hand; the rest
of each example is the draft.

**Status (first draft, 2026-11-25):** part 0's texts (§1) in full; part 3's module docstring of `jets` and the
docstrings of `probe_data_metric`, `derivative_block_weight_to_taylor` in full, the other new functions of part 3 as
today's texts with the changes listed (§2); the engine's parameter entries and module-docstring sentences (§3);
`uniform_fitting`'s module docstring and `pack_batch` (§4); the fixed substitutions of part 5 (§5). Not yet: part 1's
reference texts (`sampling_definitions`' module paragraphs and the docstrings of `w_tilde`, `f`, `r_block_sumsq`,
`Jp_norm_sq`), which change by the same substitution as the engine's (one weight to two, the rule of note §4.1).

---

## 1. Part 0: the batch knows its kind

### 1.1 `t3toolbox/backend/fitting.py:1-15`, the module docstring

Before:

    '''The least-squares oracle and its batch: what the optimizers ask of the data.

    A fit is ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` the residual of a sampling kind (probe, apply or entries) at
    a jet shape ``order``, and ``M`` the batch's data metric. :py:class:`LeastSquaresOracle` (the kind and the order,
    a frozen description with no arrays) answers what an optimizer asks at a frame or a corewise point (``f``,
    ``g``, ``Jp``, ``JtJp``, ``Kp``, the block sums, ...) through one call of the engine
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`, and checks, draws from and counts a batch.
    :py:class:`LeastSquaresBatch` holds the data, arrays only: the sample, ``y``, the weight, the minibatch scale and
    the drawn indices. What the oracle returns is bare, before any projection: ...

After (the rest of the docstring, from "What the oracle returns is bare", unchanged):

    '''The least-squares oracle and its batch: what the optimizers ask of the data.

    A fit is ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` the residual of a sampling kind (probe, apply or entries) at
    a jet shape ``order``, and ``M`` the batch's data metric. :py:class:`LeastSquaresBatch` holds the measurements:
    their kind and jet shape (static), the sample, ``y``, the block weight and the full weight, the minibatch scale and
    the drawn indices; it checks itself against a model, draws from itself, counts its samples and applies its data
    metric. :py:class:`LeastSquaresOracle` is the objective's rule, with no fields: it answers what an optimizer asks
    at a frame or a corewise point (``f``, ``g``, ``Jp``, ``JtJp``, ``Kp``, the block sums, ...) on a batch, through
    one call of the engine :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`. What the oracle
    returns is bare, before any projection: ...

### 1.2 `LeastSquaresBatch`, the class docstring (`backend/fitting.py:69-83`)

Before: quoted in the file (it says "arrays only", "Checked by `LeastSquaresOracle.check_batch`").

After (part 4's final text; part 0 keeps its sentences on `weight` as today):

    """The measurements of a least-squares fit: their kind and jet shape, the sample, the data and the weights.

    A registered pytree: ``kind`` and ``order`` are static (jax ``aux_data``: a compiled program is for one kind and
    one jet shape, the same for every draw of a fit), every other field a leaf (``None`` allowed). Every field after
    ``y`` is keyword-only, so that no positional call puts a scale in a weight:
    ``LeastSquaresBatch('probe', (2,), ww, y, block_weight=b)``.

    - ``kind``: ``'probe'``, ``'apply'`` or ``'entries'``; ``order``: the jet shape of the sample and the data, a
      tuple of nonnegative integers (``()`` for no jets; the entries kind takes ``()`` only). Both are checked when the
      batch is made, and ``order`` is kept as Python ints.
    - ``sample``: the input jets ``ww`` (probe, apply: ``len=d``, ``elm_shape=F_in_i+W+(Ni,)``, or packed
      ``(d,)+F_in+W+(N,)``, :py:func:`~t3toolbox.backend.uniform_fitting.pack_batch`), or the integer ``index``
      ``(d,)+W`` (entries).
    - ``y``: the data, exactly ``z``'s shape with the model's stack ``C``: probe ``F+W+C+(Ni,)`` per mode, apply
      ``F+W+C``, entries ``W+C``. Only its ``C`` axes may be 1 (data shared by a stack of models); a ``y`` the same for
      every sample is not a dataset.
    - ``block_weight``: one number per block, a mode at a slot (probe, ``(d,)+F``), a slot (apply, ``F``) or all of
      the data (entries, ``()``); the same array on both layers. ``full_weight``: one number per measured number,
      ``z``'s shape, each ``W`` and ``C`` axis 1 or full (a weight the same for every sample has ``W`` axes of 1).
      ``None`` is ones. The metric is ``M = gamma * block_weight**2 * full_weight**2`` (:py:meth:`data_metric`).
    - ``minibatch_scale``: ``gamma``, ``n / b`` on a batch drawn by :py:meth:`take`, ``None`` (one) on a full batch;
      ``drawn_indices``: the flat sample indices a draw took, relative to the batch it was taken from.

    The arrays are checked against a model by :py:meth:`check`, which the frontend calls where a batch enters; the
    engine checks what it reads on every query.

    Examples
    --------
    >>> import numpy as np
    >>> from t3toolbox.backend.fitting import LeastSquaresBatch
    >>> ww = [np.ones((10, N)) for N in (4, 5, 6)]                            # W = (10,)
    >>> batch = LeastSquaresBatch('apply', (), ww, np.zeros(10))
    >>> print(batch.kind, batch.order, batch.n_measurements)
    <to be pasted>

    A batch made without its kind and jet shape fails at once:

    >>> LeastSquaresBatch(ww, np.zeros(10))                                   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: ...
    """

### 1.3 `LeastSquaresBatch.data_metric` (`backend/fitting.py:95-116`)

After (part 4; part 0 keeps today's text):

    """The data metric of this batch, ``M r = gamma * block_weight**2 * full_weight**2 * r``: the metric of its kind
    (:py:func:`~t3toolbox.backend.jets.probe_data_metric`, ``apply_data_metric``, ``entries_data_metric``) with the
    batch's order, weights and minibatch scale. The call that cannot forget the scale of a drawn batch.

    >>> import numpy as np
    >>> from t3toolbox.backend.fitting import LeastSquaresBatch
    >>> y = np.ones((3, 4))                                             # apply: F + W, F = (3,), W = (4,)
    >>> batch = LeastSquaresBatch('apply', (2,), None, y, block_weight=np.array([1.0, 2.0, 3.0]),
    ...                           minibatch_scale=0.5)
    >>> print(batch.data_metric(y)[:, 0].tolist())                     # 0.5 * block_weight**2 * r, slot by slot
    <to be pasted>
    """

### 1.4 `LeastSquaresBatch.check` (from `LeastSquaresOracle.check_batch`, `backend/fitting.py:447-474`)

After:

    '''Check this batch structurally against a model's mode sizes ``shape`` and stack ``stack_shape`` (``C``; ``()``
    for an unstacked model); raise a ``ValueError`` naming what is wrong. The sample against the jet shape and
    ``shape`` (:py:func:`~t3toolbox.backend.jets.check_sample`); ``y`` exactly ``z``'s shape but for ``C`` axes of 1
    (:py:func:`~t3toolbox.backend.jets.check_data`); the full weight ``z``'s shape but for ``W`` and ``C`` axes of 1;
    the block weight exactly the block shape. Data of a stacked model checked without its ``stack_shape`` fail by
    rank. A checker: the oracle's ``query`` does not call it; the frontend does, where a batch enters. On the ragged
    form (check before packing).

    >>> import numpy as np
    >>> from t3toolbox.backend.fitting import LeastSquaresBatch
    >>> ww = [np.ones((10, N)) for N in (4, 5, 6)]                            # W = (10,)
    >>> LeastSquaresBatch('apply', (), ww, np.zeros((10, 1))).check((4, 5, 6), stack_shape=(2,))   # one y for C = (2,)
    >>> LeastSquaresBatch('apply', (), ww, np.zeros((10, 3))).check((4, 5, 6), stack_shape=(2,))
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the data y of the apply kind has shape (10, 3); ...
    '''

(Part 0's text is the same with "the weight against the sample and the stack (`check_weight`)" in place of the two
weights' sentences.)

### 1.5 `LeastSquaresBatch.take` (from `LeastSquaresOracle.take`, `backend/fitting.py:500-534`)

After (the first paragraph as today with "``batch``" read as "this batch"; the second paragraph):

    The data ``y`` and the full weight are drawn alike: ``y`` (exact on ``W``) and a full weight with a full ``W``
    axis are broadcast to the batch's ``W``, flattened and gathered at ``indices``; a full weight whose every ``W``
    axis is 1 (the same for every sample) keeps one axis of 1 in their place, with no gather. The block weight is
    constant over the samples and passes through. So a weight per sample is drawn with its samples, and a weight per
    slot stays one per slot.

The examples as today, on the method: `sub = batch.take(np.array([0, 3]))`, `per_sample = dc.replace(batch,
full_weight=np.arange(10.0))` (part 0: `weight=`).

### 1.6 `LeastSquaresBatch.n_measurements` (a property)

    '''The number of samples, ``prod(W)``: the population a draw takes from (:py:func:`~t3toolbox.backend.optimizers.flat_draw`).'''

### 1.7 `LeastSquaresOracle`, the class docstring (`backend/fitting.py:291-360`)

Before: quoted in the file. Its first two paragraphs name "the kind and the order" as the oracle's fields and its
examples construct `LeastSquaresOracle('probe', order=(1,))`.

After, the first two paragraphs (the rest, from "**The block sums**", unchanged but for "each batch's own weight" →
"each batch's own weights", part 4):

    """The least-squares oracle: ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` and ``M`` the batch's data metric
    (:py:meth:`LeastSquaresBatch.data_metric`), and what an optimizer asks of it at a frame, through one call of the
    engine.

    It is the objective's rule and has no fields (value-hashed, it rides as jax ``aux_data``); the measurements, their
    kind and their jet shape are its batch's, a :py:class:`LeastSquaresBatch`. ...

(then today's paragraph on `query`, the scale, `f_resolution`, unchanged.) The examples:

    >>> oracle = LeastSquaresOracle()
    >>> batch = LeastSquaresBatch('probe', (1,), affine_jet(ww, pp), target.probe(affine_jet(ww, pp), (1,)))
    >>> (f, g), held = oracle.query(frame, batch, ('f', 'g'))

and the trap (input jets read with `order=()`) is stated on the batch: "a batch made with `order=()` from input jets
passes every check of shapes ...", the three queries as today with `LeastSquaresBatch('probe', (), ...)` and
`LeastSquaresBatch('probe', (1,), ...)`.

### 1.8 The `Oracle` protocol (`backend/optimizers.py:181-210`)

Before, the first and last paragraphs:

    An oracle is a frozen description with no arrays (it rides as jax ``aux_data``; value-hashed); its data are its
    ``batch``, which the optimizers pass through unread (``None`` for an oracle with no data). ...

    **Optional members** ``check_batch(batch, shape, stack_shape=()) -> None`` (O-29): the structural checker of a
    batch against the mode sizes ``shape`` and the model's stack of base points ``stack_shape`` (``C``; ``()`` for an
    unstacked model), which the frontend calls, always with the model's ``stack_shape``, where a batch enters when the
    oracle has one, and nothing about the batch otherwise; and
    ``n_measurements(batch)`` with ``take(batch, indices)``, the default draw of ``mc_sgd`` and ``adam``
    (:py:func:`flat_draw`), which without them need a ``draw``.

After:

    An oracle is a frozen description with no arrays (it rides as jax ``aux_data``; value-hashed); its data are its
    ``batch``, which the optimizers pass through unread (``None`` for an oracle with no data), with one exception, the
    default draw below. ...

    **What a batch may have** (O-29; the least-squares batch has all three): ``check(shape, stack_shape=()) -> None``,
    the structural checker of the batch against the mode sizes ``shape`` and the model's stack of base points
    ``stack_shape`` (``C``; ``()`` for an unstacked model), which the frontend calls, always with the model's
    ``stack_shape``, where a batch enters; and the property ``n_measurements`` with ``take(indices)``, which the
    default draw of ``mc_sgd`` and ``adam`` calls (:py:func:`flat_draw`): a batch without them needs a ``draw``. An
    oracle that reuses :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` reads the kind and the jet shape off it.

### 1.9 `flat_draw` (`backend/optimizers.py:625-641`, under O1's recommendation)

    """The **default** minibatch draw: a uniform random subset of ``min(batch_size, n)`` of the ``n`` samples
    (``batch.n_measurements``), flat across the whole sample stack ``W``, as ``batch.take(indices)`` returns it: for
    the least-squares batch a batch with ``minibatch_scale = n / b`` and the ``drawn_indices``. Returns
    ``draw(rng) -> batch``, which calls ``rng.choice(n, size=min(batch_size, n), replace=False)`` once per draw. A user
    may pass **any** ``draw`` instead (slice ``W``, importance-sample, ...), returning a batch of the oracle. Host numpy
    indices; on a jax batch the gather runs on device (the optimizer never compiles the draw, only the step)."""

`_require_draw`'s docstring: "without a ``draw`` the batch needs ``take(indices)`` and ``n_measurements``, the default
draw (:py:func:`flat_draw`); else a ``ValueError``." The comments of `mc_sgd` and `adam`'s `draw` parameter
(`:1233`, `:1283`): `# custom draw(rng) -> a batch; None = flat_draw(batch, batch_size)`.

### 1.10 `make_newton_display` (`optimizer_display.py:549-601`)

Two sentences change. "The layout of the printed table follows the oracle's ``order`` (...; an oracle without one:
axes labelled by position)" becomes "follows the batch's ``order`` (...; a batch without one: axes labelled by
position)". After "**Each table is under its own batch's weight**: ..." (part 4: "weights") one sentence is added:

    A ``val_batch`` of another kind or jet shape than ``batch`` raises a ``ValueError`` at once: its table would set two
    different quantities side by side.

### 1.11 The frontend

- `t3toolbox/fitting.py:589-600`, `prepare_batch`'s docstring: "``oracle.check_batch(batch, x.shape,
  stack_shape=x.stack_shape)`` when the oracle has it" becomes "``batch.check(x.shape, stack_shape=x.stack_shape)``
  when the batch has it"; "a ``LeastSquaresBatch`` packed with :py:func:`pack_batch`" stays. `_is_packed`'s: "Whether
  a least-squares batch is already packed for the uniform layer (its sample a supercore, not a list)" stays.
- `t3toolbox/fitting.py:660`, `local_model`'s docstring: "It checks the batch (``batch.check``, when the batch has
  it)".
- `t3toolbox/optimizers.py:10`, the module docstring: "(1) checks the batch (``batch.check`` on a ragged batch, when
  the batch has it; structural, before any work)".
- `t3toolbox/optimizers.py:331`, `:354` (`mc_sgd`, `adam`): `oracle,  # the objective` and `batch,  # its full data
  (with n_measurements / take for the default draw)`.

---

## 2. Part 3: `backend/jets.py`

### 2.1 The module docstring (`jets.py:5-153`)

Before: quoted in the file. Its paragraphs "**``kind`` says which type an object is of**" (`:17-27`), "Five groups"
(`:29-53`), "**The weight**" (`:55-61`) and the walk-through of derivative data (`:67-152`) change; the opening
definition of a jet (`:8-15`) and "**The data coordinates are Taylor coefficients**" (`:63-65`) stay, with "the
weight" read as "the weights".

After, the changed paragraphs:

    **The kind is in the name** of a function whose computation, or the check of its arguments, depends on it: the
    metrics, the dual jet and the derivative data are ``probe_*``, ``apply_*`` (and ``entries_data_metric``). A
    ``probe_*`` function takes one array per mode, a tuple or a list (the metric also takes one packed array,
    ``(d,) + F + ... + (N,)``, and keeps its form); an ``apply_*`` function one array whose leading axes are the jet
    axes. The entries kind has data and no jets: it has a metric and nothing else here. The functions that still take
    ``kind`` (:py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`, the checkers) read the object as
    ``kind`` says, never by looking at it.

    Five groups of functions:

    - **The data metric** ``M: Y* -> Y``, ``gamma * block_weight**2 * full_weight**2 * r``: what a transpose is applied
      to for the gradient of a weighted fit. :py:func:`probe_data_metric`, :py:func:`apply_data_metric`,
      :py:func:`entries_data_metric`.
    - **The dual jet,** the engine's form of an element of ``Y``: the data metric at ``gamma = 1``, then the turn, a
      flip of every jet axis (:py:func:`probe_jet_to_dual_jet`, :py:func:`apply_jet_to_dual_jet`), and its inverse
      (:py:func:`probe_dual_jet_to_jet`, :py:func:`apply_dual_jet_to_jet`). They are for the engine's ``w_tilde`` and
      its held tilde variables, never for a transpose.
    - **Derivative data.** The library works in Taylor coefficients, everywhere. A user whose data are derivatives
      converts them before fitting: :py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`,
      :py:func:`derivative_block_weight_to_taylor`, and :py:func:`prepare_probe_derivative_data` /
      :py:func:`prepare_apply_derivative_data`, which do the data, the validation data and the block weight in one
      call. The mixed derivative ``d^k / ds^k`` of ``s^k`` is ``k_1! ... k_m!``, the factor of slot ``k``. No fitting
      function, oracle or sampling method has a keyword for derivative data.
    - **The input jets:** :py:func:`affine_jet` and :py:func:`index_to_one_hot` (as today).
    - **Checkers.** :py:func:`check_sample` and :py:func:`check_data` raise a ``ValueError`` that names what is wrong
      with the form or the shape of a sample or its data. They are checkers, not guarantees: no backend function
      calls them but the batch's :py:meth:`~t3toolbox.backend.fitting.LeastSquaresBatch.check`. Shapes are static,
      so every checker also runs under ``jax.jit``.

    **The two weights.** The measurement space is a direct sum of **blocks**: one per mode and jet slot (probe), one
    per slot (apply), one in all (entries). A fit weighs it with two weights, which multiply,
    ``M = gamma * block_weight**2 * full_weight**2``:

    - the **block weight**, one number per block, of the block shape exactly: ``(d,) + F`` (probe), ``F`` (apply),
      ``()`` (entries). It is the same array on both layers, and it sits on the leading axes of what it weighs (the
      mode and the jet axes), whatever follows them. An equalizing weight, a noise level per mode, a weight per slot
      and the ``k!`` of derivative data are block weights.
    - the **full weight**, one number per measured number, of ``z``'s shape, where an axis of the stacks ``W`` (the
      samples) or ``C`` (the base points) may be 1, meaning the same for every sample, or for every model of a stack.
      A weight per coordinate of a mode, the same for every sample, is ``F + (1,)*len(W) + (1,)*len(C) + (N_i,)``.

    A size of 1 is allowed on the stack axes only: the mode, the jet slots and the mode edge are the parts of one
    measurement, and a size of 1 there is refused. The data ``y`` are exactly ``z``'s shape, but for ``C`` axes of 1
    (data shared by a stack). **With no weight (``None``) every factor is one**: weights of 1.0 in the natural
    coordinates of the data, the plain sum of squares of the Taylor coefficient residuals. The two multiply, so a
    user who computes a relative full weight from Taylor data and also passes the ``k!`` table that
    :py:func:`prepare_apply_derivative_data` returns has both.

The walk-through (`:67-152`), after (the first block, `derivatives_to_taylor('apply', ...)`, as today until K6):

    Case 1. Taylor data. Nothing to do: the fit uses the data as given, and no weight means ones.

    Case 2. Derivative data and a block weight chosen for derivatives, here one per slot. One call prepares both:

    >>> block_derivatives = np.array([1.0, 10.0, 10.0, 10.0])               # apply: F = (4,)
    >>> y, block, _ = jets.prepare_apply_derivative_data(order, y_derivatives, block_derivatives)
    >>> print(y[:, 0].tolist())
    <to be pasted: [1.0, 1.0, 1.0, 1.0]>
    >>> print(block.tolist())
    <to be pasted: [1.0, 10.0, 20.0, 60.0]>

    The data are divided by ``t!`` and the block weight is multiplied by it: a weight goes the opposite way.

    Case 3. Derivative data and NO weight. The trap: ones on the DERIVATIVES are not ones on the Taylor coefficients.
    The same call returns the block weight ``t!``, which you then pass to the fit:

    >>> y, block, _ = jets.prepare_apply_derivative_data(order, y_derivatives)
    >>> print(block.tolist())
    <to be pasted: [1.0, 1.0, 2.0, 6.0]>

    (validation data, as today, through `prepare_apply_derivative_data(order, y_derivatives, validation=...)`)

    In every case the objective is the one you wrote down. For a model whose Taylor coefficients are ``z``:

    >>> z = np.array([[1.0], [0.9], [1.2], [0.8]])
    >>> objective = lambda block, prediction, data: 0.5 * float(np.sum((block[:, None] * (prediction - data)) ** 2))
    >>> in_derivatives = objective(block_derivatives, jets.taylor_to_derivatives('apply', order, z), y_derivatives)
    >>> y, block, _ = jets.prepare_apply_derivative_data(order, y_derivatives, block_derivatives)
    >>> print(bool(np.isclose(in_derivatives, objective(block, z, y))))
    True

    A full weight is not converted: it multiplies the block weight, which carries the factorials. (two jet variables
    as today, `prepare_apply_derivative_data((1, 2), ...)`, the block `[[1, 1, 2], [1, 1, 2]]`.)

    The probe kind has one array per mode, ``F + W + (Ni,)``, and a block weight ``(d,) + F``:

    >>> np.random.seed(0)
    >>> y_derivatives = [np.random.randn(4, 5, N) for N in (6, 7)]          # d = 2 modes, F = (4,), W = (5,)
    >>> y, block, _ = jets.prepare_probe_derivative_data((3,), y_derivatives)
    >>> print([y_i.shape for y_i in y], block.shape)
    <to be pasted: [(4, 5, 6), (4, 5, 7)] (2, 4)>
    >>> print(block.tolist())
    <to be pasted: [[1.0, 1.0, 2.0, 6.0], [1.0, 1.0, 2.0, 6.0]]>
    >>> _, block, _ = jets.prepare_probe_derivative_data((3,), y_derivatives, np.array([[1.0] * 4, [2.0] * 4]))
    >>> print(block.tolist())                                                # per mode and slot, times t!
    <to be pasted: [[1.0, 1.0, 2.0, 6.0], [2.0, 2.0, 4.0, 12.0]]>

### 2.2 `probe_data_metric` (new; `apply_data_metric` and `entries_data_metric` point to it)

    """The data metric ``M: Y* -> Y`` of the probe kind: ``gamma * block_weight**2 * full_weight**2 * r``.

    The misfit of a least-squares fit is ``f = 1/2 <M r, r>`` with ``r = z - y``, and its gradient is ``J^T (M r)``.
    ``r`` is a measurement, of the type of ``z`` (an element of ``Y*``); ``M r`` is an element of ``Y``, the space the
    transposes take, paired with ``z`` slot by slot. So the gradient of a weighted fit is a transpose applied to this
    function's result, ``t3_probe_corewise_transpose(x, ww, probe_data_metric(order, r, b, f, gamma), order)``; a
    transpose applies no weight of its own. On a batch the method
    :py:meth:`~t3toolbox.backend.fitting.LeastSquaresBatch.data_metric` passes the batch's own weights and scale.

    **The two weights** (the module docstring): ``block_weight``, one number per mode and slot, of shape ``(d,) + F``
    exactly, sits on the leading axes of ``r`` (row ``i`` on the jet axes of mode ``i``); ``full_weight``, in ``r``'s
    form, of ``r``'s rank, with the mode, the jet axes and the edge exact and each axis between them 1 or ``r``'s size.
    ``minibatch_scale`` is the scale ``gamma`` of a drawn batch, the number of samples of the full batch over the
    number drawn. All three are required: ``None`` written out means ones, or a scale of 1.

    **Its forms.** ``r`` is a tuple or a list of ``d`` arrays (one per mode), or one array packed for the uniform
    layout, ``(d,) + F + W + [K] + C + (N,)``; the full weight is in the same form, and the result too (a tuple for a
    tuple or a list). A packed ``r`` with a ragged full weight raises. Values in a packed ``r``'s padding come back
    multiplied by the weights: the function holds no masks, so it takes zero padding in (the engine and the oracle,
    which hold them, mask). ``K``, a stack of directions after ``W``, needs nothing for the block weight; a full weight
    has a size-1 axis for it.

    **What raises** (structural, so also under ``jax.jit``): a jet shape that is not a tuple of nonnegative integers;
    an ``r`` whose leading axes are not ``F``; a block weight of another shape than ``(d,) + F``, or given as a tuple or
    a list; a full weight of another form or rank than ``r``, or with an axis that is neither ``r``'s size nor, between
    the jet axes and the edge, 1; a scale that is not a scalar.

    **Coordinates.** Data are Taylor coefficients, everywhere: data given as derivatives are converted once, with their
    block weight, by :py:func:`prepare_probe_derivative_data`. A weight computed from the data is computed after the
    conversion, on the Taylor data, for instance ``equalizing_weight(data_block_norms('probe', order, y))``, a block
    weight, passed straight in. An element of ``Y`` held in derivative coordinates converts the other way than a
    measurement (:py:func:`taylor_to_derivatives`; see :py:func:`derivatives_to_taylor`).

    The result has the lowest floating precision of ``r`` and the weights (the scale is cast to it, and does not raise
    it), and is on jax if any input is.

    Examples
    --------
    One value per mode and slot, and a weight per entry of mode 1:

    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> rng = np.random.default_rng(0)
    >>> r = [rng.standard_normal((3, 4, N)) for N in (5, 6)]          # d = 2, F + W + (Ni,), F = (3,), W = (4,)
    >>> block = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])          # (d,) + F
    >>> c = jets.probe_data_metric((2,), r, block, None, None)
    >>> print(type(c).__name__, bool(np.allclose(c[1][2], 9.0 * r[1][2])))
    <to be pasted: tuple True>

    The trap of a kind-free metric, refused: a weight per slot, ``(n+1,)``, against a probe ``r`` with ``d = n+1``
    modes would be read per mode:

    >>> jets.probe_data_metric((1,), r[:2], np.array([1.0, 2.0]), None, None)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: probe_data_metric: the block weight has shape (2,); at order (1,) with d = 2 it is (d,) + F = (2, 2) ...

    The full weight on a packed ``r``: the mode axis is exact (a mode axis of 1 is refused, where the kind-free metric
    of the M arc broadcast it over the modes):

    (an example with `ut3_operations.pack_vectors`, one right and one refused; outputs to be pasted)

`apply_data_metric`'s docstring: its first paragraph and "What raises" in the apply form (``r`` one array
``F + W + [K] + C``, the block weight ``F``), the gradient example of today's `data_metric` (`jets.py:706-723`) with
`apply_data_metric((), r, None, weight, None)` and a weight per sample as the full weight. `entries_data_metric`'s: one
paragraph ("the metric of the entries kind, which has data and no jets: :py:func:`apply_data_metric` at
``order=()``; the block weight is one number") and one example.

### 2.3 The dual-jet pair

`probe_jet_to_dual_jet`'s docstring is today's `jet_to_dual_jet` (`jets.py:330-443`) with: the weight paragraph
replaced by "**The weights** are the metric's (:py:func:`probe_data_metric`): the block weight ``(d,) + F``, the
full weight in the jet's form"; the paragraph "**``kind`` says which type the jet is of**" replaced by "**The jet is
one array per mode**, a tuple or a list of ``d``: a single array (a packed jet) raises, naming
``unpack_vectors``; the result is a tuple"; the sentence "A jet with a stack of directions ``K`` has no weight of its
rank, so it raises when a weight is given" replaced by "A jet with a stack of directions ``K`` takes the block weight
as it is and a full weight with a size-1 axis for ``K``"; its examples on `probe_*` and `apply_*` with the block
weight in place of `blocks_to_weight`, and the refusal of "a weight in the old block form" replaced by the refusal of a
packed jet. `apply_jet_to_dual_jet`, `probe_dual_jet_to_jet`, `apply_dual_jet_to_jet` likewise from today's texts
(`:330-443`, `:462-523`), the divisor stated as ``block_weight**2 * full_weight**2``.

### 2.4 `derivative_block_weight_to_taylor` and `prepare_*_derivative_data`

    """The Taylor block weight that gives the misfit of a block weight chosen for derivative residuals: slot ``k``
    times ``k_1! ... k_m!``.

    A derivative residual is ``k_1! ... k_m!`` times the Taylor residual, so ``b[k] * (derivative residual)`` equals
    ``(b[k] * k_1! ... k_m!) * (Taylor residual)``. The weight goes the opposite way to the data, which are divided by
    the factorials, and it is the weight itself that is multiplied, not its square. Only the block weight carries the
    factorials: a full weight is the same in both coordinates. The block weight is a table whose trailing axes are the
    jet axes: ``(d,) + F`` (probe) or ``F`` (apply), both taken here. With no jets nothing changes.

    >>> import numpy as np
    >>> import t3toolbox.backend.jets as jets
    >>> print(jets.derivative_block_weight_to_taylor((3,), np.array([1.0, 10.0, 10.0, 10.0])).tolist())
    <to be pasted: [1.0, 10.0, 20.0, 60.0]>
    >>> print(jets.derivative_block_weight_to_taylor((3,), np.ones((2, 4))).tolist())        # probe, (d,) + F
    <to be pasted: [[1.0, 1.0, 2.0, 6.0], [1.0, 1.0, 2.0, 6.0]]>
    """

`prepare_probe_derivative_data`: today's `prepare_derivative_data` (`jets.py:1093-1169`) with: "data, the validation
data and the block weight"; "**The block weight** (``(d,) + F``, checked) chosen for the derivative residuals becomes
the Taylor block weight that gives the same misfit (:py:func:`derivative_block_weight_to_taylor`). With
``block_weight=None`` the default is ones in the data's own coordinates, the derivatives, so the block weight returned
is the ``k_1! ... k_m!`` table. A full weight is not an argument: it is the same in both coordinates, and multiplies
the returned block weight in the fit."; the recipe

    y_taylor, _, _ = prepare_probe_derivative_data(order, y_derivatives)
    block_weight = equalizing_weight(data_block_norms('probe', order, y_taylor))

and "The data are one array per mode, a tuple or a list; a single array raises, naming ``unpack_vectors``". The
entries kind has no version (no jets). `prepare_apply_derivative_data` likewise for one array and the block ``F``.

---

## 3. Part 2: the engine's parameter entries (`sampling_variables.py:1743-1752`)

Before:

    y: sequence of NDArray or NDArray, optional
        The data, of the type of ``z``: probe ``len=d`` (a tuple, a list, or one array whose first axis is the mode),
        ``elm_shape=F+W+C+(Ni,)``; apply ``shape=F+W+C``; entries ``shape=W+C``; each axis 1 or that size (a size of 1
        is the same datum along the axis: data shared by a stack have size-1 ``C`` axes). Needed for ``r`` and what
        reads it. On the uniform layout probe data may come ragged (packed here) or packed, ``(d,)+F+W+C+(e,)`` with
        the edge ``e`` 1 or ``N``.
    weight: sequence of NDArray or NDArray, optional
        ``omega``, of the type of ``z`` as ``y`` is, never with the stack of directions ``K`` (the engine places it
        against ``Jp``); it enters squared, through :py:func:`~t3toolbox.backend.jets.data_metric` at ``gamma = 1``.
        ``None`` is ones. :py:func:`~t3toolbox.backend.jets.blocks_to_weight` makes one from a value per block.

After:

    y: sequence of NDArray or NDArray, optional
        The data, ``z``'s shape exactly but for its ``C`` axes, which may be 1 (data shared by a stack): probe
        ``len=d`` (a tuple, a list, or one array whose first axis is the mode), ``elm_shape=F+W+C+(Ni,)``; apply
        ``shape=F+W+C``; entries ``shape=W+C``. Needed for ``r`` and what reads it. On the uniform layout probe data
        may come ragged (packed here) or packed, ``(d,)+F+W+C+(N,)``.
    block_weight: NDArray, optional
        One number per block: probe ``shape=(d,)+F``, apply ``F``, entries ``()``, exactly; the same array on both
        layouts, never a sequence. It sits on the leading axes of ``r`` (the mode and the jet axes). ``None`` is ones.
    full_weight: sequence of NDArray or NDArray, optional
        One number per measured number: ``z``'s shape, in ``y``'s form, each ``W`` and ``C`` axis 1 or full, never
        with the stack of directions ``K`` (the engine places it). ``None`` is ones. The two weights multiply: the
        rows that read them apply ``block_weight**2 * full_weight**2`` (the data metric at ``gamma = 1``).

The paragraph "The engine checks what it needs" (`:1704-1712`): "``y`` and the weight are of the type of ``z``, each
array of ``z``'s rank and each axis 1 or ``z``'s size (for probe exactly ``d`` arrays), or a ``ValueError``" becomes
"``y`` is ``z``'s shape but for ``C`` axes of 1, the full weight ``z``'s shape but for ``W`` and ``C`` axes of 1, the
modes of either agreeing in everything but the edge, and the block weight the block shape exactly, or a
``ValueError``". The module docstring's paragraph on the uniform layout (`:308-312`): "and probe's weight when its edge
is ``N`` (an edge of 1 has no padding, and multiplies padding that is zero)" becomes "and probe's full weight; the
block weight has no edge and is not masked"; "a ``y`` with an edge of 1 is broadcast to ``N`` by its mask, so that its
value cannot reach the padding of ``r``" goes (no edge of 1). The table rows (`:54-56`, `:84`, `:93`, `:183`) read
`block_weight`, `full_weight` for `weight`; `:122` "``weight**2`` times it" becomes "``block_weight**2 *
full_weight**2`` times it". `sampling_steps.py:909-913` (the whole steps' comment) becomes: "A data-space object (``y``,
the full weight, ``r``, ``Jp``) is of the type of ``z``; the block weight is one number per block, on the leading axes.
``_modes`` gives the arrays of a value of the type of ``z``, each with its block row and its full weight's array;
``_with_K`` places the stack of directions ``K`` in the full weight, which never carries it."

---

## 4. Part 4: `pack_batch` and the module docstring of `uniform_fitting`

`backend/uniform_fitting.py:4-11`, after:

    """Uniform-layer fitting: packing a least-squares batch for the uniform layout, and the minimal-rank reduction of
    a uniform start.

    :py:func:`pack_batch` (with :py:func:`pack_sample`) packs the loop-invariant sample and data of a
    :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` ONCE, so the optimizers run fully packed -- no
    per-matvec pack/unpack: ``newton_cg(geom, oracle, pack_batch(batch, N), (x_sc[0], x_sc[1]))`` with a uniform
    geometry (:py:mod:`t3toolbox.backend.geometry`, ``from_point``). :py:func:`uniform_minimal` reduces a uniform start
    to minimal ranks, which the uniform manifold geometry requires.
    """

`pack_batch`'s docstring, after:

    """Pack a :py:class:`~t3toolbox.backend.fitting.LeastSquaresBatch` once for the uniform layout, by its kind and
    order: the input jets with ``pack_sample`` (each jet axis brought to its largest size across the modes by zero
    coefficients, which are data: an input jet is zero beyond its slots, O-45; the modes padded to ``N`` with zeros),
    and a probe ``y`` and full weight with :py:func:`~t3toolbox.backend.ut3_operations.pack_vectors`, the one packer
    (their arrays agree in everything but the edge). The entries index, the data and the full weight of apply and
    entries, the block weight, the scale and the drawn indices are not packed. ``N=None`` is the largest mode size
    of the sample. A batch whose members disagree in their mode sizes (the sample, ``y`` and the full weight of a
    mode), or an ``N`` below the largest, raises a ``ValueError``. A member already packed is kept as it is.

    >>> import numpy as np
    >>> from t3toolbox.backend.fitting import LeastSquaresBatch
    >>> from t3toolbox.backend.uniform_fitting import pack_batch
    >>> ww = [np.ones((2, 3, 4)), np.ones((1, 3, 5))]                 # jet sizes 2 and 1, W = (3,)
    >>> y = [np.zeros((3, 3, 4)), np.zeros((3, 3, 5))]                # F = (3,): order (2,)
    >>> packed = pack_batch(LeastSquaresBatch('probe', (2,), ww, y), 6)
    >>> print(packed.sample.shape, packed.y.shape)
    <to be pasted: (2, 2, 3, 6) (2, 3, 3, 6)>
    """

---

## 5. Part 5: the one-line references elsewhere

The fixed substitution in the transposes' docstrings (`tucker_tensor_train.py:3837-3839` and its siblings for apply
and entries; `uniform_tucker_tensor_train.py`; `manifold.py:765-772`, `:923-926`; `backend/probing.py:116-118` and the
apply and entries twins): "pass ``data_metric(r, weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is
``r`` itself with no weight and no scale" becomes "pass the kind's data metric, ``probe_data_metric(order, r,
block_weight, full_weight, minibatch_scale)`` (or ``batch.data_metric(r)``), which is ``r`` itself with no weights and
no scale"; "(the Gauss-Newton gradient ``J^T(M r)`` when ``c = data_metric(r, weight, minibatch_scale)``)" likewise.
The exact list of lines is the part's handover (a scan of `data_metric(` in prose); each changed line is shown there.

`t3toolbox/fitting.py:30-49`, the module docstring's list "Preparing data, and the jet helpers": "derivative data to
the library's Taylor coefficients: :py:func:`derivatives_to_taylor`, :py:func:`taylor_to_derivatives`,
:py:func:`derivative_block_weight_to_taylor`, and :py:func:`prepare_probe_derivative_data` /
:py:func:`prepare_apply_derivative_data`, which do the data, the validation data and the block weight in one call";
"the data metric: :py:func:`probe_data_metric`, :py:func:`apply_data_metric`, :py:func:`entries_data_metric`,
``gamma * block_weight**2 * full_weight**2 * r``, what a transpose is applied to for the gradient of a weighted fit (a
batch has it as the method ``batch.data_metric(r)``)"; the item on `blocks_to_weight` goes.
