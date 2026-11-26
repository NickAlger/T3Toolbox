# K1: the library prose that is rewritten, before and after

Implementer (the second), 2026-11-25, for `K1_DESIGN_NOTE.md` §10. Line numbers are in the working tree with K0 built.
Each text describes the code **after K1** (nothing about K2 to K7). Where part 0 builds a transitional text (the batch
before its two weights), it is the final text with today's sentences on the one weight kept; the part's handover
shows it. Doctest outputs marked `<to be pasted>` are filled from a run during the build, never by hand; the rest
of each example is the draft.

**Status (revision 2, after round 1):** part 0's texts (§1) and part 1's (§6) in full, the round-1 fixes in (the
missing-`sample` example, `flat_draw`'s and `_require_draw`'s texts, the docstrings of `mc_sgd` and `adam`, the frontend
guards, the engine's `full_weight` entry, `check_data`'s O3 sentence, the `ω` lines of `backend/optimizers.py`). Parts 2
to 5 stay listed changes for the design sign-off; each is completed in full and reviewed before its part is built.

**Status (2026-11-26, 01:25 CT):** part 2's texts (§3) in full, with the predicted red of part 2 (§3.11); not yet
reviewed. Parts 3 to 5 are still listed changes.

**Status of the first draft (2026-11-25), for the record:** part 0's texts (§1) in full; part 3's module docstring of `jets` and the
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

After (the rest of the docstring unchanged, its lines as they are; the one line that holds the new sentence's end
is split after "before any" so that the next line, "projection: ``Jp = 𝒥p``, ``g = 𝒥ᵀ(M r)`` and", ends where
today's does and the lines after it stay byte for byte):

    '''The least-squares oracle and its batch: what the optimizers ask of the data.

    A fit is ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` the residual of a sampling kind (probe, apply or entries) at
    a jet shape ``order``, and ``M`` the batch's data metric. :py:class:`LeastSquaresBatch` holds the measurements:
    their kind and jet shape (static), the sample, ``y``, the block weight and the full weight, the minibatch scale and
    the drawn indices; it checks itself against a model, draws from itself, counts its samples and applies its data
    metric. :py:class:`LeastSquaresOracle` is the objective's rule, with no fields: it answers what an optimizer asks
    at a frame or a corewise point (``f``, ``g``, ``Jp``, ``JtJp``, ``Kp``, the block sums, ...) on a batch, through
    one call of the engine
    :py:func:`~t3toolbox.backend.sampling_variables.sampling_variables`. What the oracle returns is bare, before any
    projection: ``Jp = 𝒥p``, ``g = 𝒥ᵀ(M r)`` and
    ``JtJp = 𝒥ᵀ M 𝒥p``, ... (unchanged)

### 1.2 `LeastSquaresBatch`, the class docstring (`backend/fitting.py:69-83`)

Before: quoted in the file (it says "arrays only", "Checked by `LeastSquaresOracle.check_batch`").

After (part 4's final text). **Part 0's interim text** (round 2, finding 15) keeps today's sentences on `y` and on
`weight` (each axis 1 or `z`'s size), with the kind, the order, keyword-only fields and `check` / `take` /
`n_measurements` as below; the `y` sentence ("exactly `z`'s shape; only its `C` axes may be 1") arrives in part 2 (the
engine) and `check`'s in part 3 (`check_data`), each with the part that makes it true:

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

    A batch made without its kind and jet shape fails at once (here on the missing third argument; with more
    positional arguments, on the kind):

    >>> LeastSquaresBatch(ww, np.zeros(10))                                   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: LeastSquaresBatch.__init__() missing 1 required positional argument: 'sample'

(the line to be pasted from the run on the project's Python, 3.11; `+IGNORE_EXCEPTION_DETAIL` covers the floor's
`__init__() missing ...`)
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

(Part 0's text is today's `check_batch` text with the batch's own kind and order: "the data and the weight against
the sample and the model's stack ``stack_shape`` (``check_data``, ``check_weight``: each of the type of ``z``, every
axis 1 or ``z``'s size)"; part 3 brings `y`'s sentence, part 4 the two weights'.)

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

### 1.9 `flat_draw` (`backend/optimizers.py:625-641`; O1 ruled)

    """The **default** minibatch draw: a uniform random subset of ``min(batch_size, n)`` of the ``n`` samples
    (``batch.n_measurements``), flat across the whole sample stack ``W``, as ``batch.take(indices)`` returns it: for
    the least-squares batch a batch with ``minibatch_scale = n / b`` and the ``drawn_indices``. Returns
    ``draw(rng) -> batch``, which calls ``rng.choice(n, size=min(batch_size, n), replace=False)`` once per draw. A user
    may pass **any** ``draw`` instead (slice ``W``, importance-sample, ...), returning a batch of the same kind and jet
    shape as the full batch: the backend does not check it (the frontend optimizers do). Host numpy indices; on a jax
    batch the gather runs on device (the optimizer never compiles the draw, only the step)."""

`_require_draw`'s docstring:

    """Structural, at entry: without a ``draw`` the batch needs ``take(indices)`` and the property
    ``n_measurements`` (an int), the default draw (:py:func:`flat_draw`); else a ``ValueError``, which says so when
    ``n_measurements`` is a method (a batch written before it became a property)."""

The comments of the `draw` parameter of `mc_sgd` and `adam` (`:1233`, `:1283`): `# custom draw(rng) -> a batch;
None = flat_draw(batch, batch_size)`. Their docstrings (finding r1-7c): `mc_sgd` (`:1243`) "(``draw(rng)``; by
default :py:func:`flat_draw`, which needs an oracle with ``take``, else a ``ValueError`` at entry)" becomes
"(``draw(rng)``, which returns a batch of the full batch's kind and jet shape; by default :py:func:`flat_draw`, which
needs a batch with ``take`` and ``n_measurements``, else a ``ValueError`` at entry; a draw's batch is not checked
here, the frontend checks it)"; `adam` (`:1295`) "(the regularizer unscaled; an oracle without ``take`` needs a
``draw``)" becomes "(the regularizer unscaled; a batch without ``take`` and ``n_measurements`` needs a ``draw``, whose
batches, of the full batch's kind and jet shape, are not checked here)".

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
- `t3toolbox/optimizers.py:343-344`, the frontend `mc_sgd`'s docstring (and `adam`'s, which points to it): "A custom
  ``draw(rng)`` returns a batch (ragged; it is packed for a uniform ``x0``)" becomes "A custom ``draw(rng)`` returns a
  batch of the full batch's kind and jet shape, checked on each draw (another raises a ``ValueError``); ragged, it is
  packed for a uniform ``x0``".
- `_frontend_draw`'s docstring (`t3toolbox/optimizers.py:117-118`): "Each batch a user's draw returns is compared with
  the full batch, its kind and its jet shape (structural, two tuples per draw: a draw of another kind or order would
  fit another objective), and for a uniform x0 packed (not re-checked each step)."
- The frontend `newton_cg`: the comment of `val_batch` (`:450`) `# optional validation batch, of batch's kind and
  order -> a train|val table`; its docstring (`:529-530`) "pass ``val_batch`` (a batch for the same oracle)" becomes
  "pass ``val_batch`` (a batch of the same kind and jet shape as ``batch``; another raises a ``ValueError`` at entry)".

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

## 3. Part 2: the engine, its steps, the private core of `jets`, the oracle's bridge (in full)

Written 2026-11-26 by the third implementer, at `31e3d100`; each "before" is that tree's text, each "after" describes
the code **after part 2** (the forms of D1 unchanged, masking by a product until K5). Where a paragraph changes by a
sentence, the lines shown are the changed ones and their neighbours as anchors; every other line stays byte for byte.
The note's §4.2 table is the rule the engine enforces, and the texts say no more than it.

### 3.1 `sampling_variables`: the signature's shape comments (`sampling_variables.py:1687-1690`)

Before:

```python
        y:                 typ.Optional[typ.Any] = None,     # the data: Y*, of the type of z, each axis 1 or z's size:
                                                             #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        weight:            typ.Optional[typ.Any] = None,     # omega, of the type of z likewise (no K); None = ones
```

After (the note's §4.2, verbatim):

```python
        y:                 typ.Optional[typ.Any] = None,     # the data: Y*, z's shape, only its C axes may be 1:
                                                             #   probe len=d, elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C
        block_weight:      typ.Optional[NDArray] = None,     # one number per block, exact: probe (d,)+F; apply F;
                                                             #   entries (); None = ones
        full_weight:       typ.Optional[typ.Any] = None,     # one per measured number: z's shape, its W and C axes 1 or
                                                             #   full, never K; None = ones
```

`computed_names` (`:1587`), before `        weight:            typ.Optional[NDArray] = None,     # not read`; after:

```python
        block_weight:      typ.Optional[NDArray] = None,     # not read
        full_weight:       typ.Optional[typ.Any] = None,     # not read
```

and in its docstring (`:1600`) "The sample, the weight, the jet sizes and every value" becomes "The sample, the two
weights, the jet sizes and every value" (that line only; the next line is unchanged).

### 3.2 `sampling_variables`: the parameter entries (`:1743-1752`)

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
        The data, of the type of ``z``, with ``z``'s shape exactly but that each ``C`` axis may be 1 (data shared by
        a stack): probe ``len=d`` (a tuple, a list, or one array whose first axis is the mode),
        ``elm_shape=F+W+C+(Ni,)``; apply ``shape=F+W+C``; entries ``shape=W+C``. The ``d`` arrays of a probe ``y``
        have one pattern of size-1 ``C`` axes. Needed for ``r`` and what reads it. On the uniform layout probe data
        may come ragged (packed here) or packed, ``(d,)+F+W+C+(N,)``.
    block_weight: NDArray, optional
        One number per block, exactly the block shape: probe ``shape=(d,)+F`` (a mode at a slot), apply ``F`` (a
        slot), entries ``()`` (all of the data); one array (a number for entries), the same on both layouts, never a
        tuple or a list (a ``TypeError``). It sits on the leading axes of ``r`` (row ``i`` on the jet axes of mode
        ``i``), so it needs no placement against a stack of directions. ``None`` is ones.
    full_weight: sequence of NDArray or NDArray, optional
        One number per measured number, of the type of ``z``: ``z``'s shape, but that each ``W`` and each ``C`` axis
        may be 1 (the same for every sample, or every element of the stack); never the stack of directions ``K`` (the
        engine places it against ``Jp``). For probe the forms of ``y`` (on the uniform layout ragged, packed here, or
        packed ``(d,)+F+W+C+(N,)``), the ``d`` arrays with one pattern of size-1 ``W`` and ``C`` axes. ``None`` is
        ones. The two weights multiply: the rows that read them apply ``block_weight**2 * full_weight**2`` (the data
        metric at ``gamma = 1``).

### 3.3 `sampling_variables`'s docstring, "The engine checks what it needs" (`:1708-1710`)

Before (the changed lines, `:1708-1710`):

        ``w[0]``, and every direction the ``K + C`` of the first (a size-1 axis would broadcast silently); ``y`` and the
        weight are of the type of ``z``, each array of ``z``'s rank and each axis 1 or ``z``'s size (for probe exactly
        ``d`` arrays), or a ``ValueError``. It does not check the mode sizes of the sample against the frame's or the jet

After:

        ``w[0]``, and every direction the ``K + C`` of the first (a size-1 axis would broadcast silently); ``y`` has
        ``z``'s shape but for ``C`` axes of 1, the full weight ``z``'s shape but for ``W`` and ``C`` axes of 1, the
        ``d`` arrays of a probe ``y`` or full weight agree in everything but the edge, and the block weight has the
        block shape exactly, or a ``ValueError`` (a block weight given as a tuple or a list: a ``TypeError``). It does
        not check the mode sizes of the sample against the frame's or the jet

(The next line, "sizes (the checkers of :py:mod:`t3toolbox.backend.jets`), and nothing numerical: ...", is unchanged;
the inserted words force only the four lines shown.)

### 3.4 The module docstring of `sampling_variables`

**`reads` (`:24-26`).** Before: "an argument of the call (``w``, the sample; ``y``; / ``weight``; ``Vc``)." After, the
same two lines with ``weight`` replaced:

    makes; a store of the frame (``D``, ``U``, ``L``, ``R``); an argument of the call (``w``, the sample; ``y``;
    ``block_weight``; ``full_weight``; ``Vc``). A sweep's carry is its first argument and is not listed; its first

(measured: 112 and 114 columns; no other line moves.)

**The table rows** (`:54-56`, `:84`, `:93`, `:183`; the columns, `Type` included, unchanged; only the `reads` cell):

    | `f` | whole | `f_step` | `r`, `block_weight`, `full_weight` |  | no |  |
    | `r_block_sumsq` | whole | `r_block_sumsq_step` | `r`, `block_weight`, `full_weight` |  | no |  |
    | `w_tilde` | whole | `w_tilde_step` | `r`, `block_weight`, `full_weight` |  | no |  |
    | `Jp_norm_sq` | whole | `Jp_norm_sq_step` | `Jp`, `block_weight`, `full_weight` |  | yes |  |
    | `_Jp.w_tilde` | whole | `w_tilde_step` | `Jp`, `block_weight`, `full_weight` |  | yes |  |

(the last row twice, `:93` and `:183`, as today). The `Rule` rows in code (`:385-387`, `:415`, and the twin's) read
the same tuples. None of the four fits the table's 31-character `reads` column (37 and 38 characters), so the four
rows take the table's existing overflow form (`:393-406`): the row stops after `reads`, and the later columns go on a
continuation line in their own columns. No other row moves. As they will stand:

```python
    Rule('f',              ('whole', 'f_step'),             ('r', 'block_weight', 'full_weight'),
                                                                                           _NONE,             False),
    Rule('r_block_sumsq',  ('whole', 'r_block_sumsq_step'), ('r', 'block_weight', 'full_weight'),
                                                                                           _NONE,             False),
    Rule('w_tilde',        ('whole', 'w_tilde_step'),       ('r', 'block_weight', 'full_weight'),
                                                                                           _NONE,             False),
```

and at `:415`:

```python
    Rule('Jp_norm_sq',     ('whole', 'Jp_norm_sq_step'),    ('Jp', 'block_weight', 'full_weight'),
                                                                                           _NONE,             True),
```

**The held `Jp` paragraph (`:121-123`).** Before:

    **holding any vector of** ``Jp``'s **shape as** ``Jp`` **gives** ``J^T`` **of that vector** (the seed is the data metric
    of the held vector at ``gamma = 1``, ``weight**2`` times it, turned; with no weight, the vector itself). The transposes
    take this route for ``c``, an element of ``Y`` with the shape of ``z`` (they pass no weight), with a stack ``K`` read

After (the middle line only):

    of the held vector at ``gamma = 1``, ``block_weight**2 * full_weight**2`` times it, turned; with no weight, the
    vector itself). The transposes

(the inserted words force the middle line to break; the third line is unchanged, so the paragraph keeps its lines
after it byte for byte: the build joins "The transposes" to the line it breaks into.)

**The uniform layout (`:308-312`).** Before:

    ``L`` masks, never the point's (a superset at TT slack); every Tucker-shaped input, the sample and probe's ``y``
    with the shape mask on the mode axis ``N`` (the sample's jet axes untouched: their slots are coefficients; a ``y``
    with an edge of 1 is broadcast to ``N`` by its mask, so that its value cannot reach the padding of ``r``), and probe's
    weight when its edge is ``N`` (an edge of 1 has no padding, and multiplies padding that is zero); a held
    variable with its table's masks (padding is not a value). The start values carry ones on the real slots of their end

After:

    ``L`` masks, never the point's (a superset at TT slack); every Tucker-shaped input, the sample and probe's ``y``
    with the shape mask on the mode axis ``N`` (the sample's jet axes untouched: their slots are coefficients), and
    probe's full weight likewise (its edge is always ``N``; the block weight has no edge and is not masked); a held
    variable with its table's masks (padding is not a value). The start values carry ones on the real slots of their end

**`HeldVariables`' docstring (`:885-888`).** The new words make `:885` 148 columns, so the paragraph is re-filled
from `:885` to its end (`:884` and the lines before it unchanged). Before:

    ``J^T`` of ``weight**2 c``: this is how the transposes compute ``J^T c`` (they pass no weight), ``K`` read off
    ``c``. Where jax is importable it is a registered pytree: the filled slots are the leaves, the set of filled slots
    is the static structure. It refuses ``==`` and is unhashable, as every array-carrying class of the library is:
    compare the slots you mean.

After:

    ``J^T`` of ``M c``, ``M = block_weight**2 * full_weight**2``: this is how the transposes compute ``J^T c`` (they
    pass no weight), ``K`` read off ``c``. Where jax is importable it is a registered pytree: the filled slots are the
    leaves, the set of filled slots is the static structure. It refuses ``==`` and is unhashable, as every
    array-carrying class of the library is: compare the slots you mean.

**`_check_stacks`' docstring (`:1111-1114`).** Today's `:1112` is already 151 columns; the new sentence is longer, so
the paragraph is re-filled from `:1111` to its end (`:1110` and before unchanged). Before:

    operand. ``y`` and the weight are checked by ``_check_data_space`` (each axis 1 or ``z``'s size: there a size
    of 1 is meant), the jet sizes by the checkers of :py:mod:`t3toolbox.backend.jets`. Its cost is about ``5d`` comparisons of short integer tuples per
    call, at trace time only under ``jit``. (A removable check, pending O-18: delete this function, its one call and
    its test class.)'''

After:

    operand. ``y`` and the two weights are checked by ``_check_y``, ``_check_full_weight`` and ``_check_block_weight``
    (where they allow a size of 1, it is meant), the jet sizes by the checkers of :py:mod:`t3toolbox.backend.jets`. Its
    cost is about ``5d`` comparisons of short integer tuples per call, at trace time only under ``jit``. (A removable
    check, pending O-18: delete this function, its one call and its test class.)'''

### 3.5 The engine's checkers (private; `_check_data_space`, `:1172-1199`, becomes three sharing one reader)

    def _data_space_arrays(setting, name, value, edges):
        '''The arrays of a data-space object (``y``, the full weight) and the shape each must have: for probe exactly
        ``d`` arrays (a tuple, a list, or one array whose first axis is the mode), each ``F + W + C`` and the mode
        edge (``N_i`` ragged, ``N`` packed); for apply one array ``F + W + C``, for entries ``W + C``; a tuple or a
        list for apply or entries raises. Structural, static.'''

    def _check_y(setting, value, edges):
        '''``y``: ``z``'s shape exactly, each ``C`` axis 1 or ``C``'s size; the ``d`` arrays of a probe ``y`` with one
        pattern of size-1 ``C`` axes. Without it ``z - y`` would right-align a ``y`` of another rank: a ``y`` with no
        ``C`` on a stacked model, with ``W`` of the size of ``C``, would be read along ``C`` with no error. A
        ``ValueError`` naming the mode and the shapes.'''

    def _check_full_weight(setting, value, edges):
        '''The full weight: ``z``'s shape, each ``W`` and ``C`` axis 1 or its size, the jet axes and the edge exact;
        the ``d`` arrays of a probe full weight with one pattern of size-1 ``W`` and ``C`` axes (they may differ only
        in the edge; the message names ``np.broadcast_to``). Never the stack of directions ``K``: the engine places
        it (``_with_K``). A ``ValueError``.'''

    def _check_block_weight(setting, block_weight):
        '''The block weight: one array of exactly the block shape, ``(d,) + F`` (probe), ``F`` (apply), ``()``
        (entries); a tuple or a list is a ``TypeError`` (it is one array on both layouts). A ``ValueError`` otherwise,
        naming the shape it has and the block shape.'''

### 3.6 `sampling_steps.py`: the whole steps' comment (`:909-913`)

Before:

    # A whole step takes the kind as a static too, a string: the probe kind's type of z is one array per mode with a mode
    # edge, the apply kind's one array with neither. A data-space object (y, the weight, r, Jp) is of the type of z: each
    # array of z's rank, each axis 1 or z's size (the full form; the engine checks y and the weight on entry). _modes gives
    # the arrays of a value of the type of z, each with its weight's array and the number of mode-edge axes after its
    # stacks; _with_K places the stack of directions K in a weight, which never carries it.

After (the first line unchanged):

    # A whole step takes the kind as a static too, a string: the probe kind's type of z is one array per mode with a mode
    # edge, the apply kind's one array with neither. A data-space object (y, the full weight, r, Jp) is of the type of z:
    # each array of z's rank, each axis z's size or, on a stack axis, 1 (the engine checks y and the weights on entry);
    # the block weight is one number per block, on the leading axes (the mode and the jet axes). _modes gives the arrays
    # of a value of the type of z, each with its block row, its full weight's array and the number of mode-edge axes
    # after its stacks; _with_K places the stack of directions K in the full weight, which never carries it (the block
    # weight needs no placement: K comes after W, behind its axes). Every product of a weight is jets._metric's.

### 3.7 `_modes`, `_with_K`, `_weighted_sum` (`sampling_steps.py:915-962`)

Headers: the note's §4.3, verbatim. The docstrings (`_modes` has none today; one line is added):

    def _modes(...):
        '''The arrays of a value of the type of ``z`` (one per mode for probe, packed or not), each with its row of the
        block weight (``block_weight[i]`` for probe, the table for apply, the number for entries; ``None`` for ones),
        its full weight's array (``None`` for ones) and the number of mode-edge axes after its stacks.'''

`_with_K`, before:

    '''The weight placed against a variable that carries the stack of directions ``K`` (``Jp`` and the adjoint it
    seeds): ``lens.K`` axes of size 1 inserted at index ``lead + m + len(W)``, ``lead`` 1 for a mode-leading array
    (a probe weight packed, or given as one array) and 0 for each array of a tuple or a list and for apply and
    entries. A weight is constant over the directions; ``K`` is the engine's to place, never the caller's.'''

After (lines 1, 3 and 4 change; line 2 is byte for byte):

    '''The full weight placed against a variable that carries the stack of directions ``K`` (``Jp`` and the adjoint it
    seeds): ``lens.K`` axes of size 1 inserted at index ``lead + m + len(W)``, ``lead`` 1 for a mode-leading array
    (a probe full weight packed, or given as one array) and 0 for each array of a tuple or a list and for apply and
    entries. A weight is constant over the directions; ``K`` is the engine's to place, never the caller's. The block
    weight is not placed: it sits on the leading axes, before ``W`` and ``K``.'''

`_weighted_sum`, before:

    '''``<value, M value>`` at ``gamma = 1``, summed over the modes, the slots, the samples and the mode edge, keeping
    the stacks after ``W`` (``[K] + C``). The metric is ``data_metric``, the library's one ``omega**2``.'''

After (the first line unchanged):

    '''``<value, M value>`` at ``gamma = 1``, summed over the modes, the slots, the samples and the mode edge, keeping
    the stacks after ``W`` (``[K] + C``). ``M = block_weight**2 * full_weight**2``, formed by ``jets._metric``, the
    library's one place for it.'''

Its return comment: `# sum of value * M value at gamma = 1 over F, W and the edge` (the note's §4.3).

### 3.8 The four steps

Each header: today's with `weight`'s row replaced by the two rows of the note's §4.3 (`block_weight: ... # one per
block: probe (d,)+F, apply F, entries (); None = ones`, `full_weight: ... # of the type of z, no K (placed here); None
= ones`; for `f_step` and `r_block_sumsq_step`, whose value has no `K`, the full weight's comment is `# of the type of
z; None = ones`). The column of the parameter names widens to `block_weight:` (14), and the four rows of each header
are re-aligned with it (a header that changes keeps no column of the old one: house form).

`Jp_norm_sq_step`, before / after (lines 1 and 2 change):

    '''The weighted sum of squares of the direction's output, ``<Jp, M Jp>`` with ``M`` the data metric at
    ``gamma = 1`` (``data_metric``), summed over the modes, the slots, the samples and the mode edge, keeping
    ``K + C`` (no half).'''

    '''The weighted sum of squares of the direction's output, ``<Jp, M Jp>`` with ``M = block_weight**2 *
    full_weight**2`` (the data metric at ``gamma = 1``), summed over the modes, the slots, the samples and the mode
    edge, keeping ``K + C`` (no half). The full weight is placed against ``K`` (``_with_K``); the block weight is not.'''

`w_tilde_step`, before:

    '''The seed of the adjoint: the data metric at ``gamma = 1``, ``data_metric(r, weight, None)``, then the turn,
    a flip of every jet axis. The probe kind's is one array per mode (packed in, packed out on the uniform layout);
    the apply kind's one number per sample and slot, which sits on the ones that close bond ``d``. Seeded by ``Jp``
    (``_Jp.w_tilde``), ``r`` carries ``K`` and the weight is placed against it (``_with_K``).'''

After (the formula is longer than the old call, so the paragraph is re-filled from its first line; the words of
today's lines 2 to 4 are kept, and every line is within 120 columns):

    '''The seed of the adjoint: the data metric at ``gamma = 1``, ``block_weight**2 * full_weight**2 * r``, then
    the turn, a flip of every jet axis. The probe kind's is one array per mode (packed in, packed out on the uniform
    layout); the apply kind's one number per sample and slot, which sits on the ones that close bond ``d``. Seeded by
    ``Jp`` (``_Jp.w_tilde``), ``r`` carries ``K`` and the full weight is placed against it (``_with_K``). The weights
    are applied before the turn: slot ``k`` of the seed is ``M r`` at slot ``order - k``.'''

`f_step`, before / after (the first line only):

    '''The misfit: ``1/2 <r, M r>`` with ``M`` the data metric at ``gamma = 1`` (``data_metric``), summed over the

    '''The misfit: ``1/2 <r, M r>`` with ``M = block_weight**2 * full_weight**2`` (``gamma = 1``), summed over the

`r_block_sumsq_step`, before / after (the first line only):

    '''``<r, M r>`` per block, with ``M`` the data metric at ``gamma = 1`` (``data_metric``): per mode and slot for

    '''``<r, M r>`` per block, with ``M = block_weight**2 * full_weight**2`` (``gamma = 1``): per mode and slot for

`r_step`: code unchanged; its `y` comment `# the data, of the type of z: each axis 1 or z's size (checked on entry)`
becomes `# the data, of the type of z: z's shape, C axes 1 or full (checked on entry)`; its docstring's second
sentence, before "``y`` has ``z``'s rank, each axis 1 or ``z``'s size, so ``z - y`` places it by position: a size of 1
is / the same datum along that axis (shared data on a stack has size-1 ``C`` axes).", after "``y`` has ``z``'s shape
but for ``C`` axes of 1, so ``z - y`` places it by position: data shared by a stack has size-1 ``C`` axes." (the two
lines re-filled; the first sentence unchanged).

### 3.9 `jets._metric` (new, private; the note's §4.3 header)

    '''``gamma * (block_X**2 * full_X**2) * r_X`` for one array of ``r``: the data metric's one formula, the only
    place in the library where the weights are squared and multiplied. ``block_X`` is the block weight's part for
    ``r_X`` (a mode's row ``F``, the whole table ``(d,) + F`` against a packed ``r_X``, the table ``F`` for apply, the
    number for entries), reshaped here onto ``r_X``'s leading axes (left-aligned: its axes are the first of
    ``r_X``'s, every axis after them 1); ``full_X`` has ``r_X``'s rank, each axis 1 or ``r_X``'s size, and is read
    by position. **The dtype:** the result's dtype is the lowest floating precision of ``r_X`` and every weight
    given (the absorbing rule; the default float when none is floating; on jax, capped at jax's own float); ``r_X``
    and each given weight are cast to it, and each square is formed in it; the product ``block**2 * full**2`` only
    when both are given; then ``* r_X``; ``gamma`` last, cast to that dtype (it does not raise it). ``None`` is ones
    for either weight and 1 for ``gamma``. It checks nothing: its callers check (the engine on entry, the public
    metrics of :py:mod:`t3toolbox.backend.jets`). Values in a packed ``r_X``'s padding come back multiplied by the
    weights; masking them is the engine's.'''

### 3.10 The oracle's bridge (`backend/fitting.py:561`, `LeastSquaresOracle.query`)

Before:

```python
            args = dict(order=batch.order, y=batch.y, weight=batch.weight, variations=variations,
                        core_directions=core_directions, Vc=Vc, held=held)
```

After:

```python
            # K1's bridge between part 2 and part 4: the batch still has one weight, of the full form, and the engine
            # takes two; it goes to the engine as the full weight (part 4 gives the batch both fields and replaces it)
            args = dict(order=batch.order, y=batch.y, block_weight=None, full_weight=batch.weight,
                        variations=variations, core_directions=core_directions, Vc=Vc, held=held)
```

(The comment above it, "one set of arguments for the query and the call: ...", stays as it is, before the new one.)

### 3.11 What part 2 turns red, and why (as read from the code at `31e3d100`; the list itself is recorded by a run)

The engine's rule becomes stricter in three ways (a `y` exact but `C`; a full weight exact on `F` and the edge; a
block weight of the block shape), and the one library caller that passes a weight, the oracle, sends the batch's one
weight as the full weight. So red are the calls that reach the engine with a form the new rule refuses, and only
those: everything refused raises a `ValueError` at the engine's entry, never a wrong number (an apply or entries
`blocks_to_weight` output has only `F` and size-1 `W`/`C` axes, a legal full weight with today's metric bitwise).

1. **Every probe fit whose batch weight is `blocks_to_weight`'s output** (`(d,)+F+(1,)*len(W+C)+(1,)`: an edge of 1).
   Through the oracle: `tests/_fits.fit` and its callers with a probe block weight, `old_vocabulary` probe rows with a
   weight, every translated old-form probe call with `omega` (`t3toolbox/optimizers.py:217`, `:237` build it with
   `blocks_to_weight`).
2. **Every probe derivative fit, weighted or not**: with no weight `prepare_derivative_data` returns the `k!` table
   placed by `blocks_to_weight` (`jets.py:1178`, in `prepare_derivative_data`), mode-leading with an edge of 1. So `tests/_fits.derivative_fit`
   with `kind='probe'`, `old_vocabulary('probe_derivatives', ...)`, every translated `probe_derivatives` call in the
   tests (`TestTheTranslation`'s probe rows among them), and the probe rows of the derivative sweeps.
3. **Every test whose `y` has a size-1 `W` (or a structural size-1 axis) and reaches the engine**: through the oracle,
   `test_block_sums.TestSharedData`'s "shared by the samples" case (rewritten in part 4); and
   `test_block_sums.py::TestAgainstDenseSums::test_engine_and_oracle`, whose one subtest holds an engine call and the
   oracle's queries: its `y` forms "shared by W" and "edge 1" and its weights "per sample, a zero" (jet axes of 1) and
   "per block" (`blocks_to_weight`, an edge of 1 for probe) are refused from part 2. Part 2 rewrites its engine half,
   which is asserted first in the subtest; its oracle half, through the bridge, stays red until part 4 (the
   expected-red record shows it per subtest). Directly at the engine, nothing else after part 2's own rewrites.
4. **Hand-built full-form weights with a structural axis of 1** (round 2, finding 13), probe or apply, that reach the
   engine through the oracle: e.g. `tests/backend/test_optimizer_display.py:439` (`np.ones((1,) * m + (40, 1))`: a
   jet axis and an edge of 1),
   and any per-slot or per-mode weight written by hand in the oracle tests (`test_least_squares_oracle.py`,
   `test_oracle_data_space.py`, `test_block_sums.py`'s oracle half, `backend/test_curvature.py`,
   `test_uniform_curvature.py`, `test_sharing.py`, `test_fitting.py`): each is part 4's, and the run lists which.
5. **Engine calls in tests outside the note's part-2 list that pass `weight=`** would raise a `TypeError` (the keyword
   is gone). Measured at `31e3d100` (`grep` of `sampling_variables(`/`computed_names(` with `weight=` in `tests/`):
   besides the note's list, `tests/test_contractions_sharding.py` (2, `test_g_and_Kp_shard_over_W`) and
   `tests/test_sampling_graph.py` (2, `:462`, `:468`). **I add both to part 2's rewrite** (keyword only), so they are
   not red. No library module but the oracle passes a weight to the engine (`probing`, `apply`, `entries`,
   `ut3_sampling`, `utv_sampling` pass none).

Not red: unweighted fits of every kind but probe derivatives (the engine's arithmetic with no weight is unchanged);
apply and entries fits with a `blocks_to_weight` weight (legal full weights, the same numbers); every engine test
part 2 rewrites. Per the note, the red list is recorded per subtest from the `-v` log, each failing subtest classified
by its label, and part 4's green run is the first check of part 2's engine on the cases a red node hides.

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

---

## 6. Part 1: the dense reference (`backend/sampling_definitions.py`), in full

The functions keep their structure; the reference reads the two weights by its own code (no import of `jets`), and
its rule is the one of note §4.1 (no `C`, no `K`: `y` exact; the full weight's `F` and edge exact, its `W` axes 1 or
full; the block weight exact).

### 6.1 The module docstring

`:85-91`, before:

    **The seed, and the turn.** The data ``y`` and the residual ``r = z - y`` are of the type of ``z``, in forward
    order. The seed is the residual made into a dual jet: weighted, then turned, a flip of every jet axis,
    ``w_tilde[k] = weight[order - k]**2 * r[order - k]`` (``k`` a multi-index, the weight read at the same entry as the
    residual). As a polynomial in ``s`` the seed is
    ``sum_k w_tilde[k] s^k``, so ... Pass ``y`` (and ``weight``) for the seed
    of the residual, or ``w_tilde`` for a seed of your own; ``w_tilde`` wins where both are given.

After:

    **The seed, and the turn.** The data ``y`` and the residual ``r = z - y`` are of the type of ``z``, in forward
    order. The seed is the residual made into a dual jet: weighted, then turned, a flip of every jet axis,
    ``w_tilde[k] = b[order - k]**2 * f[order - k]**2 * r[order - k]`` (``k`` a multi-index; ``b`` the block weight's
    row of the mode, read at the slot; ``f`` the full weight, read at the same entry as the residual). As a polynomial
    in ``s`` the seed is ``sum_k w_tilde[k] s^k``, so ... Pass ``y`` (and the weights) for the seed of the residual, or
    ``w_tilde`` for a seed of your own; ``w_tilde`` wins where both are given.

`:108-115`, before:

    - ``y``: the data, of the type of ``z``: probe ``len=d``, ``elm_shape=F+W+(Ni,)``; apply ``F+W``; entries ``W``.
    - ``weight``: ``omega``, of the type of ``z`` too, entering squared; ``None`` is ones.

    ``y`` and the weight are **data-space objects**: of the type of ``z`` (for probe exactly ``d`` arrays, a tuple, a
    list, or one array whose first axis is the mode), each array of the rank of ``z``'s, each axis of size 1 or of
    ``z``'s size. A size of 1 means the same value along that axis: a weight per slot has size-1 sample and edge axes,
    one per sample size-1 jet and edge axes, one per entry ``z``'s shape. They are read by position, with no reshape, and
    any other form raises a ``ValueError`` (a weight of one value per mode and slot, ``(d,) + F``, is not of this type).

After:

    - ``y``: the data, exactly ``z``'s shape: probe ``len=d``, ``elm_shape=F+W+(Ni,)``; apply ``F+W``; entries ``W``.
    - ``block_weight``: one number per block, a mode at a slot (probe, ``shape=(d,)+F``), a slot (apply, ``F``) or all
      of the data (entries, ``()``), exactly that shape; it enters squared, on the leading axes (row ``i`` on the jet
      axes of mode ``i``). ``None`` is ones.
    - ``full_weight``: one number per measured number, of the type of ``z``, with ``z``'s shape except that each
      sample axis ``W`` may be 1 (the same for every sample); it enters squared, entry by entry. ``None`` is ones.

    ``y`` and the full weight are **data-space objects**, of the type of ``z``: for probe exactly ``d`` arrays (a
    tuple, a list, or one array whose first axis is the mode), each of the rank of ``z``'s array. ``y`` has ``z``'s
    shape exactly; the full weight has its jet axes and its mode edge exactly, its sample axes 1 or full. A size of 1
    elsewhere (a jet axis, the edge) raises a ``ValueError``: a weight per slot or per mode is a block weight. They are
    read by position, with no reshape. The two weights multiply: the misfit is ``1/2 sum b**2 f**2 r**2``.

### 6.2 `w_tilde`

The docstring's second paragraph, after:

    The seed is what the adjoint (tilde) variables are built on. With ``y`` given it is the residual ``r = z - y``
    made into a dual jet: weighted, then turned, a flip of every jet axis (``k`` a multi-index of the box ``F``),

        ``w_tilde[i][k] = b[i][order - k]**2 * f[i][order - k]**2 * r[i][order - k]``   (probe; no mode index for the
                                                                                     others; b a number for entries),

    the block weight ``b`` read at the slot, the full weight ``f`` at the same entry as the residual (positionally; a
    sample axis of 1 is constant).

    The turn is what makes its pairing with a jet the coefficient of the corner ``s_1^n_1 ... s_m^n_m`` of their
    product: ``sum_k w_tilde[order - k] * a[k] = sum_k b[k]**2 f[k]**2 r[k] a[k]``, the weighted inner product with
    the residual. With no jets there is nothing to turn. With ``w_tilde`` given, that seed is returned as it is.

The example, after (the frame, `ww`, `pp`, `line`, `y` as today):

    >>> block = 0.5 + rng.random((3, 3))                                   # (d,) + F: one number per mode and slot
    >>> args = dict(order=(2,), y=y, block_weight=block)
    >>> r1 = sdef.r('probe', frame, line, 1, **args)
    >>> w_tilde1 = sdef.w_tilde('probe', frame, line, 1, **args)
    >>> print(w_tilde1.shape)                                              # F + W + (N1,)
    <to be pasted: (3, 7, 5)>
    >>> print(bool(np.allclose(w_tilde1[0], block[1, 2] ** 2 * r1[2])))    # slot 0 holds order 2: turned
    <to be pasted: True>
    >>> print(bool(np.allclose(w_tilde1[2], block[1, 0] ** 2 * r1[0])))
    <to be pasted: True>

(`rng.random((3, 3))` draws the same nine numbers as today's `rng.random((3, 3, 1, 1))`, so the rest of the example's
stream is unchanged; the apply example after it, with no weight, is unchanged.)

### 6.3 The tilde variables' examples (`xi_tilde`, `mu_tilde`, `nu_tilde`, `eta_tilde`, `nu_L_tilde`, `eta_D_tilde`)

Each has `seed = dict(y=y, weight=0.5 + rng.random((3, 1, 1)))  # the seed of the residual z - y`, a weight per mode
(`(d,) + W + edge` of 1, illegal now). After:

    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode

(the same three draws; the outputs are shapes and `True`, re-pasted from a run).

### 6.4 `JtJp` and `Jp_norm_sq`

`JtJp`'s first line, after: "``JtJp``: the Gauss-Newton product. ``Jp`` made into a seed (weighted by both weights,
then turned), and the transpose applied to it: ...". Its example's weight `0.5 + rng.random((3, 7, 1))` ("a weight per
mode and sample", an edge of 1, illegal now) becomes a full weight written out on the edge:

    >>> per_sample = 0.5 + rng.random((3, 7, 1))                          # one number per mode and sample ...
    >>> full = [np.broadcast_to(per_sample[i], (7, n)) for i, n in enumerate(N)]   # ... written out on the edge
    >>> p = dict(variations=(V, H), full_weight=full)

(the rest as today, outputs pasted). `Jp_norm_sq`'s first line: "``Jp_norm_sq``: ``<Jp, M Jp>`` with ``M = b**2
f**2``, the squared weighted norm of the direction's output, summed as ``f`` is (without the half)"; its example the
same rewrite, its check `sum(np.sum(full[i] ** 2 * Jp[i] ** 2) for i in range(3))`.

### 6.5 `f`

First line, after: "``f``: the misfit, ``1/2 <r, M r>`` with ``M = b**2 f**2``: half the sum, over the modes, the
orders, the samples and the mode edge, of the squared weighted residual, slot by slot. Needs ``y``." The example, after
(the setting as today):

    >>> full = [0.5 + rng.random((7, n)) for n in N]                      # the full weight: one per entry
    >>> r = sdef.r('probe', frame, ww, y=y)
    >>> f = sdef.f('probe', frame, ww, y=y, full_weight=full)
    >>> print(bool(np.allclose(f, 0.5 * sum(np.sum(full[i] ** 2 * r[i] ** 2) for i in range(3)))))
    <to be pasted: True>

    One value per mode is a block weight, ``(d,)`` with no jets. Given as a full weight with size-1 sample and edge
    axes it raises: the edge is part of the measurement, never 1:

    >>> per_mode = np.array([1.0, 2.0, 3.0])
    >>> sdef.f('probe', frame, ww, y=y, full_weight=per_mode.reshape(3, 1, 1))     # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the full weight (mode 0) has shape (1, 1); ... the mode edge exactly ...
    >>> print(bool(np.allclose(sdef.f('probe', frame, ww, y=y, block_weight=per_mode),
    ...                        0.5 * sum(per_mode[i] ** 2 * np.sum(r[i] ** 2) for i in range(3)))))
    <to be pasted: True>

(The draw `rng.random(3)` of today's refused example goes; the reader sees the refusal of the old form and the right
form side by side.)

### 6.6 `r_block_sumsq`

First line, after: "``r_block_sumsq``: ``<r, M r>`` in each block, ``M = b**2 f**2``, the weighted sum of squares of
the residual (the sum of squares with no weight). A block is ... (as today)". The weighted example, after:

    Weighted, one full weight per sample (written out on the slots and the edge): a zero drops the sample from every
    block.

    >>> full = [np.ones((3, 7, n)) for n in N]                            # F + W + (Ni,)
    >>> for omega in full:
    ...     omega[:, 3, :] = 0.0
    >>> sums = sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y, full_weight=full)
    >>> print(bool(np.allclose(sums[1, 2], np.sum(np.delete(r[1][2], 3, axis=0) ** 2))))
    <to be pasted: True>

    A block weight scales its block: ``block[i, k]**2`` times the block's sum.

    >>> block = 0.5 + rng.random((3, 3))
    >>> print(bool(np.allclose(sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y, block_weight=block),
    ...                        block ** 2 * sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y))))
    <to be pasted: True>

### 6.7 The private helpers (shown here because the reference is part 1's subject)

- `_check_data_space(what, i, shape, of_shape)` becomes `_check_data_space(what, i, shape, of_shape, may_be_one)`,
  `may_be_one` the positions (the sample axes) where a 1 is allowed (`()` for `y`); its docstring: "The rule of a
  data-space object (``y``, the full weight): the rank of ``z``'s array, each axis ``z``'s size, except at
  ``may_be_one`` (the full weight's sample axes), where 1 is allowed too. A structural ``ValueError`` otherwise."
- `_check_block_weight(S, block_weight)` (new): "The block weight's shape is the block shape exactly: ``(d,) + F``
  (probe), ``F`` (apply), ``()`` (entries); a sequence is refused (a ``TypeError``)."
- `_weight_squared(S, block_weight, full_weight, i, shape)`: "``b**2 f**2`` for one array of the type of ``z`` (mode
  ``i`` for probe): the block row of the mode (or the table, apply; the number, entries) reshaped onto the leading jet
  axes, times the full weight's square read by position. ``1.0`` for a factor that is ``None``. No reshape of the
  full weight."
- `_dual_jet(S, jet, block_weight, full_weight)` and `_the_seed(S, sample, y, block_weight, full_weight, w_tilde,
  name)`: the comments say "the weights" for "the weight".

---

## 7. The round-1 additions to the later parts' prose (listed, completed before each part)

- **Part 3, `check_data`'s docstring** (O3): one sentence after its rule: "It is the checker of ``y`` (and of a
  residual): a full weight, whose sample axes may be 1, is refused by it, and is checked by the batch's
  :py:meth:`~t3toolbox.backend.fitting.LeastSquaresBatch.check`."
- **Part 4, `backend/optimizers.py:1849-1851` and `:2096-2098`** (finding r1-8): "``eps (|omega y| |omega r| +
  |f|)``" and "``ε·(‖ω⊙y‖·‖ω⊙r‖ + |f|)``" become "``eps (|y|_M |r|_M + |f|)``" and "``ε·(‖y‖_M·‖r‖_M + |f|)``", with
  "``|v|_M = <v, M v>^(1/2)``, the norm of the data metric" said once at the first; the 2026.5.0 floor's quotation
  keeps its old letters inside the quotation marks.
- **Part 4, the oracle's class docstring:** its sentence on `f_resolution` stays as is; nothing in it promises that
  the padding of `y` may hold `inf` (note §9).
