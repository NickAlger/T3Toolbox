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

import math
import typing as typ
from dataclasses import dataclass

import numpy as np

import t3toolbox.backend.sampling_variables as sampling_variables
import t3toolbox.backend.jets as jets
import t3toolbox.backend.ut3_operations as ut3_operations
from t3toolbox.backend.common import *

if jax_available:
    import jax

__all__ = [
    'LeastSquaresOracle',
    'LeastSquaresBatch',
]


# --------------------------------------------------------------------------------------------------
# The flat draw over the sample stack W: flatten W's axes into one and gather the drawn positions. The
# oracle's `take` uses it for the sample (and, through `_take_W`, for y and the weight), wherever W sits in
# the array. A USER-supplied draw bypasses it entirely.
# --------------------------------------------------------------------------------------------------
def _flat_gather(
        arr:     NDArray,    # an array carrying the probe stack W
        w_start: int,        # axis where W begins
        n_w:     int,        # number of W axes (flattened to one, then gathered)
        idx:     NDArray,    # int -- the drawn flat positions over the combined W axis
) -> NDArray:                # arr with its W axes replaced by one axis of len(idx)
    '''Flatten ``arr``'s ``n_w`` W-axes (from ``w_start``) into one and gather ``idx`` along it.'''
    use_jax = is_jax_ndarray(arr)
    xnp, _, _ = get_backend(False, use_jax)
    sh = arr.shape
    return xnp.take(arr.reshape(sh[:w_start] + (-1,) + sh[w_start + n_w:]), idx, axis=w_start)


#####################################################################
#########    The least-squares oracle and its batch (P14)    ########
#####################################################################
# The fitting layer's extension point (graded messages, Part 1): an oracle answers a query at a frame through the
# engine, its batch holds the data.

_KINDS = ('probe', 'apply', 'entries')
_HOLD_LIST = ('xi', 'mu', 'nu', 'eta', 'r')            # Part 1's fixed hold list (7.2, "as today")
_SCALED = ('f', 'g', 'JtJp', 'Jp_norm_sq', 'Kp', 'r_block_sumsq')   # the minibatch scale's reach (O-21), with the
                                                                      #   oracle's own y_block_sumsq and f_resolution


@dataclass(frozen=True, eq=False)
class LeastSquaresBatch:
    """The data of a least-squares fit, arrays only: a registered pytree, every field a leaf (``None`` allowed).

    ``sample`` is the input jets ``ww`` (probe, apply: ``len=d``, ``elm_shape=F_in_i+W+(Ni,)``, or packed
    ``(d,)+F_in+W+(N,)`` on the uniform layout, :py:func:`~t3toolbox.backend.uniform_fitting.pack_batch`), or the
    integer ``index`` ``(d,)+W`` (entries). ``y`` and ``weight`` are **of the type of** ``z``, with the model's stack
    of base points ``C``: probe ``F+W+C+(Ni,)`` per mode (a tuple or a list of ``d`` arrays, or one array whose first
    axis is the mode; packed ``(d,)+F+W+C+(e,)`` on the uniform layout), apply ``F+W+C``, entries ``W+C``; each axis 1
    or that size, a size of 1 meaning the same along the axis (data shared by a stack have size-1 ``C`` axes, a
    weight per slot size-1 ``W``, ``C`` and edge axes). :py:func:`~t3toolbox.backend.jets.blocks_to_weight` makes a
    weight from one value per block. ``weight=None`` is ones. ``minibatch_scale`` is ``n / batch_size`` on a drawn
    batch (:py:meth:`LeastSquaresOracle.take`), ``None`` (one) on a full batch; ``drawn_indices`` the flat sample
    indices a draw took, relative to the batch it was taken from. Checked by
    :py:meth:`LeastSquaresOracle.check_batch`, never by the oracle's ``query`` (the engine checks the rule of ``y``
    and the weight on entry).
    """
    sample:           typ.Any                                    # P_in_i⊗V_i (the input jets ww) | index^d (entries)
    y:                typ.Any = None                             # Y*: probe len=d, F+W+C+(Ni,); apply F+W+C; entries W+C
    weight:           typ.Any = None                             # omega, of the type of z likewise; None = ones
    minibatch_scale:  typ.Optional[typ.Any] = None               # gamma: n / batch_size on a drawn batch; None = 1
    drawn_indices:    typ.Optional[NDArray] = None               # int, (batch_size,): the draw, flat over W

    def __eq__(self, other):
        raise TypeError('LeastSquaresBatch: == is not defined on array-carrying records; compare the fields you mean')

    __hash__ = None

    def data_metric(
            self,
            r:  typ.Any,        # Y*, of the type of z (no K), on this batch's samples
    ) -> typ.Any:               # M r: Y, minibatch_scale * weight**2 * r elementwise, r's structure
        """The data metric of this batch: :py:func:`~t3toolbox.backend.jets.data_metric` with the batch's own weight
        and minibatch scale, ``data_metric(r, self.weight, self.minibatch_scale)``. The call that cannot forget the
        scale of a drawn batch.

        The weight must be of the type of ``z`` (each axis 1 or ``z``'s size), the form
        :py:func:`~t3toolbox.backend.jets.blocks_to_weight` makes. A batch whose weight is one value per block,
        ``(d,) + F``, ``F`` or ``()``, is not of that type, and raises here by rank, except where the two forms
        coincide (an apply or entries batch with no sample or stack axes).

        >>> import numpy as np
        >>> from t3toolbox.backend.fitting import LeastSquaresBatch
        >>> from t3toolbox.backend.jets import blocks_to_weight
        >>> y = np.ones((3, 4))                                             # apply: F + W, F = (3,), W = (4,)
        >>> batch = LeastSquaresBatch(None, y, blocks_to_weight('apply', (2,), np.array([1.0, 2.0, 3.0]), y), 0.5)
        >>> print(batch.data_metric(y)[:, 0].tolist())                     # 0.5 * weight**2 * r
        [0.5, 2.0, 4.5]
        """
        return jets.data_metric(r, self.weight, self.minibatch_scale)


_BATCH_FIELDS = ('sample', 'y', 'weight', 'minibatch_scale', 'drawn_indices')

if jax_available:
    jax.tree_util.register_pytree_node(
        LeastSquaresBatch,
        lambda b: (tuple(getattr(b, name) for name in _BATCH_FIELDS), None),
        lambda aux, children: LeastSquaresBatch(*children))


def _same_leaves(
        a:  typ.Any,                        # a direction: a pair of tuples of arrays, or a pair of supercores
        b:  typ.Any,
) -> bool:                                  # the same object, or the same arrays leaf by leaf (identity)
    '''Whether two directions are the one direction: the same object, or containers of the very same arrays (a rebuilt
    tuple of the same arrays is the same direction). Never compares values.'''
    if a is b:
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same_leaves(p, q) for p, q in zip(a, b))
    return False


def _scale_tree(
        value:  typ.Any,                    # an array, or nested tuples of arrays (the blocks)
        scale:  typ.Any,                    # a float or a 0-d array (traced under jit)
) -> typ.Any:
    if isinstance(value, (tuple, list)):
        return tuple(_scale_tree(item, scale) for item in value)
    return value * scale


def _arrays_of(
        kind:   str,                        # 'probe' | 'apply' | 'entries'
        value:  typ.Any,                    # of the type of z: probe a tuple or a list of d arrays, or one array whose
                                            #   first axis is the mode (packed or not); else one array; None
        shape:  typ.Tuple[int, ...],        # the mode sizes N_i (a packed probe object is sliced to them)
) -> typ.List[typ.Any]:                     # the arrays, one per mode (probe) or one; [None] * count for None
    '''The arrays of a data-space object, unpadded: a packed probe object (``(d,) + F + W + C + (e,)``, the edge ``e``
    1 or the padded ``N``) sliced to each mode's ``N_i`` (an edge of 1 stays 1).'''
    count = len(shape) if kind == 'probe' else 1
    if value is None:
        return [None] * count
    if kind != 'probe':
        return [value]
    if isinstance(value, (tuple, list)):
        return list(value)
    return list(ut3_operations.unpack_vectors(value, shape))


def _stack_of_frame(
        frame:  typ.Tuple,                  # any of the four forms (sampling_variables.frame_form)
) -> typ.Tuple[int, ...]:                   # C, the stack of base points
    '''The stack of base points ``C`` of a frame or a corewise point: the leading axes of the first Tucker core
    (``D[0]`` or ``B[0]``) on the ragged layout, the axes after ``d`` of the Tucker supercore on the uniform one.
    The data and the weight cannot give it: shared by the stack, they have size-1 ``C`` axes.'''
    layout, _ = sampling_variables.frame_form(frame)
    if layout == 'uniform':
        return tuple(int(n) for n in np.shape(frame[0])[1:-2])
    return tuple(int(n) for n in np.shape(frame[0][0])[:-2])


def _y_times_metric_y(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape
        y:       typ.Any,                   # Y*, the data, of the type of z: probe len=d (or (d,)+..., packed or not),
                                            #   elm_shape=F+W+C+(Ni,); apply F+W+C; entries W+C; each axis 1 or z's size
        weight:  typ.Any,                   # omega, of the type of z likewise; None = ones
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points
        shape:   typ.Tuple[int, ...],       # the mode sizes N_i
) -> typ.List[NDArray]:                     # y * M y at gamma = 1, entrywise at z's shape: per mode (probe), or one
    '''``y * data_metric(y, omega, None)`` at ``z``'s shape, one array per mode for probe (one otherwise): ``y`` is
    broadcast to ``z``'s shape first, so an axis where ``y`` has size 1 and ``z`` does not counts every entry of ``z``
    along it, as ``f`` and ``r_block_sumsq`` do. The two data sums below reduce it.'''
    F = tuple(n + 1 for n in order)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))
    if kind == 'probe':
        targets = [F + W + C + (N,) for N in shape]
    else:
        targets = [(F + W + C) if kind == 'apply' else (W + C)]
    products = []
    for y_i, omega_i, z_shape in zip(_arrays_of(kind, y, shape), _arrays_of(kind, weight, shape), targets):
        y_full = xnp.broadcast_to(xnp.asarray(y_i), z_shape)
        products.append(y_full * jets.data_metric(y_full, omega_i, None))
    return products


def _y_metric_y(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape
        y:       typ.Any,                   # Y*, the data, of the type of z (_y_times_metric_y)
        weight:  typ.Any,                   # omega, of the type of z likewise; None = ones
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points (the shape of f)
        shape:   typ.Tuple[int, ...],       # the mode sizes N_i
) -> NDArray:                               # <y, M y> at gamma = 1 over z's shape, keeping C. shape=C
    '''``<y, M y>`` with ``M`` the data metric at ``gamma = 1``, summed over every entry of ``z`` (the modes, the slots
    of the box ``F``, the samples ``W`` and the mode edge), one sum per element of ``C``, like ``f``. ``y`` is
    broadcast to ``z``'s shape first: an axis where ``y`` has size 1 and ``z`` does not counts every entry of ``z``
    along it, as ``f`` does (the floor bounds the rounding of ``f``, which sums every sample). The metric is
    :py:func:`~t3toolbox.backend.jets.data_metric`, the library's one ``omega**2``.'''
    m = len(order)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))
    total = None
    for product in _y_times_metric_y(kind, order, y, weight, W, C, shape):
        summed_axes = tuple(range(m + len(W))) + ((product.ndim - 1,) if kind == 'probe' else ())
        summed = xnp.sum(product, axis=summed_axes)
        total = summed if total is None else total + summed
    return total


def _y_block_metric_y(
        kind:    str,                       # 'probe' | 'apply' | 'entries'
        order:   typ.Tuple[int, ...],       # the jet shape
        y:       typ.Any,                   # Y*, the data, of the type of z (_y_times_metric_y)
        weight:  typ.Any,                   # omega, of the type of z likewise; None = ones
        W:       typ.Tuple[int, ...],       # the sample stack, read off the sample
        C:       typ.Tuple[int, ...],       # the stack of base points
        shape:   typ.Tuple[int, ...],       # the mode sizes N_i
) -> NDArray:                               # <y, M y> at gamma = 1 per block: the block shape (d,)+F | F | ()
    '''``<y, M y>`` per block, with ``M`` the data metric at ``gamma = 1``, over ``z``'s shape: per mode and slot for
    probe, per slot for apply, one number for entries; summed over the samples, the stack ``C`` and the mode edge. The
    data's half of ``r_block_sumsq``: the same blocks, the same metric, the same entries counted.'''
    m = len(order)
    xnp, _, _ = get_backend(False, tree_contains_jax((y, weight)))
    sums = [xnp.sum(product, axis=tuple(range(m, product.ndim)))
            for product in _y_times_metric_y(kind, order, y, weight, W, C, shape)]
    return xnp.stack(sums) if kind == 'probe' else sums[0]


def _take_W(
        array:    NDArray,                  # one array of a data-space object, its W axes at w_start
        w_start:  int,                      # m (an apply or entries array, a ragged probe mode); 1 + m (mode-leading)
        W:        typ.Tuple[int, ...],      # the batch's sample stack
        indices:  NDArray,                  # int, (b,): flat positions over W
        what:     str,                      # 'the data y' or 'the weight', for the message
) -> NDArray:                               # W replaced by one axis: (1,) if every W axis was 1, else (b,)
    '''One array of ``y`` or the weight, drawn: its ``W`` axes (each 1 or ``W``'s size) kept as one axis of 1 when
    every one is 1, else broadcast to ``W``, flattened and gathered at ``indices``.'''
    n_w = len(W)
    shape = tuple(np.shape(array))
    if len(shape) < w_start + n_w or any(a not in (1, b) for a, b in zip(shape[w_start:w_start + n_w], W)):
        raise ValueError('LeastSquaresOracle.take: %s has shape %s, whose axes %d to %d are not the sample stack W = '
                         '%s (each axis 1 or its size): it is not of the type of z'
                         % (what, shape, w_start, w_start + n_w - 1, W))
    xnp, _, _ = get_backend(False, is_jax_ndarray(array))
    if all(a == 1 for a in shape[w_start:w_start + n_w]):
        return xnp.reshape(array, shape[:w_start] + (1,) + shape[w_start + n_w:])
    full = xnp.broadcast_to(array, shape[:w_start] + W + shape[w_start + n_w:])
    return _flat_gather(full, w_start, n_w, indices)


def _take_data_space(
        kind:     str,                      # 'probe' | 'apply' | 'entries'
        m:        int,                      # the number of jet axes
        W:        typ.Tuple[int, ...],      # the batch's sample stack
        value:    typ.Any,                  # y or the weight, of the type of z: probe a tuple or a list of d arrays, or
                                            #   one mode-leading array (packed or not); else one array; None
        indices:  NDArray,                  # int, (b,): flat positions over W
        what:     str,                      # 'the data y' or 'the weight', for the message
) -> typ.Any:                               # value drawn, its form kept; None for None
    '''``y`` or the weight of a drawn batch (:py:meth:`LeastSquaresOracle.take`): its ``W`` axes start at ``m`` in an
    apply or entries array and in each array of a ragged probe object, and at ``1 + m`` in a mode-leading probe array.'''
    if value is None:
        return None
    if kind == 'probe' and isinstance(value, (tuple, list)):
        return [_take_W(array, m, W, indices, what) for array in value]
    return _take_W(value, (1 + m) if kind == 'probe' else m, W, indices, what)


@dataclass(frozen=True, eq=False)
class LeastSquaresOracle(ValueHashedFields):
    """The least-squares oracle: ``f = 1/2 <M r, r>`` with ``r = z(x) - y`` and ``M`` the batch's data metric
    (:py:meth:`LeastSquaresBatch.data_metric`, ``minibatch_scale * weight**2`` elementwise), and what an optimizer asks
    of it at a frame, through one call of the engine.

    It is a frozen description with no arrays (its fields are statics, value-hashed: it rides as jax ``aux_data``);
    the data are its batch, a :py:class:`LeastSquaresBatch`. ``order`` is the jet shape of the sample and the data, a
    tuple, **required for every kind** (O-44): ``LeastSquaresOracle('apply', order=())`` for a plain fit,
    ``('probe', order=(2,))`` along a line; the entries kind has no jets and takes ``order=()`` only (O-46).

    :py:meth:`query` answers ``want`` at a frame (a manifold frame, or a corewise point ``(B, G)``, told from its
    structure by :py:func:`~t3toolbox.backend.sampling_variables.frame_form`). On a corewise point a direction has one
    form, the raw one: the oracle passes ``core_directions`` when it is given, else the ``variations`` it is given, as
    ``core_directions``; given both, they must be the same direction (the same object, or the same arrays), else a
    ``ValueError``. The **minibatch scale** of the batch (O-21) is part of ``M``: it multiplies ``f``, ``g``,
    ``JtJp``, ``Jp_norm_sq``, ``Kp``, ``f_resolution`` and the two block sums, and nothing else: ``r`` and ``Jp`` are
    the plain vectors at the drawn samples. ``f_resolution`` is ``eps * (sqrt(<y, M y>) * sqrt(2 f) + |f|)``, ``eps``
    of ``f``'s dtype: a bound on the rounding of ``f``, per element of ``C`` like ``f``, with ``<y, M y>`` summed over
    every entry of ``z`` (``y`` broadcast to ``z``'s shape, so a ``y`` shared by the samples counts once per sample, as
    ``f`` does). It needs the batch's ``y`` even when ``f`` comes from a held ``r`` (a ``ValueError`` otherwise).

    **The block sums** ``r_block_sumsq`` and ``y_block_sumsq`` are ``<r, M r>`` and ``<y, M y>`` per block, the block
    shape ``(d,) + F`` (probe: a mode at a slot), ``F`` (apply: a slot) or ``()`` (entries), each summed over the
    samples, the stack ``C`` and the mode edge, over ``z``'s shape (``y`` broadcast to it, as for ``f_resolution``).
    Both carry the scale, so their ratio, ``‖omega⊙r_b‖² / ‖omega⊙y_b‖²``, does not depend on it; with no weight they
    are sums of squares. On a drawn batch they estimate the full batch's without bias, as ``f`` does; a ``y`` and a
    weight that are the same for every sample (``W`` axes of 1, kept as one axis of 1 by :py:meth:`take`) give the full
    batch's ``y_block_sumsq`` exactly. Each batch's sums use that batch's own weight: a validation batch with another
    weight (or none) gives the ratio under that weight.

    **The outputs, typed** (the glossary's Symbols): ``r`` and ``Jp`` are in ``Y*``, of the type of ``z``; ``g``,
    ``JtJp`` and ``Kp`` in ``X*``, the blocks before the geometry's projection; ``f``, ``Jp_norm_sq`` and
    ``f_resolution`` are numbers, ``ℝ`` per element of ``C`` (``K + C`` for ``Jp_norm_sq``); the two block sums
    ``ℝ`` per block.

    **What it holds.** ``query`` returns, with its values, a
    :py:class:`~t3toolbox.backend.sampling_variables.HeldVariables`: the ``held`` it was given plus each of ``xi``,
    ``mu``, ``nu``, ``eta``, ``r`` that the call computed anyway (never one computed only to be held). Hand it back
    to a later query **at the same frame and the same batch**: a held ``r`` belongs to its batch's ``y``, and the
    sweeps to its sample. The record carries no identity of either.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
    >>> from t3toolbox.backend.jets import affine_jet
    >>> np.random.seed(0)
    >>> x, target = (t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)) for _ in range(2))
    >>> frame = t3m.MANIFOLD.frame(x).data
    >>> ww, pp = ([np.random.randn(7, N) for N in x.shape] for _ in range(2))          # W = (7,)
    >>> oracle = LeastSquaresOracle('probe', order=(1,))
    >>> batch = LeastSquaresBatch(affine_jet(ww, pp), target.probe(affine_jet(ww, pp), (1,)))
    >>> (f, g), held = oracle.query(frame, batch, ('f', 'g'))
    >>> print(f.shape, [b.shape for b in g[0]], held.keys())
    () [(2, 4), (3, 5), (2, 6)] ('xi', 'mu', 'nu', 'eta', 'r')

    The trap: input jets read with ``order=()`` pass every check of shapes (the jet axis is read as one more sample
    axis), and fit a different problem -- the probes at ``ww`` and at ``pp`` separately, not along the line:

    >>> plain = LeastSquaresOracle('probe', order=())
    >>> (f_wrong,), _ = plain.query(frame, LeastSquaresBatch(affine_jet(ww, pp), x.probe(affine_jet(ww, pp))), ('f',))
    >>> (f_right,), _ = oracle.query(frame, LeastSquaresBatch(affine_jet(ww, pp), x.probe(affine_jet(ww, pp), (1,))),
    ...                              ('f',))
    >>> print(float(f_wrong) < 1e-20, float(f_right) < 1e-20)                 # each fits its own data exactly
    True True
    >>> (f_mixed,), _ = oracle.query(frame, LeastSquaresBatch(affine_jet(ww, pp), x.probe(affine_jet(ww, pp))), ('f',))
    >>> print(float(f_mixed) > 1.0)                    # the slots of the line are not the probes at ww and pp
    True
    """
    kind:   str                                         # 'probe' | 'apply' | 'entries'
    order:  typ.Tuple[int, ...]                         # the jet shape; required (O-44); entries: () only (O-46)

    optional_outputs: typ.ClassVar[typ.Tuple[str, ...]] = (
        'Kp', 'Jp', 'r', 'r_block_sumsq', 'y_block_sumsq', 'f_resolution')

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError('LeastSquaresOracle: unknown kind %r; expected one of %s' % (self.kind, _KINDS))
        if not isinstance(self.order, tuple) or not all(isinstance(n, (int, np.integer)) and not isinstance(n, bool)
                                                         and n >= 0 for n in self.order):
            raise TypeError('LeastSquaresOracle: order is the jet shape, a tuple of nonnegative ints (() for no jets); '
                            'got %r' % (self.order,))
        if self.kind == 'entries' and self.order != ():
            raise ValueError("LeastSquaresOracle: the entries kind has no jets: its order is (), got %r "
                             "(a derivative of entries is the apply kind at the one-hot jet)" % (self.order,))

    # ------------------------------------------------------------------ the operation
    def query(
            self,
            frame:            typ.Tuple,                      # a manifold frame or a corewise point (frame_form)
            batch:            LeastSquaresBatch,              # its data
            want:             typ.Sequence[str],              # f, g, JtJp, Jp_norm_sq; optional: optional_outputs
            Vc:               typ.Optional[typ.Sequence[NDArray]] = None,
            variations:       typ.Optional[typ.Tuple] = None, # X, (V, H): JtJp, Jp_norm_sq, Jp; raw on a point
            core_directions:  typ.Optional[typ.Tuple] = None, # X, (D_dot, L_dot): Kp
            held:             typ.Any = None,                 # a HeldVariables from an earlier query here, or None
    ) -> typ.Tuple[
            typ.Tuple[typ.Any, ...],                          # values, aligned with want: r, Jp Y*; g, JtJp, Kp X*
                                                              #   (before the projection); f, Jp_norm_sq, f_resolution
                                                              #   ℝ; r_block_sumsq, y_block_sumsq ℝ per block
            typ.Any,                                          # held: a HeldVariables
    ]:
        '''The values ``want`` at ``frame`` on ``batch``, and what to hold for a later query here (the class doc).'''
        want = tuple(want)
        _, call = sampling_variables.frame_form(frame)
        if call == 'corewise' and variations is not None:
            if core_directions is None:
                core_directions = variations
            elif not _same_leaves(variations, core_directions):
                raise ValueError('LeastSquaresOracle.query: a corewise point has one form of a direction, and both '
                                 'variations and core_directions were given, not the same direction (pass one, or '
                                 'the same arrays as both)')
            variations = None
        local = ('y_block_sumsq', 'f_resolution')
        engine_want = tuple(name for name in want if name not in local)
        if 'f_resolution' in want and 'f' not in engine_want:
            engine_want = engine_want + ('f',)
        values = {}
        held_out = held if held is not None else sampling_variables.HeldVariables()
        if engine_want:
            # one set of arguments for the query and the call: the hold-list names the call computes anyway are
            # added to want and split off (none is computed only to be held; TARGET_DECISIONS, the hold list)
            args = dict(order=self.order, y=batch.y, weight=batch.weight, variations=variations,
                        core_directions=core_directions, Vc=Vc, held=held)
            free = sampling_variables.computed_names(self.kind, frame, batch.sample, engine_want, **args)
            extras = tuple(name for name in _HOLD_LIST if name in free and name not in engine_want)
            outs = sampling_variables.sampling_variables(self.kind, frame, batch.sample, engine_want + extras, **args)
            values = dict(zip(engine_want + extras, outs))
            kept = {name: values[name] for name in _HOLD_LIST if name in free}
            held_out = held_out.with_values(**kept) if kept else held_out
        if 'y_block_sumsq' in want:                   # the data's half of the block sums: <y, M y> per block
            if batch.y is None:
                raise ValueError('LeastSquaresOracle.query: y_block_sumsq is <y, M y> per block and needs the '
                                 'batch\'s data y, which is None')
            values['y_block_sumsq'] = _y_block_metric_y(self.kind, self.order, batch.y, batch.weight, self._W(batch),
                                                        _stack_of_frame(frame), sampling_variables.frame_shape(frame))
        if 'f_resolution' in want:                    # from the unscaled batch values; the scale once, below
            if batch.y is None:                         # f can come from a held r; <y, M y> cannot
                raise ValueError('LeastSquaresOracle.query: f_resolution is eps (sqrt(<y, M y>) sqrt(2 f) + |f|) and '
                                 'needs the batch\'s data y, which is None (f alone can come from a held r; the '
                                 'resolution cannot)')
            f = values['f']
            xnp, _, _ = get_backend(False, tree_contains_jax((f, batch.y, batch.weight)))
            y_norm_sq = _y_metric_y(self.kind, self.order, batch.y, batch.weight, self._W(batch),
                                    _stack_of_frame(frame), sampling_variables.frame_shape(frame))
            eps = float(np.finfo(np.dtype(f.dtype)).eps)
            values['f_resolution'] = eps * (xnp.sqrt(y_norm_sq) * xnp.sqrt(2.0 * f) + xnp.abs(f))
        scale = batch.minibatch_scale
        if scale is not None:
            for name in _SCALED + ('y_block_sumsq', 'f_resolution'):
                if name in values and name in want:
                    values[name] = _scale_tree(values[name], scale)
        return tuple(values[name] for name in want), held_out

    # ------------------------------------------------------------------ the batch
    def check_batch(
            self,
            batch:        LeastSquaresBatch,                  # ragged (check before packing)
            shape:        typ.Sequence[int],                  # the mode sizes (N0, ..., N(d-1))
            stack_shape:  typ.Tuple[int, ...] = (),           # C, the model's stack of base points; () unstacked
    ) -> None:
        '''Check a batch structurally, on its ragged form: the sample against the jet shape and ``shape``
        (:py:func:`~t3toolbox.backend.jets.check_sample`), the data and the weight against the sample and the model's
        stack ``stack_shape`` (``check_data``, ``check_weight``: each of the type of ``z``, every axis 1 or ``z``'s
        size); raise a ``ValueError`` naming what is wrong. Data of a stacked model checked without its
        ``stack_shape`` fail by rank. A checker: :py:meth:`query` does not call it.

        >>> import numpy as np
        >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
        >>> oracle = LeastSquaresOracle('apply', order=())
        >>> ww = [np.ones((10, N)) for N in (4, 5, 6)]                            # W = (10,)
        >>> shared = LeastSquaresBatch(ww, np.zeros((10, 1)))                     # one y for a stack C = (2,)
        >>> oracle.check_batch(shared, (4, 5, 6), stack_shape=(2,))
        >>> oracle.check_batch(LeastSquaresBatch(ww, np.zeros((10, 3))), (4, 5, 6), stack_shape=(2,))
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: the data y of the apply kind has shape (10, 3); it is of the type of z, ...
        '''
        jets.check_sample(self.kind, self.order, batch.sample, shape)
        if batch.y is not None:
            jets.check_data(self.kind, self.order, batch.sample, batch.y, stack_shape)
        jets.check_weight(self.kind, self.order, batch.sample, batch.weight, stack_shape)

    def _w_axes(self, batch):
        '''(where W starts in the sample, the number of W axes): after the jets (and the packed d), or after the
        mode axis of an index.'''
        m = len(self.order)
        if self.kind == 'entries':
            return 1, np.ndim(batch.sample) - 1
        if isinstance(batch.sample, (tuple, list)):
            return m, np.ndim(batch.sample[0]) - m - 1
        return 1 + m, np.ndim(batch.sample) - m - 2

    def _W(self, batch) -> typ.Tuple[int, ...]:
        '''The sample stack ``W``, read off the sample.'''
        start, n_w = self._w_axes(batch)
        lead = batch.sample if self.kind == 'entries' or not isinstance(batch.sample, (tuple, list)) \
            else batch.sample[0]
        return tuple(int(n) for n in np.shape(lead)[start:start + n_w])

    def n_measurements(
            self,
            batch:  LeastSquaresBatch,
    ) -> int:                                                 # |W|, the number of samples
        '''The number of samples of a batch, ``prod(W)``: the population a draw takes from.'''
        return math.prod(self._W(batch))

    def take(
            self,
            batch:    LeastSquaresBatch,
            indices:  NDArray,                                # int, (b,): flat positions over W; len(indices) static
    ) -> LeastSquaresBatch:                                   # the drawn batch, W flattened to (b,)
        '''A draw: the samples at ``indices`` (flat over ``W``, which becomes one axis of ``b = len(indices)``), with
        ``minibatch_scale = n / b`` (times the batch's own scale) and ``drawn_indices = indices``, relative to
        ``batch``. The scale makes the drawn ``f`` (and every scaled output) an unbiased estimate of the full one under
        a uniform draw of ``b`` of the ``n`` samples. Traceable, ``b`` static. For a plain subset with no scale, such
        as a fixed validation slice, drop it: ``dataclasses.replace(sub, minibatch_scale=None)``.

        The data ``y`` and the weight are of the type of ``z``, each axis 1 or ``z``'s size, and are drawn alike: an
        object whose every ``W`` axis is 1 (the same for every sample) keeps one axis of 1 in their place, with no
        gather; any other is broadcast to the batch's ``W``, flattened and gathered at ``indices``. So a weight per
        sample is drawn with its samples, and a weight per slot stays one per slot.

        >>> import dataclasses as dc
        >>> import numpy as np
        >>> from t3toolbox.backend.fitting import LeastSquaresOracle, LeastSquaresBatch
        >>> oracle = LeastSquaresOracle('apply', order=())
        >>> batch = LeastSquaresBatch([np.ones((10, N)) for N in (4, 5, 6)], np.zeros(10))
        >>> sub = oracle.take(batch, np.array([0, 3]))
        >>> print(sub.y.shape, sub.minibatch_scale, sub.drawn_indices)
        (2,) 5.0 [0 3]
        >>> print(dc.replace(sub, minibatch_scale=None).minibatch_scale)          # a plain subset
        None

        A weight per sample is drawn with the samples; one constant over the samples keeps its axis of 1:

        >>> per_sample = dc.replace(batch, weight=np.arange(10.0))
        >>> print(oracle.take(per_sample, np.array([0, 3])).weight)
        [0. 3.]
        >>> print(oracle.take(dc.replace(batch, weight=np.full((1,), 2.0)), np.array([0, 3])).weight)
        [2.]
        '''
        n = self.n_measurements(batch)
        start, n_w = self._w_axes(batch)
        W = self._W(batch)
        m = len(self.order)
        if self.kind == 'entries':
            sample = _flat_gather(batch.sample, 1, n_w, indices)
        elif isinstance(batch.sample, (tuple, list)):
            sample = [_flat_gather(w, m, n_w, indices) for w in batch.sample]
        else:
            sample = _flat_gather(batch.sample, 1 + m, n_w, indices)
        y = _take_data_space(self.kind, m, W, batch.y, indices, 'the data y')
        weight = _take_data_space(self.kind, m, W, batch.weight, indices, 'the weight')
        scale = n / len(indices)
        if batch.minibatch_scale is not None:
            scale = batch.minibatch_scale * scale
        return LeastSquaresBatch(sample, y, weight, scale, indices)
