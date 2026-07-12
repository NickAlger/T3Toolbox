# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Backend infrastructure: numpy/jax dispatch, array predicates, scans/maps, shared mixins.

``get_backend(is_uniform, use_jax) -> (xnp, xmap, xscan)`` selects the array module and loop
machinery; ``is_ndarray``/``is_jax_ndarray``/``tree_contains_jax`` are the type-inference
predicates behind the no-``use_jax``-parameter convention (dispatch is inferred from the input
arrays at the lowest level); ``ValueHashedMasks`` is the value-based hash/eq mixin that keeps a
rebuilt-but-identical mask holder on the same jit cache key.
"""
import numpy as np
import typing as typ
import functools as ft

__all__ = [
    'jax_available',
    #
    'NDArray',
    'is_ndarray',
    'is_boolean_ndarray',
    'is_jax_ndarray',
    'is_numpy_ndarray',
    'to_jax',
    'to_numpy',
    #
    'ValueHashedMasks',
    #
    'ragged_scan',
    'numpy_scan',
    'jax_scan',
    #
    'ragged_map',
    'numpy_map',
    'jax_map',
    #
    'get_backend',
    'xwhile',
    'xcat',
    'xappend',
    'xprepend',
    'tree_contains_jax',
    'items_are_uniform',
    #
    'randn',
    #
    'save_core_families',
    'load_core_families',
]

jax_available = False
try:
    import jax.numpy as jnp
    import jax
    jax_available = True
except ImportError:
    print('Unable to import Jax. Using numpy instead.')

NDArray = np.ndarray
if jax_available:
    NDArray = typ.Union[np.ndarray, jnp.ndarray]

is_ndarray = lambda x: isinstance(x, np.ndarray)
if jax_available:
    is_ndarray = lambda x: (isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray))


is_jax_ndarray = lambda x: False
if jax_available:
    is_jax_ndarray = lambda x: isinstance(x, jnp.ndarray)


is_numpy_ndarray = lambda x: isinstance(x, np.ndarray)


def is_boolean_ndarray(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.bool_)
    else:
        return False

if jax_available:
    def is_boolean_ndarray(x):
        if isinstance(x, jnp.ndarray):
            return jnp.issubdtype(x.dtype, jnp.bool_)
        elif isinstance(x, np.ndarray):
            return np.issubdtype(x.dtype, np.bool_)
        else:
            return False

to_jax = lambda x: np.array(x)
if jax_available:
    to_jax = lambda x: jnp.array(x)

to_numpy = lambda x: np.array(x)


#


CarryType = typ.TypeVar('CarryType')

def ragged_scan(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],   # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],   # len=num_outputs
            ],
        ],
        init: CarryType,
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray], # len=scan_length
                NDArray, # shape[0]=scan_length
            ]
        ], # len=num_inputs
) -> typ.Tuple[
    CarryType,
    typ.Tuple[
        typ.Tuple[NDArray, ...], # len=scan_length
        ...
    ],  # len=num_outputs,
]:
    """Similar to jax.lax.scan, except for ragged-sized arrays
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html

    """
    scan_length = len(xs[0])
    carry = init

    ys_list = []
    for ii in range(scan_length):
        x = tuple([x[ii] for x in xs])
        carry, y = f(carry, x)

        if ii==0:
            ys_list = [[] for _ in range(len(y))]

        for l, elm in zip(ys_list, y):
            l.append(elm)

    return carry, tuple([tuple(y) for y in ys_list])


def numpy_scan(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],  # len=num_outputs
            ],
        ],
        init: CarryType,
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray],  # len=scan_length
                NDArray,  # shape[0]=scan_length
            ]
        ],  # len=num_inputs
) -> typ.Tuple[
    CarryType,
    typ.Tuple[
        NDArray, # shape[0]=scan_length
        ...
    ],  # len=num_outputs,
]:
    """Similar to jax.lax.scan, except returns numpy arrays instead of jax arrays.
    """
    xs_list = [list(x) for x in xs]
    carry, ys_list = ragged_scan(f, init, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    return carry, ys


def ragged_map(
        f: typ.Callable[
            [
                typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
                typ.Sequence[NDArray],  # len=num_outputs
            ],
        ],
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray],  # len=map_length
                NDArray,  # shape[0]=map_length
            ]
        ],  # len=num_inputs
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # len=map_length
    ...
]:  # len=num_outputs
    map_length = len(xs[0])

    ys_list = []
    for ii in range(map_length):
        x = tuple([elm[ii] for elm in xs])
        y = f(x)

        if ii==0:
            ys_list = [[] for _ in range(len(y))]

        for l, elm in zip(ys_list, y):
            l.append(elm)

    return tuple([tuple(y) for y in ys_list])


def numpy_map(
        f: typ.Callable[
            [CarryType,
             typ.Sequence[NDArray],  # len=num_inputs
             ],
            typ.Tuple[
                CarryType,
                typ.Sequence[NDArray],  # len=num_outputs
            ],
        ],
        xs: typ.Sequence[
            typ.Union[
                typ.Sequence[NDArray],  # len=map_length
                NDArray,  # shape[0]=map_length
            ]
        ],  # len=num_inputs
) -> typ.Tuple[
    NDArray,  # shape[0]=map_length
    ...
]:  # len=num_outputs,
    xs_list = [list(x) for x in xs]
    ys_list = ragged_map(f, xs_list)
    ys = tuple([np.stack(y) for y in ys_list])
    return ys


jax_scan = numpy_scan
jax_map = numpy_map
if jax_available:
    jax_scan = jax.lax.scan
    jax_map = jax.lax.map


def get_backend(
        is_uniform: bool,
        use_jax: bool,
):
    if is_uniform:
        if use_jax:
            xmap = jax_map
            xscan = jax_scan
        else:
            xmap = numpy_map
            xscan = numpy_scan
    else:
        xmap = ragged_map
        xscan = ragged_scan

    if use_jax:
        xnp = jnp
    else:
        xnp = np

    return xnp, xmap, xscan


def xwhile(
        cond,                  # state -> 0-d bool (eager) / traced bool (jit): loop while True
        body,                  # state -> state (a pytree of the SAME structure and leaf shapes)
        init_state,            # the loop-carried state (pytree)
        use_jit: bool = False, # True + jax state -> jax.lax.while_loop; else a Python while loop
):
    """Data-dependent ``while`` with the numpy / eager-jax / jit dispatch -- the ``xscan`` precedent for a
    ``while``. With ``use_jit`` and a jax state it compiles via ``jax.lax.while_loop``; otherwise (numpy,
    eager jax, or jax not installed) it runs ``while bool(cond(state)): state = body(state)``, so it works
    on every backend and **silently falls back to eager** when jit is unavailable. Write ``cond``/``body``
    backend-agnostically (``cond`` returns a 0-d boolean; ``body`` uses ``xnp.where``, not Python branches)
    so the SAME pair drives both paths."""
    if use_jit and jax_available and tree_contains_jax(init_state):
        import jax
        return jax.lax.while_loop(cond, body, init_state)
    state = init_state
    while bool(cond(state)):
        state = body(state)
    return state


def xcat(
        x: typ.Union[NDArray, typ.Sequence],  # array (concat on axis 0) or sequence (concat as tuples); same kind as y
        y: typ.Union[NDArray, typ.Sequence],  # same kind as x
) -> typ.Union[NDArray, typ.Tuple]:           # x and y concatenated
    """Concatenate arrays or sequences.
    """
    if is_ndarray(x):
        assert(is_ndarray(y))
        if is_jax_ndarray(x) or is_jax_ndarray(y):
            return jnp.concatenate([x, y], axis=0)
        else:
            return np.concatenate([x, y], axis=0)

    assert(isinstance(x, typ.Sequence) and isinstance(y, typ.Sequence))

    if len(x) == 0:
        return y
    elif len(y) == 0:
        return x
    else:
        return tuple(x) + tuple(y)


def xappend(
        S: typ.Union[NDArray, typ.Sequence],  # array or sequence to append to
        x,                                     # slice (if S is an array) or element (if S is a sequence)
) -> typ.Union[NDArray, typ.Tuple]:            # S with x appended
    """Append slice to array or element to sequence
    """
    if is_ndarray(S):
        assert(is_ndarray(x))
        if is_jax_ndarray(S) or is_jax_ndarray(x):
            return jnp.concatenate([S, x.reshape((1,)+x.shape)], axis=0)
        else:
            return np.concatenate([S, x.reshape((1,)+x.shape)], axis=0)

    assert(isinstance(S, typ.Sequence))

    if len(S) == 0:
        return (x,)
    else:
        return tuple(S) + (x,)


def xprepend(
        x,                                     # slice (if S is an array) or element (if S is a sequence)
        S: typ.Union[NDArray, typ.Sequence],   # array or sequence to prepend to
) -> typ.Union[NDArray, typ.Tuple]:            # S with x prepended
    """Prepend slice to array or element to sequence
    """
    if is_ndarray(S):
        assert(is_ndarray(x))
        if is_jax_ndarray(S) or is_jax_ndarray(x):
            return jnp.concatenate([x.reshape((1,)+x.shape), S], axis=0)
        else:
            return np.concatenate([x.reshape((1,)+x.shape), S], axis=0)

    assert(isinstance(S, typ.Sequence))

    if len(S) == 0:
        return (x,)
    else:
        return (x,) + tuple(S)



def randn(*args, use_jax: bool):
    if use_jax:
        return jnp.array(np.random.randn(*args)) # should convert this to pure jax
    else:
        return np.random.randn(*args)


def tree_contains_jax(T):
    if isinstance(T, typ.Sequence):
        return any([tree_contains_jax(t) for t in T])
    return is_jax_ndarray(T)


def items_are_uniform(
        xx,
) -> bool:
    """Checks if an object can be treated as uniform for the purposes of jax.scan and jax.map.

    True if x is an array, or a sequence of arrays which all have the same shape. False otherwise.
    """
    if is_ndarray(xx):
        return True

    elif isinstance(xx, typ.Sequence):
        if all([is_ndarray(xi) for xi in xx]):
            if len(xx) == 0:
                return True

            shape = xx[0].shape
            if all([xi.shape == shape for xi in xx]):
                return True

    return False


def save_core_families(
        file,      # path or open file object to write the .npz to
        families,  # sequence of core-families, each a sequence of arrays
) -> None:
    """Save a sequence of core-families (each a sequence of arrays) to a ``.npz`` file.

    Uses ``'f{family_index}_{core_index}'`` keys so :py:func:`load_core_families` can regroup them.
    Shared by the frontend ``save`` methods (T3Frame, T3Variations, T3Tangent).
    """
    np.savez(file, **{'f%d_%d' % (fi, ci): np.asarray(c)
                      for fi, fam in enumerate(families) for ci, c in enumerate(fam)})


def load_core_families(
        file,  # path or open file object to read the .npz from
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # one core-family per element
    ...,
]:
    """Inverse of :py:func:`save_core_families`: load a ``.npz`` file into a tuple of core-families.

    The number of families is inferred from the highest ``'f{fi}_...'`` key; each family is returned
    as a tuple of arrays ordered by core index.
    """
    npz = np.load(file)
    num_families = 1 + max(int(k.split('_', 1)[0][1:]) for k in npz.files)
    def fam(fi):
        ks = sorted((k for k in npz.files if k.startswith('f%d_' % fi)),
                    key=lambda k: int(k.split('_', 1)[1]))
        return tuple(npz[k] for k in ks)
    return tuple(fam(fi) for fi in range(num_families))


class ValueHashedMasks:
    """Mixin giving uniform-layer mask holders VALUE-based ``__hash__``/``__eq__`` over mask content.

    Why this matters (PERFORMANCE, not cosmetics): a mask holder rides as jax ``aux_data``, so its
    ``__hash__``/``__eq__`` are part of the jit compilation cache key. Identity hash/eq (the bare
    ``@dataclass(eq=False)`` default) makes a fresh-but-array-identical holder a NEW key -> jit
    RECOMPILES. In a manifold-optimization loop the orthogonal frame is rebuilt every iteration (via
    ``ut3_orthogonal_representations``), producing fresh holders whose rank structure is identical;
    identity hashing would recompile every step, dwarfing the actual compute -- the opposite of the
    uniform layer's whole point. Value-based hash/eq makes the cache key reflect the rank STRUCTURE:
    identical structure -> cache hit (no recompile); genuinely different structure -> recompile (correct).
    See ``docs/contributor/uniform_pytree_composition.md``.

    Subclasses must be ``@dataclass(frozen=True, eq=False)`` and expose ``.data`` as a tuple of the
    (HOST numpy, concrete) mask arrays. The content hash is cached (the holder is frozen/immutable); eq
    short-circuits on identity, then compares by ``np.array_equal``.
    """

    @ft.cached_property
    def _content_hash(self) -> int:
        # shape + dtype + bytes so distinct-shape/dtype masks never collide; masks are HOST numpy.
        return hash(tuple((m.shape, m.dtype.str, m.tobytes()) for m in self.data))

    def __hash__(self) -> int:
        return self._content_hash

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return all(np.array_equal(a, b) for a, b in zip(self.data, other.data))
