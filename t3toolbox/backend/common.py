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
import warnings
import typing as typ
import functools as ft
import dataclasses as dc
import weakref

__all__ = [
    'jax_available',
    #
    'NDArray',
    'is_ndarray',
    'is_boolean_ndarray',
    'is_jax_ndarray',
    'is_numpy_ndarray',
    'to_jax',
    'jax_or_warn',
    'to_numpy',
    #
    'ValueHashedMasks',
    'ValueHashedFields',
    'StaticSkeleton',
    'partition_static',
    'rebuild_static',
    'require_concrete_masks',
    'prefix_mask',
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
    'tree_to_jax',
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
    pass  # jax is optional: numpy-only operation is a fully supported configuration
          # (probe `jax_available` or install the `t3toolbox[jax]` extra)

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

_JAX_WARNED = set()


def jax_or_warn(what: str) -> bool:
    """The jax-absent policy: a request for jax (``use_jax=True``, ``to_jax``, ``use_jit=True``) on a machine
    without jax runs on numpy and WARNS, once per call site -- never an error (a project developed without
    jax on one machine and deployed with it on another must run unchanged on both), never silent (the old
    ``to_jax = np.array`` fallback). Returns ``jax_available``."""
    if not jax_available and what not in _JAX_WARNED:
        _JAX_WARNED.add(what)
        warnings.warn('%s: jax was requested but is not installed -- running on numpy instead. Install the '
                      't3toolbox[jax] extra for jax.' % what, RuntimeWarning, stacklevel=3)
    return jax_available


def to_jax(x):
    """``jnp.array(x)`` when jax is available; otherwise ``np.array(x)`` with a warning (see ``jax_or_warn``)."""
    return jnp.array(x) if jax_or_warn('to_jax') else np.array(x)

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

    _jax_map_adapters = weakref.WeakKeyDictionary()   # body -> its scan adapter (see jax_map)

    def _jax_map_adapter(f):
        f_ref = weakref.ref(f)      # the adapter must NOT strongly reference f: it is the VALUE in a
                                    # WeakKeyDictionary keyed on f, so a strong ref would make every
                                    # entry immortal and pin each throwaway body (and its arrays).

        def g(carry, x):
            return carry, f_ref()(x)
        return g

    def jax_map(f, xs):
        """``jax.lax.map`` with a per-body cached scan adapter.

        ``lax.map`` is a ``scan`` that builds a FRESH ``lambda _, x: ((), f(x))`` every call, and
        ``scan`` keys its trace/compile cache on that lambda's identity -- so ``lax.map`` recompiles on
        every call however stable ``f`` is, which no amount of hoisting at the call sites can fix
        (:doc:`/contributor/scan_body_principles`). Reusing one adapter per ``f`` restores the hit, so a
        module-level body compiles once, as it already does under ``xscan``.

        The keys are weak so the cache discriminates by itself: a module-level body outlives the process
        and keeps its entry, while a body still built inline dies with the call and takes its entry --
        and everything it closed over -- with it.

        ``batch_size`` is deliberately unsupported: nothing in the library passes it, and it selects a
        chunked-``vmap`` strategy rather than the sequential ``scan`` the memory-lean sampling paths
        depend on.
        """
        try:
            g = _jax_map_adapters.get(f)
        except TypeError:                       # not weak-referenceable -- fall back, uncached
            return jax.lax.map(f, xs)
        if g is None:
            g = _jax_map_adapter(f)
            _jax_map_adapters[f] = g
        return jax.lax.scan(g, (), xs)[1]


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

    if use_jax and not jax_or_warn('get_backend(use_jax=True)'):
        use_jax = False                       # jax absent: numpy, with the one-time warning
        xmap, xscan = (numpy_map, numpy_scan) if is_uniform else (ragged_map, ragged_scan)
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
        if not is_ndarray(y):
            raise TypeError('expected an ndarray to append, got %s' % type(y).__name__)
        if is_jax_ndarray(x) or is_jax_ndarray(y):
            return jnp.concatenate([x, y], axis=0)
        else:
            return np.concatenate([x, y], axis=0)

    if not (isinstance(x, typ.Sequence) and isinstance(y, typ.Sequence)):
        raise TypeError('expected two sequences, got %s and %s' % (type(x).__name__, type(y).__name__))

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
        if not is_ndarray(x):
            raise TypeError('expected an ndarray, got %s' % type(x).__name__)
        if is_jax_ndarray(S) or is_jax_ndarray(x):
            return jnp.concatenate([S, x.reshape((1,)+x.shape)], axis=0)
        else:
            return np.concatenate([S, x.reshape((1,)+x.shape)], axis=0)

    if not isinstance(S, typ.Sequence):
        raise TypeError('expected a sequence, got %s' % type(S).__name__)

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
        if not is_ndarray(x):
            raise TypeError('expected an ndarray, got %s' % type(x).__name__)
        if is_jax_ndarray(S) or is_jax_ndarray(x):
            return jnp.concatenate([x.reshape((1,)+x.shape), S], axis=0)
        else:
            return np.concatenate([x.reshape((1,)+x.shape), S], axis=0)

    if not isinstance(S, typ.Sequence):
        raise TypeError('expected a sequence, got %s' % type(S).__name__)

    if len(S) == 0:
        return (x,)
    else:
        return (x,) + tuple(S)



def randn(*args, use_jax: bool):
    if use_jax and jax_or_warn('randn(use_jax=True)'):
        return jnp.array(np.random.randn(*args)) # should convert this to pure jax
    else:
        return np.random.randn(*args)


def tree_contains_jax(T):
    if isinstance(T, typ.Sequence):
        return any([tree_contains_jax(t) for t in T])
    return is_jax_ndarray(T)


def tree_to_jax(T):
    """Move every array leaf of a pytree (nested tuples/lists of arrays) onto jax, preserving the tree
    structure; leaves already jax are a no-op (``jnp.asarray``). Numpy ``float64`` leaves become jax
    ``float32`` unless jax x64 is enabled -- the caller opts into jax precision. Without jax: numpy leaves
    and a one-time warning (the jax-absent policy, :py:func:`jax_or_warn`)."""
    if isinstance(T, typ.Sequence):
        return type(T)(tree_to_jax(t) for t in T)   # preserve list-vs-tuple
    return jnp.asarray(T) if jax_or_warn('tree_to_jax') else np.asarray(T)


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


class ValueHashedFields:
    """Mixin giving a frozen dataclass VALUE-based ``__hash__``/``__eq__`` over **all** its fields,
    with numpy arrays compared by CONTENT rather than by identity.

    The generalization of :py:class:`ValueHashedMasks` (which value-hashes one designated ``.data``
    tuple of masks) to a dataclass whose *parameters are its fields*. Both exist for the same reason:
    the object rides as jax ``aux_data``, so its ``__hash__``/``__eq__`` are part of the jit
    compilation cache key, and identity semantics make a rebuilt-but-identical object a NEW key ->
    jit RECOMPILES. A geometry rebuilt at each rank-continuation level, or a sampling kind rebuilt at
    each outer step, must be the SAME cache key when its parameters are the same.

    Use it instead of the dataclass ``eq=True`` default whenever a field may hold a numpy array
    (``eq=True`` would raise "truth value of an array is ambiguous"), and instead of ``eq=False``
    whenever the object is a jit aux (identity semantics recompile). Subclasses must be
    ``@dataclass(frozen=True, eq=False)``.

    The alternative this replaces is a hand-maintained parallel ``identity`` tuple restating the
    fields -- correct while someone keeps it in sync, silently identity-based for a user-built object
    that omits it. Here the fields ARE the key, so there is nothing to keep in sync.
    """

    @staticmethod
    def _value_key(v):
        """A hashable, value-equal key for one field: arrays by (shape, dtype, bytes); tuples/lists
        recursively; anything else by itself (it must already hash by value)."""
        if isinstance(v, np.ndarray):
            return ('__array__', v.shape, v.dtype.str, v.tobytes())
        if isinstance(v, (tuple, list)):
            return tuple(ValueHashedFields._value_key(x) for x in v)
        return v


    def require_parameters_are_fields(self) -> None:
        """Reject a parameter that was stashed on the instance instead of declared as a **field**.

        The value identity is built from ``dc.fields(self)``, so anything else an instance carries is
        invisible to it -- two objects that differ only in such a value compare EQUAL, and since these
        objects are jax ``aux_data`` one of them silently receives the other's compiled program. That
        makes "parameters are fields" a correctness rule, not a style preference, and this turns the
        three ways of breaking it (a hand-written ``__init__``, a bare class attribute assigned per
        instance, a class that forgot its ``@dataclass`` decorator) into a construction-time error.

        Call it from ``__post_init__``. Values cached by ``functools.cached_property`` also live in
        ``__dict__`` by design and are skipped -- they are derived, never parameters."""
        declared = {f.name for f in dc.fields(self)}
        cached = {name for cls in type(self).__mro__
                  for name, attr in vars(cls).items() if isinstance(attr, ft.cached_property)}
        stray = set(self.__dict__) - declared - cached
        if stray:
            raise TypeError(
                "%s carries %s outside its dataclass fields. Parameters must be declared as fields: the "
                "value identity (and hence the jax compilation cache key) is built from the fields alone, "
                "so an instance attribute set some other way is invisible to it and two differently "
                "parameterized objects would compare equal -- one would silently get the other's compiled "
                "program. Declare it as an annotated field on a @dataclass(frozen=True, eq=False) class."
                % (type(self).__name__, ', '.join(sorted(repr(k) for k in stray))))

    @ft.cached_property
    def _fields_key(self) -> typ.Tuple:
        # Checked HERE, not in __post_init__: a hand-written __init__ sets its attributes AFTER calling
        # the dataclass __init__ (which is what runs __post_init__), so at that point there is nothing to
        # see. First hash/eq is both late enough to catch it and exactly the moment it would do harm.
        self.require_parameters_are_fields()
        return tuple(self._value_key(getattr(self, f.name)) for f in dc.fields(self))

    def __hash__(self) -> int:
        return hash((type(self), self._fields_key))   # type included: field-free siblings must not collide

    def __eq__(self, other) -> bool:
        if self is other:
            return True
        if type(other) is not type(self):
            return NotImplemented
        return self._fields_key == other._fields_key



@dc.dataclass(frozen=True, eq=False)
class StaticSkeleton(ValueHashedFields):
    """The static half of a partitioned data tuple -- tree shape plus the static values, hashed by value
    so it can ride as jax ``aux_data``. Produced by :py:func:`partition_static`."""
    tree: typ.Any = ()


def _is_static_leaf(x) -> bool:
    """Whether a leaf is STATIC STRUCTURE rather than traced data, per the uniform layer's contract.

    Two cases, and only two: a Python ``int``/``bool`` (a shape entry, an axis count) and a **host numpy
    boolean array** (a rank mask). Masks are always host numpy by construction and never traced
    (``docs/uniform_masks_vs_ranks.md``); a jax boolean array is therefore NOT a mask and stays traced,
    and an integer numpy *array* (the ``entries`` index sample) is data and stays traced too."""
    if isinstance(x, (bool, int)) and not isinstance(x, np.ndarray):
        return True
    return isinstance(x, np.ndarray) and x.dtype == np.bool_


def partition_static(
        tree,   # a tuple/list tree of arrays and static structure (a frame's .data, a frame sweep, ...)
) -> typ.Tuple[
    typ.Tuple,        # the dynamic leaves, in traversal order -- the jax pytree children
    StaticSkeleton,   # the tree shape + static values -- the jax pytree aux_data
]:
    """Split a raw backend data tuple into what jax should TRACE and what it must keep STATIC.

    Backend data tuples mix both -- a uniform frame is ``(4 supercores, shape, masks)`` -- and a bare
    tuple is a jax pytree whose every element is a leaf, so flattening one naively traces the masks and
    the shape. That raises (``require_concrete_masks``) or silently produces a program that cannot do
    host-integer shape arithmetic. The frontend avoids it by giving ``UT3Frame`` a registered pytree with
    the masks as aux; the backend keeps plain tuples, so the split happens here instead.

    Round-trips exactly through :py:func:`rebuild_static`."""
    dynamic = []

    def walk(node):
        if isinstance(node, (tuple, list)):
            return ('list' if type(node) is list else 'tuple', tuple(walk(e) for e in node))
        if _is_static_leaf(node):
            return ('static', node)
        dynamic.append(node)
        return ('dynamic', None)

    skeleton = walk(tree)                  # must run BEFORE tuple(dynamic) -- it is what fills it
    return tuple(dynamic), StaticSkeleton(skeleton)


def rebuild_static(
        dynamic:   typ.Sequence,   # the dynamic leaves from partition_static (or their traced/updated twins)
        skeleton:  StaticSkeleton,  # the matching skeleton
):
    """Reassemble the tuple tree :py:func:`partition_static` split, substituting ``dynamic`` in order."""
    it = iter(dynamic)

    def walk(node):
        tag, payload = node
        if tag == 'static':
            return payload
        if tag == 'dynamic':
            return next(it)
        rebuilt = [walk(e) for e in payload]
        return rebuilt if tag == 'list' else tuple(rebuilt)

    return walk(skeleton.tree)


def require_concrete_masks(
        *masks: NDArray,  # HOST bool, static -- the uniform structure masks (must NOT be traced)
) -> None:
    """Guard the uniform-mask contract: masks are concrete host (numpy) arrays, never jax tracers.

    Under jit any ``jnp`` op on a mask returns a tracer, which breaks the layer two ways: host-int
    shape/rank extraction (``int(mask.sum())``) raises ``ConcretizationTypeError``, and recomputed masks
    leak as tracers into the (identity-hashed, never-inspected) output ``aux_data`` -- silently invalid.
    So a traced mask here means the masks were passed *among* the traced jit args; the fix is functional,
    not numerical (raise, per the structural-vs-numerical philosophy). See
    ``docs/contributor/uniform_pytree_composition.md``.

    Infrastructure, not part of the ``ut3_*`` family (``docs/naming_conventions.md``): it guards the mask
    representation itself, so it serves every uniform object -- plain, frame, variations, and weights.
    """
    if not jax_available:
        return
    for m in masks:
        if isinstance(m, jax.core.Tracer):
            raise ValueError(
                'uniform masks must be concrete host (numpy) arrays, but a traced mask was seen -- you '
                'likely jitted a backend function with the masks among the traced args. Close over the '
                'masks as constants and trace only the supercores (the masks are static structure). '
                'See docs/contributor/uniform_pytree_composition.md.')


def prefix_mask(
        ranks: NDArray,  # HOST int, any shape (e.g. (d,)+stack_shape, or a plain int tuple); the real extent
        pad:   int,      # padded width of the edge
) -> NDArray:            # HOST bool, static, shape = ranks.shape + (pad,)
    """Boolean prefix indicator: slot ``j`` is real iff ``j < rank`` -- the canonical (prefix) form.

    The shared primitive under every prefix structure in the uniform layer: the rank edge masks
    (``ut3_make_masks`` / ``ufv_make_frame_masks``), the physical shape mask rebuilt from the static
    ``shape`` ints, the orthogonalization rank recurrences, and the weight layer's edge masks.

    It is deliberately **neutral** -- it belongs to neither the masking layer nor the weighting layer.
    Masks are boolean *structure*; weights are float *parameters* (opposite jax treatment: static aux vs
    traced leaf -- ``docs/contributor/uniform_rank_masks_rationale.md``). Both legitimately need prefix
    indicators, but the weighting layer must never route its *operations* through the masking layer, so
    the shared mechanics live here and each side calls this (``docs/contributor/weighted_internals.md`` §2).

    HOST numpy (``np``, never ``xnp``): a prefix mask is static structure, and a jax mask becomes a tracer
    under jit -- breaking ``int(mask.sum())`` extraction and leaking tracers into ``aux_data``. See
    ``docs/contributor/uniform_pytree_composition.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from t3toolbox.backend.common import prefix_mask
    >>> prefix_mask(np.array([2, 3]), 4).astype(int).tolist()   # per-edge ranks -> (2, 4) prefix rows
    [[1, 1, 0, 0], [1, 1, 1, 0]]
    >>> prefix_mask(np.array(1), 3).astype(int).tolist()        # a scalar rank -> one row
    [1, 0, 0]
    """
    return np.arange(pad) < np.asarray(ranks)[..., None]
