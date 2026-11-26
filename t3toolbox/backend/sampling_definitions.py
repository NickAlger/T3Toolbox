# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The sampling variables by their definitions: one slow, dense function per variable. A reference.

Sampling a Tucker tensor train at input vectors makes a family of intermediate arrays, the *sampling variables*,
and a few outputs. This module says what each of them **is**. Every function here is named exactly as its
variable and computes it from its definition, as the dense contraction of a part of the network, with no sweep
along the chain and no reuse of one variable by another. It is slow and meant for small problems. Use it to
check a fast computation against (the library's own tests do), to see what a variable means, or to check a
modification of your own. Each variable's algebraic type (``P``, ``Y*``, a turned ``P*`` or ``Y``, ``X*``) is the
last column of the variable table of :py:mod:`t3toolbox.backend.sampling_variables`.

**The network** has ``d`` sites. Site ``i`` holds a TT core over a Tucker core over an input vector::

     bond 0     bond 1     bond 2            bond d
       |--[TT 0]--|--[TT 1]--|--  ...  --[TT d-1]--|          the TT cores
            |          |                     |                Tucker edge i, below TT core i
         [Tu 0]     [Tu 1]               [Tu d-1]             the Tucker cores
            |          |                     |                mode edge i, below Tucker core i
          w[0]       w[1]                 w[d-1]              the inputs

Bond ``i`` is the edge to the left of site ``i``, for ``i = 0 .. d``; bonds ``0`` and ``d`` are the two ends, each
closed with a vector of ones (so cores with tails are allowed).

**A frame** ``(D, U, L, R)`` holds four stores, ``d`` cores each. At site ``i``: ``D[i]``, the down Tucker core,
``(nDi, Ni)``; ``L[i]``, the left TT core, ``(rLi, nDi, rL(i+1))``; ``R[i]``, the right TT core,
``(rRi, nDi, rR(i+1))``; ``U[i]``, the up TT core, ``(rLi, nUi, rR(i+1))``. The base point is ``(D, L)``: the
tensor is every ``L[i]`` over its ``D[i]``. Nothing here needs the stores to be orthogonal, or even to agree
with each other: each definition says which cores it contracts.

**A name** is ``<letter>[_<basis>][_tilde][_dot]``.

- The letter is the side of an edge that is contracted: ``mu`` the left of a bond (sites ``0 .. i-1``), ``nu`` the
  right of a bond (sites ``i .. d-1``), ``xi`` the leaf below a Tucker edge (Tucker core ``i`` on its input),
  ``eta`` the trunk above a Tucker edge (everything but that leaf).
- The basis is the store whose basis the open edge is written in. It is spelled only where it is not the usual
  one (``L`` for ``mu``, ``R`` for ``nu``, ``D`` for ``xi``, ``U`` for ``eta``): ``nu_L``, ``eta_D``, ``mu_R``,
  ``xi_U``.
- ``_dot``: a direction is put in at one site, summed over the sites. In the frame's form the direction is
  ``variations = (V, H)``: at its site it is ``H[j]`` over ``D[j]``, or ``U[j]`` over ``V[j]``, with ``L`` cores
  before it and ``R`` cores after it. In the raw form it is ``core_directions = (D_dot, L_dot)`` at the base
  point: ``L_dot[j]`` over ``D[j]``, or ``L[j]`` over ``D_dot[j]``, every other TT core an ``L`` core.
- ``_tilde``: the seed ``w_tilde`` is put in. For ``probe`` it takes the place of one input, summed over the
  inputs. For ``apply`` and ``entries`` the seed is one number per sample and multiplies the vector of ones that
  closes bond ``d``. A side that reaches bond ``d`` (a right side, a trunk) then has the seed as a factor: its
  tilde variable is the seed times the plain one (``nu_tilde``, ``nu_L_tilde``, ``eta_tilde``, ``eta_D_tilde``).
  A side that does not (a left side, a leaf) has no place for it, and its tilde variable is zero (``xi_tilde``,
  ``mu_tilde``).
- Both marks: both are put in, summed over both places. A tilde-dot variable is the derivative of its tilde
  variable along the raw direction with the seed held. For ``apply`` and ``entries``: the seed times the raw dot
  (``nu_L_tilde_dot``, ``eta_D_tilde_dot``), or zero (``xi_tilde_dot``, ``mu_tilde_dot``).

**The multipliers.** A tilde variable is the multiplier of the dot of the other side of its edge, in the same
basis: ``nu_tilde`` of ``mu_R_dot``, and ``eta_tilde`` of ``xi_U_dot``, for every kind; ``mu_tilde`` of
``nu_L_dot``, and ``xi_tilde`` of ``eta_D_dot``, for the probe kind only. For ``apply`` and ``entries`` the seed
is on the right of every bond and in every trunk, so the last two do not hold there (those tilde variables are
zero).

**The three kinds** differ in the inputs and the outputs. ``'probe'`` and ``'apply'`` take vectors ``ww``
(input jets, below); ``'entries'`` takes integer grid points ``index``, whose inputs are the unit vectors
``e_index``, and has no jets. ``'probe'`` has one output per mode, ``z[i]``, with mode edge ``i`` left open;
``'apply'`` and ``'entries'`` have one number per sample.

**Jets.** The jet shape ``order = (n_1, ..., n_m)`` is a tuple of maximum orders, one per jet variable ``s_1 ...
s_m``; ``order = ()`` is no jets. The inputs are then **input jets**: ``ww[i]`` has ``m`` leading jet axes of
sizes of its own, ``F_in_i``, and its slot at the multi-index ``k`` is the Taylor coefficient of
``s_1^k_1 ... s_m^k_m`` of the input vector (zero beyond its sizes). The line ``w + s p`` is the input jet of two
slots, ``np.stack([w, p])``; two directions make a jet in two variables; a curve has more slots. Every variable is
then a polynomial in ``s``, returned as its Taylor coefficients (not derivatives) on the box ``F = (n_1 + 1, ...,
n_m + 1)``, on ``m`` leading jet axes. **The declared jet sizes**: the input ``w[i]`` has its own, as given; the
leaves ``xi[i]``, ``xi_dot[i]`` and ``xi_U_dot[i]`` have the input's, clipped to ``F`` (exact, of the input's
degree); every other variable has ``F``. Input slots beyond the box never reach a variable and are not read.

The coefficients are found by evaluating the dense contractions on a grid of complex roots of unity, ``M_j`` of
them in ``s_j``, and taking the discrete Fourier transform of the values along each axis. With ``a_j`` the largest
input size along axis ``j`` (clipped to ``n_j + 1``), every variable has degree at most ``d (a_j - 1) + n_j`` in
``s_j`` (a product of ``d`` inputs and one seed), so ``M_j = d (a_j - 1) + n_j + 1`` is exact; for the apply kind
with every input of ``a_j`` slots that degree is reached, and one node fewer would not do. It loses no digits as the
order grows (the transform is unitary up to a scale), and a coefficient that is zero in truth comes out at rounding
level. The outputs and the blocks do not interpolate at all. The cost grows with the grid, ``M_1 ... M_m``
evaluations per variable, so the module is for small problems.

**The seed, and the turn.** The data ``y`` and the residual ``r = z - y`` are of the type of ``z``, in forward
order. The seed is the residual made into a dual jet: weighted, then turned, a flip of every jet axis,
``w_tilde[k] = b[order - k]**2 * f[order - k]**2 * r[order - k]`` (``k`` a multi-index; ``b`` the block weight's
row of the mode, read at the slot; ``f`` the full weight, read at the same entry as the residual). As a polynomial
in ``s`` the seed is
``sum_k w_tilde[k] s^k``, so the tilde and tilde-dot variables are stored turned too: paired with a jet, the
number they stand for is the coefficient of the corner ``s_1^n_1 ... s_m^n_m`` of the product. Pass ``y`` (and the weights) for the seed
of the residual, or ``w_tilde`` for a seed of your own; ``w_tilde`` wins where both are given.

**The arguments** are the same for every function:

- ``kind``: ``'probe'``, ``'apply'`` or ``'entries'``.
- ``frame``: ``(D, U, L, R)``, each ``len=d``. A corewise point ``(B, G)`` is the frame ``(B, G, G, G)``; on it
  ``nu_L`` and ``nu`` are one array, and so are ``eta_D`` and ``eta``.
- ``sample``: the input jets ``ww`` (``len=d``, ``elm_shape=F_in_i+W+(Ni,)``; with ``order = ()``, ``W+(Ni,)``),
  or for ``'entries'`` the grid points ``index`` (int, ``shape=(d,)+W``). ``order`` says how many leading axes of
  ``ww[i]`` are jet axes, never the shape of the array.
- ``i``: the index of the variable: a bond ``0 .. d`` for ``mu`` and ``nu`` (in any basis and grade), a site
  ``0 .. d-1`` for ``xi``, ``eta``, ``w`` and the blocks, a mode ``0 .. d-1`` for ``z``, ``r``, ``Jp`` and
  ``w_tilde`` of the probe kind. ``None`` (the default) returns the whole variable: a tuple over the index, or,
  for a variable with no index, the array itself.
- ``order``: the jet shape, a tuple; ``()`` (the default) for no jets, and the only one ``'entries'`` takes.
- ``variations``: ``(V, H)``, ``V[i]`` of shape ``(nUi, Ni)`` and ``H[i]`` of shape ``(rLi, nDi, rR(i+1))``.
- ``core_directions``: ``(D_dot, L_dot)``, with the shapes of ``D`` and ``L``.
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
- ``w_tilde``: a seed, of the type of ``z``, stored turned.

A function reads only the arguments its variable depends on, and raises a ``ValueError`` if one of those is
missing.

**What is not here.** The stacks ``K`` (of directions) and ``C`` (of base points): apply these functions element
by element. The uniform layout: this module is for the ragged layout only, and a uniform ``.data`` must not be
passed (that of a uniform Tucker tensor train also has four entries and would be read as four stores, with no
error). Complex arrays (nothing checks for them: with jets the imaginary part is dropped without an error). Problems with more than 16 sites. The sample stack ``W`` is supported, with any number of
axes, and the blocks (``g``, ``JtJp``, ``Kp``, ``V_tilde``, ...) are summed over it.

**How each is computed.** An edge variable (the four letters, in every basis and grade) is one dense ``einsum``
over the cores and inputs of its side, summed over the places of the direction and of the seed. ``z`` is the
dense tensor of ``(D, L)`` through :py:func:`~t3toolbox.backend.probing.dense_probe`,
:py:func:`~t3toolbox.backend.apply.dense_apply` or :py:func:`~t3toolbox.backend.entries.dense_entries`. ``Jp``
is the same function of the dense tangent tensor. ``g`` and ``JtJp`` are the dense Jacobian, built by applying
the forward map to a basis of directions, transposed onto the seed. ``Kp`` is the second derivative of the dense
objective ``<seed, z>`` in the cores of the base point: by central finite differences for numpy arrays (exact
to rounding, the objective being a polynomial along the direction) and by automatic differentiation for jax
arrays.
"""
import math
import typing as typ
import dataclasses as dc

import numpy as np

import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
from t3toolbox.backend.common import *

if jax_available:
    import jax

__all__ = [
    # The inputs and the seed
    'w',
    'w_tilde',
    # Plain
    'xi',
    'mu',
    'nu',
    'eta',
    'nu_L',
    'eta_D',
    # Dot, the frame's form of the direction
    'xi_U_dot',
    'mu_R_dot',
    'nu_L_dot',
    'eta_D_dot',
    # Dot, the raw form of the direction
    'xi_dot',
    'mu_dot',
    # Tilde
    'xi_tilde',
    'mu_tilde',
    'nu_tilde',
    'eta_tilde',
    'nu_L_tilde',
    'eta_D_tilde',
    # Tilde-dot
    'xi_tilde_dot',
    'mu_tilde_dot',
    'nu_L_tilde_dot',
    'eta_D_tilde_dot',
    # Outputs and blocks
    'z',
    'r',
    'Jp',
    'g',
    'JtJp',
    'Kp',
    'V_tilde',
    'H_tilde',
    'V_tilde_dot',
    'H_tilde_dot',
    # Sums
    'f',
    'r_block_sumsq',
    'Jp_norm_sq',
]


######################################################
########    The arguments of one definition    #######
######################################################

_KINDS = ('probe', 'apply', 'entries')


def _read_jet_shape(
        order:  typ.Any,            # the jet shape as given
) -> typ.Tuple[int, ...]:           # the jet shape: a tuple of Python ints >= 0, one per jet variable
    '''The jet shape: a tuple of maximum orders, ``()`` for no jets, ``(n,)`` for one variable. A tuple only, so an
    integer or a list raises; each entry a nonnegative integer (a Python ``int`` or a numpy integer scalar, never a
    ``bool``), returned as Python ints. The rule of the library; written here again, since this module shares no
    code with the library but the dense functions.'''
    if not isinstance(order, tuple):
        raise TypeError('order is the jet shape, a tuple of maximum orders, one per jet variable (() for none, (n,) '
                        'for one), got %r' % (order,))
    for n in order:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):       # a numpy bool is not an np.integer
            raise TypeError('every entry of the jet shape is an integer, got %r' % (order,))
        if n < 0:
            raise ValueError('every entry of the jet shape is >= 0, got %r' % (order,))
    return tuple(int(n) for n in order)


@dc.dataclass(frozen=True, eq=False)
class _Setting:
    '''What every definition reads, worked out once from the common arguments: the stores, and the inputs evaluated
    on the grid of nodes. ``M = G + W`` below: the grid's axes ``G = (M_1, ..., M_m)``, then the sample stack.'''
    kind:        str                       # 'probe' | 'apply' | 'entries'
    d:           int                       # the number of sites
    order:       typ.Tuple[int, ...]       # the jet shape (n_1, ..., n_m)
    box:         typ.Tuple[int, ...]       # F = (n_1 + 1, ..., n_m + 1)
    D:           typ.List[NDArray]         # len=d, elm_shape=(nDi,Ni)
    U:           typ.List[NDArray]         # len=d, elm_shape=(rLi,nUi,rR(i+1))
    L:           typ.List[NDArray]         # len=d, elm_shape=(rLi,nDi,rL(i+1))
    R:           typ.List[NDArray]         # len=d, elm_shape=(rRi,nDi,rR(i+1))
    ww:          typ.List[NDArray]         # len=d, elm_shape=F_in_i+W+(Ni,): the input jets as given (unit vectors
                                           #   for entries)
    leaf_sizes:  typ.List[typ.Tuple[int, ...]]     # len=d: the input jet sizes clipped to F, min(F_in_i, F)
    nodes:       typ.Tuple[np.ndarray, ...]        # HOST complex128, len=m, elm_shape=(M_j,): the values of s_j
    inputs:      typ.List[NDArray]         # len=d, elm_shape=M+(Ni,): the input jets on the grid; complex if m >= 1
    M:           typ.Tuple[int, ...]       # G + W


def _setting(
        kind:    str,           # 'probe' | 'apply' | 'entries'
        frame:   typ.Tuple,     # (D, U, L, R)
        sample:  typ.Any,       # ww (the input jets, len=d, elm_shape=F_in_i+W+(Ni,)) | index (entries)
        order:   typ.Any,       # the jet shape
) -> _Setting:
    '''Read the common arguments. Raises only on what cannot be run: a jet shape that is not one, an unknown kind,
    an entries call with jets, a frame that is not four stores, more sites than the einsum letters allow.

    The grid. Along jet variable ``j`` the inputs, clipped to the box ``F``, have at most ``a_j`` slots, so every
    edge variable, a product of at most ``d`` inputs and at most one seed of ``n_j + 1`` slots, has degree at most
    ``d (a_j - 1) + n_j`` in ``s_j``. ``M_j = d (a_j - 1) + n_j + 1`` roots of unity along that axis are then enough
    for the discrete Fourier transform to give its coefficients exactly (see `_taylor`).'''
    order = _read_jet_shape(order)
    if kind not in _KINDS:
        raise ValueError("unknown kind %r: expected 'probe', 'apply' or 'entries'" % (kind,))
    if kind == 'entries' and order != ():
        raise ValueError("the entries kind has no jets: its order is (), got %r. Along a curve that leaves the "
                         "grid point, use 'apply' at an input jet whose slot 0 is the one-hot vectors" % (order,))
    if len(frame) != 4:
        raise ValueError('a frame is the four stores (D, U, L, R); got %d. A corewise point (B, G) is the frame '
                         '(B, G, G, G)' % len(frame))
    D, U, L, R = (list(store) for store in frame)
    d = len(D)
    if d > len(_BOND) - 1:
        raise ValueError('the definitions are for small problems: at most %d sites, got %d' % (len(_BOND) - 1, d))
    xnp, _, _ = get_backend(False, tree_contains_jax((frame, sample)))

    m, box = len(order), tuple(n + 1 for n in order)
    if kind == 'entries':
        index = xnp.asarray(sample)
        ww = [xnp.eye(D[i].shape[1], dtype=D[i].dtype)[index[i]] for i in range(d)]     # the unit vectors e_index
    else:
        ww = list(sample)
    clipped = [w[tuple(slice(0, f) for f in box)] for w in ww]                         # slots beyond F reach nothing
    leaf_sizes = [tuple(w.shape[:m]) for w in clipped]
    largest = [max(sizes[j] for sizes in leaf_sizes) for j in range(m)]
    nodes = tuple(np.exp(2j * np.pi * np.arange(M_j) / M_j)
                  for M_j in (d * (a - 1) + n + 1 for a, n in zip(largest, order)))
    inputs = [_on_grid(nodes, w) for w in clipped]
    return _Setting(kind, d, order, box, D, U, L, R, ww, leaf_sizes, nodes, inputs, tuple(inputs[0].shape[:-1]))


def _complex(
        values:  np.ndarray,    # HOST complex128
        like:    NDArray,       # a real array of the computation
) -> NDArray:                   # the values in the array module of `like`, complex in its precision
    '''Host constants (the powers of the nodes) as a complex array beside ``like``: complex64 for float32,
    complex128 for float64, numpy or jax as ``like`` is.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((like,)))
    return xnp.asarray(values, dtype=xnp.result_type(like.dtype, np.complex64))


def _on_grid(
        nodes:  typ.Tuple[np.ndarray, ...],     # HOST complex128, len=m, elm_shape=(M_j,)
        jet:    NDArray,                        # shape=P+X: a polynomial in s_1 .. s_m, its m leading axes the slots
) -> NDArray:                                   # shape=G+X: its values on the grid, G = (M_1, ..., M_m)
    '''A jet evaluated on the grid of nodes: the sum over its slots ``k`` of ``jet[k] s_1^k_1 ... s_m^k_m``. With no
    jet variable, the jet itself.'''
    m = len(nodes)
    if m == 0:
        return jet
    grid = tuple(len(s_j) for s_j in nodes)
    rest = jet.ndim - m
    value = None
    for k in np.ndindex(*jet.shape[:m]):
        powers = np.ones(grid, dtype=complex)                   # s_1^k_1 ... s_m^k_m on the grid, on the host
        for j, (s_j, k_j) in enumerate(zip(nodes, k)):
            powers = powers * (s_j ** k_j).reshape((1,) * j + (len(s_j),) + (1,) * (m - j - 1))
        term = _complex(powers, jet).reshape(grid + (1,) * rest) * jet[k]
        value = term if value is None else value + term
    return value


def _taylor(
        S:          _Setting,
        values_MX:  NDArray,                                    # shape=M+X: a polynomial in s on the grid
        sizes:      typ.Optional[typ.Tuple[int, ...]] = None,   # the jet sizes kept; None: the box F
) -> NDArray:                                                   # shape=sizes+W+X: the Taylor coefficients
    '''The Taylor coefficients of a polynomial in ``s_1 ... s_m`` from its values on the grid, on the box ``sizes``.

    Along axis ``j`` the nodes are the ``M_j``-th roots of unity, ``s_k = exp(2 pi i k / M_j)``. There a polynomial of
    degree below ``M_j`` has the values ``f(s_k) = sum_t c_t exp(2 pi i k t / M_j)``, so its coefficients are the
    averages ``c_t = mean_k f(s_k) exp(-2 pi i k t / M_j)``: the discrete Fourier transform of the values, divided
    by ``M_j``, along each axis in turn. That is exact for such a polynomial, and it loses no digits at any order
    (the transform is unitary up to a scale, where real nodes give a Vandermonde system that loses digits as the
    order grows). The cores and the inputs are real, so the coefficients are, and the imaginary part, which is
    rounding, is dropped. With no jet variable there is no grid, and nothing to transform.'''
    m = len(S.order)
    if m == 0:
        return values_MX
    xnp, _, _ = get_backend(False, tree_contains_jax((values_MX,)))
    sizes = S.box if sizes is None else sizes
    coefficients = xnp.fft.fftn(values_MX, axes=tuple(range(m))) / math.prod(len(s_j) for s_j in S.nodes)
    return coefficients[tuple(slice(0, size) for size in sizes)].real


def _each(
        i:        typ.Optional[int],    # an index, or None for all
        indices:  typ.Sequence[int],    # the valid indices
        at:       typ.Callable,         # i -> the variable at i
        name:     str,
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:
    '''The variable at index ``i``, or the tuple over every index when ``i`` is None.'''
    if i is None:
        return tuple(at(j) for j in indices)
    if i not in indices:
        raise ValueError('%s[i] is defined for i = %d .. %d; got i = %r' % (name, indices[0], indices[-1], i))
    return at(i)


def _per_mode(
        S:      _Setting,
        i:      typ.Optional[int],      # a mode (probe only), or None for the whole output
        value:  typ.Any,                # of the type of z: a tuple of d (probe), or one array (apply, entries)
        name:   str,
) -> typ.Any:
    '''An output of the type of ``z`` at mode ``i``: the probe kind has one array per mode, the others one array.'''
    if S.kind == 'probe':
        return _each(i, range(S.d), lambda j: value[j], name)
    if i is not None:
        raise ValueError("%s has no index for the kind %r (one number per sample); got i = %r" % (name, S.kind, i))
    return value


def _no_index(i: typ.Optional[int], name: str) -> None:
    if i is not None:
        raise ValueError('%s has no index; got i = %r' % (name, i))


def _need(argument: typ.Any, name: str, what: str) -> typ.Any:
    '''The argument, or a ``ValueError`` saying that the variable ``name`` needs it.'''
    if argument is None:
        raise ValueError('%s needs %s' % (name, what))
    return argument


def _swap(items: typ.Sequence, j: int, item: typ.Any) -> typ.List:
    '''A copy of a list with entry ``j`` replaced.'''
    return list(items[:j]) + [item] + list(items[j + 1:])


def _others(d: int, i: int) -> typ.List[int]:
    '''Every site but ``i``.'''
    return [j for j in range(d) if j != i]


################################################################
########    The subnetworks: one dense einsum for each    #######
################################################################
#
# A network is given by its TT cores `tt`, its Tucker cores `tucker` and its inputs `inputs`, one of each per site.
# The three functions below contract one side of an edge of it, all at once.  The einsum letters: bond j, Tucker
# edge j and mode edge j each have their own.

_BOND   = 'abcdefghijklmnopq'       # bond j, j = 0 .. d
_TUCKER = 'rstuvwxyzABCDEFGH'       # Tucker edge j
_MODE   = 'IJKLMNOPQRSTUVWXY'       # mode edge j


def _einsum(
        subscripts:  str,
        operands:    typ.Sequence[NDArray],
) -> NDArray:
    '''One ``einsum`` of many operands (a whole subnetwork: ``2d`` cores and more). Both backends are told to order
    the pairwise contractions greedily. numpy left alone loops over every index of the subnetwork at once. jax
    left alone uses its default ``optimize``: on jax 0.4.30 (the CI floor) that is ``'optimal'``, opt_einsum's
    exhaustive path search, exponential in the number of operands, which never finished on a four-site network.
    So this is a deliberate exception to the house convention of a bare ``jnp.einsum``, which holds for the
    library's einsums of two or three operands; do not remove ``optimize`` here. The order does not change what
    is computed.'''
    if tree_contains_jax((operands,)):
        xnp, _, _ = get_backend(False, True)
        return xnp.einsum(subscripts, *operands, optimize='greedy')
    return np.einsum(subscripts, *operands, optimize='greedy')


def _subnetwork(
        tt:          typ.Sequence[NDArray],     # len=d, elm_shape=(ri,ni,r(i+1))
        tucker:      typ.Sequence[NDArray],     # len=d, elm_shape=(ni,Ni)
        inputs:      typ.Sequence[NDArray],     # len=d, elm_shape=M+(Ni,)
        tt_sites:    typ.Sequence[int],         # the sites whose TT core is in the subnetwork
        leaf_sites:  typ.Sequence[int],         # the sites whose leaf (Tucker core on its input) is in it
        open_leg:    str,                       # the einsum letter of the one open edge
) -> NDArray:                                   # shape=M+(size of the open edge,)
    '''Contract a part of the network in one ``einsum``: the TT cores of ``tt_sites``, the leaves of
    ``leaf_sites``, and the vector of ones at an end bond that a TT core of the part touches.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((tt, tucker, inputs)))
    d = len(tt)
    subscripts, operands = [], []
    if 0 in tt_sites:
        subscripts.append(_BOND[0])
        operands.append(xnp.ones(tt[0].shape[0], dtype=tt[0].dtype))            # the left end
    if d - 1 in tt_sites:
        subscripts.append(_BOND[d])
        operands.append(xnp.ones(tt[d - 1].shape[2], dtype=tt[d - 1].dtype))    # the right end
    for j in tt_sites:
        subscripts.append(_BOND[j] + _TUCKER[j] + _BOND[j + 1])
        operands.append(tt[j])
    for j in leaf_sites:
        subscripts.append(_TUCKER[j] + _MODE[j])
        operands.append(tucker[j])
        subscripts.append('...' + _MODE[j])
        operands.append(inputs[j])
    stack = '...' if len(leaf_sites) > 0 else ''                                # no input: nothing carries M
    value = _einsum(','.join(subscripts) + '->' + stack + open_leg, operands)
    return xnp.broadcast_to(value, tuple(inputs[0].shape[:-1]) + tuple(value.shape[-1:]))


def _left_side(
        tt:      typ.Sequence[NDArray],     # len=d, elm_shape=(ri,ni,r(i+1))
        tucker:  typ.Sequence[NDArray],     # len=d, elm_shape=(ni,Ni)
        inputs:  typ.Sequence[NDArray],     # len=d, elm_shape=M+(Ni,)
        i:       int,                       # a bond, 0 .. d
) -> NDArray:                               # shape=M+(ri,)
    '''The left of bond ``i``: sites ``0 .. i-1``, each TT core over its Tucker core and its input, from the ones
    at bond 0. The open leg is bond ``i``. At ``i = 0`` it is the ones themselves.'''
    if i == 0:
        xnp, _, _ = get_backend(False, tree_contains_jax((tt, inputs)))
        return xnp.ones(tuple(inputs[0].shape[:-1]) + (tt[0].shape[0],), dtype=tt[0].dtype)
    return _subnetwork(tt, tucker, inputs, range(i), range(i), _BOND[i])


def _right_side(
        tt:      typ.Sequence[NDArray],     # len=d, elm_shape=(ri,ni,r(i+1))
        tucker:  typ.Sequence[NDArray],     # len=d, elm_shape=(ni,Ni)
        inputs:  typ.Sequence[NDArray],     # len=d, elm_shape=M+(Ni,)
        i:       int,                       # a bond, 0 .. d
) -> NDArray:                               # shape=M+(ri,)
    '''The right of bond ``i``: sites ``i .. d-1``, each TT core over its Tucker core and its input, up to the ones
    at bond ``d``. The open leg is bond ``i``. At ``i = d`` it is the ones themselves.'''
    d = len(tt)
    if i == d:
        xnp, _, _ = get_backend(False, tree_contains_jax((tt, inputs)))
        return xnp.ones(tuple(inputs[0].shape[:-1]) + (tt[d - 1].shape[2],), dtype=tt[d - 1].dtype)
    return _subnetwork(tt, tucker, inputs, range(i, d), range(i, d), _BOND[i])


def _trunk(
        tt:      typ.Sequence[NDArray],     # len=d, elm_shape=(ri,ni,r(i+1))
        tucker:  typ.Sequence[NDArray],     # len=d, elm_shape=(ni,Ni); tucker[i] is not read
        inputs:  typ.Sequence[NDArray],     # len=d, elm_shape=M+(Ni,); inputs[i] is not read
        i:       int,                       # a site, 0 .. d-1
) -> NDArray:                               # shape=M+(ni,)
    '''The trunk above Tucker edge ``i``: the whole network but the leaf of site ``i``. The open leg is the Tucker
    edge of ``tt[i]``.'''
    d = len(tt)
    return _subnetwork(tt, tucker, inputs, range(d), _others(d, i), _TUCKER[i])


def _leaf(
        tucker_core:  NDArray,      # shape=(ni,Ni)
        input_MN:     NDArray,      # shape=M+(Ni,)
) -> NDArray:                       # shape=M+(ni,)
    '''The leaf below a Tucker edge: a Tucker core applied to its input.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((tucker_core, input_MN)))
    return xnp.einsum('io,...o->...i', tucker_core, input_MN)


def _tensor(
        tt:      typ.Sequence[NDArray],     # len=d, elm_shape=(ri,ni,r(i+1)); one core of the network may carry
        tucker:  typ.Sequence[NDArray],     # len=d, elm_shape=(ni,Ni);          leading stack axes X
) -> NDArray:                               # shape=X+(N0,...,N(d-1))
    '''The dense tensor of a network: every TT core over its Tucker core, the mode edges open, the end bonds
    closed with ones. One ``einsum``.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((tt, tucker)))
    d = len(tt)
    subscripts = [_BOND[0], _BOND[d]]
    operands = [xnp.ones(tt[0].shape[-3], dtype=tt[0].dtype), xnp.ones(tt[d - 1].shape[-1], dtype=tt[d - 1].dtype)]
    for j in range(d):
        subscripts.append('...' + _BOND[j] + _TUCKER[j] + _BOND[j + 1])
        operands.append(tt[j])
        subscripts.append('...' + _TUCKER[j] + _MODE[j])
        operands.append(tucker[j])
    return _einsum(','.join(subscripts) + '->...' + _MODE[:d], operands)


def _zeros(
        S:     _Setting,
        size:  int,
) -> NDArray:               # shape=M+(size,): an empty sum of subnetworks
    xnp, _, _ = get_backend(False, tree_contains_jax((S.inputs,)))
    return xnp.zeros(S.M + (size,), dtype=S.D[0].dtype)


def _bond_size(tt: typ.Sequence[NDArray], i: int) -> int:
    '''The size of bond ``i`` in the basis of the TT cores ``tt``.'''
    return tt[i].shape[0] if i < len(tt) else tt[-1].shape[2]


#####################################################################
########    Where the direction goes, and where the seed goes    #####
#####################################################################

def _frame_form(
        S:           _Setting,
        variations:  typ.Tuple,             # (V, H)
        sites:       typ.Sequence[int],     # the sites of the subnetwork
        leaf_sites:  typ.Sequence[int],     # those of them whose leaf is in the subnetwork
) -> typ.List[typ.Tuple[typ.List[NDArray], typ.List[NDArray]]]:    # the networks (tt, tucker), one per place
    '''The frame's form of a direction, at each place it can be put: ``H[j]`` over ``D[j]``, or ``U[j]`` over
    ``V[j]``, with ``L`` cores before site ``j`` and ``R`` cores after it.'''
    V, H = variations
    networks = []
    for j in sites:
        networks.append((S.L[:j] + [H[j]] + S.R[j + 1:], S.D))
        if j in leaf_sites:
            networks.append((S.L[:j] + [S.U[j]] + S.R[j + 1:], _swap(S.D, j, V[j])))
    return networks


def _raw_form(
        S:                _Setting,
        core_directions:  typ.Tuple,            # (D_dot, L_dot)
        sites:            typ.Sequence[int],    # the sites of the subnetwork
        leaf_sites:       typ.Sequence[int],    # those of them whose leaf is in the subnetwork
) -> typ.List[typ.Tuple[typ.List[NDArray], typ.List[NDArray]]]:    # the networks (tt, tucker), one per place
    '''The raw form of a direction, at each place it can be put: ``L_dot[j]`` over ``D[j]``, or ``L[j]`` over
    ``D_dot[j]``, every other TT core an ``L`` core.'''
    D_dot, L_dot = core_directions
    networks = []
    for j in sites:
        networks.append((_swap(S.L, j, L_dot[j]), S.D))
        if j in leaf_sites:
            networks.append((S.L, _swap(S.D, j, D_dot[j])))
    return networks


def _either_form(
        S:                _Setting,
        variations:       typ.Optional[typ.Tuple],
        core_directions:  typ.Optional[typ.Tuple],
        sites:            typ.Sequence[int],
        leaf_sites:       typ.Sequence[int],
        name:             str,
) -> typ.List[typ.Tuple[typ.List[NDArray], typ.List[NDArray]]]:
    '''For the two names that both forms of the direction share (``nu_L_dot``, ``eta_D_dot``): the raw form
    where ``core_directions`` is given, also when ``variations`` is given too; else the frame's form.'''
    if core_directions is not None:
        return _raw_form(S, core_directions, sites, leaf_sites)
    return _frame_form(S, _need(variations, name, 'variations (V, H) or core_directions (D_dot, L_dot)'),
                       sites, leaf_sites)


def _seeded(
        S:              _Setting,
        seed_M:         typ.Any,                # the seed at the nodes: probe len=d, elm_shape=M+(Ni,); else M
        leaf_sites:     typ.Sequence[int],      # the sites whose leaf is in the subnetwork
        has_right_end:  bool,                   # whether the subnetwork reaches the ones at bond d
) -> typ.List[typ.Tuple[typ.List[NDArray], typ.Optional[NDArray]]]:    # (inputs, factor), one per place
    '''The seed, at each place it can be put. For probe: in place of one input of the subnetwork, ``factor`` None.
    For apply and entries: on the ones at bond ``d``, which it multiplies, so the inputs stay and ``factor`` is
    the seed; a subnetwork that does not reach bond ``d`` has no place for it.'''
    if S.kind == 'probe':
        return [(_swap(S.inputs, k, seed_M[k]), None) for k in leaf_sites]
    return [(S.inputs, seed_M)] if has_right_end else []


def _times(
        value_MX:  NDArray,                     # shape=M+(size,)
        factor_M:  typ.Optional[NDArray],       # shape=M, or None for one
) -> NDArray:
    return value_MX if factor_M is None else value_MX * factor_M[..., None]


def _sum(
        terms:  typ.Sequence[NDArray],
        zero:   NDArray,                # what an empty sum is
) -> NDArray:
    total = zero
    for term in terms:
        total = total + term
    return total


############################################################
########    The outputs, the seed and the pairing    ########
############################################################

def _dense_sample(
        S:       _Setting,
        tensor:  NDArray,       # shape=X+(N0,...,N(d-1))
) -> typ.Any:                   # of the type of z, with X after W: probe len=d, elm_shape=F+W+X+(Ni,); else F+W+X
    '''The kind's sampling of a dense tensor, through the library's dense function for the kind. For entries the
    inputs are already unit vectors (with no jets), so it is the dense apply at them. The dense functions take the
    input jets as given and read none of their slots beyond the box.'''
    if S.kind == 'probe':
        return probing.dense_probe(tensor, S.ww, S.order)
    return apply.dense_apply(tensor, S.ww, S.order)


def _z(S: _Setting, sample: typ.Any) -> typ.Any:
    '''``z``: the dense tensor of the base point ``(D, L)``, sampled. For entries through ``dense_entries``.'''
    tensor = _tensor(S.L, S.D)
    if S.kind == 'entries':                         # no jets: indexing
        return entries.dense_entries(tensor, sample)
    return _dense_sample(S, tensor)


def _Jp(S: _Setting, variations: typ.Tuple) -> typ.Any:
    '''``Jp``: the dense tangent tensor (the direction at one site, summed over the sites), sampled.'''
    tensors = [_tensor(tt, tucker) for tt, tucker in _frame_form(S, variations, range(S.d), range(S.d))]
    return _dense_sample(S, _sum(tensors[1:], tensors[0]))


def _r(S: _Setting, sample: typ.Any, y: typ.Any) -> typ.Any:
    '''``r = z - y``: the residual, of the type of ``z``, in forward order. ``y`` is of the type of ``z``, with ``z``'s
    shape exactly (checked, `_check_data_space`).'''
    def residual(i, z_TX, y_TX):
        _check_data_space('the data y', i, np.shape(y_TX), np.shape(z_TX), ())
        return z_TX - y_TX
    return _map_type_of_z(S, residual, _z(S, sample), _modes_of_data(S, y, 'the data y'))


def _map_type_of_z(S: _Setting, function: typ.Callable, *values: typ.Any) -> typ.Any:
    '''Apply ``function(i, *arrays)`` to objects of the type of ``z``: mode by mode for probe (``i`` the mode),
    once for apply and entries (``i`` None).'''
    if S.kind == 'probe':
        return tuple(function(i, *(value[i] for value in values)) for i in range(S.d))
    return function(None, *values)


def _modes_of_data(
        S:      _Setting,
        value:  typ.Any,                    # of the type of z: probe d arrays (a tuple, a list, or one array whose
                                            #   first axis is the mode); apply and entries one array
        what:   str,                        # for the messages
) -> typ.Any:                               # probe: a tuple of d arrays; otherwise the value
    '''The arrays of a data-space object, read by the reference's own rule: for probe exactly ``d`` of them, along the
    first axis of an array or the items of a tuple or a list (a mode axis of 1 is not broadcast).'''
    if S.kind != 'probe':
        if isinstance(value, (tuple, list)):
            raise ValueError('%s of the %s kind is one array, got a %s' % (what, S.kind, type(value).__name__))
        return value
    count = len(value) if isinstance(value, (tuple, list)) else (np.shape(value)[0] if np.ndim(value) else 0)
    if count != S.d:
        raise ValueError('%s of the probe kind has exactly d = %d arrays, one per mode; got %d' % (what, S.d, count))
    return tuple(value[i] for i in range(S.d))


def _check_data_space(
        what:        str,
        i:           typ.Optional[int],             # the mode (probe), None otherwise
        shape:       typ.Tuple[int, ...],           # the data-space array's shape
        of_shape:    typ.Tuple[int, ...],           # the shape of the array of z it goes with
        may_be_one:  typ.Tuple[int, ...],           # the axes where 1 is allowed too: the full weight's W; () for y
) -> None:
    '''The rule of a data-space object (``y``, the full weight): the rank of ``z``'s array, each axis ``z``'s size,
    except at ``may_be_one`` (the full weight's sample axes), where 1 is allowed too. A structural ``ValueError``
    otherwise.'''
    where = '' if i is None else ' (mode %d)' % i
    if len(shape) != len(of_shape) or any(a != b and not (a == 1 and axis in may_be_one)
                                          for axis, (a, b) in enumerate(zip(shape, of_shape))):
        rule = ('each sample axis 1 or its size, the jet axes and the mode edge exactly' if may_be_one else
                'exactly that shape')
        raise ValueError('%s%s has shape %s; it is of the type of z, whose array here has shape %s: the same rank, %s'
                         % (what, where, tuple(shape), tuple(of_shape), rule))


def _check_block_weight(
        S:             _Setting,
        block_weight:  typ.Any,                 # per block: probe (d,)+F, apply F, entries ()
) -> None:
    '''The block weight's shape is the block shape exactly: ``(d,) + F`` (probe), ``F`` (apply), ``()`` (entries); a
    sequence is refused (a ``TypeError``).'''
    if isinstance(block_weight, (tuple, list)):
        raise TypeError('the block weight is one array (a number for entries), not a %s: per mode and slot (d,) + F '
                        'for probe, per slot F for apply' % type(block_weight).__name__)
    blocks = ((S.d,) if S.kind == 'probe' else ()) + S.box
    if tuple(np.shape(block_weight)) != blocks:
        raise ValueError('the block weight has shape %s; the block shape of the %s kind at order %s is %s exactly: '
                         'probe (d,) + F, apply F, entries ()' % (tuple(np.shape(block_weight)), S.kind, S.order,
                                                                  blocks))


def _weight_squared(
        S:             _Setting,
        block_weight:  typ.Any,                 # per block: probe (d,)+F, apply F, entries (); None = ones
        full_weight:   typ.Any,                 # of the type of z: z's shape, each W axis 1 or W's size; None = ones
        i:             typ.Optional[int],       # the mode (probe), None otherwise
        shape:         typ.Tuple[int, ...],     # the shape of the array of the type of z it will multiply
) -> typ.Any:                                   # b**2 * f**2, broadcast positionally against `shape`, or 1.0
    '''``b**2 f**2`` for one array of the type of ``z`` (mode ``i`` for probe): the block row of the mode (or the
    table, apply; the number, entries) reshaped onto the leading jet axes, times the full weight's square read by
    position. ``1.0`` for a factor that is ``None``. No reshape of the full weight.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((block_weight, full_weight)))
    m = len(S.order)
    squared = 1.0
    if block_weight is not None:
        _check_block_weight(S, block_weight)
        row = xnp.asarray(block_weight if i is None else block_weight[i])            # F (probe, apply); () entries
        squared = xnp.reshape(row ** 2, tuple(row.shape) + (1,) * (len(shape) - m))  # left-aligned: the jet axes
    if full_weight is not None:
        f_X = _modes_of_data(S, full_weight, 'the full weight')
        f_X = xnp.asarray(f_X if i is None else f_X[i])
        _check_data_space('the full weight', i, f_X.shape, shape, tuple(range(m, len(S.M))))   # the sample axes W
        squared = squared * f_X ** 2
    return squared


def _turn(S: _Setting, jet_TX: NDArray) -> NDArray:
    '''Turn the jets: a flip of every jet axis, so slot ``k`` and slot ``order - k`` change places. Nothing with no
    jets.'''
    return jet_TX[(slice(None, None, -1),) * len(S.order)]


def _dual_jet(
        S:             _Setting,
        jet:           typ.Any,                 # of the type of z
        block_weight:  typ.Any,                 # per block: probe (d,)+F, apply F, entries (); None = ones
        full_weight:   typ.Any,                 # of the type of z: z's shape, each W axis 1 or W's size; None = ones
) -> typ.Any:                                   # of the type of z: b**2 f**2 jet, then every jet axis flipped
    '''The seed made from a jet: the weights, then the turn.'''
    return _map_type_of_z(S, lambda i, jet_TX: _turn(S, _weight_squared(S, block_weight, full_weight, i, jet_TX.shape)
                                                     * jet_TX), jet)


def _the_seed(
        S:             _Setting,
        sample:        typ.Any,
        y:             typ.Any,
        block_weight:  typ.Any,
        full_weight:   typ.Any,
        w_tilde:       typ.Any,
        name:          str,
) -> typ.Any:                                   # the seed, of the type of z, stored turned
    '''The seed a tilde variable is built on: ``w_tilde`` where it is given, else the one made from the residual
    ``z - y`` and the weights.'''
    if w_tilde is not None:
        return w_tilde
    y = _need(y, name, 'the data y (with optional weights), or a seed w_tilde')
    return _dual_jet(S, _r(S, sample, y), block_weight, full_weight)


def _at_nodes(
        S:     _Setting,
        seed:  typ.Any,         # of the type of z, stored turned: probe len=d, elm_shape=F+W+(Ni,); else F+W
) -> typ.Any:                   # the seed as a polynomial in s, at the nodes: probe len=d, elm_shape=M+(Ni,); else M
    '''``sum_k seed[k] s^k`` on the grid. This is where "stored turned" is used: slot ``k`` of the stored seed is the
    coefficient of ``s_1^k_1 ... s_m^k_m``.'''
    def at_nodes(i, seed_TX):
        xnp, _, _ = get_backend(False, tree_contains_jax((seed_TX,)))
        seed_TX = xnp.asarray(seed_TX)
        return _on_grid(S.nodes, seed_TX)
    return _map_type_of_z(S, at_nodes, seed)


def _pair(
        S:     _Setting,
        seed:  typ.Any,         # of the type of z, stored turned: probe len=d, elm_shape=F+W+(Ni,); else F+W
        jet:   typ.Any,         # of the type of z, with a stack X after W: F+W+X+(Ni,); else F+W+X
) -> NDArray:                   # shape=X
    '''The number a seed and a jet stand for together: the coefficient of ``s^order`` of their product, summed
    over the samples (and, for probe, over the modes and the mode edge). The same as the plain sum, slot by slot,
    of the jet against the seed turned back.'''
    def pair(i, seed_TX, jet_TX):
        xnp, _, _ = get_backend(False, tree_contains_jax((seed_TX, jet_TX)))
        functional_TX = _turn(S, xnp.asarray(seed_TX))                     # forward order again
        if i is not None:                                                   # probe: the mode edge next to W
            jet_TX = xnp.moveaxis(jet_TX, -1, functional_TX.ndim - 1)
        return xnp.tensordot(functional_TX, jet_TX, axes=functional_TX.ndim)
    pairs = _map_type_of_z(S, pair, seed, jet)
    return _sum(pairs[1:], pairs[0]) if S.kind == 'probe' else pairs


def _basis(
        shape:  typ.Tuple[int, ...],    # the shape of a core
        like:   NDArray,                # for the dtype and the array module
) -> NDArray:                           # shape=shape+shape: the unit cores, stacked on leading axes
    '''A basis of the cores of one shape, as a stack: ``basis[a, b, ...]`` is the core that is one at ``[a, b,
    ...]`` and zero elsewhere.'''
    xnp, _, _ = get_backend(False, tree_contains_jax((like,)))
    return xnp.eye(math.prod(shape), dtype=like.dtype).reshape(tuple(shape) + tuple(shape))


def _jacobian_transpose(
        S:     _Setting,
        seed:  typ.Any,                         # of the type of z, stored turned
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                # V blocks. len=d, elm_shape=(nUi,Ni)
        typ.Tuple[NDArray, ...],                # H blocks. len=d, elm_shape=(rLi,nDi,rR(i+1))
]:
    '''The dense Jacobian of ``(V, H) -> Jp``, transposed onto a seed. The forward map is applied to a basis of
    directions (one unit core at one site, every other core of the direction zero), and each result is paired
    with the seed: entry ``[a, o]`` of block ``i`` is the pairing of the seed with ``Jp`` along the unit ``V[i]``
    at ``[a, o]``.'''
    V_blocks, H_blocks = [], []
    for i in range(S.d):
        V_basis = _basis((S.U[i].shape[1], S.D[i].shape[1]), S.D[i])                      # U[i] over a unit V[i]
        Jp_basis = _dense_sample(S, _tensor(S.L[:i] + [S.U[i]] + S.R[i + 1:], _swap(S.D, i, V_basis)))
        V_blocks.append(_pair(S, seed, Jp_basis))
        H_basis = _basis((S.L[i].shape[0], S.D[i].shape[0], S.R[i].shape[2]), S.L[i])     # a unit H[i] over D[i]
        Jp_basis = _dense_sample(S, _tensor(S.L[:i] + [H_basis] + S.R[i + 1:], S.D))
        H_blocks.append(_pair(S, seed, Jp_basis))
    return tuple(V_blocks), tuple(H_blocks)


#############################################################################
########    Kp: the second derivative of the dense objective <seed, z>    ####
#############################################################################
#
# phi(B, G) = <seed, z(B, G)> is the pairing of a fixed seed with the sampled dense tensor of the cores (B, G).
# Kp is its second derivative at the base point (D, L), applied to (D_dot, L_dot).

def _objective(
        S:       _Setting,
        seed:    typ.Any,                   # of the type of z, stored turned
        tucker:  typ.Sequence[NDArray],     # B. len=d, elm_shape=(nDi,Ni); one core may carry a stack X
        tt:      typ.Sequence[NDArray],     # G. len=d, elm_shape=(rLi,nDi,rL(i+1))
) -> NDArray:                               # shape=X
    '''The dense objective ``phi(B, G) = <seed, z(B, G)>``.'''
    return _pair(S, seed, _dense_sample(S, _tensor(tt, tucker)))


def _Kp_by_finite_differences(
        S:                _Setting,
        seed:             typ.Any,          # of the type of z, stored turned
        core_directions:  typ.Tuple,        # (D_dot, L_dot)
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],            # V_tilde_dot. len=d, elm_shape=(nDi,Ni)
        typ.Tuple[NDArray, ...],            # H_tilde_dot. len=d, elm_shape=(rLi,nDi,rL(i+1))
]:
    '''``Kp`` by central finite differences of the gradient of the dense objective.

    ``phi`` is linear in each core, so its gradient in one core is ``phi`` with a basis in place of that core,
    with no step. Along ``(D, L) + e (D_dot, L_dot)`` that gradient is a polynomial in ``e`` of degree at most
    ``2 d - 1``, and its derivative at ``e = 0`` is taken by the central difference on the ``2 d`` points
    ``e = +-h, ..., +-d h``, which is exact for such a polynomial. So the result is right to rounding, with no
    truncation error.'''
    D_dot, L_dot = core_directions
    d, h = S.d, 1.0 / S.d
    stencil = [(-1.0) ** (k + 1) * math.factorial(d) ** 2 / (k * math.factorial(d - k) * math.factorial(d + k))
               for k in range(1, d + 1)]                        # f'(0) = sum_k stencil[k] (f(k h) - f(-k h)) / h

    def gradient(e):                                            # of phi at (D, L) + e (D_dot, L_dot), block by block
        tucker = [B + e * B_dot for B, B_dot in zip(S.D, D_dot)]
        tt = [G + e * G_dot for G, G_dot in zip(S.L, L_dot)]
        V_blocks = [_objective(S, seed, _swap(tucker, i, _basis(S.D[i].shape, S.D[i])), tt) for i in range(d)]
        H_blocks = [_objective(S, seed, tucker, _swap(tt, i, _basis(S.L[i].shape, S.L[i]))) for i in range(d)]
        return V_blocks + H_blocks

    blocks = None
    for k, c_k in zip(range(1, d + 1), stencil):
        terms = [(c_k / h) * (plus - minus) for plus, minus in zip(gradient(k * h), gradient(-k * h))]
        blocks = terms if blocks is None else [block + term for block, term in zip(blocks, terms)]
    return tuple(blocks[:d]), tuple(blocks[d:])


def _Kp_by_jax(
        S:                _Setting,
        seed:             typ.Any,          # of the type of z, stored turned
        core_directions:  typ.Tuple,        # (D_dot, L_dot)
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],            # V_tilde_dot. len=d, elm_shape=(nDi,Ni)
        typ.Tuple[NDArray, ...],            # H_tilde_dot. len=d, elm_shape=(rLi,nDi,rL(i+1))
]:
    '''``Kp`` by automatic differentiation: the derivative, along the direction, of the gradient of the dense
    objective. In jax, in the precision jax is set to.'''
    D_dot, L_dot = core_directions
    gradient = jax.grad(lambda cores: _objective(S, seed, cores[0], cores[1]))
    cores = (tree_to_jax(S.D), tree_to_jax(S.L))
    directions = (tree_to_jax(list(D_dot)), tree_to_jax(list(L_dot)))
    V_blocks, H_blocks = jax.jvp(gradient, (cores,), (directions,))[1]
    return tuple(V_blocks), tuple(H_blocks)


def _Kp(S: _Setting, seed: typ.Any, core_directions: typ.Tuple) -> typ.Tuple:
    '''``Kp``, by the route the arrays say, as everything in the library is dispatched: automatic differentiation
    where an array it reads is a jax array, finite differences where all are numpy (so a numpy result does not
    depend on whether jax is installed). The two agree to rounding.'''
    if tree_contains_jax((S.D, S.L, S.ww, seed, core_directions)):
        return _Kp_by_jax(S, seed, core_directions)
    return _Kp_by_finite_differences(S, seed, core_directions)


###############################################
########    The inputs and the seed    ########
###############################################


def w(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # w[i]. shape=min(F_in_i,F)+W+(Ni,)
    """``w[i]``: the input of site ``i``, as a jet.

    With no jets it is the input vector itself: ``ww[i]``, or for ``'entries'`` the unit vector ``e_index[i]``.
    With a jet shape it is the input jet ``ww[i]`` as it is given, of its own jet sizes ``F_in_i`` (slots beyond the
    box ``F`` are kept here, though no other variable reads them). For the line ``w + s p`` it has two slots,
    ``w[i][0] = ww[i]`` and ``w[i][1] = pp[i]``, whatever the order.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``w[i]``. shape=F_in_i+W+(Ni,)

    Examples
    --------
    The input as a jet: the input jet of the line ``ww + s pp``, two slots at any order:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> pp = [rng.standard_normal((7, n)) for n in N]                      # the direction the inputs move in
    >>> line = [np.stack([w, p]) for w, p in zip(ww, pp)]                  # the input jets: F_in = (2,)
    >>> w1 = sdef.w('probe', frame, line, 1, order=(3,))
    >>> print(w1.shape)                                                    # F_in + W + (N1,) = (2,) + (7,) + (5,)
    (2, 7, 5)
    >>> print(bool(np.all(w1[0] == ww[1])), bool(np.all(w1[1] == pp[1])))
    True True

    For ``'entries'`` the inputs are the unit vectors at the grid points:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> index = np.array([[3, 0, 1], [4, 4, 2], [0, 5, 5]])                # (d,)+W, W=(3,)
    >>> w0 = sdef.w('entries', frame, index, 0)
    >>> print(w0.shape, w0[0].tolist())                                    # W + (N0,); sample 0 is at index 3
    (3, 4) [0.0, 0.0, 0.0, 1.0]
    """
    S = _setting(kind, frame, sample, order)
    return _each(i, range(S.d), lambda i: S.ww[i], 'w')


def w_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the mode, 0 .. d-1 (probe only); None = all of it
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Any:                                                # w_tilde[i]. probe F+W+(Ni,); apply F+W; entries W
    """``w_tilde``: the seed, of the type of ``z``, stored turned.

    The seed is what the adjoint (tilde) variables are built on. With ``y`` given it is the residual ``r = z - y``
    made into a dual jet: weighted, then turned, a flip of every jet axis (``k`` a multi-index of the box ``F``),

        ``w_tilde[i][k] = b[i][order - k]**2 * f[i][order - k]**2 * r[i][order - k]``   (probe; no mode index for the
                                                                                     others; b a number for entries),

    the block weight ``b`` read at the slot, the full weight ``f`` at the same entry as the residual (positionally; a
    sample axis of 1 is constant).

    The turn is what makes its pairing with a jet the coefficient of the corner ``s_1^n_1 ... s_m^n_m`` of their
    product: ``sum_k w_tilde[order - k] * a[k] = sum_k b[k]**2 f[k]**2 r[k] a[k]``, the weighted inner product with
    the residual. With no jets there is nothing to turn. With ``w_tilde`` given, that seed is returned as it is.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; for probe with ``i`` None, a tuple of ``d`` of them.
        The seed. probe: ``w_tilde[i]``, shape=F+W+(Ni,); apply: shape=F+W; entries: shape=W

    Examples
    --------
    The seed of the residual: the weights, then the turn (here one jet variable, the line ``ww + s pp``):

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> pp = [rng.standard_normal((7, n)) for n in N]                      # the direction the inputs move in
    >>> line = [np.stack([w, p]) for w, p in zip(ww, pp)]                  # the input jets: F_in = (2,)
    >>> y = [rng.standard_normal((3, 7, n)) for n in N]                    # the data: F + W + (Ni,), order (2,)
    >>> block = 0.5 + rng.random((3, 3))                                   # (d,) + F: one number per mode and slot
    >>> args = dict(order=(2,), y=y, block_weight=block)
    >>> r1 = sdef.r('probe', frame, line, 1, **args)
    >>> w_tilde1 = sdef.w_tilde('probe', frame, line, 1, **args)
    >>> print(w_tilde1.shape)                                              # F + W + (N1,)
    (3, 7, 5)
    >>> print(bool(np.allclose(w_tilde1[0], block[1, 2] ** 2 * r1[2])))    # slot 0 holds order 2: turned
    True
    >>> print(bool(np.allclose(w_tilde1[2], block[1, 0] ** 2 * r1[0])))
    True

    For ``'apply'`` the seed is one number per sample (here with no jets and no weight: the residual):

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = rng.standard_normal(7)
    >>> print(bool(np.allclose(sdef.w_tilde('apply', frame, ww, y=y), sdef.z('apply', frame, ww) - y)))
    True
    """
    S = _setting(kind, frame, sample, order)
    return _per_mode(S, i, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'w_tilde'), 'w_tilde')


#############################
########    Plain    ########
#############################


def xi(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # xi[i]. shape=min(F_in_i,F)+W+(nDi,)
    """``xi[i]``: the leaf below Tucker edge ``i``, in the ``D`` basis: ``D[i]`` applied to the input ``w[i]``.

    Recurrence: ``xi[i] = xi_term(D[i], w[i])``, the contraction ``'io,Wo->Wi'``. It is a polynomial of the input's
    degree, so with jets it has the input's jet sizes, clipped to the box ``F`` (two slots each for a line).

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``xi[i]``. shape=min(F_in_i,F)+W+(nDi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> xi1 = sdef.xi('probe', frame, ww, 1)
    >>> print(xi1.shape)                                                   # W + (nD1,)
    (7, 3)
    >>> print(bool(np.allclose(xi1, np.einsum('io,Wo->Wi', D[1], ww[1]))))
    True
    """
    S = _setting(kind, frame, sample, order)
    return _each(i, range(S.d), lambda i: _taylor(S, _leaf(S.D[i], S.inputs[i]), S.leaf_sizes[i]), 'xi')


def mu(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # mu[i]. shape=F+W+(rLi,)
    """``mu[i]``: the left of bond ``i``, in the ``L`` basis: the ``L`` cores of sites ``0 .. i-1``, each over its
    ``D`` core and its input, from the ones at bond 0.

    Recurrence: ``mu[i+1] = mu_term(mu[i], L[i], xi[i])``, the contraction ``'Wa,aib,Wi->Wb'``, from
    ``mu[0] = 1``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``mu[i]``. shape=F+W+(rLi,)

    Examples
    --------
    One array per bond; the definition (one dense contraction of the whole left side) agrees with the recurrence:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> mu = sdef.mu('probe', frame, ww)                                   # i=None: every bond
    >>> print([m.shape for m in mu])                                       # W + (rLi,), i = 0 .. d
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> xi = sdef.xi('probe', frame, ww)
    >>> print(bool(np.all(mu[0] == 1)))
    True
    >>> print(bool(np.allclose(mu[2], np.einsum('Wa,aib,Wi->Wb', mu[1], L[1], xi[1]))))
    True

    With ``order``, the Taylor coefficients along ``ww + s pp`` on a leading axis; slot 0 is the plain variable:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> pp = [rng.standard_normal((7, n)) for n in N]                      # the direction the inputs move in
    >>> line = [np.stack([w, p]) for w, p in zip(ww, pp)]                  # the input jets of ww + s pp
    >>> mu2 = sdef.mu('probe', frame, line, 2, order=(3,))
    >>> print(mu2.shape)                                                   # F + W + (rL2,) = (4,) + (7,) + (3,)
    (4, 7, 3)
    >>> print(bool(np.allclose(mu2[0], sdef.mu('probe', frame, ww, 2))))
    True
    >>> print(bool(np.allclose(mu2[3], 0)))                                # two sites to the left: degree 2 in s
    True

    In two jet variables, along ``ww + s_1 pp + s_2 qq`` (input jets of sizes ``(2, 2)``, slot ``(1, 1)`` zero), at
    the jet shape ``(1, 2)``: two jet axes, and the slots of total degree above 2 are zero:

    >>> qq = [rng.standard_normal((7, n)) for n in N]
    >>> plane = [np.stack([np.stack([w, q]), np.stack([p, 0 * p])]) for w, p, q in zip(ww, pp, qq)]
    >>> mu2 = sdef.mu('probe', frame, plane, 2, order=(1, 2))
    >>> print(mu2.shape)                                                   # F + W + (rL2,) = (2, 3) + (7,) + (3,)
    (2, 3, 7, 3)
    >>> print(bool(np.allclose(mu2[0, 0], sdef.mu('probe', frame, ww, 2))), bool(np.allclose(mu2[1, 2], 0)))
    True True
    """
    S = _setting(kind, frame, sample, order)
    return _each(i, range(S.d + 1), lambda i: _taylor(S, _left_side(S.L, S.D, S.inputs, i)), 'mu')


def nu(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # nu[i]. shape=F+W+(rRi,)
    """``nu[i]``: the right of bond ``i``, in the ``R`` basis: the ``R`` cores of sites ``i .. d-1``, each over its
    ``D`` core and its input, up to the ones at bond ``d``.

    Recurrence: ``nu[i] = nu_term(R[i], xi[i], nu[i+1])``, the contraction ``'aib,Wi,Wb->Wa'``, from
    ``nu[d] = 1``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``nu[i]``. shape=F+W+(rRi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> nu = sdef.nu('probe', frame, ww)
    >>> print([n.shape for n in nu])                                       # W + (rRi,), i = 0 .. d
    [(7, 1), (7, 3), (7, 2), (7, 1)]
    >>> xi = sdef.xi('probe', frame, ww)
    >>> print(bool(np.allclose(nu[1], np.einsum('aib,Wi,Wb->Wa', R[1], xi[1], nu[2]))))
    True
    """
    S = _setting(kind, frame, sample, order)
    return _each(i, range(S.d + 1), lambda i: _taylor(S, _right_side(S.R, S.D, S.inputs, i)), 'nu')


def eta(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # eta[i]. shape=F+W+(nUi,)
    """``eta[i]``: the trunk above Tucker edge ``i``, in the ``U`` basis: ``U[i]`` with ``L`` cores to its left and
    ``R`` cores to its right, every site but ``i`` over its ``D`` core and its input.

    Recurrence: ``eta[i] = eta_term(mu[i], U[i], nu[i+1])``, the contraction ``'Wa,aib,Wb->Wi'``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``eta[i]``. shape=F+W+(nUi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> eta1 = sdef.eta('probe', frame, ww, 1)
    >>> print(eta1.shape)                                                  # W + (nU1,)
    (7, 2)
    >>> mu1, nu2 = sdef.mu('probe', frame, ww, 1), sdef.nu('probe', frame, ww, 2)
    >>> print(bool(np.allclose(eta1, np.einsum('Wa,aib,Wb->Wi', mu1, U[1], nu2))))
    True
    """
    S = _setting(kind, frame, sample, order)

    def eta_at(i):
        return _taylor(S, _trunk(S.L[:i] + [S.U[i]] + S.R[i + 1:], S.D, S.inputs, i))
    return _each(i, range(S.d), eta_at, 'eta')


def nu_L(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # nu_L[i]. shape=F+W+(rLi,)
    """``nu_L[i]``: the right of bond ``i``, in the ``L`` basis: as ``nu[i]`` with ``L`` cores in place of ``R`` cores.
    It is the right side at the base point ``(D, L)``.

    Recurrence: ``nu_L[i] = nu_term(L[i], xi[i], nu_L[i+1])``, from ``nu_L[d] = 1``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``nu_L[i]``. shape=F+W+(rLi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> nu_L = sdef.nu_L('probe', frame, ww)
    >>> print([n.shape for n in nu_L])                                     # W + (rLi,): the L basis, unlike nu
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> xi = sdef.xi('probe', frame, ww)
    >>> print(bool(np.allclose(nu_L[1], np.einsum('aib,Wi,Wb->Wa', L[1], xi[1], nu_L[2]))))
    True
    """
    S = _setting(kind, frame, sample, order)
    return _each(i, range(S.d + 1), lambda i: _taylor(S, _right_side(S.L, S.D, S.inputs, i)), 'nu_L')


def eta_D(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # eta_D[i]. shape=F+W+(nDi,)
    """``eta_D[i]``: the trunk above Tucker edge ``i``, in the ``D`` basis: as ``eta[i]`` with ``L[i]`` in place of
    ``U[i]`` and ``L`` cores to the right, so every TT core is an ``L`` core. It is the trunk at the base point
    ``(D, L)``, and the value is read off it: ``z[i] = z_term(eta_D[i], D[i])``.

    Recurrence: ``eta_D[i] = eta_term(mu[i], L[i], nu_L[i+1])``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``eta_D[i]``. shape=F+W+(nDi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> eta_D1 = sdef.eta_D('probe', frame, ww, 1)
    >>> print(eta_D1.shape)                                                # W + (nD1,)
    (7, 3)
    >>> mu1, nu_L2 = sdef.mu('probe', frame, ww, 1), sdef.nu_L('probe', frame, ww, 2)
    >>> print(bool(np.allclose(eta_D1, np.einsum('Wa,aib,Wb->Wi', mu1, L[1], nu_L2))))
    True
    >>> print(bool(np.allclose(sdef.z('probe', frame, ww, 1), np.einsum('Wi,io->Wo', eta_D1, D[1]))))   # the value
    True
    """
    S = _setting(kind, frame, sample, order)
    return _each(i, range(S.d), lambda i: _taylor(S, _trunk(S.L, S.D, S.inputs, i)), 'eta_D')


######################################################################
########    Dot: the frame's form of the direction, (V, H)    ########
######################################################################


def xi_U_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # xi_U_dot[i]. shape=min(F_in_i,F)+W+(nUi,)
    """``xi_U_dot[i]``: the leaf below Tucker edge ``i`` with the direction in it, in the ``U`` basis: ``V[i]`` applied
    to the input ``w[i]``. Needs ``variations``.

    Recurrence: ``xi_U_dot[i] = xi_term(V[i], w[i])``. With jets, the input's jet sizes clipped to ``F``, as ``xi``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``xi_U_dot[i]``. shape=min(F_in_i,F)+W+(nUi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> V = [rng.standard_normal((nU[i], N[i])) for i in range(3)]         # a direction at the frame: (V, H)
    >>> H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> xi_U_dot1 = sdef.xi_U_dot('probe', frame, ww, 1, variations=(V, H))
    >>> print(xi_U_dot1.shape)                                             # W + (nU1,): the U basis, unlike xi
    (7, 2)
    >>> print(bool(np.allclose(xi_U_dot1, np.einsum('io,Wo->Wi', V[1], ww[1]))))
    True
    """
    S = _setting(kind, frame, sample, order)
    V, H = _need(variations, 'xi_U_dot', 'variations (V, H)')
    return _each(i, range(S.d), lambda i: _taylor(S, _leaf(V[i], S.inputs[i]), S.leaf_sizes[i]), 'xi_U_dot')


def mu_R_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # mu_R_dot[i]. shape=F+W+(rRi,)
    """``mu_R_dot[i]``: the left of bond ``i`` with the direction at one of the sites ``0 .. i-1``, summed over the
    sites, in the ``R`` basis. Needs ``variations``.

    At its site ``j`` the direction is ``H[j]`` over ``D[j]``, or ``U[j]`` over ``V[j]``; the TT cores before it are
    ``L`` cores and those after it are ``R`` cores, which is why the open bond is in the ``R`` basis.

    Recurrence: ``mu_R_dot[i+1] = mu_term(mu_R_dot[i], R[i], xi[i]) + mu_term(mu[i], H[i], xi[i])
    + mu_term(mu[i], U[i], xi_U_dot[i])``, from ``mu_R_dot[0] = 0``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``mu_R_dot[i]``. shape=F+W+(rRi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> V = [rng.standard_normal((nU[i], N[i])) for i in range(3)]         # a direction at the frame: (V, H)
    >>> H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> mu_R_dot = sdef.mu_R_dot('probe', frame, ww, variations=(V, H))
    >>> print([m.shape for m in mu_R_dot])                                 # W + (rRi,): the R basis, unlike mu
    [(7, 1), (7, 3), (7, 2), (7, 1)]
    >>> mu, xi = sdef.mu('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_U_dot = sdef.xi_U_dot('probe', frame, ww, variations=(V, H))
    >>> mu_term = lambda mu, G, xi: np.einsum('Wa,aib,Wi->Wb', mu, G, xi)
    >>> step = mu_term(mu_R_dot[1], R[1], xi[1]) + mu_term(mu[1], H[1], xi[1]) + mu_term(mu[1], U[1], xi_U_dot[1])
    >>> print(bool(np.all(mu_R_dot[0] == 0)), bool(np.allclose(mu_R_dot[2], step)))
    True True
    """
    S = _setting(kind, frame, sample, order)
    _need(variations, 'mu_R_dot', 'variations (V, H)')

    def mu_R_dot_at(i):
        places = _frame_form(S, variations, range(i), range(i))
        terms = [_left_side(tt, tucker, S.inputs, i) for tt, tucker in places]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.R, i))))
    return _each(i, range(S.d + 1), mu_R_dot_at, 'mu_R_dot')


def nu_L_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # nu_L_dot[i]. shape=F+W+(rLi,)
    """``nu_L_dot[i]``: the right of bond ``i`` with the direction at one of the sites ``i .. d-1``, summed over the
    sites, in the ``L`` basis. Needs ``variations`` or ``core_directions``.

    One name for both forms of the direction. With ``variations`` it is the frame's form: at its site ``j`` the
    direction is ``H[j]`` over ``D[j]``, or ``U[j]`` over ``V[j]``, with ``L`` cores before it and ``R`` cores after
    it. With ``core_directions`` it is the raw form: ``L_dot[j]`` over ``D[j]``, or ``L[j]`` over ``D_dot[j]``, every
    other core an ``L`` core. With both given it is the raw form, the one ``Kp`` reads. The two are the same array
    when ``(D_dot, L_dot)`` is the lift of ``(V, H)``.

    Recurrence, the frame's form: ``nu_L_dot[i] = nu_term(L[i], xi[i], nu_L_dot[i+1])
    + nu_term(H[i], xi[i], nu[i+1]) + nu_term(U[i], xi_U_dot[i], nu[i+1])``. The raw form:
    ``nu_term(L[i], xi[i], nu_L_dot[i+1]) + nu_term(L_dot[i], xi[i], nu_L[i+1]) + nu_term(L[i], xi_dot[i], nu_L[i+1])``.
    Both from ``nu_L_dot[d] = 0``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``nu_L_dot[i]``. shape=F+W+(rLi,)

    Examples
    --------
    The frame's form, with ``variations``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> V = [rng.standard_normal((nU[i], N[i])) for i in range(3)]         # a direction at the frame: (V, H)
    >>> H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> nu_L_dot = sdef.nu_L_dot('probe', frame, ww, variations=(V, H))
    >>> print([n.shape for n in nu_L_dot])                                 # W + (rLi,)
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> nu, xi = sdef.nu('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_U_dot = sdef.xi_U_dot('probe', frame, ww, variations=(V, H))
    >>> nu_term = lambda G, xi, nu: np.einsum('aib,Wi,Wb->Wa', G, xi, nu)
    >>> step = nu_term(L[1], xi[1], nu_L_dot[2]) + nu_term(H[1], xi[1], nu[2]) + nu_term(U[1], xi_U_dot[1], nu[2])
    >>> print(bool(np.all(nu_L_dot[3] == 0)), bool(np.allclose(nu_L_dot[1], step)))
    True True

    The raw form, with ``core_directions``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> nu_L_dot = sdef.nu_L_dot('probe', frame, ww, core_directions=(D_dot, L_dot))
    >>> nu_L, xi = sdef.nu_L('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_dot = sdef.xi_dot('probe', frame, ww, core_directions=(D_dot, L_dot))
    >>> nu_term = lambda G, xi, nu: np.einsum('aib,Wi,Wb->Wa', G, xi, nu)
    >>> step = nu_term(L[1], xi[1], nu_L_dot[2]) + nu_term(L_dot[1], xi[1], nu_L[2]) + nu_term(L[1], xi_dot[1], nu_L[2])
    >>> print(bool(np.allclose(nu_L_dot[1], step)))
    True

    Gotcha -- a dotted variable needs its direction:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> sdef.nu_L_dot('probe', frame, ww)   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError
    """
    S = _setting(kind, frame, sample, order)

    def nu_L_dot_at(i):
        places = _either_form(S, variations, core_directions, range(i, S.d), range(i, S.d), 'nu_L_dot')
        terms = [_right_side(tt, tucker, S.inputs, i) for tt, tucker in places]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.L, i))))
    return _each(i, range(S.d + 1), nu_L_dot_at, 'nu_L_dot')


def eta_D_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # eta_D_dot[i]. shape=F+W+(nDi,)
    """``eta_D_dot[i]``: the trunk above Tucker edge ``i`` with the direction at one site, summed over the sites, in
    the ``D`` basis. Needs ``variations`` or ``core_directions``.

    One name for both forms of the direction, chosen as for :py:func:`nu_L_dot`. In the frame's form the direction
    at site ``i`` itself can only be ``H[i]`` (the leaf of site ``i`` is not in the trunk); at another site it is
    ``H``, or ``U`` over ``V``. In the raw form it can only be ``L_dot[i]`` at site ``i``. The direction's output is
    read off it: ``Jp[i] = z_term(eta_D_dot[i], D[i]) + z_term(eta[i], V[i])``.

    Recurrence, the frame's form: ``eta_D_dot[i] = eta_term(mu_R_dot[i], R[i], nu[i+1])
    + eta_term(mu[i], H[i], nu[i+1]) + eta_term(mu[i], L[i], nu_L_dot[i+1])``. The raw form:
    ``eta_term(mu_dot[i], L[i], nu_L[i+1]) + eta_term(mu[i], L_dot[i], nu_L[i+1])
    + eta_term(mu[i], L[i], nu_L_dot[i+1])``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``eta_D_dot[i]``. shape=F+W+(nDi,)

    Examples
    --------
    The frame's form: the recurrence, and the direction's output read off it:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> V = [rng.standard_normal((nU[i], N[i])) for i in range(3)]         # a direction at the frame: (V, H)
    >>> H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> p = dict(variations=(V, H))
    >>> eta_D_dot1 = sdef.eta_D_dot('probe', frame, ww, 1, **p)
    >>> print(eta_D_dot1.shape)                                            # W + (nD1,)
    (7, 3)
    >>> mu1, nu2 = sdef.mu('probe', frame, ww, 1), sdef.nu('probe', frame, ww, 2)
    >>> mu_R_dot1, nu_L_dot2 = sdef.mu_R_dot('probe', frame, ww, 1, **p), sdef.nu_L_dot('probe', frame, ww, 2, **p)
    >>> eta_term = lambda mu, G, nu: np.einsum('Wa,aib,Wb->Wi', mu, G, nu)
    >>> print(bool(np.allclose(eta_D_dot1, eta_term(mu_R_dot1, R[1], nu2) + eta_term(mu1, H[1], nu2)
    ...                                    + eta_term(mu1, L[1], nu_L_dot2))))
    True
    >>> Jp1 = np.einsum('Wi,io->Wo', eta_D_dot1, D[1]) + np.einsum('Wi,io->Wo', sdef.eta('probe', frame, ww, 1), V[1])
    >>> print(bool(np.allclose(sdef.Jp('probe', frame, ww, 1, **p), Jp1)))
    True
    """
    S = _setting(kind, frame, sample, order)

    def eta_D_dot_at(i):
        places = _either_form(S, variations, core_directions, range(S.d), _others(S.d, i), 'eta_D_dot')
        terms = [_trunk(tt, tucker, S.inputs, i) for tt, tucker in places]
        return _taylor(S, _sum(terms, _zeros(S, S.D[i].shape[0])))
    return _each(i, range(S.d), eta_D_dot_at, 'eta_D_dot')


##########################################################################
########    Dot: the raw form of the direction, (D_dot, L_dot)    ########
##########################################################################


def xi_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # xi_dot[i]. shape=min(F_in_i,F)+W+(nDi,)
    """``xi_dot[i]``: the leaf below Tucker edge ``i`` with the raw direction in it, in the ``D`` basis: ``D_dot[i]``
    applied to the input ``w[i]``. Needs ``core_directions``.

    Recurrence: ``xi_dot[i] = xi_term(D_dot[i], w[i])``. With jets, the input's jet sizes clipped to ``F``, as ``xi``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``xi_dot[i]``. shape=min(F_in_i,F)+W+(nDi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> xi_dot1 = sdef.xi_dot('probe', frame, ww, 1, core_directions=(D_dot, L_dot))
    >>> print(xi_dot1.shape)                                               # W + (nD1,)
    (7, 3)
    >>> print(bool(np.allclose(xi_dot1, np.einsum('io,Wo->Wi', D_dot[1], ww[1]))))
    True
    """
    S = _setting(kind, frame, sample, order)
    D_dot, L_dot = _need(core_directions, 'xi_dot', 'core_directions (D_dot, L_dot)')
    return _each(i, range(S.d), lambda i: _taylor(S, _leaf(D_dot[i], S.inputs[i]), S.leaf_sizes[i]), 'xi_dot')


def mu_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # mu_dot[i]. shape=F+W+(rLi,)
    """``mu_dot[i]``: the left of bond ``i`` with the raw direction at one of the sites ``0 .. i-1``, summed over the
    sites, in the ``L`` basis. Needs ``core_directions``.

    At its site ``j`` the direction is ``L_dot[j]`` over ``D[j]``, or ``L[j]`` over ``D_dot[j]``; every other TT core
    is an ``L`` core. It is the derivative of ``mu[i]`` along ``(D_dot, L_dot)``.

    Recurrence: ``mu_dot[i+1] = mu_term(mu_dot[i], L[i], xi[i]) + mu_term(mu[i], L_dot[i], xi[i])
    + mu_term(mu[i], L[i], xi_dot[i])``, from ``mu_dot[0] = 0``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``mu_dot[i]``. shape=F+W+(rLi,)

    Examples
    --------
    The recurrence; and ``mu_dot`` is the derivative of ``mu`` along the direction, here by a central difference:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> mu_dot = sdef.mu_dot('probe', frame, ww, core_directions=(D_dot, L_dot))
    >>> print([m.shape for m in mu_dot])                                   # W + (rLi,)
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> mu, xi = sdef.mu('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_dot = sdef.xi_dot('probe', frame, ww, core_directions=(D_dot, L_dot))
    >>> mu_term = lambda mu, G, xi: np.einsum('Wa,aib,Wi->Wb', mu, G, xi)
    >>> step = mu_term(mu_dot[1], L[1], xi[1]) + mu_term(mu[1], L_dot[1], xi[1]) + mu_term(mu[1], L[1], xi_dot[1])
    >>> print(bool(np.allclose(mu_dot[2], step)))
    True
    >>> moved = lambda e: ([B + e * B_dot for B, B_dot in zip(D, D_dot)], U,
    ...                    [G + e * G_dot for G, G_dot in zip(L, L_dot)], R)
    >>> difference = (sdef.mu('probe', moved(1e-5), ww, 3) - sdef.mu('probe', moved(-1e-5), ww, 3)) / 2e-5
    >>> print(bool(np.allclose(mu_dot[3], difference)))
    True
    """
    S = _setting(kind, frame, sample, order)
    _need(core_directions, 'mu_dot', 'core_directions (D_dot, L_dot)')

    def mu_dot_at(i):
        places = _raw_form(S, core_directions, range(i), range(i))
        terms = [_left_side(tt, tucker, S.inputs, i) for tt, tucker in places]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.L, i))))
    return _each(i, range(S.d + 1), mu_dot_at, 'mu_dot')


##############################################
########    Tilde: the seed put in    ########
##############################################


def xi_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # xi_tilde[i]. shape=F+W+(nDi,)
    """``xi_tilde[i]``: the leaf below Tucker edge ``i`` with the seed in place of its input, in the ``D`` basis:
    ``D[i]`` applied to the seed ``w_tilde[i]``. Needs ``y`` or ``w_tilde``.

    That is the probe kind, whose seed has one vector per input. Recurrence: ``xi_tilde[i] = xi_term(D[i],
    w_tilde[i])``. The seed is a full dual jet, so this has the full jet sizes ``F``, not the input's, and is stored
    turned.
    For the probe kind it is the multiplier of ``eta_D_dot[i]``.

    For ``'apply'`` and ``'entries'`` the seed is one number per sample at bond ``d``, not at an input, and
    ``xi_tilde`` is zero (with the jet sizes ``F`` all the same); it is then not the multiplier of ``eta_D_dot[i]``
    (that is the seed times ``xi[i]``, which has no name).

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``xi_tilde[i]``. shape=F+W+(nDi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode
    >>> xi_tilde1 = sdef.xi_tilde('probe', frame, ww, 1, **seed)
    >>> print(xi_tilde1.shape)                                             # W + (nD1,)
    (7, 3)
    >>> w_tilde1 = sdef.w_tilde('probe', frame, ww, 1, **seed)
    >>> print(bool(np.allclose(xi_tilde1, np.einsum('io,Wo->Wi', D[1], w_tilde1))))
    True
    """
    S = _setting(kind, frame, sample, order)
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'xi_tilde'))

    def xi_tilde_at(i):
        if kind == 'probe':
            return _taylor(S, _leaf(S.D[i], seed_M[i]))
        return _taylor(S, _zeros(S, S.D[i].shape[0]))               # apply, entries: no input takes the seed
    return _each(i, range(S.d), xi_tilde_at, 'xi_tilde')


def mu_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # mu_tilde[i]. shape=F+W+(rLi,)
    """``mu_tilde[i]``: ``mu[i]`` with the seed in place of one of the inputs ``0 .. i-1``, summed over them. In the
    ``L`` basis. Needs ``y`` or ``w_tilde``.

    That is the probe kind. There is one ``mu_tilde`` per bond, not one per mode: it combines the seeds of every
    mode on its side. For the probe kind it is the multiplier of ``nu_L_dot[i]``: with the direction wholly to the
    right of bond ``i``, the seed paired with ``Jp`` over the modes to the left of the bond is ``mu_tilde[i]`` paired
    with ``nu_L_dot[i]``.

    For ``'apply'`` and ``'entries'`` the seed sits at bond ``d``, which no left side reaches, and ``mu_tilde`` is
    zero; it is then not the multiplier of ``nu_L_dot[i]`` (that is the seed times ``mu[i]``, which has no name).

    Recurrence, every kind: ``mu_tilde[i+1] = mu_term(mu_tilde[i], L[i], xi[i]) + mu_term(mu[i], L[i],
    xi_tilde[i])``, from ``mu_tilde[0] = 0``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``mu_tilde[i]``. shape=F+W+(rLi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode
    >>> mu_tilde = sdef.mu_tilde('probe', frame, ww, **seed)
    >>> print([m.shape for m in mu_tilde])                                 # W + (rLi,)
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> mu, xi = sdef.mu('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_tilde = sdef.xi_tilde('probe', frame, ww, **seed)
    >>> mu_term = lambda mu, G, xi: np.einsum('Wa,aib,Wi->Wb', mu, G, xi)
    >>> print(bool(np.all(mu_tilde[0] == 0)))
    True
    >>> print(bool(np.allclose(mu_tilde[2], mu_term(mu_tilde[1], L[1], xi[1]) + mu_term(mu[1], L[1], xi_tilde[1]))))
    True
    """
    S = _setting(kind, frame, sample, order)
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'mu_tilde'))

    def mu_tilde_at(i):
        terms = [_times(_left_side(S.L, S.D, inputs, i), factor)
                 for inputs, factor in _seeded(S, seed_M, range(i), False)]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.L, i))))
    return _each(i, range(S.d + 1), mu_tilde_at, 'mu_tilde')


def nu_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # nu_tilde[i]. shape=F+W+(rRi,)
    """``nu_tilde[i]``: ``nu[i]`` with the seed put in, in the ``R`` basis. Needs ``y`` or ``w_tilde``.

    For ``'probe'`` the seed takes the place of one of the inputs ``i .. d-1``, summed over them. For ``'apply'``
    and ``'entries'`` it multiplies the ones at bond ``d``, so ``nu_tilde[i]`` is the seed times ``nu[i]`` (as
    polynomials in ``s``, when there are jets). For every kind it is the multiplier of ``mu_R_dot[i]``: with the
    direction wholly to the left of bond ``i``, the seed paired with ``Jp`` (for probe, over the modes to the right
    of the bond) is ``nu_tilde[i]`` paired with ``mu_R_dot[i]``.

    Recurrence, probe: ``nu_tilde[i] = nu_term(R[i], xi[i], nu_tilde[i+1]) + nu_term(R[i], xi_tilde[i], nu[i+1])``,
    from ``nu_tilde[d] = 0``. Apply and entries: ``nu_tilde[i] = nu_term(R[i], xi[i], nu_tilde[i+1])``, from
    ``nu_tilde[d] = w_tilde`` times the ones.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``nu_tilde[i]``. shape=F+W+(rRi,)

    Examples
    --------
    Probe: the seed in place of one input, summed over the inputs of the right side:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode
    >>> nu_tilde = sdef.nu_tilde('probe', frame, ww, **seed)
    >>> print([n.shape for n in nu_tilde])                                 # W + (rRi,)
    [(7, 1), (7, 3), (7, 2), (7, 1)]
    >>> nu, xi = sdef.nu('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_tilde = sdef.xi_tilde('probe', frame, ww, **seed)
    >>> nu_term = lambda G, xi, nu: np.einsum('aib,Wi,Wb->Wa', G, xi, nu)
    >>> print(bool(np.all(nu_tilde[3] == 0)))
    True
    >>> print(bool(np.allclose(nu_tilde[1], nu_term(R[1], xi[1], nu_tilde[2]) + nu_term(R[1], xi_tilde[1], nu[2]))))
    True

    Apply: the seed is one number per sample and enters at bond ``d``, so ``nu_tilde`` is the seed times ``nu``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> c = rng.standard_normal(7)                                         # a seed of our own: W
    >>> nu_tilde1 = sdef.nu_tilde('apply', frame, ww, 1, w_tilde=c)
    >>> print(bool(np.allclose(nu_tilde1, c[:, None] * sdef.nu('apply', frame, ww, 1))))
    True
    """
    S = _setting(kind, frame, sample, order)
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'nu_tilde'))

    def nu_tilde_at(i):
        terms = [_times(_right_side(S.R, S.D, inputs, i), factor)
                 for inputs, factor in _seeded(S, seed_M, range(i, S.d), True)]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.R, i))))
    return _each(i, range(S.d + 1), nu_tilde_at, 'nu_tilde')


def eta_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # eta_tilde[i]. shape=F+W+(nUi,)
    """``eta_tilde[i]``: ``eta[i]`` with the seed put in, in the ``U`` basis. Needs ``y`` or ``w_tilde``.

    For ``'probe'`` the seed takes the place of one input other than input ``i`` (the leaf of site ``i`` is not in
    the trunk), summed over them. For ``'apply'`` and ``'entries'`` it multiplies the ones at bond ``d``, so
    ``eta_tilde[i]`` is the seed times ``eta[i]``. For every kind it is the multiplier of ``xi_U_dot[i]``: with a
    direction that is ``V[i]`` alone, the seed paired with ``Jp`` (for probe, over the modes other than ``i``) is
    ``eta_tilde[i]`` paired with ``xi_U_dot[i]``.

    Recurrence, every kind: ``eta_tilde[i] = eta_term(mu_tilde[i], U[i], nu[i+1]) + eta_term(mu[i], U[i],
    nu_tilde[i+1])``; for apply and entries the first term is zero.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``eta_tilde[i]``. shape=F+W+(nUi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode
    >>> eta_tilde1 = sdef.eta_tilde('probe', frame, ww, 1, **seed)
    >>> print(eta_tilde1.shape)                                            # W + (nU1,)
    (7, 2)
    >>> mu1, nu2 = sdef.mu('probe', frame, ww, 1), sdef.nu('probe', frame, ww, 2)
    >>> mu_tilde1 = sdef.mu_tilde('probe', frame, ww, 1, **seed)
    >>> nu_tilde2 = sdef.nu_tilde('probe', frame, ww, 2, **seed)
    >>> eta_term = lambda mu, G, nu: np.einsum('Wa,aib,Wb->Wi', mu, G, nu)
    >>> print(bool(np.allclose(eta_tilde1, eta_term(mu_tilde1, U[1], nu2) + eta_term(mu1, U[1], nu_tilde2))))
    True
    """
    S = _setting(kind, frame, sample, order)
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'eta_tilde'))

    def eta_tilde_at(i):
        tt = S.L[:i] + [S.U[i]] + S.R[i + 1:]
        terms = [_times(_trunk(tt, S.D, inputs, i), factor)
                 for inputs, factor in _seeded(S, seed_M, _others(S.d, i), True)]
        return _taylor(S, _sum(terms, _zeros(S, S.U[i].shape[1])))
    return _each(i, range(S.d), eta_tilde_at, 'eta_tilde')


def nu_L_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # nu_L_tilde[i]. shape=F+W+(rLi,)
    """``nu_L_tilde[i]``: ``nu_L[i]`` with the seed put in, in the ``L`` basis: as ``nu_tilde[i]`` with ``L`` cores in
    place of ``R`` cores. Needs ``y`` or ``w_tilde``.

    It is the seeded right side at the base point ``(D, L)``, which ``Kp`` reads. For ``'probe'`` the seed takes
    the place of one of the inputs ``i .. d-1``, summed over them; for ``'apply'`` and ``'entries'`` it is the seed
    times ``nu_L[i]``.

    Recurrence, probe: ``nu_L_tilde[i] = nu_term(L[i], xi[i], nu_L_tilde[i+1])
    + nu_term(L[i], xi_tilde[i], nu_L[i+1])``, from ``nu_L_tilde[d] = 0``. Apply and entries: the first term only,
    from ``nu_L_tilde[d] = w_tilde`` times the ones.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``nu_L_tilde[i]``. shape=F+W+(rLi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode
    >>> nu_L_tilde = sdef.nu_L_tilde('probe', frame, ww, **seed)
    >>> print([n.shape for n in nu_L_tilde])                               # W + (rLi,)
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> nu_L, xi = sdef.nu_L('probe', frame, ww), sdef.xi('probe', frame, ww)
    >>> xi_tilde = sdef.xi_tilde('probe', frame, ww, **seed)
    >>> nu_term = lambda G, xi, nu: np.einsum('aib,Wi,Wb->Wa', G, xi, nu)
    >>> step = nu_term(L[1], xi[1], nu_L_tilde[2]) + nu_term(L[1], xi_tilde[1], nu_L[2])
    >>> print(bool(np.allclose(nu_L_tilde[1], step)))
    True
    """
    S = _setting(kind, frame, sample, order)
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'nu_L_tilde'))

    def nu_L_tilde_at(i):
        terms = [_times(_right_side(S.L, S.D, inputs, i), factor)
                 for inputs, factor in _seeded(S, seed_M, range(i, S.d), True)]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.L, i))))
    return _each(i, range(S.d + 1), nu_L_tilde_at, 'nu_L_tilde')


def eta_D_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # eta_D_tilde[i]. shape=F+W+(nDi,)
    """``eta_D_tilde[i]``: ``eta_D[i]`` with the seed put in, in the ``D`` basis: as ``eta_tilde[i]`` with every TT
    core an ``L`` core. Needs ``y`` or ``w_tilde``.

    It is the seeded trunk at the base point ``(D, L)``: what the gradient reads on the corewise frame
    ``(B, G, G, G)``, where it is the same array as ``eta_tilde``. For ``'probe'`` the seed takes the place of one
    input other than input ``i``, summed over them; for ``'apply'`` and ``'entries'`` it is the seed times
    ``eta_D[i]``.

    Recurrence, every kind: ``eta_D_tilde[i] = eta_term(mu_tilde[i], L[i], nu_L[i+1]) + eta_term(mu[i], L[i],
    nu_L_tilde[i+1])``; for apply and entries the first term is zero.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``eta_D_tilde[i]``. shape=F+W+(nDi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> seed = dict(y=y, block_weight=0.5 + rng.random(3))                 # the seed of z - y; one weight per mode
    >>> eta_D_tilde1 = sdef.eta_D_tilde('probe', frame, ww, 1, **seed)
    >>> print(eta_D_tilde1.shape)                                          # W + (nD1,)
    (7, 3)
    >>> mu1, nu_L2 = sdef.mu('probe', frame, ww, 1), sdef.nu_L('probe', frame, ww, 2)
    >>> mu_tilde1 = sdef.mu_tilde('probe', frame, ww, 1, **seed)
    >>> nu_L_tilde2 = sdef.nu_L_tilde('probe', frame, ww, 2, **seed)
    >>> eta_term = lambda mu, G, nu: np.einsum('Wa,aib,Wb->Wi', mu, G, nu)
    >>> print(bool(np.allclose(eta_D_tilde1, eta_term(mu_tilde1, L[1], nu_L2) + eta_term(mu1, L[1], nu_L_tilde2))))
    True
    >>> corewise = (D, L, L, L)                                            # one basis on each edge
    >>> print(bool(np.allclose(sdef.eta_tilde('probe', corewise, ww, 1, **seed),
    ...                        sdef.eta_D_tilde('probe', corewise, ww, 1, **seed))))
    True
    """
    S = _setting(kind, frame, sample, order)
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'eta_D_tilde'))

    def eta_D_tilde_at(i):
        terms = [_times(_trunk(S.L, S.D, inputs, i), factor)
                 for inputs, factor in _seeded(S, seed_M, _others(S.d, i), True)]
        return _taylor(S, _sum(terms, _zeros(S, S.D[i].shape[0])))
    return _each(i, range(S.d), eta_D_tilde_at, 'eta_D_tilde')


########################################################################
########    Tilde-dot: the seed and the raw direction put in    ########
########################################################################


def xi_tilde_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # xi_tilde_dot[i]. shape=F+W+(nDi,)
    """``xi_tilde_dot[i]``: the leaf below Tucker edge ``i`` with the raw direction and the seed in it: ``D_dot[i]``
    applied to the seed ``w_tilde[i]``. In the ``D`` basis. Needs ``core_directions``, and ``y`` or ``w_tilde``.

    That is the probe kind. Recurrence: ``xi_tilde_dot[i] = xi_term(D_dot[i], w_tilde[i])``. Zero for ``'apply'``
    and ``'entries'``, whose seed is not at an input. For every kind it is the derivative of ``xi_tilde[i]`` along
    ``(D_dot, L_dot)`` with the seed held fixed.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``xi_tilde_dot[i]``. shape=F+W+(nDi,)

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> both = dict(core_directions=(D_dot, L_dot), y=y)                   # the raw direction; the seed of the residual
    >>> xi_tilde_dot1 = sdef.xi_tilde_dot('probe', frame, ww, 1, **both)
    >>> print(xi_tilde_dot1.shape)                                         # W + (nD1,)
    (7, 3)
    >>> w_tilde1 = sdef.w_tilde('probe', frame, ww, 1, y=y)
    >>> print(bool(np.allclose(xi_tilde_dot1, np.einsum('io,Wo->Wi', D_dot[1], w_tilde1))))
    True
    """
    S = _setting(kind, frame, sample, order)
    D_dot, L_dot = _need(core_directions, 'xi_tilde_dot', 'core_directions (D_dot, L_dot)')
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'xi_tilde_dot'))

    def xi_tilde_dot_at(i):
        if kind == 'probe':
            return _taylor(S, _leaf(D_dot[i], seed_M[i]))
        return _taylor(S, _zeros(S, S.D[i].shape[0]))               # apply, entries: no input takes the seed
    return _each(i, range(S.d), xi_tilde_dot_at, 'xi_tilde_dot')


def mu_tilde_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # mu_tilde_dot[i]. shape=F+W+(rLi,)
    """``mu_tilde_dot[i]``: the left of bond ``i`` with the raw direction at one of the sites ``0 .. i-1`` and the
    seed in place of one of the inputs ``0 .. i-1``, summed over both. In the ``L`` basis. Needs
    ``core_directions``, and ``y`` or ``w_tilde``.

    That is the probe kind. The two may be at the same site: ``L_dot[j]`` over a ``D[j]`` that is applied to the
    seed, or ``L[j]`` over ``D_dot[j]`` applied to the seed. Zero for ``'apply'`` and ``'entries'``, whose seed no
    left side reaches. For every kind it is the derivative of ``mu_tilde[i]`` along ``(D_dot, L_dot)`` with the seed
    held fixed.

    Recurrence, six terms, every kind (all zero for apply and entries): ``mu_tilde_dot[i+1]`` is the sum of
    ``mu_term`` of ``(mu_tilde_dot, L, xi)``, ``(mu_dot, L, xi_tilde)``, ``(mu_tilde, L, xi_dot)``,
    ``(mu, L, xi_tilde_dot)``, ``(mu_tilde, L_dot, xi)`` and ``(mu, L_dot, xi_tilde)`` at site ``i``, from
    ``mu_tilde_dot[0] = 0``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``mu_tilde_dot[i]``. shape=F+W+(rLi,)

    Examples
    --------
    The six-term recurrence, from the other grades of ``mu`` and ``xi``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> both = dict(core_directions=(D_dot, L_dot), y=y)                   # the raw direction; the seed of the residual
    >>> get = lambda name: getattr(sdef, name)('probe', frame, ww, **both)
    >>> mu, mu_dot, mu_tilde, mu_tilde_dot = get('mu'), get('mu_dot'), get('mu_tilde'), get('mu_tilde_dot')
    >>> xi, xi_dot, xi_tilde, xi_tilde_dot = get('xi'), get('xi_dot'), get('xi_tilde'), get('xi_tilde_dot')
    >>> print([m.shape for m in mu_tilde_dot])                             # W + (rLi,)
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> mu_term = lambda mu, G, xi: np.einsum('Wa,aib,Wi->Wb', mu, G, xi)
    >>> step = (mu_term(mu_tilde_dot[1], L[1], xi[1]) + mu_term(mu_dot[1], L[1], xi_tilde[1])
    ...         + mu_term(mu_tilde[1], L[1], xi_dot[1]) + mu_term(mu[1], L[1], xi_tilde_dot[1])
    ...         + mu_term(mu_tilde[1], L_dot[1], xi[1]) + mu_term(mu[1], L_dot[1], xi_tilde[1]))
    >>> print(bool(np.all(mu_tilde_dot[0] == 0)), bool(np.allclose(mu_tilde_dot[2], step)))
    True True
    """
    S = _setting(kind, frame, sample, order)
    _need(core_directions, 'mu_tilde_dot', 'core_directions (D_dot, L_dot)')
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'mu_tilde_dot'))

    def mu_tilde_dot_at(i):
        terms = [_times(_left_side(tt, tucker, inputs, i), factor)
                 for tt, tucker in _raw_form(S, core_directions, range(i), range(i))
                 for inputs, factor in _seeded(S, seed_M, range(i), False)]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.L, i))))
    return _each(i, range(S.d + 1), mu_tilde_dot_at, 'mu_tilde_dot')


def nu_L_tilde_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the bond, 0 .. d; None = every bond
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # nu_L_tilde_dot[i]. shape=F+W+(rLi,)
    """``nu_L_tilde_dot[i]``: the right of bond ``i`` with the raw direction at one of the sites ``i .. d-1`` and the
    seed put in, summed over both. In the ``L`` basis. Needs ``core_directions``, and ``y`` or ``w_tilde``.

    For ``'probe'`` the seed takes the place of one of the inputs ``i .. d-1``, which may be at the site of the
    direction. For ``'apply'`` and ``'entries'`` it multiplies the ones at bond ``d``, so this is the seed times
    ``nu_L_dot[i]`` in its raw form. For every kind it is the derivative of ``nu_L_tilde[i]`` along
    ``(D_dot, L_dot)`` with the seed held fixed.

    Recurrence, six terms, every kind: ``nu_L_tilde_dot[i]`` is the sum of ``nu_term`` of
    ``(L, xi, nu_L_tilde_dot)``, ``(L, xi_tilde, nu_L_dot)``, ``(L, xi_dot, nu_L_tilde)``,
    ``(L, xi_tilde_dot, nu_L)``, ``(L_dot, xi, nu_L_tilde)`` and ``(L_dot, xi_tilde, nu_L)`` at site ``i``, from
    ``nu_L_tilde_dot[d] = 0``. For apply and entries the three terms with ``xi_tilde`` or ``xi_tilde_dot`` are zero.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d + 1`` of them.
        ``nu_L_tilde_dot[i]``. shape=F+W+(rLi,)

    Examples
    --------
    The six-term recurrence, from the other grades of ``nu_L`` and ``xi``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> both = dict(core_directions=(D_dot, L_dot), y=y)                   # the raw direction; the seed of the residual
    >>> get = lambda name: getattr(sdef, name)('probe', frame, ww, **both)
    >>> nu, nu_dot, nu_tilde, nu_tilde_dot = get('nu_L'), get('nu_L_dot'), get('nu_L_tilde'), get('nu_L_tilde_dot')
    >>> xi, xi_dot, xi_tilde, xi_tilde_dot = get('xi'), get('xi_dot'), get('xi_tilde'), get('xi_tilde_dot')
    >>> print([n.shape for n in nu_tilde_dot])                             # W + (rLi,)
    [(7, 1), (7, 2), (7, 3), (7, 1)]
    >>> nu_term = lambda G, xi, nu: np.einsum('aib,Wi,Wb->Wa', G, xi, nu)
    >>> step = (nu_term(L[1], xi[1], nu_tilde_dot[2]) + nu_term(L[1], xi_tilde[1], nu_dot[2])
    ...         + nu_term(L[1], xi_dot[1], nu_tilde[2]) + nu_term(L[1], xi_tilde_dot[1], nu[2])
    ...         + nu_term(L_dot[1], xi[1], nu_tilde[2]) + nu_term(L_dot[1], xi_tilde[1], nu[2]))
    >>> print(bool(np.all(nu_tilde_dot[3] == 0)), bool(np.allclose(nu_tilde_dot[1], step)))
    True True
    """
    S = _setting(kind, frame, sample, order)
    _need(core_directions, 'nu_L_tilde_dot', 'core_directions (D_dot, L_dot)')
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'nu_L_tilde_dot'))

    def nu_L_tilde_dot_at(i):
        terms = [_times(_right_side(tt, tucker, inputs, i), factor)
                 for tt, tucker in _raw_form(S, core_directions, range(i, S.d), range(i, S.d))
                 for inputs, factor in _seeded(S, seed_M, range(i, S.d), True)]
        return _taylor(S, _sum(terms, _zeros(S, _bond_size(S.L, i))))
    return _each(i, range(S.d + 1), nu_L_tilde_dot_at, 'nu_L_tilde_dot')


def eta_D_tilde_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # eta_D_tilde_dot[i]. shape=F+W+(nDi,)
    """``eta_D_tilde_dot[i]``: the trunk above Tucker edge ``i`` with the raw direction at one site and the seed put
    in, summed over both. In the ``D`` basis. Needs ``core_directions``, and ``y`` or ``w_tilde``.

    At site ``i`` itself the direction can only be ``L_dot[i]``, and for ``'probe'`` the seed takes the place of one
    input other than input ``i``: the leaf of site ``i`` is not in the trunk. For ``'apply'`` and ``'entries'`` this
    is the seed times ``eta_D_dot[i]`` in its raw form. For every kind it is the derivative of ``eta_D_tilde[i]``
    along ``(D_dot, L_dot)`` with the seed held fixed.

    Recurrence, six terms, every kind: ``eta_D_tilde_dot[i]`` is the sum of ``eta_term`` of
    ``(mu_tilde_dot, L, nu_L)``, ``(mu_dot, L, nu_L_tilde)``, ``(mu_tilde, L, nu_L_dot)``,
    ``(mu, L, nu_L_tilde_dot)``, ``(mu_tilde, L_dot, nu_L)`` and ``(mu, L_dot, nu_L_tilde)``, with ``mu`` at bond
    ``i``, the core of site ``i`` and ``nu_L`` at bond ``i+1``. For apply and entries the three terms with
    ``mu_tilde`` or ``mu_tilde_dot`` are zero.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``eta_D_tilde_dot[i]``. shape=F+W+(nDi,)

    Examples
    --------
    The six-term formula, from the four grades of ``mu`` at bond 1 and of ``nu_L`` at bond 2:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> both = dict(core_directions=(D_dot, L_dot), y=y)                   # the raw direction; the seed of the residual
    >>> get = lambda name, i: getattr(sdef, name)('probe', frame, ww, i, **both)
    >>> mu, mu_dot, mu_tilde, mu_tilde_dot = get('mu', 1), get('mu_dot', 1), get('mu_tilde', 1), get('mu_tilde_dot', 1)
    >>> nu, nu_dot = get('nu_L', 2), get('nu_L_dot', 2)
    >>> nu_tilde, nu_tilde_dot = get('nu_L_tilde', 2), get('nu_L_tilde_dot', 2)
    >>> eta_D_tilde_dot1 = get('eta_D_tilde_dot', 1)
    >>> print(eta_D_tilde_dot1.shape)                                      # W + (nD1,)
    (7, 3)
    >>> eta_term = lambda mu, G, nu: np.einsum('Wa,aib,Wb->Wi', mu, G, nu)
    >>> print(bool(np.allclose(eta_D_tilde_dot1,
    ...                        eta_term(mu_tilde_dot, L[1], nu) + eta_term(mu_dot, L[1], nu_tilde)
    ...                        + eta_term(mu_tilde, L[1], nu_dot) + eta_term(mu, L[1], nu_tilde_dot)
    ...                        + eta_term(mu_tilde, L_dot[1], nu) + eta_term(mu, L_dot[1], nu_tilde))))
    True
    """
    S = _setting(kind, frame, sample, order)
    _need(core_directions, 'eta_D_tilde_dot', 'core_directions (D_dot, L_dot)')
    seed_M = _at_nodes(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'eta_D_tilde_dot'))

    def eta_D_tilde_dot_at(i):
        terms = [_times(_trunk(tt, tucker, inputs, i), factor)
                 for tt, tucker in _raw_form(S, core_directions, range(S.d), _others(S.d, i))
                 for inputs, factor in _seeded(S, seed_M, _others(S.d, i), True)]
        return _taylor(S, _sum(terms, _zeros(S, S.D[i].shape[0])))
    return _each(i, range(S.d), eta_D_tilde_dot_at, 'eta_D_tilde_dot')


##########################################
########    Outputs and blocks    ########
##########################################


def z(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the mode, 0 .. d-1 (probe only); None = all of it
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Any:                                                # z[i]. probe F+W+(Ni,); apply F+W; entries W
    """``z``: the value. The whole network at the base point ``(D, L)``, sampled.

    For ``'probe'`` there is one output per mode: ``z[i]`` is the dense tensor of ``(D, L)`` contracted with the
    inputs of every mode but ``i``, mode edge ``i`` left open. For ``'apply'`` it is the tensor contracted with the
    inputs of every mode, and for ``'entries'`` its entry at ``index``: one number per sample. With a jet shape it
    is the jet of that at the input jets, on the box ``F`` (the entries kind has no jets). Computed from the dense
    tensor, through
    :py:func:`~t3toolbox.backend.probing.dense_probe`, :py:func:`~t3toolbox.backend.apply.dense_apply` or
    :py:func:`~t3toolbox.backend.entries.dense_entries`.

    In the engine: ``z[i] = z_term(eta_D[i], D[i])``, the contraction ``'Wi,io->Wo'`` (or, off the frame with the
    Tucker centers, ``z_term(eta[i], Vc[i])``); for apply and entries, ``mu[d]`` summed over bond ``d``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; for probe with ``i`` None, a tuple of ``d`` of them.
        The value. probe: ``z[i]``, shape=F+W+(Ni,); apply: shape=F+W; entries: shape=W

    Examples
    --------
    The three kinds, against the dense tensor of the base point ``(D, L)``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> T = np.einsum('aib,bjc,ckd,io,jp,kq->opq', L[0], L[1], L[2], D[0], D[1], D[2])
    >>> z = sdef.z('probe', frame, ww)
    >>> print([z_i.shape for z_i in z])                                    # W + (Ni,), one per mode
    [(7, 4), (7, 5), (7, 6)]
    >>> print(bool(np.allclose(z[1], np.einsum('opq,Wo,Wq->Wp', T, ww[0], ww[2]))))
    True
    >>> print(bool(np.allclose(sdef.z('apply', frame, ww), np.einsum('opq,Wo,Wp,Wq->W', T, *ww))))
    True
    >>> index = np.array([[3, 0, 1], [4, 4, 2], [0, 5, 5]])                # (d,)+W, W=(3,)
    >>> print(bool(np.allclose(sdef.z('entries', frame, index), T[index[0], index[1], index[2]])))
    True

    Jets: the Taylor coefficients along ``ww + s pp`` (the input jet of two slots) on a leading axis. They sum to
    the value at ``s = 1``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> pp = [rng.standard_normal((7, n)) for n in N]                      # the direction the inputs move in
    >>> line = [np.stack([w, p]) for w, p in zip(ww, pp)]                  # the input jets: F_in = (2,)
    >>> z1 = sdef.z('probe', frame, line, 1, order=(2,))
    >>> print(z1.shape)                                                    # F + W + (N1,) = (3,) + (7,) + (5,)
    (3, 7, 5)
    >>> at_one = sdef.z('probe', frame, [w + p for w, p in zip(ww, pp)], 1)
    >>> print(bool(np.allclose(z1.sum(axis=0), at_one)))                   # z[1] has degree d-1 = 2 in s
    True

    Gotchas: the jet shape is a tuple, and the entries kind has no jets (off the grid it is the apply kind at an
    input jet whose slot 0 is the one-hot vectors):

    >>> sdef.z('probe', frame, line, 1, order=2)                          # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    TypeError: order is the jet shape, a tuple of maximum orders, one per jet variable (() for none, (n,) for one), got 2
    >>> sdef.z('entries', frame, np.array([[3], [4], [0]]), order=(1,))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the entries kind has no jets: its order is (), got (1,). ...
    """
    S = _setting(kind, frame, sample, order)
    return _per_mode(S, i, _z(S, sample), 'z')


def r(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the mode, 0 .. d-1 (probe only); None = all of it
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Any:                                                # r[i]. probe F+W+(Ni,); apply F+W; entries W
    """``r = z - y``: the residual, of the type of ``z``, in forward order (slot ``k`` is the coefficient of ``s^k``).
    Needs ``y``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; for probe with ``i`` None, a tuple of ``d`` of them.
        The residual. probe: ``r[i]``, shape=F+W+(Ni,); apply and entries: shape=F+W

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> r1 = sdef.r('probe', frame, ww, 1, y=y)
    >>> print(r1.shape)                                                    # W + (N1,)
    (7, 5)
    >>> print(bool(np.all(r1 == sdef.z('probe', frame, ww, 1) - y[1])))
    True
    """
    S = _setting(kind, frame, sample, order)
    return _per_mode(S, i, _r(S, sample, _need(y, 'r', 'the data y')), 'r')


def Jp(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the mode, 0 .. d-1 (probe only); None = all of it
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Any:                                                # Jp[i]. probe F+W+(Ni,); apply F+W; entries W
    """``Jp``: the direction's output. The whole network with the direction ``(V, H)`` at one site, summed over the
    sites, sampled as ``z`` is. Needs ``variations``.

    It is the kind's sampling of the dense tangent tensor: the sum over the sites ``j`` of the tensor with ``H[j]``
    over ``D[j]``, plus the tensor with ``U[j]`` over ``V[j]``, in both cases with ``L`` cores before site ``j`` and
    ``R`` cores after it. It is linear in ``(V, H)``; on a frame of a point it is the derivative of ``z`` along the
    tangent vector.

    In the engine: ``Jp[i] = z_term(eta_D_dot[i], D[i]) + z_term(eta[i], V[i])``; for apply and entries,
    ``mu_R_dot[d]`` summed over bond ``d``.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; for probe with ``i`` None, a tuple of ``d`` of them.
        The direction's output. probe: ``Jp[i]``, shape=F+W+(Ni,); apply and entries: shape=F+W

    Examples
    --------
    On a corewise point, where a direction is a change of the cores, ``Jp`` is the derivative of ``z`` (here by a
    central difference):

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> moved = lambda e: ([B + e * B_dot for B, B_dot in zip(D, D_dot)],
    ...                    [G + e * G_dot for G, G_dot in zip(L, L_dot)])
    >>> corewise = lambda B, G: (B, G, G, G)                               # a corewise point, as a frame
    >>> Jp1 = sdef.Jp('probe', corewise(D, L), ww, 1, variations=(D_dot, L_dot))
    >>> print(Jp1.shape)                                                   # W + (N1,)
    (7, 5)
    >>> z1 = lambda e: sdef.z('probe', corewise(*moved(e)), ww, 1)
    >>> print(bool(np.allclose(Jp1, (z1(1e-5) - z1(-1e-5)) / 2e-5)))
    True
    """
    S = _setting(kind, frame, sample, order)
    return _per_mode(S, i, _Jp(S, _need(variations, 'Jp', 'variations (V, H)')), 'Jp')


def g(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # None: it has no index
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                             # V_tilde. len=d, elm_shape=(nUi,Ni)
        typ.Tuple[NDArray, ...],                             # H_tilde. len=d, elm_shape=(rLi,nDi,rR(i+1))
]:
    """``g = (V_tilde, H_tilde)``: the transpose of ``(V, H) -> Jp``, applied to the seed. Needs ``y`` or ``w_tilde``.

    With the seed of the residual (``y`` given) it is the gradient of ``f`` in the coordinates ``(V, H)``, before any
    projection onto a tangent space. With a seed of your own (``w_tilde``) it is the bare transpose. Computed from
    the dense Jacobian: the forward map is applied to each unit direction, and the result is paired with the seed.
    Summed over the samples.

    Returns
    -------
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]
        ``(V_tilde, H_tilde)``: blocks with the shapes of ``V`` and ``H``, ``(nUi, Ni)`` and ``(rLi, nDi, rR(i+1))``

    Examples
    --------
    The blocks have the shapes of ``V`` and ``H``, and ``g`` is the gradient of ``f``: on a corewise point, the
    derivative of ``f`` along a change of the cores is that change paired with ``g``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> moved = lambda e: ([B + e * B_dot for B, B_dot in zip(D, D_dot)],
    ...                    [G + e * G_dot for G, G_dot in zip(L, L_dot)])
    >>> corewise = lambda B, G: (B, G, G, G)                               # a corewise point, as a frame
    >>> V_tilde, H_tilde = sdef.g('probe', frame, ww, y=y)
    >>> print([b.shape for b in V_tilde], [b.shape for b in H_tilde])
    [(3, 4), (2, 5), (3, 6)] [(1, 2, 3), (2, 3, 2), (3, 2, 1)]
    >>> V_tilde, H_tilde = sdef.g('probe', corewise(D, L), ww, y=y)
    >>> slope = sum(np.sum(a * b) for a, b in zip(V_tilde + H_tilde, D_dot + L_dot))
    >>> f = lambda e: sdef.f('probe', corewise(*moved(e)), ww, y=y)
    >>> print(bool(np.allclose(slope, (f(1e-5) - f(-1e-5)) / 2e-5)))
    True
    """
    _no_index(i, 'g')
    S = _setting(kind, frame, sample, order)
    return _jacobian_transpose(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'g'))


def JtJp(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # None: it has no index
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                             # V_tilde. len=d, elm_shape=(nUi,Ni)
        typ.Tuple[NDArray, ...],                             # H_tilde. len=d, elm_shape=(rLi,nDi,rR(i+1))
]:
    """``JtJp``: the Gauss-Newton product. ``Jp`` made into a seed (weighted by both weights, then turned), and the
    transpose applied to it: the blocks of :py:func:`g` with that seed in place of the residual's. Needs ``variations``.

    Returns
    -------
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]
        Blocks with the shapes of ``V`` and ``H``, ``(nUi, Ni)`` and ``(rLi, nDi, rR(i+1))``

    Examples
    --------
    The direction paired with ``JtJp`` is the weighted norm of ``Jp``, squared:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> V = [rng.standard_normal((nU[i], N[i])) for i in range(3)]         # a direction at the frame: (V, H)
    >>> H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> per_sample = 0.5 + rng.random((3, 7, 1))                          # one number per mode and sample ...
    >>> full = [np.broadcast_to(per_sample[i], (7, n)) for i, n in enumerate(N)]   # ... written out on the edge
    >>> p = dict(variations=(V, H), full_weight=full)
    >>> JtJp_V, JtJp_H = sdef.JtJp('probe', frame, ww, **p)
    >>> print([b.shape for b in JtJp_V], [b.shape for b in JtJp_H])
    [(3, 4), (2, 5), (3, 6)] [(1, 2, 3), (2, 3, 2), (3, 2, 1)]
    >>> paired = sum(np.sum(a * b) for a, b in zip(JtJp_V + JtJp_H, V + H))
    >>> print(bool(np.allclose(paired, sdef.Jp_norm_sq('probe', frame, ww, **p))))
    True
    """
    _no_index(i, 'JtJp')
    S = _setting(kind, frame, sample, order)
    return _jacobian_transpose(S, _dual_jet(S, _Jp(S, _need(variations, 'JtJp', 'variations (V, H)')), block_weight,
                                            full_weight))


def Kp(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # None: it has no index
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Tuple[
        typ.Tuple[NDArray, ...],                             # V_tilde_dot. len=d, elm_shape=(nDi,Ni)
        typ.Tuple[NDArray, ...],                             # H_tilde_dot. len=d, elm_shape=(rLi,nDi,rL(i+1))
]:
    """``Kp = (V_tilde_dot, H_tilde_dot)``: the curvature term. The second derivative of the dense objective
    ``phi(B, G) = <seed, z(B, G)>`` in the cores of the base point ``(D, L)``, applied to
    ``core_directions = (D_dot, L_dot)``, with the seed held fixed. Needs ``core_directions``, and ``y`` or
    ``w_tilde``.

    ``phi`` pairs a fixed seed with the sampled dense tensor of the cores ``(B, G)``; its first derivative at
    ``(D, L)`` is the gradient on the corewise frame, and ``Kp`` is the derivative of that along the direction. It is
    not the second derivative of ``f``, which has the Gauss-Newton term ``JtJp`` as well. The route is read off
    the arrays, as everywhere in the library: with numpy arrays, central finite differences of the gradient, which
    are exact to rounding because ``phi`` is a polynomial along the direction; with jax arrays, automatic
    differentiation, in the precision jax is set to. The two agree to rounding.

    Returns
    -------
    typ.Tuple[typ.Tuple[NDArray, ...], typ.Tuple[NDArray, ...]]
        ``(V_tilde_dot, H_tilde_dot)``: blocks with the shapes of ``D`` and ``L``

    Examples
    --------
    ``Kp`` is the derivative, along the direction, of the gradient on the corewise frame with the seed held fixed (here
    by a central difference):

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> moved = lambda e: ([B + e * B_dot for B, B_dot in zip(D, D_dot)],
    ...                    [G + e * G_dot for G, G_dot in zip(L, L_dot)])
    >>> corewise = lambda B, G: (B, G, G, G)                               # a corewise point, as a frame
    >>> V_tilde_dot, H_tilde_dot = sdef.Kp('probe', frame, ww, core_directions=(D_dot, L_dot), y=y)
    >>> print([b.shape for b in V_tilde_dot], [b.shape for b in H_tilde_dot])     # the shapes of D and of L
    [(2, 4), (3, 5), (2, 6)] [(1, 2, 2), (2, 3, 3), (3, 2, 1)]
    >>> seed = sdef.w_tilde('probe', frame, ww, y=y)                       # held fixed while the cores move
    >>> gradient = lambda e: sum(sdef.g('probe', corewise(*moved(e)), ww, w_tilde=seed), ())
    >>> difference = [(a - b) / 2e-5 for a, b in zip(gradient(1e-5), gradient(-1e-5))]
    >>> print(all(bool(np.allclose(a, b)) for a, b in zip(V_tilde_dot + H_tilde_dot, difference)))
    True
    """
    _no_index(i, 'Kp')
    S = _setting(kind, frame, sample, order)
    return _Kp(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'Kp'),
               _need(core_directions, 'Kp', 'core_directions (D_dot, L_dot)'))


def V_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # V_tilde[i]. shape=(nUi,Ni)
    """``V_tilde[i]``: the Tucker block of ``g``, with the shape of ``V[i]``: the whole network with the leaf core under
    ``U[i]`` taken out and the seed put in, summed over the samples. Needs ``y`` or ``w_tilde``.

    In the engine it is the coefficient of ``s^order`` of ``V_term(eta_tilde[i], w[i]) + V_term(eta[i], w_tilde[i])``,
    the contraction ``'Wi,Wo->io'``; the second term, the seed at input ``i`` itself, is there for ``'probe'`` only.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``V_tilde[i]``. shape=(nUi,Ni)

    Examples
    --------
    The block from the dense Jacobian equals the readout of the tilde variables:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> V_tilde1 = sdef.V_tilde('probe', frame, ww, 1, y=y)
    >>> print(V_tilde1.shape)                                              # the shape of V[1]: (nU1, N1)
    (2, 5)
    >>> eta1, eta_tilde1 = sdef.eta('probe', frame, ww, 1), sdef.eta_tilde('probe', frame, ww, 1, y=y)
    >>> w_tilde1 = sdef.w_tilde('probe', frame, ww, 1, y=y)
    >>> V_term = lambda eta, w: np.einsum('Wi,Wo->io', eta, w)
    >>> print(bool(np.allclose(V_tilde1, V_term(eta_tilde1, ww[1]) + V_term(eta1, w_tilde1))))
    True
    """
    S = _setting(kind, frame, sample, order)
    V_blocks, H_blocks = _jacobian_transpose(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'V_tilde'))
    return _each(i, range(S.d), lambda i: V_blocks[i], 'V_tilde')


def H_tilde(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # H_tilde[i]. shape=(rLi,nDi,rR(i+1))
    """``H_tilde[i]``: the TT block of ``g``, with the shape of ``H[i]``: the whole network with the TT core of site
    ``i`` taken out (``L`` cores to its left, ``R`` cores to its right) and the seed put in, summed over the samples.
    Needs ``y`` or ``w_tilde``.

    In the engine it is the coefficient of ``s^order`` of ``H_term(mu_tilde[i], xi[i], nu[i+1])
    + H_term(mu[i], xi_tilde[i], nu[i+1]) + H_term(mu[i], xi[i], nu_tilde[i+1])``, the contraction
    ``'Wa,Wi,Wb->aib'``. For ``'apply'`` and ``'entries'`` only the third term is there: ``mu_tilde`` and
    ``xi_tilde`` are zero.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``H_tilde[i]``. shape=(rLi,nDi,rR(i+1))

    Examples
    --------
    The block from the dense Jacobian equals the readout of the tilde variables:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> H_tilde1 = sdef.H_tilde('probe', frame, ww, 1, y=y)
    >>> print(H_tilde1.shape)                                              # the shape of H[1]: (rL1, nD1, rR2)
    (2, 3, 2)
    >>> get = lambda name, i: getattr(sdef, name)('probe', frame, ww, i, y=y)
    >>> H_term = lambda mu, xi, nu: np.einsum('Wa,Wi,Wb->aib', mu, xi, nu)
    >>> print(bool(np.allclose(H_tilde1, H_term(get('mu_tilde', 1), get('xi', 1), get('nu', 2))
    ...                                  + H_term(get('mu', 1), get('xi_tilde', 1), get('nu', 2))
    ...                                  + H_term(get('mu', 1), get('xi', 1), get('nu_tilde', 2)))))
    True
    """
    S = _setting(kind, frame, sample, order)
    V_blocks, H_blocks = _jacobian_transpose(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'H_tilde'))
    return _each(i, range(S.d), lambda i: H_blocks[i], 'H_tilde')


def V_tilde_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # V_tilde_dot[i]. shape=(nDi,Ni)
    """``V_tilde_dot[i]``: the Tucker block of ``Kp``, with the shape of ``D[i]``: the network at the base point with
    ``D[i]`` taken out, the raw direction at one site and the seed put in, summed over the samples. At site ``i`` the
    direction can only be ``L_dot[i]``. Needs ``core_directions``, and ``y`` or ``w_tilde``.

    In the engine it is the coefficient of ``s^order`` of ``V_term(eta_D_tilde_dot[i], w[i])
    + V_term(eta_D_dot[i], w_tilde[i])``, with ``eta_D_dot`` in its raw form; the second term is there for
    ``'probe'`` only.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``V_tilde_dot[i]``. shape=(nDi,Ni)

    Examples
    --------
    The block from the second derivative equals the readout of the tilde-dot variables:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> both = dict(core_directions=(D_dot, L_dot), y=y)                   # the raw direction; the seed of the residual
    >>> V_tilde_dot1 = sdef.V_tilde_dot('probe', frame, ww, 1, **both)
    >>> print(V_tilde_dot1.shape == D[1].shape)
    True
    >>> get = lambda name: getattr(sdef, name)('probe', frame, ww, 1, **both)
    >>> print(bool(np.allclose(V_tilde_dot1, np.einsum('Wi,Wo->io', get('eta_D_tilde_dot'), ww[1])
    ...                                      + np.einsum('Wi,Wo->io', get('eta_D_dot'), get('w_tilde')))))
    True
    """
    S = _setting(kind, frame, sample, order)
    V_blocks, H_blocks = _Kp(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'V_tilde_dot'),
                             _need(core_directions, 'V_tilde_dot', 'core_directions (D_dot, L_dot)'))
    return _each(i, range(S.d), lambda i: V_blocks[i], 'V_tilde_dot')


def H_tilde_dot(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # the site, 0 .. d-1; None = every site
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> typ.Union[NDArray, typ.Tuple[NDArray, ...]]:            # H_tilde_dot[i]. shape=(rLi,nDi,rL(i+1))
    """``H_tilde_dot[i]``: the TT block of ``Kp``, with the shape of ``L[i]``: the network at the base point with
    ``L[i]`` taken out, the raw direction at one site and the seed put in, summed over the samples. At site ``i`` the
    direction can only be ``D_dot[i]``. Needs ``core_directions``, and ``y`` or ``w_tilde``.

    In the engine it is the coefficient of ``s^order`` of nine ``H_term`` s of ``mu[i]``, ``xi[i]`` and
    ``nu_L[i+1]``: the tilde on one of the three and the dot on one of the three. Three terms have both on the same
    variable (its tilde-dot); six have them on two different ones. For ``'apply'`` and ``'entries'`` only the three
    with the tilde on ``nu_L`` are there (``mu_dot`` with ``nu_L_tilde``, ``xi_dot`` with ``nu_L_tilde``, and
    ``nu_L_tilde_dot``): every tilde grade of ``mu`` and ``xi`` is zero.

    Returns
    -------
    NDArray or tuple of NDArray
        One array; with ``i`` None, a tuple of ``d`` of them.
        ``H_tilde_dot[i]``. shape=(rLi,nDi,rL(i+1))

    Examples
    --------
    The block from the second derivative equals the nine-term readout of the grades of ``mu``, ``xi``, ``nu_L``:

    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> D_dot = [rng.standard_normal(B.shape) for B in D]                  # a raw core direction at (D, L)
    >>> L_dot = [rng.standard_normal(G.shape) for G in L]
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> both = dict(core_directions=(D_dot, L_dot), y=y)                   # the raw direction; the seed of the residual
    >>> H_tilde_dot1 = sdef.H_tilde_dot('probe', frame, ww, 1, **both)
    >>> print(H_tilde_dot1.shape == L[1].shape)
    True
    >>> grades = ('', '_dot', '_tilde', '_tilde_dot')
    >>> mu = {g: getattr(sdef, 'mu' + g)('probe', frame, ww, 1, **both) for g in grades}
    >>> xi = {g: getattr(sdef, 'xi' + g)('probe', frame, ww, 1, **both) for g in grades}
    >>> nu = {g: getattr(sdef, 'nu_L' + g)('probe', frame, ww, 2, **both) for g in grades}
    >>> H_term = lambda mu, xi, nu: np.einsum('Wa,Wi,Wb->aib', mu, xi, nu)
    >>> same = [('_tilde_dot', '', ''), ('', '_tilde_dot', ''), ('', '', '_tilde_dot')]   # both marks on one variable
    >>> three = ('', '_dot', '_tilde')
    >>> apart = [(a, b, c) for a in three for b in three for c in three
    ...          if sorted((a, b, c)) == sorted(three)]                                   # on two different ones
    >>> print(len(same), len(apart))
    3 6
    >>> print(bool(np.allclose(H_tilde_dot1, sum(H_term(mu[a], xi[b], nu[c]) for a, b, c in same + apart))))
    True
    """
    S = _setting(kind, frame, sample, order)
    V_blocks, H_blocks = _Kp(S, _the_seed(S, sample, y, block_weight, full_weight, w_tilde, 'H_tilde_dot'),
                             _need(core_directions, 'H_tilde_dot', 'core_directions (D_dot, L_dot)'))
    return _each(i, range(S.d), lambda i: H_blocks[i], 'H_tilde_dot')


############################
########    Sums    ########
############################


def f(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # None: it has no index
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> NDArray:                                                # f. shape=()
    """``f``: the misfit, ``1/2 <r, M r>`` with ``M = b**2 f**2``: half the sum, over the modes, the orders, the
    samples and the mode edge, of the squared weighted residual, slot by slot. Needs ``y``.

    Returns
    -------
    NDArray
        The misfit, one number. shape=()

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> y = [rng.standard_normal((7, n)) for n in N]                       # the data: of the type of z
    >>> full = [0.5 + rng.random((7, n)) for n in N]                      # the full weight: one per entry
    >>> r = sdef.r('probe', frame, ww, y=y)
    >>> f = sdef.f('probe', frame, ww, y=y, full_weight=full)
    >>> print(bool(np.allclose(f, 0.5 * sum(np.sum(full[i] ** 2 * r[i] ** 2) for i in range(3)))))
    True

    One value per mode is a block weight, ``(d,)`` with no jets. Given as a full weight with size-1 sample and edge
    axes it raises: the edge is part of the measurement, never 1:

    >>> per_mode = np.array([1.0, 2.0, 3.0])
    >>> sdef.f('probe', frame, ww, y=y, full_weight=per_mode.reshape(3, 1, 1))     # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: the full weight (mode 0) has shape (1, 1); ... the mode edge exactly ...
    >>> print(bool(np.allclose(sdef.f('probe', frame, ww, y=y, block_weight=per_mode),
    ...                        0.5 * sum(per_mode[i] ** 2 * np.sum(r[i] ** 2) for i in range(3)))))
    True
    """
    _no_index(i, 'f')
    S = _setting(kind, frame, sample, order)
    r_ = _r(S, sample, _need(y, 'f', 'the data y'))
    xnp, _, _ = get_backend(False, tree_contains_jax((r_,)))
    halves = _map_type_of_z(S, lambda i, r_TX: 0.5 * xnp.sum(_weight_squared(S, block_weight, full_weight, i,
                                                                              r_TX.shape) * r_TX ** 2), r_)
    return _sum(halves[1:], halves[0]) if kind == 'probe' else halves


def r_block_sumsq(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # None: it has no index
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> NDArray:                                                # the block shape: probe (d,)+F; apply F; entries ()
    """``r_block_sumsq``: ``<r, M r>`` in each block, ``M = b**2 f**2``, the weighted sum of squares of the residual
    (the sum of squares with no weight). A block is one mode at one slot of the box ``F`` (probe) or one slot (apply;
    entries has the one block), so the result has the block shape; the sum is over the samples and the mode edge.
    Needs ``y``.

    Returns
    -------
    NDArray
        The block sums. probe: shape=(d,)+F; apply: shape=F; entries: shape=()

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> pp = [rng.standard_normal((7, n)) for n in N]                      # the direction the inputs move in
    >>> line = [np.stack([w, p]) for w, p in zip(ww, pp)]                  # the input jets: F_in = (2,)
    >>> y = [rng.standard_normal((3, 7, n)) for n in N]                    # the data at (2,): F + W + (Ni,)
    >>> sums = sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y)
    >>> print(sums.shape)                                                  # (d,) + F: one number per mode and slot
    (3, 3)
    >>> r = sdef.r('probe', frame, line, order=(2,), y=y)
    >>> print(bool(np.allclose(sums[1, 2], np.sum(r[1][2] ** 2))))
    True

    Weighted, one full weight per sample (written out on the slots and the edge): a zero drops the sample from every
    block.

    >>> full = [np.ones((3, 7, n)) for n in N]                            # F + W + (Ni,)
    >>> for omega in full:
    ...     omega[:, 3, :] = 0.0
    >>> sums = sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y, full_weight=full)
    >>> print(bool(np.allclose(sums[1, 2], np.sum(np.delete(r[1][2], 3, axis=0) ** 2))))
    True

    A block weight scales its block: ``block[i, k]**2`` times the block's sum.

    >>> block = 0.5 + rng.random((3, 3))
    >>> print(bool(np.allclose(sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y, block_weight=block),
    ...                        block ** 2 * sdef.r_block_sumsq('probe', frame, line, order=(2,), y=y))))
    True
    """
    _no_index(i, 'r_block_sumsq')
    S = _setting(kind, frame, sample, order)
    r_ = _r(S, sample, _need(y, 'r_block_sumsq', 'the data y'))
    xnp, _, _ = get_backend(False, tree_contains_jax((r_,)))
    m = len(S.order)

    def sumsq(i, r_FX):
        return xnp.sum(_weight_squared(S, block_weight, full_weight, i, r_FX.shape) * r_FX ** 2,
                       axis=tuple(range(m, r_FX.ndim)))
    sums = _map_type_of_z(S, sumsq, r_)
    return xnp.stack(sums) if kind == 'probe' else sums


def Jp_norm_sq(
        kind:             str,                               # 'probe' | 'apply' | 'entries'
        frame:            typ.Tuple,                         # (D, U, L, R), each len=d
        sample:           typ.Any,                           # ww (input jets, F_in_i+W+(Ni,)) | index (entries)
        i:                typ.Optional[int] = None,          # None: it has no index
        order:            typ.Tuple[int, ...] = (),          # the jet shape (n_1, ..., n_m); F = (n_1 + 1, ...)
        *,
        variations:       typ.Optional[typ.Tuple] = None,    # (V, H), elm_shapes (nUi,Ni), (rLi,nDi,rR(i+1))
        core_directions:  typ.Optional[typ.Tuple] = None,    # (D_dot, L_dot), the shapes of D and L
        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)
        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();
                                                             #   None = ones
        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones
        w_tilde:          typ.Optional[typ.Any] = None,      # a seed of the type of z, stored turned
) -> NDArray:                                                # Jp_norm_sq. shape=()
    """``Jp_norm_sq``: ``<Jp, M Jp>`` with ``M = b**2 f**2``, the squared weighted norm of the direction's output,
    summed as ``f`` is (without the half). Needs ``variations``.

    Returns
    -------
    NDArray
        One number. shape=()

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.sampling_definitions as sdef
    >>> rng = np.random.default_rng(0)
    >>> N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
    >>> D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
    >>> U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
    >>> L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
    >>> R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> frame = (D, U, L, R)
    >>> ww = [rng.standard_normal((7, n)) for n in N]                      # W=(7,)
    >>> V = [rng.standard_normal((nU[i], N[i])) for i in range(3)]         # a direction at the frame: (V, H)
    >>> H = [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]
    >>> per_sample = 0.5 + rng.random((3, 7, 1))                          # one number per mode and sample ...
    >>> full = [np.broadcast_to(per_sample[i], (7, n)) for i, n in enumerate(N)]   # ... written out on the edge
    >>> Jp = sdef.Jp('probe', frame, ww, variations=(V, H))
    >>> Jp_norm_sq = sdef.Jp_norm_sq('probe', frame, ww, variations=(V, H), full_weight=full)
    >>> print(bool(np.allclose(Jp_norm_sq, sum(np.sum(full[i] ** 2 * Jp[i] ** 2) for i in range(3)))))
    True
    """
    _no_index(i, 'Jp_norm_sq')
    S = _setting(kind, frame, sample, order)
    Jp_ = _Jp(S, _need(variations, 'Jp_norm_sq', 'variations (V, H)'))
    xnp, _, _ = get_backend(False, tree_contains_jax((Jp_,)))
    squares = _map_type_of_z(S, lambda i, Jp_TX: xnp.sum(_weight_squared(S, block_weight, full_weight, i,
                                                                         Jp_TX.shape) * Jp_TX ** 2), Jp_)
    return _sum(squares[1:], squares[0]) if kind == 'probe' else squares
