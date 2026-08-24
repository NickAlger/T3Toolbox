# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Uniform mirror of ``frame_variations_format``: ``UT3Frame`` + ``UT3Variations``.

Supercores + host-numpy masks instead of ragged core tuples. ``ut3_orthogonal_representations``
returns the frontend objects (the backend ``ufv_conversions`` twin returns raw data). Class
asymmetries vs the ragged classes (e.g. no ``save``/``to_vector``) are deliberate design, not gaps.
"""
import numpy as np
import typing as typ
import t3toolbox.safety as safety_mod
import functools as ft
from dataclasses import dataclass

import t3toolbox.backend.fv_conversions
import t3toolbox.backend.ufv_conversions as ufv_conversions
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.corewise as cw
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ufv_operations as ufv_operations
import t3toolbox.backend.ufv_masking as masking
import t3toolbox.backend.common as common
from t3toolbox.backend.common import *


__all__ = [
    'UT3Frame',
    'UT3Variations',
    'ut3_orthogonal_representations',
    'ut3svd_orthogonal_representations',
    'UT3FrameWeights',
    'ufv_absorb_weights',
    'check_ufw_pair',
]


@dataclass(frozen=True, eq=False)  # eq=False -> the mixin's VALUE-based __hash__/__eq__ stand
class UT3FrameMasks(common.ValueHashedMasks):
    """The static rank structure of a :py:class:`UT3Frame`: its four boolean edge masks.

    Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form). Held as a
    separate object so it can ride as jax ``aux_data``; hash/eq are **value-based** (the
    :py:class:`~t3toolbox.backend.common.ValueHashedMasks` mixin) so a rebuilt-but-identical frame is the
    *same* jit cache key (no per-iteration recompile when the orthogonal frame is rebuilt in an
    optimization loop). The plain-layer :py:class:`~t3toolbox.uniform_tucker_tensor_train.UT3Masks`
    pattern; see ``docs/contributor/uniform_pytree_composition.md``. (The physical ``shape`` is a separate static int
    tuple on :py:class:`UT3Frame` -- not a mask, and value-hashable.)
    """
    up_mask:          NDArray  # dtype=bool, (d,)  +stack_shape+(nU,)
    down_mask:        NDArray  # dtype=bool, (d,)  +stack_shape+(nD,)
    frame_left_mask:  NDArray  # dtype=bool, (d+1,)+stack_shape+(rL,)
    frame_right_mask: NDArray  # dtype=bool, (d+1,)+stack_shape+(rR,)

    def __post_init__(self):
        # Defensive READ-ONLY copies: the value key is cached on this frozen holder, so an aliased
        # writeable caller array mutated in place would leave a stale jit cache key (review H1-5).
        for _f in ('up_mask', 'down_mask', 'frame_left_mask', 'frame_right_mask'):
            _m = np.array(getattr(self, _f), copy=True)
            _m.setflags(write=False)
            object.__setattr__(self, _f, _m)

    @property
    def data(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:
        """The four raw rank-mask arrays, ``(up_mask, down_mask, frame_left_mask, frame_right_mask)``."""
        return self.up_mask, self.down_mask, self.frame_left_mask, self.frame_right_mask


@dataclass(frozen=True, eq=False)   # eq=False -> the ExplicitEquality mixin stands
class UT3Frame(common.ExplicitEquality):
    """Frame (orthogonal frame) for the frame-variations representation of uniform Tucker tensor trains.

    Uniform analog of :py:class:`~t3toolbox.frame_variations_format.T3Frame`: four padded supercores
    (``up_tucker``, ``down_tt``, ``left_tt``, ``right_tt``) + the static physical ``shape`` (an int tuple,
    shared across the stack) + a :py:class:`UT3FrameMasks` holder (the four per-stack rank masks).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_frame_variations_format as ubcf
    >>> d, N, nU, nD, rL, rR = 3, 6, 4, 5, 3, 2
    >>> up    = np.random.randn(d, nU, N)
    >>> down  = np.random.randn(d, rL, nD, rR)
    >>> left  = np.random.randn(d, rL, nU, rL)
    >>> right = np.random.randn(d, rR, nU, rR)
    >>> shape = (4, 5, 6)                                          # real mode dims (each <= N=6)
    >>> up_mask    = np.arange(nU) < np.array([[2],[3],[4]])       # (d, nU); per-mode up rank
    >>> down_mask  = np.arange(nD) < np.array([[3],[4],[5]])       # (d, nD)
    >>> left_mask  = np.arange(rL) < np.array([[1],[2],[3],[1]])   # (d+1, rL); boundary TT ranks
    >>> right_mask = np.arange(rR) < np.array([[1],[2],[2],[1]])   # (d+1, rR)
    >>> masks = ubcf.UT3FrameMasks(up_mask, down_mask, left_mask, right_mask)
    >>> B = ubcf.UT3Frame(up, down, left, right, shape, masks)
    >>> B.shape
    (4, 5, 6)
    >>> np.asarray(B.up_ranks).tolist()
    [2, 3, 4]
    >>> B.stack_shape
    ()
    """
    up_tucker_supercore:    NDArray              # B_dxo B_dyo   = I_dxy, shape = (d,)+stack_shape+(nU, N)
    down_tt_supercore:      NDArray              # R_dixj R_diyj = I_dxy  shape = (d,)+stack_shape+(rL, nD, rR)
    left_tt_supercore:      NDArray              # P_diax P_diay = I_dxy, shape = (d,)+stack_shape+(rL, nU, rL)
    right_tt_supercore:     NDArray              # Q_dxaj Q_dyaj = I_dxy  shape = (d,)+stack_shape+(rR, nU, rR)
    shape:                  typ.Tuple[int, ...]  # len=d; (N0,...,N(d-1)) real mode dims, shared across the stack
    masks:                  UT3FrameMasks        # static rank structure (the four edge masks)

    @ft.cached_property
    def data(self) -> typ.Tuple[
        NDArray,                                   # up_tucker_supercore
        NDArray,                                   # down_tt_supercore
        NDArray,                                   # left_tt_supercore
        NDArray,                                   # right_tt_supercore
        typ.Tuple[int, ...],                       # shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (up_mask, down_mask, frame_left_mask, frame_right_mask)
    ]:
        return (
            self.up_tucker_supercore, self.down_tt_supercore,
            self.left_tt_supercore, self.right_tt_supercore,
            self.shape, self.masks.data,
        )

    @ft.cached_property
    def d(self) -> int:
        return self.up_tucker_supercore.shape[0]

    @ft.cached_property
    def N(self) -> int:
        return self.up_tucker_supercore.shape[-1]

    @ft.cached_property
    def nU(self) -> int:
        return self.up_tucker_supercore.shape[-2]

    @ft.cached_property
    def nD(self) -> int:
        return self.down_tt_supercore.shape[-2]

    @ft.cached_property
    def rL(self) -> int:
        return self.left_tt_supercore.shape[-1]

    @ft.cached_property
    def rR(self) -> int:
        return self.right_tt_supercore.shape[-1]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        return self.up_tucker_supercore.shape[1:-2]

    @ft.cached_property
    def uniform_structure(self) -> typ.Tuple[
        int, # d
        int, # N
        int, # nU
        int, # nD
        int, # rL
        int, # rR
        typ.Tuple[int,...], # stack_shape
    ]:
        return self.d, self.N, self.nU, self.nD, self.rL, self.rR, self.stack_shape

    @ft.cached_property
    def uniform_variation_shapes(self) -> typ.Tuple[
        typ.Tuple[int,...], # uniform_tucker_variations_shape = (d, nD, N)
        typ.Tuple[int,...], # uniform_tt_variations_shape = (d, rL, nU, rR)
    ]: # does not include stack_shape
        uniform_tucker_variations_shape = (self.d, self.nD, self.N)
        uniform_tt_variations_shape = (self.d, self.rL, self.nU, self.rR)
        return uniform_tucker_variations_shape, uniform_tt_variations_shape

    # (`shape` is a stored field above -- the real mode dims, shared across the stack.)

    @ft.cached_property
    def up_ranks(self) -> NDArray:
        return self.masks.up_mask.sum(axis=-1)

    @ft.cached_property
    def down_ranks(self) -> NDArray:
        return self.masks.down_mask.sum(axis=-1)

    @ft.cached_property
    def left_ranks(self) -> NDArray:
        return self.masks.frame_left_mask.sum(axis=-1)

    @ft.cached_property
    def right_ranks(self) -> NDArray:
        return self.masks.frame_right_mask.sum(axis=-1)

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int, ...], # shape
        NDArray, # up_ranks
        NDArray,  # down_tt_ranks
        NDArray, # left_ranks
        NDArray, # right_ranks
        typ.Tuple[int,...], # stack_shape
    ]:
        return (
            self.shape, self.up_ranks, self.down_ranks,
            self.left_ranks, self.right_ranks,
            self.stack_shape,
        )

    def apply_masks(self) -> 'UT3Frame':
        """Apply masks to the frame supercores, zeroing out unmasked entries.
        """
        up_sc, down_sc, left_sc, right_sc = masking.ufv_apply_frame_masks(self.data)
        return UT3Frame(
            up_sc, down_sc, left_sc, right_sc,
            self.shape, self.masks,
        )

    # ------------------------------------------------------------- validity checkers (per stack element)
    @ft.cached_property
    def orthogonality_residual(self) -> NDArray:  # shape = stack_shape (scalar/0-d when unstacked)
        """Max deviation of the (masked) frame supercores from orthonormality, **per stack element**
        (uniform analog of :py:attr:`~t3toolbox.frame_variations_format.T3Frame.orthogonality_residual`;
        masked-Gram over the four senses -- see
        :py:func:`~t3toolbox.backend.ufv_operations.ufv_frame_orthogonality_residual`). **cached**."""
        return ufv_operations.ufv_frame_orthogonality_residual(self.data)

    def is_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        """True (per stack element) if the frame supercores are orthogonal in their respective senses on
        the real (masked) content. Per-element bool array (reduce with ``.all()`` for a single verdict);
        verified against the ragged oracle ``to_t3frame(self).is_orthogonal()``."""
        return self.orthogonality_residual <= atol

    @ft.cached_property
    def minimal_ranks(self):  # (min_tucker (d,)+stack, min_tt (d+1,)+stack) -- host int, per stack element
        """Structural minimal ranks ``(min_tucker_ranks, min_tt_ranks)`` for this frame's shape/ranks,
        **per stack element** (the ranks vary across the stack -- the determinantal variety). Host ints."""
        return ranks.compute_minimal_ranks(self.shape, self.up_ranks, self.left_ranks)

    @ft.cached_property
    def has_minimal_ranks(self) -> NDArray:  # bool array, shape = stack_shape (per element; uniform ranks vary)
        """True (per stack element) if the frame has structurally minimal ranks: ``left==right``,
        ``up==down``, and up/left equal the minimal ranks for the shape (reduced over the mode axis).
        Non-enforcing; not a correctness precondition (``docs/numerical_contracts.md``)."""
        up, down = np.asarray(self.up_ranks), np.asarray(self.down_ranks)        # (d,)+stack
        left, right = np.asarray(self.left_ranks), np.asarray(self.right_ranks)  # (d+1,)+stack
        mn_tk, mn_tt = self.minimal_ranks
        return (np.all(left == right, axis=0) & np.all(up == down, axis=0)
                & np.all(up == np.asarray(mn_tk), axis=0) & np.all(left == np.asarray(mn_tt), axis=0))

    def has_numerically_minimal_ranks(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape
        """True (per stack element) if the frame is **numerically** minimal.

        Equals ``is_orthogonal(atol) & has_minimal_ranks`` (orthonormal cores are full-rank, so
        orthogonal + structurally minimal => numerically minimal; no SVD). Mirrors
        ``T3Frame.has_numerically_minimal_ranks``."""
        return self.is_orthogonal(atol=atol) & self.has_minimal_ranks

    def is_consistent(self, rtol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape
        """``True`` (per stack element) if the left- and right-canonical reconstructions of the base point
        agree: ``||up·left - up·right|| <= rtol * ||up·right||`` (dense Frobenius norm). EXPENSIVE
        (densifies both). Consistent by construction for a frame from :py:func:`ut3_orthogonal_representations`."""
        left_ut3 = ut3.UniformTuckerTensorTrain(
            self.up_tucker_supercore, self.left_tt_supercore, self.shape,
            ut3.UT3Masks(self.masks.up_mask, self.masks.frame_left_mask))
        right_ut3 = self.to_ut3()
        return (left_ut3 - right_ut3).norm() <= rtol * right_ut3.norm()

    def allclose(
            self,
            other: 'UT3Frame',

            rtol:  typ.Optional[float] = None,  # None: the ambient jax-aware default (safety.comparison_rtol)
            atol:  float = 0.0,
    ) -> NDArray:  # bool array, shape = stack_shape; scalar unstacked; reduce with .all()
        """``True`` (per stack element) if ``other`` represents the same BASE POINT as ``self``
        (gauge-invariant): ``||self.to_ut3() - other.to_ut3()|| <= atol + rtol * max(norms)``. Three
        distinct equality questions exist for a frame; say which you mean: this method (same base
        point, any gauge); ``safety.frames_equal`` on the masked supercores (the same-FRAME /
        same-tangent-space question); :py:meth:`corewise_equal` (bitwise incl. padding). ``==`` is
        intentionally not defined."""
        a, b = self.to_ut3(), other.to_ut3()
        if rtol is None:
            rtol = safety_mod.comparison_rtol(a.supercores + b.supercores)
        use_jax = common.tree_contains_jax(a.supercores + b.supercores)
        xnp, _, _ = common.get_backend(True, use_jax)
        dn = (a - b).norm()
        rn = xnp.maximum(a.norm(), b.norm())                    # xnp: jit-safe (traced norms)
        return dn <= atol + rtol * rn

    # ------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3frame(
            frame: bvf.T3Frame,
            N:  typ.Optional[int] = None,   # padded mode dim   (default max(shape))
            nU: typ.Optional[int] = None,   # padded up rank    (default max(up_ranks))
            nD: typ.Optional[int] = None,   # padded down rank  (default max(down_ranks))
            rL: typ.Optional[int] = None,   # padded left rank  (default max(left_ranks))
            rR: typ.Optional[int] = None,   # padded right rank (default max(right_ranks))
    ) -> 'UT3Frame':
        """Pack a ragged :py:class:`~t3toolbox.frame_variations_format.T3Frame` into a uniform frame.

        A single ragged frame has ranks shared across its ``C`` stack, so the masks come out **uniform
        across the stack** (varying-rank uniform batches arise only by ``stack``-ing a heterogeneous tree).
        """
        uc, dc, lc, rc, shape, masks = ufv_conversions.t3frame_to_ut3frame(
            frame.data, N=N, nU=nU, nD=nD, rL=rL, rR=rR)
        return UT3Frame(uc, dc, lc, rc, shape, UT3FrameMasks(*masks))


    def corewise_equal(
            self,
            other: 'UT3Frame',
    ) -> bool:
        """Bitwise equality of the whole stored representation -- shape, rank masks, and raw
        supercores including padding (``False`` on any mismatch, never raises)."""
        return (type(other) is type(self) and self.shape == other.shape
                and cw.corewise_equal(self.masks.data, other.masks.data)
                and cw.corewise_equal(self.data[:4], other.data[:4]))

    def to_t3frame(self):  # -> bvf.T3Frame, or a nested tree (shaped like stack_shape) of them if stacked
        """Convert to a ragged :py:class:`~t3toolbox.frame_variations_format.T3Frame` (or, if stacked, an
        array-like tree of them)."""
        d = self.d
        result = ufv_conversions.ut3frame_to_t3frame(self.apply_masks().data)
        return stacking.apply_func_to_leaf_subtrees(
            result,
            lambda c: bvf.T3Frame(*c),
            ((None,) * d,) * 4,  # leaf_structure: 4 core-families, each a length-d tuple
        )

    # ------------------------------------------------------------- base point / orthogonal frame
    @staticmethod
    def from_ut3(x: ut3.UniformTuckerTensorTrain) -> 'UT3Frame':
        """Orthogonal frame at the point ``x`` (the frame part of :py:func:`ut3_orthogonal_representations`).
        Uniform analog of :py:meth:`~t3toolbox.frame_variations_format.T3Frame.from_t3`."""
        return ut3_orthogonal_representations(x)[0]

    def to_ut3(self) -> ut3.UniformTuckerTensorTrain:
        """The base point this frame represents, as a :py:class:`UniformTuckerTensorTrain` (right-canonical:
        the Tucker supercore over the right-orthogonal TT supercore). Uniform analog of ``T3Frame.to_t3``;
        the plain-UT3 tt edge mask is the frame's ``frame_right_mask`` (the right TT ranks)."""
        return ut3.UniformTuckerTensorTrain(
            self.up_tucker_supercore, self.right_tt_supercore,
            self.shape, ut3.UT3Masks(self.masks.up_mask, self.masks.frame_right_mask))

    def to_dense(self) -> NDArray:
        """Dense tensor of the base point this frame represents (``= to_ut3().to_dense()``)."""
        return self.to_ut3().to_dense()

    def reverse(self) -> 'UT3Frame':
        """Reverse the mode order. Left/right supercores (and masks) **swap roles** -- reversing a
        left-orthogonal chain yields a right-orthogonal one -- so the redundant L/R store makes this exact
        with no re-orthogonalization. Commutes with conversion: ``B.reverse().to_t3frame() ==
        B.to_t3frame().reverse()``."""
        up, down, left, right, shape, masks = ufv_operations.ufv_frame_reverse(self.data)
        return UT3Frame(up, down, left, right, shape, UT3FrameMasks(*masks))

    def orthogonalize(self) -> 'UT3Frame':
        """Orthogonal representation of the base point this frame reconstructs to (``= from_ut3(to_ut3())``).
        For an already-orthogonal frame, an equivalent orthogonal frame; for a drifted one, a genuinely
        orthogonal (minimal-rank) frame for the right-canonical base point."""
        return UT3Frame.from_ut3(self.to_ut3())

    @staticmethod
    def random_orthogonal(
            shape:        typ.Sequence[int],   # (N0,...,N(d-1))
            tucker_ranks,                       # int | len-d seq | (d,)+stack array (the variety)
            tt_ranks,                           # int | len-(d+1) seq | (d+1,)+stack array
            stack_shape:  typ.Tuple[int, ...] = (),
            use_jax:      bool = False,
    ) -> 'UT3Frame':
        """Orthogonal representation of a *random* uniform T3 -- a genuine random base point (orthogonal,
        consistent), not iid-random supercores. Equals ``from_ut3(UniformTuckerTensorTrain.randn(...))``."""
        x = ut3.UniformTuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks,
                                               stack_shape=tuple(stack_shape), use_jax=use_jax)
        return UT3Frame.from_ut3(x)

    @staticmethod
    def random_orthogonal_like(frame: 'UT3Frame') -> 'UT3Frame':
        """A random orthogonal frame with the same shape / ranks / stack as ``frame``."""
        return UT3Frame.random_orthogonal(frame.shape, frame.up_ranks, frame.left_ranks,
                                          stack_shape=frame.stack_shape, use_jax=frame.contains_jax)

    # ------------------------------------------------------------- save / load
    def save(self, file) -> None:
        """Save the four supercores + ``shape`` + masks to a ``.npz`` file (load with :py:meth:`load`)."""
        ufv_operations.ufv_save(file, self.data)

    @staticmethod
    def load(file, use_jax: bool = False) -> 'UT3Frame':
        """Load a frame saved by :py:meth:`save`. Supercores follow ``use_jax``; masks stay host numpy."""
        up, down, left, right, shape, masks = ufv_operations.ufv_load(file, use_jax=use_jax)
        return UT3Frame(up, down, left, right, shape, UT3FrameMasks(*masks))

    # ------------------------------------------------------------- dtype / copy / repr
    @ft.cached_property
    def supercores(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:
        """The four padded supercores ``(up, down, left, right)`` (the data; masks ride separately)."""
        return (self.up_tucker_supercore, self.down_tt_supercore,
                self.left_tt_supercore, self.right_tt_supercore)

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any supercore is a jax array (the masks are always host numpy)."""
        return tree_contains_jax(self.supercores)

    def to_jax(self) -> 'UT3Frame':
        # Supercores -> jax; masks stay host numpy (a jax mask is a tracer under jit). See docs/uniform_*.
        return UT3Frame(*(to_jax(sc) for sc in self.supercores), self.shape, self.masks)

    def to_numpy(self) -> 'UT3Frame':
        # Supercores -> numpy; the masks are already host numpy, so reuse the holder.
        return UT3Frame(*(to_numpy(sc) for sc in self.supercores), self.shape, self.masks)

    def copy(self) -> 'UT3Frame':
        # Deep-copy the supercores (the data leaves), like ragged T3Frame.copy; the static aux (shape +
        # masks) is shared (immutable structure, the same way `shape` is not duplicated).
        return UT3Frame(*(sc.copy() for sc in self.supercores), self.shape, self.masks)

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return ('UT3Frame(shape=%s, N=%d, nU=%d, nD=%d, rL=%d, rR=%d%s)'
                % (self.shape, self.N, self.nU, self.nD, self.rL, self.rR, ss))

    def validate(self) -> None:
        '''Check rank and shape consistency of a uniform Tucker tensor train frame (`UT3Frame`).

        Raises
        ------
        ValueError
            If the supercores / masks have inconsistent shapes, or `shape` is not a length-d tuple of
            mode dims within the padded N.
        '''
        up_mask, down_mask, frame_left_mask, frame_right_mask = self.masks.data

        UU_good = self.up_tucker_supercore.shape  == (self.d,) + self.stack_shape + (self.nU, self.N)
        DD_good = self.down_tt_supercore.shape    == (self.d,) + self.stack_shape + (self.rL, self.nD, self.rR)
        LL_good = self.left_tt_supercore.shape    == (self.d,) + self.stack_shape + (self.rL, self.nU, self.rL)
        RR_good = self.right_tt_supercore.shape   == (self.d,) + self.stack_shape + (self.rR, self.nU, self.rR)

        UM_good = up_mask.shape          == (self.d,) + self.stack_shape + (self.nU,)
        DM_good = down_mask.shape        == (self.d,) + self.stack_shape + (self.nD,)
        LM_good = frame_left_mask.shape  == (self.d + 1,) + self.stack_shape + (self.rL,)
        RM_good = frame_right_mask.shape == (self.d + 1,) + self.stack_shape + (self.rR,)

        SH_good = (len(self.shape) == self.d) and all(0 <= Ni <= self.N for Ni in self.shape)

        bad_str = lambda x: ' <-- Bad' if not x else ''

        shapes_string = ''
        shapes_string += 'up_tucker_supercore.shape = ' + str(self.up_tucker_supercore.shape)   + ' =? (d,) + stack_shape + (nU, N)' + bad_str(UU_good) + '\n'
        shapes_string += 'down_tt_supercore.shape   = ' + str(self.down_tt_supercore.shape)     + ' =? (d,) + stack_shape + (rL, nD, rR)' + bad_str(DD_good) + '\n'
        shapes_string += 'left_tt_supercore.shape   = ' + str(self.left_tt_supercore.shape)     + ' =? (d,) + stack_shape + (rL, nU, rL)' + bad_str(LL_good) + '\n'
        shapes_string += 'right_tt_supercore.shape  = ' + str(self.right_tt_supercore.shape)    + ' =? (d,) + stack_shape + (rR, nU, rR)' + bad_str(RR_good) + '\n'
        shapes_string += 'up_mask.shape             = ' + str(up_mask.shape) + ' =? (d,) + stack_shape + (nU,)' + bad_str(UM_good) + '\n'
        shapes_string += 'down_mask.shape           = ' + str(down_mask.shape) + ' =? (d,) + stack_shape + (nD,)' + bad_str(DM_good) + '\n'
        shapes_string += 'frame_left_mask.shape     = ' + str(frame_left_mask.shape) + ' =? (d+1,) + stack_shape + (rL,)' + bad_str(LM_good) + '\n'
        shapes_string += 'frame_right_mask.shape    = ' + str(frame_right_mask.shape) + ' =? (d+1,) + stack_shape + (rR,)' + bad_str(RM_good) + '\n'
        shapes_string += 'shape                     = ' + str(self.shape) + ' =? length-d ints in [0, N]' + bad_str(SH_good)

        if not (UU_good and DD_good and LL_good and RR_good and UM_good and DM_good and LM_good and RM_good and SH_good):
            raise ValueError(
                'Inconsistent shapes for UT3Frame.\n'
                + shapes_string
            )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack a stacked UT3Frame into an array-like tree (shaped like ``stack_shape``) of UT3Frame."""
        return stacking.apply_func_to_leaf_subtrees(
            ufv_operations.ufv_unstack(self.data, 4),
            lambda leaf: UT3Frame(leaf[0], leaf[1], leaf[2], leaf[3], leaf[4], UT3FrameMasks(*leaf[5])),
            ufv_operations.ufv_leaf_structure(self.d, 4),
        )

    @staticmethod
    def stack(xx):  # Array-like tree of UT3Frame
        """Stack an array-like tree of UT3Frame into a single stacked UT3Frame."""
        data_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda b: b.data, None)
        up, down, left, right, shape, masks = ufv_operations.ufv_stack(data_tree, 4)
        return UT3Frame(up, down, left, right, shape, UT3FrameMasks(*masks))


@dataclass(frozen=True, eq=False)  # eq=False -> the mixin's VALUE-based __hash__/__eq__ stand
class UT3VariationsMasks(common.ValueHashedMasks):
    """The static rank structure of a :py:class:`UT3Variations`: its four boolean edge masks.

    Value-hashed (the :py:class:`~t3toolbox.backend.common.ValueHashedMasks` mixin) so a rebuilt-but-
    identical holder is the same jit cache key. NOTE: the left/right masks are ``(d,)`` here (not
    ``(d+1,)`` as on :py:class:`UT3FrameMasks`) -- a variation occupies one TT slot, not a boundary edge.
    """
    variations_up_mask:    NDArray  # dtype=bool, (d,)+stack_shape+(nU,)
    variations_down_mask:  NDArray  # dtype=bool, (d,)+stack_shape+(nD,)
    variations_left_mask:  NDArray  # dtype=bool, (d,)+stack_shape+(rL,)
    variations_right_mask: NDArray  # dtype=bool, (d,)+stack_shape+(rR,)

    def __post_init__(self):
        # Defensive READ-ONLY copies: the value key is cached on this frozen holder, so an aliased
        # writeable caller array mutated in place would leave a stale jit cache key (review H1-5).
        for _f in ('variations_up_mask', 'variations_down_mask', 'variations_left_mask', 'variations_right_mask'):
            _m = np.array(getattr(self, _f), copy=True)
            _m.setflags(write=False)
            object.__setattr__(self, _f, _m)

    @property
    def data(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:
        """``(variations_up_mask, variations_down_mask, variations_left_mask, variations_right_mask)``."""
        return (self.variations_up_mask, self.variations_down_mask,
                self.variations_left_mask, self.variations_right_mask)


@dataclass(frozen=True, eq=False)   # eq=False -> the ExplicitEquality mixin stands
class UT3Variations(common.ExplicitEquality):
    """Variation cores for the frame-variations representation of uniform Tucker tensor trains.

    Uniform analog of :py:class:`~t3toolbox.frame_variations_format.T3Variations`: two padded supercores
    (``tucker_variations``, ``tt_variations``) + the static physical ``shape`` (an int tuple, shared across
    the stack) + a :py:class:`UT3VariationsMasks` holder. The variations fit in the "holes" of a
    :py:class:`UT3Frame`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_frame_variations_format as ubcf
    >>> d, N, nU, nD, rL, rR = 3, 6, 4, 5, 3, 2
    >>> tkv = np.random.randn(d, nD, N)
    >>> ttv = np.random.randn(d, rL, nU, rR)
    >>> shape = (4, 5, 6)
    >>> up    = np.arange(nU) < np.array([[2],[3],[4]])       # (d, nU)
    >>> down  = np.arange(nD) < np.array([[3],[4],[5]])       # (d, nD)
    >>> left  = np.arange(rL) < np.array([[1],[2],[3]])       # (d, rL) -- note d, not d+1
    >>> right = np.arange(rR) < np.array([[1],[2],[2]])       # (d, rR)
    >>> masks = ubcf.UT3VariationsMasks(up, down, left, right)
    >>> V = ubcf.UT3Variations(tkv, ttv, shape, masks)
    >>> V.uniform_structure
    (3, 6, 4, 5, 3, 2, ())
    >>> V.uniform_variation_shapes
    ((3, 5, 6), (3, 3, 4, 2))
    """
    tucker_variations: NDArray              # (d,) + stack_shape + (nD, N)
    tt_variations:     NDArray              # (d,) + stack_shape + (rL, nU, rR)
    shape:             typ.Tuple[int, ...]  # len=d; (N0,...,N(d-1)) real mode dims, shared across the stack
    masks:             UT3VariationsMasks   # static rank structure (the four edge masks)

    @ft.cached_property
    def d(self) -> int:
        return self.tucker_variations.shape[0]

    @ft.cached_property
    def N(self) -> int:
        return self.tucker_variations.shape[-1]

    @ft.cached_property
    def nU(self) -> int:
        return self.tt_variations.shape[-2]

    @ft.cached_property
    def nD(self) -> int:
        return self.tucker_variations.shape[-2]

    @ft.cached_property
    def rL(self) -> int:
        return self.tt_variations.shape[-3]

    @ft.cached_property
    def rR(self) -> int:
        return self.tt_variations.shape[-1]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        return self.tucker_variations.shape[1:-2]

    @ft.cached_property
    def uniform_structure(self) -> typ.Tuple[
        int, # d
        int, # N
        int, # nU
        int, # nD
        int, # rL
        int, # rR
        typ.Tuple[int,...], # stack_shape
    ]:
        return self.d, self.N, self.nU, self.nD, self.rL, self.rR, self.stack_shape

    @ft.cached_property
    def uniform_variation_shapes(self) -> typ.Tuple[
        typ.Tuple[int,...], # uniform_tucker_variations_shape = (d, nD, N)
        typ.Tuple[int,...], # uniform_tt_variations_shape = (d, rL, nU, rR)
    ]: # does not include stack_shape
        uniform_tucker_variations_shape = (self.d, self.nD, self.N)
        uniform_tt_variations_shape = (self.d, self.rL, self.nU, self.rR)
        return uniform_tucker_variations_shape, uniform_tt_variations_shape

    # (`shape` is a stored field above.)

    @ft.cached_property
    def up_ranks(self) -> NDArray:
        return self.masks.variations_up_mask.sum(axis=-1)

    @ft.cached_property
    def down_ranks(self) -> NDArray:
        return self.masks.variations_down_mask.sum(axis=-1)

    @ft.cached_property
    def variation_left_ranks(self) -> NDArray:
        return self.masks.variations_left_mask.sum(axis=-1)

    @ft.cached_property
    def variation_right_ranks(self) -> NDArray:
        return self.masks.variations_right_mask.sum(axis=-1)

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int, ...], # shape
        NDArray, # up_ranks
        NDArray, # down_tt_ranks
        NDArray, # variations_left_ranks
        NDArray, # variations_right_ranks
        typ.Tuple[int,...], # stack_shape
    ]:
        return (
            self.shape, self.up_ranks, self.down_ranks,
            self.variation_left_ranks, self.variation_right_ranks,
            self.stack_shape,
        )

    @ft.cached_property
    def data(self) -> typ.Tuple[
        NDArray,                                        # tucker_variations
        NDArray,                                        # tt_variations
        typ.Tuple[int, ...],                            # shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (variations up, down, left, right masks)
    ]:
        return (self.tucker_variations, self.tt_variations, self.shape, self.masks.data)

    def allclose(
            self,
            other: 'UT3Variations',

            rtol:  typ.Optional[float] = None,  # None: the ambient jax-aware default (safety.comparison_rtol)
            atol:  typ.Optional[float] = None,  # None: 0.0
    ) -> NDArray:  # bool, shape = stack_shape (K+C); scalar unstacked; reduce with .all()
        """True where the MASKED variation supercores are numerically equal (coordinate-level,
        padding don't-care), per stack element (norm-based ``atol + rtol * max`` reference).
        Different shape or rank masks raise. Bitwise incl. padding: :py:meth:`corewise_equal`;
        ``==`` is undefined."""
        if self.shape != other.shape or not cw.corewise_equal(self.masks.data, other.masks.data):
            raise ValueError('UT3Variations.allclose: different shape or rank masks')
        if rtol is None:
            rtol = safety_mod.comparison_rtol(self.supercores + other.supercores)
        atol = 0.0 if atol is None else atol
        use_jax = common.tree_contains_jax(self.supercores + other.supercores)
        xnp, _, _ = common.get_backend(True, use_jax)
        def sq(pair):   # (tucker (d,)+stack+(nD,N), tt (d,)+stack+(rL,nU,rR)) -> stack
            tk, tt = pair
            return (xnp.sum(tk * tk, axis=(0, -2, -1)) + xnp.sum(tt * tt, axis=(0, -3, -2, -1)))
        a, b = self.apply_masks().supercores, other.apply_masks().supercores
        dn = sq(tuple(x - y for x, y in zip(a, b))) ** 0.5
        rn = xnp.maximum(sq(a), sq(b)) ** 0.5
        return dn <= atol + rtol * rn

    def corewise_equal(
            self,
            other: 'UT3Variations',
    ) -> bool:
        """Bitwise equality of the whole stored representation -- shape, masks, raw supercores
        including padding (``False`` on any mismatch, never raises)."""
        return (type(other) is type(self) and self.shape == other.shape
                and cw.corewise_equal(self.masks.data, other.masks.data)
                and cw.corewise_equal(self.supercores, other.supercores))

    def apply_masks(self) -> 'UT3Variations':
        """Apply masks to the variation supercores, zeroing out unmasked entries."""
        masked_tk_supercore, masked_tt_supercore = masking.ufv_apply_variations_masks(self.data)
        return UT3Variations(masked_tk_supercore, masked_tt_supercore, self.shape, self.masks)

    # ------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3variations(
            variations: bvf.T3Variations,
            N:  typ.Optional[int] = None,   # padded mode dim   (default max(shape))
            nU: typ.Optional[int] = None,   # padded up rank    (default max(up_ranks))
            nD: typ.Optional[int] = None,   # padded down rank  (default max(down_ranks))
            rL: typ.Optional[int] = None,   # padded left rank  (default max(left_ranks))
            rR: typ.Optional[int] = None,   # padded right rank (default max(right_ranks))
    ) -> 'UT3Variations':
        """Pack a ragged :py:class:`~t3toolbox.frame_variations_format.T3Variations` into uniform variations."""
        tkv, ttv, shape, masks = ufv_conversions.t3variations_to_ut3variations(
            variations.data, N=N, nU=nU, nD=nD, rL=rL, rR=rR)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def to_t3variations(self):  # -> bvf.T3Variations, or a nested tree (shaped like stack_shape) of them if stacked
        """Convert to ragged :py:class:`~t3toolbox.frame_variations_format.T3Variations` (or, if stacked, an
        array-like tree of them)."""
        d = self.d
        result = ufv_conversions.ut3variations_to_t3variations(self.apply_masks().data)
        return stacking.apply_func_to_leaf_subtrees(
            result,
            lambda c: bvf.T3Variations(*c),
            ((None,) * d,) * 2,  # leaf_structure: 2 core-families, each a length-d tuple
        )

    # ------------------------------------------------------------- dtype / copy / repr
    @ft.cached_property
    def supercores(self) -> typ.Tuple[NDArray, NDArray]:
        """The two padded supercores ``(tucker_variations, tt_variations)`` (the data; masks ride separately)."""
        return (self.tucker_variations, self.tt_variations)

    @ft.cached_property
    def contains_jax(self) -> bool:
        """True if any supercore is a jax array (the masks are always host numpy)."""
        return tree_contains_jax(self.supercores)

    def to_jax(self) -> 'UT3Variations':
        # Supercores -> jax; masks stay host numpy (a jax mask is a tracer under jit). See docs/uniform_*.
        return UT3Variations(*(to_jax(sc) for sc in self.supercores), self.shape, self.masks)

    def to_numpy(self) -> 'UT3Variations':
        return UT3Variations(*(to_numpy(sc) for sc in self.supercores), self.shape, self.masks)

    def copy(self) -> 'UT3Variations':
        # Deep-copy the supercores (the data leaves), like ragged T3Variations.copy; the static aux (shape +
        # masks) is shared (immutable structure, the same way `shape` is not duplicated).
        return UT3Variations(*(sc.copy() for sc in self.supercores), self.shape, self.masks)

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return ('UT3Variations(shape=%s, N=%d, nU=%d, nD=%d, rL=%d, rR=%d%s)'
                % (self.shape, self.N, self.nU, self.nD, self.rL, self.rR, ss))

    def validate(self) -> None:
        '''Check rank and shape consistency of a uniform Tucker tensor train variations (`UT3Variations`).
        Raises ValueError.'''
        up_mask, down_mask, left_mask, right_mask = self.masks.data

        TK_good = self.tucker_variations.shape  == (self.d,) + self.stack_shape + (self.nD, self.N)
        TT_good = self.tt_variations.shape      == (self.d,) + self.stack_shape + (self.rL, self.nU, self.rR)

        UM_good = up_mask.shape    == (self.d,) + self.stack_shape + (self.nU,)
        DM_good = down_mask.shape  == (self.d,) + self.stack_shape + (self.nD,)
        LM_good = left_mask.shape  == (self.d,) + self.stack_shape + (self.rL,)
        RM_good = right_mask.shape == (self.d,) + self.stack_shape + (self.rR,)

        SH_good = (len(self.shape) == self.d) and all(0 <= Ni <= self.N for Ni in self.shape)

        bad_str = lambda x: ' <-- Bad' if not x else ''
        shapes_string = ''
        shapes_string += 'tucker_variations.shape     = ' + str(self.tucker_variations.shape) + ' =? (d,) + stack_shape + (nD, N)'      + bad_str(TK_good) + '\n'
        shapes_string += 'tt_variations.shape         = ' + str(self.tt_variations.shape)     + ' =? (d,) + stack_shape + (rL, nU, rR)' + bad_str(TT_good) + '\n'
        shapes_string += 'variations_up_mask.shape    = ' + str(up_mask.shape)    + ' =? (d,) + stack_shape + (nU,)' + bad_str(UM_good) + '\n'
        shapes_string += 'variations_down_mask.shape  = ' + str(down_mask.shape)  + ' =? (d,) + stack_shape + (nD,)' + bad_str(DM_good) + '\n'
        shapes_string += 'variations_left_mask.shape  = ' + str(left_mask.shape)  + ' =? (d,) + stack_shape + (rL,)' + bad_str(LM_good) + '\n'
        shapes_string += 'variations_right_mask.shape = ' + str(right_mask.shape) + ' =? (d,) + stack_shape + (rR,)' + bad_str(RM_good) + '\n'
        shapes_string += 'shape                       = ' + str(self.shape) + ' =? length-d ints in [0, N]' + bad_str(SH_good)

        if not (TK_good and TT_good and UM_good and DM_good and LM_good and RM_good and SH_good):
            raise ValueError('Inconsistent shapes for UT3Variations.\n' + shapes_string)

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack a stacked UT3Variations into an array-like tree (shaped like ``stack_shape``) of them."""
        return stacking.apply_func_to_leaf_subtrees(
            ufv_operations.ufv_unstack(self.data, 2),
            lambda leaf: UT3Variations(leaf[0], leaf[1], leaf[2], UT3VariationsMasks(*leaf[3])),
            ufv_operations.ufv_leaf_structure(self.d, 2),
        )

    @staticmethod
    def stack(xx):  # array-like tree of UT3Variations
        """Stack an array-like tree of UT3Variations into a single stacked UT3Variations."""
        data_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda v: v.data, None)
        tkv, ttv, shape, masks = ufv_operations.ufv_stack(data_tree, 2)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    # ------------------------------------------------------------- linear algebra (fixed-rank vector space)
    def _check_same_tangent_structure(self, other: 'UT3Variations') -> None:
        """Structural precondition for the vector-space ops: ``other`` must be the same fixed-rank tangent
        structure (padded dims + stack, real ``shape``, and the four rank masks). Uniform padding hides a
        mismatch that ragged would catch as a numpy shape error, so it is enforced explicitly; masks are
        host numpy, so the check is a cheap ``array_equal`` valid even under jit. The variation-level analog
        of the same-frame guard; see the "tangent vector-space ops" section of ``docs/uniform_masks_vs_ranks.md``.
        """
        if self.uniform_structure != other.uniform_structure:
            raise ValueError('UT3Variations op: structures differ (%s vs %s).'
                             % (self.uniform_structure, other.uniform_structure))
        if self.shape != other.shape:
            raise ValueError('UT3Variations op: shapes differ (%s vs %s).' % (self.shape, other.shape))
        if self.masks != other.masks:
            raise ValueError('UT3Variations op: rank masks differ (different tangent spaces).')

    def __add__(self, other: 'UT3Variations') -> 'UT3Variations':
        """Corewise sum (variations form a vector space at a fixed frame; the mask is unchanged)."""
        self._check_same_tangent_structure(other)
        tkv, ttv = cw.corewise_add(self.supercores, other.supercores)
        return UT3Variations(tkv, ttv, self.shape, self.masks)

    def __sub__(self, other: 'UT3Variations') -> 'UT3Variations':
        """Corewise difference (mask unchanged)."""
        self._check_same_tangent_structure(other)
        tkv, ttv = cw.corewise_sub(self.supercores, other.supercores)
        return UT3Variations(tkv, ttv, self.shape, self.masks)

    def __mul__(self, scalar) -> 'UT3Variations':
        """Corewise scalar multiplication (mask unchanged)."""
        tkv, ttv = cw.corewise_scale(self.supercores, scalar)
        return UT3Variations(tkv, ttv, self.shape, self.masks)

    __rmul__ = __mul__

    def __neg__(self) -> 'UT3Variations':
        """Corewise negation (mask unchanged)."""
        tkv, ttv = cw.corewise_neg(self.supercores)
        return UT3Variations(tkv, ttv, self.shape, self.masks)

    def reverse(self) -> 'UT3Variations':
        """Reverse the mode order (corewise): the tucker-variation supercore reverses; the tt-variation
        supercore reverses with a bond swap; the per-slot left/right masks swap. Matches
        :py:meth:`UT3Frame.reverse` so a tangent reverses by reversing both components."""
        tkv, ttv, shape, masks = ufv_operations.ufv_variations_reverse(self.data)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def save(self, file) -> None:
        """Save the two supercores + ``shape`` + masks to a ``.npz`` file (load with :py:meth:`load`)."""
        ufv_operations.ufv_save(file, self.data)

    @staticmethod
    def load(file, use_jax: bool = False) -> 'UT3Variations':
        """Load variations saved by :py:meth:`save`. Supercores follow ``use_jax``; masks stay host numpy."""
        tkv, ttv, shape, masks = ufv_operations.ufv_load(file, use_jax=use_jax)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def sum_stack(self, axis=None) -> 'UT3Variations':
        """Corewise sum over stack axes (a batch of variations -> their sum; the tangent sum, by linearity).
        ``axis`` indexes the stack (default: the whole stack). The mask ORs over the summed axes -- a no-op
        for a same-mask (single-frame) stack; see ``docs/uniform_masks_vs_ranks.md``."""
        tkv, ttv, shape, masks = ufv_operations.ufv_variations_sum_stack(self.data, axis)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def allclose(self, other: 'UT3Variations', rtol: float = 1e-9, atol: float = 0.0) -> NDArray:  # bool array, stack
        """``True`` (per stack element) if ``other`` matches ``self`` on the **real (masked)** content.

        The comparison is ``||self - other|| <= atol + rtol * ||other||``, the corewise (Euclidean)
        norm reduced over the **non-stack** axes -- the leading mode index ``d`` and the core axes --
        keeping the stack (reduce with ``.all()`` for a single verdict)."""
        n_stack = len(self.stack_shape)
        use_jax = self.contains_jax or other.contains_jax
        xnp, _, _ = get_backend(True, use_jax)

        def stack_norm(supercores):  # sqrt of summed squares over non-stack axes (d + core), keep the stack
            total = 0.0
            for sc in supercores:
                total = total + xnp.sum(sc ** 2, axis=(0,) + tuple(range(1 + n_stack, sc.ndim)))
            return xnp.sqrt(total)

        dn = stack_norm((self - other).apply_masks().supercores)
        rn = stack_norm(other.apply_masks().supercores)
        return dn <= atol + rtol * rn

    # ------------------------------------------------------------- constructors (fill complete; masks optional)
    @staticmethod
    def _filled(uniform_variation_shapes, shape, stack_shape, masks, use_jax, fill):
        (d, nD, N), (_d2, rL, nU, rR) = uniform_variation_shapes
        stack = tuple(stack_shape)
        tk_shape = (d,) + stack + (nD, N)
        tt_shape = (d,) + stack + (rL, nU, rR)
        if fill == 'randn':
            tkv = common.randn(*tk_shape, use_jax=use_jax)
            ttv = common.randn(*tt_shape, use_jax=use_jax)
        else:  # zeros
            tkv, ttv = np.zeros(tk_shape), np.zeros(tt_shape)
            if use_jax:
                tkv, ttv = to_jax(tkv), to_jax(ttv)
        if masks is None:  # default: all-True (full rank) -- the user fills the supercores completely
            masks = UT3VariationsMasks(
                np.ones((d,) + stack + (nU,), dtype=bool), np.ones((d,) + stack + (nD,), dtype=bool),
                np.ones((d,) + stack + (rL,), dtype=bool), np.ones((d,) + stack + (rR,), dtype=bool))
        return UT3Variations(tkv, ttv, tuple(shape), masks)

    @staticmethod
    def zeros(
            uniform_variation_shapes,                    # ((d, nD, N), (d, rL, nU, rR)) -- the padded supercore dims
            shape:       typ.Sequence[int],              # (N0,...,N(d-1)) real mode dims
            stack_shape: typ.Tuple[int, ...] = (),
            masks:       typ.Optional['UT3VariationsMasks'] = None,  # None -> all-True (full rank)
            use_jax:     bool = False,
    ) -> 'UT3Variations':
        """Zero variations filling the padded supercores completely (additive identity). ``masks`` optional;
        ``None`` -> all-True. (See :py:meth:`zeros_like` to take the structure -- incl. gauge masks -- from
        a :py:class:`UT3Frame` / :py:class:`UT3Variations`.)"""
        return UT3Variations._filled(uniform_variation_shapes, shape, stack_shape, masks, use_jax, 'zeros')

    @staticmethod
    def randn(
            uniform_variation_shapes,
            shape:       typ.Sequence[int],
            stack_shape: typ.Tuple[int, ...] = (),
            masks:       typ.Optional['UT3VariationsMasks'] = None,
            use_jax:     bool = False,
    ) -> 'UT3Variations':
        """Variations with i.i.d. N(0,1) supercore entries (filled completely; ungauged). See :py:meth:`randn_like`."""
        return UT3Variations._filled(uniform_variation_shapes, shape, stack_shape, masks, use_jax, 'randn')

    @staticmethod
    def unit(
            uniform_variation_shapes,
            shape:       typ.Sequence[int],
            index:       typ.Tuple[bool, int, typ.Sequence[int]],  # (use_tt_coordinate, i, within_index)
            stack_shape: typ.Tuple[int, ...] = (),
            masks:       typ.Optional['UT3VariationsMasks'] = None,
            use_jax:     bool = False,
    ) -> 'UT3Variations':
        """Canonical unit variation: zero supercores except a single entry set to 1 (broadcast over the
        stack). ``index = (use_tt_coordinate, i, within_index)`` selects the family, the mode ``i``, and the
        within-core entry. (For a meaningful unit the entry must land in a real slot -- automatic under the
        all-True default.)"""
        (d, nD, N), (_d2, rL, nU, rR) = uniform_variation_shapes
        stack = tuple(stack_shape)
        use_tt, i, within = index
        tkv = np.zeros((d,) + stack + (nD, N))
        ttv = np.zeros((d,) + stack + (rL, nU, rR))
        (ttv if use_tt else tkv)[(i, Ellipsis) + tuple(within)] = 1.0   # i = mode (leading axis); ... = stack
        if use_jax:
            tkv, ttv = to_jax(tkv), to_jax(ttv)
        if masks is None:
            masks = UT3VariationsMasks(
                np.ones((d,) + stack + (nU,), dtype=bool), np.ones((d,) + stack + (nD,), dtype=bool),
                np.ones((d,) + stack + (rL,), dtype=bool), np.ones((d,) + stack + (rR,), dtype=bool))
        return UT3Variations(tkv, ttv, tuple(shape), masks)

    @staticmethod
    def _variation_masks_of(x) -> 'UT3VariationsMasks':
        # x: UT3Frame (gauge-shift its frame masks -> left[:-1], right[1:]) or UT3Variations (its own masks).
        if isinstance(x, UT3Frame):
            bm = x.masks
            return UT3VariationsMasks(bm.up_mask, bm.down_mask, bm.frame_left_mask[:-1], bm.frame_right_mask[1:])
        return x.masks

    @staticmethod
    def zeros_like(x) -> 'UT3Variations':
        """Zero variations matching the structure of ``x`` (a :py:class:`UT3Frame` or :py:class:`UT3Variations`).
        For a frame this is the zero tangent carrying the frame's gauge masks."""
        return UT3Variations.zeros(x.uniform_variation_shapes, x.shape, stack_shape=x.stack_shape,
                                   masks=UT3Variations._variation_masks_of(x), use_jax=x.contains_jax)

    @staticmethod
    def randn_like(x) -> 'UT3Variations':
        """Random variations matching the structure (incl. gauge masks) of ``x`` (a UT3Frame or UT3Variations)."""
        return UT3Variations.randn(x.uniform_variation_shapes, x.shape, stack_shape=x.stack_shape,
                                   masks=UT3Variations._variation_masks_of(x), use_jax=x.contains_jax)


def check_ufv_pair(frame: UT3Frame, variations: UT3Variations) -> None:
    """Check rank and shape consistency between UT3Frame and UT3Variations.

    This ensures that the variation cores (V, H) have the correct dimensions
     to interface with the frame cores (U, L, R, O).

    The variations may carry an EXTRA leading tangent (``K``) stack on top of the frame's core
    (``C``) stack -- i.e. ``variations.stack_shape == K + frame.stack_shape`` -- because a bundle of
    tangent vectors all live at a single base point (the ``W+K+C`` convention, see
    ``docs/batching_and_stacking.md``). So the check compares the *stack-free* structure
    ``(d, N, nU, nD, rL, rR)``, requires the frame ``C`` stack to be the trailing suffix of the
    variations ``K+C`` stack, and matches the rank masks broadcast over the excess ``K`` (each
    variation mask must be constant along ``K`` and equal the frame's gauge-shifted mask).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_frame_variations_format as ubcf
    >>> stack_shape = ()                                    # unstacked, for a small readable example
    >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
    >>> uc = np.random.randn(*((d,) + stack_shape + (nU, N)))
    >>> dc = np.random.randn(*((d,) + stack_shape + (rL, nD, rR)))
    >>> lc = np.random.randn(*((d,) + stack_shape + (rL, nU, rL)))
    >>> rc = np.random.randn(*((d,) + stack_shape + (rR, nU, rR)))
    >>> shape = (10, 11, 12)
    >>> up = np.arange(nU) < np.array([[2],[3],[4]])        # (d, nU)
    >>> dn = np.arange(nD) < np.array([[3],[4],[5]])        # (d, nD)
    >>> bl = np.arange(rL) < np.array([[1],[2],[3],[1]])    # (d+1, rL) frame left
    >>> br = np.arange(rR) < np.array([[1],[2],[2],[1]])    # (d+1, rR) frame right
    >>> B = ubcf.UT3Frame(uc, dc, lc, rc, shape, ubcf.UT3FrameMasks(up, dn, bl, br))
    >>> tkv = np.random.randn(*((d,) + stack_shape + (nD, N)))
    >>> ttv = np.random.randn(*((d,) + stack_shape + (rL, nU, rR)))
    >>> V = ubcf.UT3Variations(tkv, ttv, shape,
    ...                        ubcf.UT3VariationsMasks(up, dn, bl[:-1], br[1:]))
    >>> ubcf.check_ufv_pair(B, V)   # consistent frame/variations pair -> no error

    A bundle of ``K`` tangent vectors at the SAME frame carries an extra leading stack; the masks must
    be constant along it and equal the frame's gauge-shifted masks:

    >>> K = (2,)
    >>> bcast = lambda m: np.broadcast_to(m[:, None], (m.shape[0],) + K + m.shape[1:])
    >>> VK = ubcf.UT3Variations(np.random.randn(*((d,) + K + (nD, N))),
    ...                         np.random.randn(*((d,) + K + (rL, nU, rR))), shape,
    ...                         ubcf.UT3VariationsMasks(bcast(up), bcast(dn), bcast(bl[:-1]), bcast(br[1:])))
    >>> ubcf.check_ufv_pair(B, VK)   # frame (C=()) vs K-stacked variations -> still consistent
    """
    # Compare the stack-free structure (d, N, nU, nD, rL, rR) -- NOT the full 7-tuple, whose trailing
    # stack_shape would wrongly reject a legitimate tangent (K) stack on the variations.
    if frame.uniform_structure[:6] != variations.uniform_structure[:6]:
        raise ValueError(
            'Inconsistent (UT3Frame, UT3Variations) pair: stack-free structures differ.\n'
            + str(frame.uniform_structure[:6]) + ' (frame) != '
            + str(variations.uniform_structure[:6]) + ' (variations)')

    if frame.shape != variations.shape:
        raise ValueError('Inconsistent (UT3Frame, UT3Variations) pair: shapes differ (%s vs %s).'
                         % (frame.shape, variations.shape))

    # The frame core (C) stack must be the trailing suffix of the variations (K+C) stack.
    frame_stack, var_stack = frame.stack_shape, variations.stack_shape
    n_K = len(var_stack) - len(frame_stack)
    if n_K < 0 or var_stack[n_K:] != frame_stack:
        raise ValueError(
            'Inconsistent (UT3Frame, UT3Variations) pair: frame stack_shape %s is not a trailing suffix '
            'of variations stack_shape %s (expected variations.stack_shape == K + frame.stack_shape).'
            % (frame_stack, var_stack))

    # Rank masks must match, broadcast over the excess K: reshape each frame mask to insert n_K size-1
    # axes after the leading core (d) axis, then broadcast up to the variations mask shape. Masks are
    # host numpy (static aux), so this stays on `np` (see CLAUDE.md: supercores -> xnp, masks -> np).
    bm, vm = frame.masks, variations.masks
    for a, b, name in (
            (bm.up_mask,              vm.variations_up_mask,    'up'),
            (bm.down_mask,            vm.variations_down_mask,  'down'),
            (bm.frame_left_mask[:-1], vm.variations_left_mask,  'left'),
            (bm.frame_right_mask[1:], vm.variations_right_mask, 'right'),
    ):
        a_bcast = np.broadcast_to(a.reshape(a.shape[:1] + (1,) * n_K + a.shape[1:]), b.shape)
        if not np.array_equal(a_bcast, b):
            raise ValueError(
                'Inconsistent (UT3Frame, UT3Variations) pair: %s rank masks differ '
                '(variation mask must be constant along the K stack and equal the frame mask).' % name)


def ut3_orthogonal_representations(
        x: ut3.UniformTuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash_tails: bool = True,
) -> typ.Tuple[
    UT3Frame,  # orthogonal frame
    UT3Variations,  # variations
]:
    '''Construct frame-variation representations of UniformTuckerTensorTrain with orthogonal frame.

    Input TuckerTensorTrain::

                  1 -- G0 -- G1 -- G2 -- G3 -- 1
        X    =         |     |     |     |
                       B0    B1    B2    B3
                       |     |     |     |

    Frame-variation representation with non-orthogonal TT-backend H1::

                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |

    Frame-variation representation with non-orthogonal tucker backend V2::

                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_tucker_cores     = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "frame cores" are:
        - tucker_cores       = (U0,U1, U2, U3), up orthogonal
        - left_tt_cores     = (L0, L1, L2),     left orthogonal
        - right_tt_cores    = (R1, R2, R3),     right orthogonal
        - outer_tt_cores    = (O0, O1, O2, O3), down orthogonal
    The "variation cores" are:
        - tucker_variations  = (V0, V1, V2, V3)
        - tt_variations     = (H0, H1, H2, H3)

    Parameters
    ----------
    x: TuckerTensorTrain
        Input TuckerTensorTrain
        x = (x_tucker_cores, x_tt_cores)
        x_tucker_cores = (B0, ..., B(d-1))
        x_tt_cores = (G0, ..., G(d-1))
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    T3Base
        Orthogonal frame for frame-variation representations of x.
    T3Variation
        Variation for frame-variation representaions of x.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_frame_variations_format as ubvf
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> frame, variations = ubvf.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
    >>> type(frame).__name__, type(variations).__name__
    ('UT3Frame', 'UT3Variations')
    >>> frame.shape
    (4, 5, 6)
    >>> # the orthogonal frame still represents the original tensor:
    >>> bool(np.allclose(frame.to_t3frame().to_dense(), x.to_dense()))
    True

    '''
    # Thin wrapper: the backend twin carries the logic (orthogonalize + build the SVD-justified prefix
    # masks); this only wraps the raw frame/variation .data into the OO classes.
    frame_data, variation_data = ufv_conversions.ut3_orthogonal_representations(
        x.data, already_left_orthogonal=already_left_orthogonal, squash_tails=squash_tails)
    uc, dc, lc, rc, shape, frame_masks = frame_data
    tkv, ttv, _, variation_masks = variation_data
    return (UT3Frame(uc, dc, lc, rc, shape, UT3FrameMasks(*frame_masks)),
            UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*variation_masks)))


# (`ufv_to_ut3` -- the uniform analog of `fv_to_t3`, substituting one variation core into the frame -- was
# dropped: the left/right subchains become differently-shaped supercores glued by the variation, with no
# clean single uniform supercore op. Low importance. See dev/archive/uniform_fix_plan.md "Refinements (round 2)".)



###########################################


@dataclass(frozen=True, eq=False)   # eq=False -> the ExplicitEquality mixin stands
class UT3FrameWeights(common.ExplicitEquality):
    """Diagonal weights defining a **metric on the tangent coordinates** of a :py:class:`UT3Frame` -- the
    uniform twin of :py:class:`~t3toolbox.frame_variations_format.T3FrameWeights`.

    Four families, each ``len=d`` (one per variation core), packed into supercores + a
    :py:class:`UT3VariationsMasks` holder: ``up`` (on ``H``'s ``nU`` leg), ``down`` (on ``V``'s ``nD``
    leg), ``left`` (``H``'s ``rL``), ``right`` (``H``'s ``rR``). Absorbed into the **variation**
    supercores, leaving the frame orthonormal and untouched -- so it is ``O(ranks)``.

    **Batching: a weight is FRAME-like** (it is *absorbed into* the variations, but it *batches with* the
    frame -- do not conflate the two). Every supercore is ``(d,) + C + (size,)`` where ``C`` is the
    **frame** stack, not the variations' ``K + C``: one metric per base point, shared by all ``K`` tangent
    vectors at that frame, broadcast over ``K`` for free (``C`` is innermost). There is no ``shape``
    field: weights live only on internal edges.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_frame_variations_format as ubvf
    >>> import t3toolbox.uniform_manifold as ut3m
    >>> np.random.seed(0)
    >>> x  = t3.TuckerTensorTrain.randn((6, 7, 8), (2, 2, 2), (1, 2, 2, 1), stack_shape=(2,))
    >>> ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    >>> frame, _ = ubvf.ut3_orthogonal_representations(ux)
    >>> v = ut3m.UNIFORM_COREWISE.randn(frame, stack_shape=(3,))   # 3 tangents at each of 2 base points
    >>> print(frame.stack_shape, v.stack_shape)            # C, then K + C
    (2,) (3, 2)

    The metric is built from the base point's singular values, so it carries ``C`` -- and pairs directly
    with the ``K``-stack of tangents there:

    >>> W = ubvf.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ux)).reciprocal()
    >>> print(W.stack_shape)                               # C, NOT K + C
    (2,)
    >>> print(np.asarray(v.weighted_norm(W)).shape)        # one norm per stacked tangent
    (3, 2)
    """
    up_weight_supercore:    NDArray             # (d,)+C+(nU,)
    down_weight_supercore:  NDArray             # (d,)+C+(nD,)
    left_weight_supercore:  NDArray             # (d,)+C+(rL,)
    right_weight_supercore: NDArray             # (d,)+C+(rR,)
    masks:                  UT3VariationsMasks  # static rank structure: the four variation edge masks, at C

    # ----------------------------------------------------------------- views
    @ft.cached_property
    def supercores(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:
        """``(up, down, left, right)`` weight supercores."""
        return (self.up_weight_supercore, self.down_weight_supercore,
                self.left_weight_supercore, self.right_weight_supercore)

    @ft.cached_property
    def data(self) -> typ.Tuple:
        """Raw-array view, mirroring the fields: ``(up, down, left, right, (4 rank masks))``. Backend
        ``ufv_*_weights`` / ``utv_weighted_*`` functions take this layout. A **5-tuple** -- one shorter
        than ``UT3Frame.data``, which also carries ``shape``."""
        return self.supercores + (self.masks.data,)

    # ------------------------------------------------- padded (uniform) structure
    @ft.cached_property
    def d(self) -> int:
        return self.up_weight_supercore.shape[0]

    @ft.cached_property
    def nU(self) -> int:
        return self.up_weight_supercore.shape[-1]

    @ft.cached_property
    def nD(self) -> int:
        return self.down_weight_supercore.shape[-1]

    @ft.cached_property
    def rL(self) -> int:
        return self.left_weight_supercore.shape[-1]

    @ft.cached_property
    def rR(self) -> int:
        return self.right_weight_supercore.shape[-1]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        """The **frame** stack ``C`` (``()`` if unstacked) -- never the variations' ``K + C``."""
        return self.up_weight_supercore.shape[1:-1]

    # ------------------------------------------------- original (real) structure
    @ft.cached_property
    def up_ranks(self) -> NDArray:  # dtype=int, (d,)+C
        return self.masks.variations_up_mask.sum(axis=-1)

    @ft.cached_property
    def down_ranks(self) -> NDArray:  # dtype=int, (d,)+C
        return self.masks.variations_down_mask.sum(axis=-1)

    @ft.cached_property
    def left_ranks(self) -> NDArray:  # dtype=int, (d,)+C
        return self.masks.variations_left_mask.sum(axis=-1)

    @ft.cached_property
    def right_ranks(self) -> NDArray:  # dtype=int, (d,)+C
        return self.masks.variations_right_mask.sum(axis=-1)

    # ----------------------------------------------------------------- validation
    def validate(self) -> None:
        """Structural: the four masks are boolean and match their supercores (same ``(d,)+C+(size,)``)."""
        for m, name in zip(self.masks.data, ('variations_up_mask', 'variations_down_mask',
                                             'variations_left_mask', 'variations_right_mask')):
            if not common.is_boolean_ndarray(m):
                raise ValueError('UT3FrameWeights: %s must be a boolean array (got %s).'
                                 % (name, getattr(m, 'dtype', type(m))))
        d, ss = self.d, self.stack_shape
        for sc, m, name in zip(self.supercores, self.masks.data, ('up', 'down', 'left', 'right')):
            if tuple(sc.shape) != (d,) + ss + (sc.shape[-1],):
                raise ValueError('Inconsistent UT3FrameWeights: %s_weight_supercore.shape = %s, expected '
                                 '(d=%d,) + stack_shape=%s + (size,).' % (name, tuple(sc.shape), d, ss))
            if tuple(m.shape) != tuple(sc.shape):
                raise ValueError('Inconsistent UT3FrameWeights: %s mask shape %s != supercore shape %s.'
                                 % (name, tuple(m.shape), tuple(sc.shape)))

    def __post_init__(self):
        self.validate()

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return ('UT3FrameWeights(d=%d, nU=%d, nD=%d, rL=%d, rR=%d%s)'
                % (self.d, self.nU, self.nD, self.rL, self.rR, ss))

    # ----------------------------------------------------------------- operations
    def is_consistent_with(self, tangent) -> bool:
        """True iff this metric can be absorbed into a ``UT3Tangent``'s (or ``UT3Variations``') variations
        (non-raising): padded widths match, this weight's stack ``C`` is the **trailing part** of the
        variation stack ``K + C``, and the masks agree (broadcast constant over ``K``).

        Like the variations themselves, this is **blind to the frame**. Whether the metric belongs to a
        particular tangent's frame needs the frame, and is checked by :py:func:`check_ufw_pair`.
        """
        variations = tangent.variations if hasattr(tangent, 'variations') else tangent
        return ufv_operations.ufv_weights_consistent(variations.data, self.data)

    def reciprocal(self) -> 'UT3FrameWeights':
        """Elementwise ``1/w`` on the real slots of all four families (e.g. the inverse-singular-value /
        Grasedyck-Kramer metric); the padding stays a canonical, **finite** zero rather than becoming
        ``inf``. Masks unchanged. See
        :py:func:`~t3toolbox.backend.ufv_operations.ufv_reciprocal_weights`."""
        return _frame_weights_from_data(ufv_operations.ufv_reciprocal_weights(self.data))

    def sqrt(self) -> 'UT3FrameWeights':
        """Elementwise ``sqrt`` on the real slots of all four families; the padding stays a canonical,
        finite zero (masks unchanged)."""
        return _frame_weights_from_data(ufv_operations.ufv_sqrt_weights(self.data))

    def concatenate(self, other: 'UT3FrameWeights') -> 'UT3FrameWeights':
        """Per-edge concatenation (the ``+`` combine; ranks add). Output masks may go gappy."""
        return _frame_weights_from_data(ufv_operations.ufv_concatenate_weights(self.data, other.data))

    def kronecker(self, other: 'UT3FrameWeights') -> 'UT3FrameWeights':
        """Per-edge Kronecker product (the Hadamard combine; ranks multiply). Output masks are strided."""
        return _frame_weights_from_data(ufv_operations.ufv_kronecker_weights(self.data, other.data))

    # ----------------------------------------------------------------- constructors
    @classmethod
    def from_ut3weights(cls, weights: 'ut3.UT3Weights') -> 'UT3FrameWeights':
        """Build a tangent metric from uniform base-point edge weights (e.g.
        ``UT3Weights.from_ut3svd(x)``): ``up = down = tucker``, ``left = tt[:-1]``, ``right = tt[1:]``, on
        the supercores **and** the masks. The TT slicing follows the ``H_i`` bond convention (``H_i``'s
        left bond is TT bond ``i``, its right bond is bond ``i+1``) -- simple but convention-dependent,
        hence a named method.

        The result pairs with a **minimal-rank** tangent (where the complement rank ``nD`` equals the
        Tucker rank ``nU``, as for ``ut3svd`` output); a non-minimal tangent has ``nD < nU`` and is
        rejected by :py:meth:`is_consistent_with` rather than silently absorbed. The Grasedyck-Kramer
        metric is ``UT3FrameWeights.from_ut3weights(UT3Weights.from_ut3svd(x)).reciprocal()``.
        """
        return _frame_weights_from_data(ufv_operations.ufv_weights_from_ut3_weights(weights.data))

    # ----------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3frameweights(
            weights: 'bvf.T3FrameWeights',
            nU: typ.Optional[int] = None,   # padded up rank    (default max); pass to match the tangent's pad
            nD: typ.Optional[int] = None,   # padded down rank  (default max)
            rL: typ.Optional[int] = None,   # padded left rank  (default max)
            rR: typ.Optional[int] = None,   # padded right rank (default max)
    ) -> 'UT3FrameWeights':
        """Pack a ragged :py:class:`~t3toolbox.frame_variations_format.T3FrameWeights` into a uniform one.

        Pass the padded sizes to match the tangent these weights will pair with (e.g. from
        ``frame.uniform_structure``); the defaults pad tightly to the weights' own max ranks.
        """
        return _frame_weights_from_data(
            ufv_conversions.t3frameweights_to_ut3frameweights(weights.data, nU=nU, nD=nD, rL=rL, rR=rR))

    def allclose(
            self,
            other: 'UT3FrameWeights',

            rtol:  typ.Optional[float] = None,  # None: the ambient jax-aware default (safety.comparison_rtol)
            atol:  typ.Optional[float] = None,  # None: 0.0
    ) -> NDArray:  # bool, shape = stack_shape (C); scalar unstacked; reduce with .all()
        """True where the MASKED weight supercores are numerically equal, per stack element
        (norm-based ``atol + rtol * max`` reference). Different masks raise. Bitwise incl. padding:
        :py:meth:`corewise_equal`; ``==`` is undefined."""
        if not cw.corewise_equal(self.masks.data, other.masks.data):
            raise ValueError('UT3FrameWeights.allclose: different rank masks')
        if rtol is None:
            rtol = safety_mod.comparison_rtol(self.supercores + other.supercores)
        atol = 0.0 if atol is None else atol
        use_jax = common.tree_contains_jax(self.supercores + other.supercores)
        xnp, _, _ = common.get_backend(True, use_jax)
        masked = lambda w: tuple(sc * m for sc, m in zip(w.supercores, w.masks.data))
        sq = lambda fams: sum(xnp.sum(a * a, axis=(0, -1)) for a in fams)   # (d,)+stack+(w,) -> stack
        a, b = masked(self), masked(other)
        dn = sq(tuple(x - y for x, y in zip(a, b))) ** 0.5
        rn = xnp.maximum(sq(a), sq(b)) ** 0.5
        return dn <= atol + rtol * rn

    def corewise_equal(
            self,
            other: 'UT3FrameWeights',
    ) -> bool:
        """Bitwise equality of the stored representation -- masks and raw supercores including
        padding (``False`` on any mismatch, never raises)."""
        return (type(other) is type(self)
                and cw.corewise_equal(self.masks.data, other.masks.data)
                and cw.corewise_equal(self.supercores, other.supercores))

    def to_t3frameweights(self):  # -> T3FrameWeights (unstacked) or a nested tree (shaped like C) of them
        """Convert back to ragged form. Unstacked: one
        :py:class:`~t3toolbox.frame_variations_format.T3FrameWeights`. Stacked: a nested tree of them (a
        varying-rank stack has no single stacked ragged weight -- the :py:meth:`UT3Frame.to_t3frame`
        pattern)."""
        def _wrap(res):
            if common.is_ndarray(res[0][0]):   # res = (up, down, left, right) leaf
                return bvf.T3FrameWeights(*res)
            return tuple(_wrap(w) for w in res)

        return _wrap(ufv_conversions.ut3frameweights_to_t3frameweights(self.data))


def _frame_weights_from_data(data: typ.Tuple) -> 'UT3FrameWeights':
    """Wrap a backend frame-weights ``.data`` tuple ``(up, down, left, right, (4 masks))`` into a
    :py:class:`UT3FrameWeights`."""
    return UT3FrameWeights(*data[:4], UT3VariationsMasks(*data[4]))


def ut3svd_orthogonal_representations(
        x: ut3.UniformTuckerTensorTrain,
        **t3svd_kwargs,                 # passed to UniformTuckerTensorTrain.t3svd (max_*_ranks, sharing, ...)
) -> typ.Tuple[
    UT3Frame,            # orthogonal frame at the t3svd result, in the t3svd GAUGE
    UT3Variations,       # the variations of that representation
    'ut3.UT3Weights',    # the singular values (the result's masks), ready for UT3FrameWeights.from_ut3weights
]:
    '''Uniform twin of :py:func:`~t3toolbox.frame_variations_format.t3svd_orthogonal_representations`:
    the orthogonal frame of ``x`` in the T3-SVD gauge (``already_left_orthogonal=True``, so the Tucker
    basis is the singular basis and the returned singular values weight the right coordinates), with one
    SVD instead of two.'''
    xs, tucker_svals, tt_svals = x.t3svd(**t3svd_kwargs)
    frame, variations = ut3_orthogonal_representations(xs, already_left_orthogonal=True)
    return frame, variations, ut3.UT3Weights(tucker_svals, tt_svals, xs.masks)


def check_ufw_pair(
        frame:   UT3Frame,          # stack_shape = C
        weights: UT3FrameWeights,   # stack_shape = C -- a weight is FRAME-LIKE: one metric per base point
) -> None:
    """Check that ``weights`` is a metric on the tangent coordinates **at this frame**.

    The uniform twin of :py:func:`~t3toolbox.frame_variations_format.check_fw_pair`, and the weight
    analog of :py:func:`check_ufv_pair`. The stack must equal ``frame.stack_shape`` **exactly** (not
    merely be a trailing part of it, as when pairing with variations alone), and the four families must
    match the frame's variation holes -- ``up`` <-> ``nU``, ``down`` <-> ``nD``, ``left`` <-> ``rL``,
    ``right`` <-> ``rR`` -- in both padded width and **rank mask**.

    Two things uniform must check that ragged gets for free:

    - **The exact stack.** Absorption only needs the weight's stack to be the *trailing* part of the
      variation stack (:py:meth:`UT3FrameWeights.is_consistent_with`, blind to the frame). A ``K + C``
      weight satisfies that too -- it reads as ``C_w = K + C`` -- so it would silently weight one frame's
      ``K`` tangents with ``K`` different metrics. Only here are both objects present.
    - **The masks.** Uniform pads every family to a common width, so a mask mismatch is invisible to the
      shapes and would silently zero a real variation slot. The frame's masks are gauge-shifted to the
      variation families exactly as in :py:func:`check_ufv_pair` (``frame_left_mask[:-1]`` /
      ``frame_right_mask[1:]``: the ``d+1``-th left/right cores are base-point padding, not tangent edges).

    Structural (shapes + host-numpy masks) -> raises in both safety modes; jit-safe.
    """
    if weights.stack_shape != frame.stack_shape:
        raise ValueError(
            'Inconsistent (UT3Frame, UT3FrameWeights) pair.\n'
            'A UT3FrameWeights is a metric at a base point, so it carries the FRAME stack C exactly (the\n'
            'variations carry K + C; a K-batch of tangents at one frame shares the one metric).\n'
            + str(weights.stack_shape) + ' = weights.stack_shape != '
            + str(frame.stack_shape) + ' = frame.stack_shape')

    frame_masks = (frame.masks.up_mask, frame.masks.down_mask,
                   frame.masks.frame_left_mask[:-1], frame.masks.frame_right_mask[1:])
    sizes = (frame.nU, frame.nD, frame.rL, frame.rR)
    names = ('up', 'down', 'left', 'right')
    for name, supercore, weight_mask, frame_mask, size in zip(
            names, weights.supercores, weights.masks.data, frame_masks, sizes):
        if supercore.shape[-1] != size:
            raise ValueError(
                'Inconsistent (UT3Frame, UT3FrameWeights) pair.\n%s_weight_supercore has padded width %d, '
                'but the frame\'s %s variation hole is %d wide.'
                % (name, supercore.shape[-1], name, size))
        if not np.array_equal(weight_mask, frame_mask):
            raise ValueError(
                'Inconsistent (UT3Frame, UT3FrameWeights) pair.\nThe %s rank mask differs from the frame\'s '
                '%s variation-hole mask -- absorbing it would silently zero a real variation slot.'
                % (name, name))


def ufv_absorb_weights(variations: UT3Variations, weights: UT3FrameWeights) -> UT3Variations:
    """Absorb the metric ``weights`` into the variation supercores (``down``->V, ``up``/``left``/``right``
    ->H), returning the weighted :py:class:`UT3Variations` (the frame is unchanged, and the masks are
    preserved -- absorb is rank-preserving).

    The ``C``-stacked metric broadcasts over the variations' ``K`` for free. Uniform twin of
    :py:func:`t3toolbox.frame_variations_format.fv_absorb_weights`; see
    :py:func:`~t3toolbox.backend.ufv_operations.ufv_absorb_weights`."""
    if not weights.is_consistent_with(variations):
        raise ValueError(
            'Inconsistent (UT3Variations, UT3FrameWeights) pair in absorb_weights.\n'
            'The metric must fit the variation holes and declare the SAME rank masks (broadcast over K).\n'
            'variations: stack_shape=%s ; weights: stack_shape=%s'
            % (variations.stack_shape, weights.stack_shape))
    return UT3Variations(*ufv_operations.ufv_absorb_weights(variations.data, weights.data)[:2],
                         variations.shape, variations.masks)


if common.jax_available:
    # UT3Frame as a jax pytree: the four supercores are the (traced) children; the static aux_data is
    # (shape, UT3FrameMasks). `shape` is a value-hashable int tuple (same shape -> same jit cache key);
    # UT3FrameMasks hashes/compares by VALUE over its mask content (common.ValueHashedMasks), so a rebuilt but
    # identical holder is the same jit cache key -- the load-bearing perf contract (test_mask_rebuild_does_not_recompile).
    # Mirrors UniformTuckerTensorTrain. See docs/contributor/uniform_pytree_composition.md.
    import jax
    jax.tree_util.register_pytree_node(
        UT3Frame,
        lambda x: ((x.up_tucker_supercore, x.down_tt_supercore,
                    x.left_tt_supercore, x.right_tt_supercore),
                   (x.shape, x.masks)),
        lambda aux, children: UT3Frame(children[0], children[1], children[2], children[3],
                                       aux[0], aux[1]),
    )

    # UT3Variations: the two variation supercores are the (traced) children; (shape, UT3VariationsMasks)
    # is the value-keyed static aux (same pattern as UT3Frame / the plain UT3).
    jax.tree_util.register_pytree_node(
        UT3Variations,
        lambda x: ((x.tucker_variations, x.tt_variations), (x.shape, x.masks)),
        lambda aux, children: UT3Variations(children[0], children[1], aux[0], aux[1]),
    )
    # UT3FrameWeights: the four weight supercores are traced children (float PARAMETERS), the mask holder
    # is value-hashed static aux (boolean STRUCTURE). No `shape` -- weights have no physical legs.
    jax.tree_util.register_pytree_node(
        UT3FrameWeights,
        lambda w: (w.supercores, w.masks),
        lambda aux, children: UT3FrameWeights(*children, aux),
    )
