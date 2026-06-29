# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.backend.bv_conversions
import t3toolbox.backend.ubv_conversions as ubv_conversions
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.corewise as cw
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ubv_operations as ubv_operations
import t3toolbox.backend.ubv_masking as masking
import t3toolbox.backend.common as common
from t3toolbox.backend.common import *


__all__ = [
    'UT3Basis',
    'UT3Variations',
    'ut3_orthogonal_representations',
]


@dataclass(frozen=True, eq=False)  # eq=False -> the mixin's VALUE-based __hash__/__eq__ stand
class UT3BasisMasks(common.ValueHashedMasks):
    """The static rank structure of a :py:class:`UT3Basis`: its four boolean edge masks.

    Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form). Held as a
    separate object so it can ride as jax ``aux_data``; hash/eq are **value-based** (the
    :py:class:`~t3toolbox.backend.common.ValueHashedMasks` mixin) so a rebuilt-but-identical frame is the
    *same* jit cache key (no per-iteration recompile when the orthogonal frame is rebuilt in an
    optimization loop). The plain-layer :py:class:`~t3toolbox.uniform_tucker_tensor_train.UT3Masks`
    pattern; see ``docs/uniform_pytree_composition.md``. (The physical ``shape`` is a separate static int
    tuple on :py:class:`UT3Basis` -- not a mask, and value-hashable.)
    """
    up_mask:          NDArray  # dtype=bool, (d,)  +stack_shape+(nU,)
    down_mask:        NDArray  # dtype=bool, (d,)  +stack_shape+(nD,)
    basis_left_mask:  NDArray  # dtype=bool, (d+1,)+stack_shape+(rL,)
    basis_right_mask: NDArray  # dtype=bool, (d+1,)+stack_shape+(rR,)

    @property
    def data(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:
        """The four raw rank-mask arrays, ``(up_mask, down_mask, basis_left_mask, basis_right_mask)``."""
        return self.up_mask, self.down_mask, self.basis_left_mask, self.basis_right_mask


@dataclass(frozen=True)
class UT3Basis:
    """Basis (orthogonal frame) for the basis-variations representation of uniform Tucker tensor trains.

    Uniform analog of :py:class:`~t3toolbox.basis_variations_format.T3Basis`: four padded supercores
    (``up_tucker``, ``down_tt``, ``left_tt``, ``right_tt``) + the static physical ``shape`` (an int tuple,
    shared across the stack) + a :py:class:`UT3BasisMasks` holder (the four per-stack rank masks).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_basis_variations_format as ubcf
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
    >>> masks = ubcf.UT3BasisMasks(up_mask, down_mask, left_mask, right_mask)
    >>> B = ubcf.UT3Basis(up, down, left, right, shape, masks)
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
    masks:                  UT3BasisMasks        # static rank structure (the four edge masks)

    @ft.cached_property
    def data(self) -> typ.Tuple[
        NDArray,                                   # up_tucker_supercore
        NDArray,                                   # down_tt_supercore
        NDArray,                                   # left_tt_supercore
        NDArray,                                   # right_tt_supercore
        typ.Tuple[int, ...],                       # shape
        typ.Tuple[NDArray, NDArray, NDArray, NDArray],  # (up_mask, down_mask, basis_left_mask, basis_right_mask)
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
        return self.masks.basis_left_mask.sum(axis=-1)

    @ft.cached_property
    def right_ranks(self) -> NDArray:
        return self.masks.basis_right_mask.sum(axis=-1)

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

    def apply_masks(self) -> 'UT3Basis':
        """Apply masks to the basis supercores, zeroing out unmasked entries.
        """
        up_sc, down_sc, left_sc, right_sc = masking.apply_basis_masks(self.data)
        return UT3Basis(
            up_sc, down_sc, left_sc, right_sc,
            self.shape, self.masks,
        )

    # ------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3basis(
            basis: bvf.T3Basis,
            N:  typ.Optional[int] = None,   # padded mode dim   (default max(shape))
            nU: typ.Optional[int] = None,   # padded up rank    (default max(up_ranks))
            nD: typ.Optional[int] = None,   # padded down rank  (default max(down_ranks))
            rL: typ.Optional[int] = None,   # padded left rank  (default max(left_ranks))
            rR: typ.Optional[int] = None,   # padded right rank (default max(right_ranks))
    ) -> 'UT3Basis':
        """Pack a ragged :py:class:`~t3toolbox.basis_variations_format.T3Basis` into a uniform frame.

        A single ragged frame has ranks shared across its ``C`` stack, so the masks come out **uniform
        across the stack** (varying-rank uniform batches arise only by ``stack``-ing a heterogeneous tree).
        """
        uc, dc, lc, rc, shape, masks = ubv_conversions.t3basis_to_ut3basis(
            basis.data, N=N, nU=nU, nD=nD, rL=rL, rR=rR)
        return UT3Basis(uc, dc, lc, rc, shape, UT3BasisMasks(*masks))

    def to_t3basis(self):  # -> bvf.T3Basis, or a nested tree (shaped like stack_shape) of them if stacked
        """Convert to a ragged :py:class:`~t3toolbox.basis_variations_format.T3Basis` (or, if stacked, an
        array-like tree of them)."""
        d = self.d
        result = ubv_conversions.ut3basis_to_t3basis(self.apply_masks().data)
        return stacking.apply_func_to_leaf_subtrees(
            result,
            lambda c: bvf.T3Basis(*c),
            ((None,) * d,) * 4,  # leaf_structure: 4 core-families, each a length-d tuple
        )

    # ------------------------------------------------------------- base point / orthogonal frame
    @staticmethod
    def from_ut3(x: ut3.UniformTuckerTensorTrain) -> 'UT3Basis':
        """Orthogonal frame at the point ``x`` (the frame part of :py:func:`ut3_orthogonal_representations`).
        Uniform analog of :py:meth:`~t3toolbox.basis_variations_format.T3Basis.from_t3`."""
        return ut3_orthogonal_representations(x)[0]

    def to_ut3(self) -> ut3.UniformTuckerTensorTrain:
        """The base point this frame represents, as a :py:class:`UniformTuckerTensorTrain` (right-canonical:
        the Tucker supercore over the right-orthogonal TT supercore). Uniform analog of ``T3Basis.to_t3``;
        the plain-UT3 tt edge mask is the frame's ``basis_right_mask`` (the right TT ranks)."""
        return ut3.UniformTuckerTensorTrain(
            self.up_tucker_supercore, self.right_tt_supercore,
            self.shape, ut3.UT3Masks(self.masks.up_mask, self.masks.basis_right_mask))

    def to_dense(self) -> NDArray:
        """Dense tensor of the base point this frame represents (``= to_ut3().to_dense()``)."""
        return self.to_ut3().to_dense()

    def reverse(self) -> 'UT3Basis':
        """Reverse the mode order. Left/right supercores (and masks) **swap roles** -- reversing a
        left-orthogonal chain yields a right-orthogonal one -- so the redundant L/R store makes this exact
        with no re-orthogonalization. Commutes with conversion: ``B.reverse().to_t3basis() ==
        B.to_t3basis().reverse()``."""
        up, down, left, right, shape, masks = ubv_operations.ubv_reverse_basis(self.data)
        return UT3Basis(up, down, left, right, shape, UT3BasisMasks(*masks))

    def orthogonalize(self) -> 'UT3Basis':
        """Orthogonal representation of the base point this frame reconstructs to (``= from_ut3(to_ut3())``).
        For an already-orthogonal frame, an equivalent orthogonal frame; for a drifted one, a genuinely
        orthogonal (minimal-rank) frame for the right-canonical base point."""
        return UT3Basis.from_ut3(self.to_ut3())

    @staticmethod
    def random_orthogonal(
            shape:        typ.Sequence[int],   # (N0,...,N(d-1))
            tucker_ranks,                       # int | len-d seq | (d,)+stack array (the variety)
            tt_ranks,                           # int | len-(d+1) seq | (d+1,)+stack array
            stack_shape:  typ.Tuple[int, ...] = (),
            use_jax:      bool = False,
    ) -> 'UT3Basis':
        """Orthogonal representation of a *random* uniform T3 -- a genuine random base point (orthogonal,
        consistent), not iid-random supercores. Equals ``from_ut3(UniformTuckerTensorTrain.randn(...))``."""
        x = ut3.UniformTuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks,
                                               stack_shape=tuple(stack_shape), use_jax=use_jax)
        return UT3Basis.from_ut3(x)

    @staticmethod
    def random_orthogonal_like(basis: 'UT3Basis') -> 'UT3Basis':
        """A random orthogonal frame with the same shape / ranks / stack as ``basis``."""
        return UT3Basis.random_orthogonal(basis.shape, basis.up_ranks, basis.left_ranks,
                                          stack_shape=basis.stack_shape, use_jax=basis.contains_jax)

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

    def to_jax(self) -> 'UT3Basis':
        # Supercores -> jax; masks stay host numpy (a jax mask is a tracer under jit). See docs/uniform_*.
        return UT3Basis(*(to_jax(sc) for sc in self.supercores), self.shape, self.masks)

    def to_numpy(self) -> 'UT3Basis':
        # Supercores -> numpy; the masks are already host numpy, so reuse the holder.
        return UT3Basis(*(to_numpy(sc) for sc in self.supercores), self.shape, self.masks)

    def copy(self) -> 'UT3Basis':
        # Deep-copy the supercores (the data leaves), like ragged T3Basis.copy; the static aux (shape +
        # masks) is shared (immutable structure, the same way `shape` is not duplicated).
        return UT3Basis(*(sc.copy() for sc in self.supercores), self.shape, self.masks)

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return ('UT3Basis(shape=%s, N=%d, nU=%d, nD=%d, rL=%d, rR=%d%s)'
                % (self.shape, self.N, self.nU, self.nD, self.rL, self.rR, ss))

    def validate(self) -> None:
        '''Check rank and shape consistency of a uniform Tucker tensor train basis (`UT3Basis`).

        Raises
        ------
        ValueError
            If the supercores / masks have inconsistent shapes, or `shape` is not a length-d tuple of
            mode dims within the padded N.
        '''
        up_mask, down_mask, basis_left_mask, basis_right_mask = self.masks.data

        UU_good = self.up_tucker_supercore.shape  == (self.d,) + self.stack_shape + (self.nU, self.N)
        DD_good = self.down_tt_supercore.shape    == (self.d,) + self.stack_shape + (self.rL, self.nD, self.rR)
        LL_good = self.left_tt_supercore.shape    == (self.d,) + self.stack_shape + (self.rL, self.nU, self.rL)
        RR_good = self.right_tt_supercore.shape   == (self.d,) + self.stack_shape + (self.rR, self.nU, self.rR)

        UM_good = up_mask.shape          == (self.d,) + self.stack_shape + (self.nU,)
        DM_good = down_mask.shape        == (self.d,) + self.stack_shape + (self.nD,)
        LM_good = basis_left_mask.shape  == (self.d + 1,) + self.stack_shape + (self.rL,)
        RM_good = basis_right_mask.shape == (self.d + 1,) + self.stack_shape + (self.rR,)

        SH_good = (len(self.shape) == self.d) and all(0 <= Ni <= self.N for Ni in self.shape)

        bad_str = lambda x: ' <-- Bad' if not x else ''

        shapes_string = ''
        shapes_string += 'up_tucker_supercore.shape = ' + str(self.up_tucker_supercore.shape)   + ' =? (d,) + stack_shape + (nU, N)' + bad_str(UU_good) + '\n'
        shapes_string += 'down_tt_supercore.shape   = ' + str(self.down_tt_supercore.shape)     + ' =? (d,) + stack_shape + (rL, nD, rR)' + bad_str(DD_good) + '\n'
        shapes_string += 'left_tt_supercore.shape   = ' + str(self.left_tt_supercore.shape)     + ' =? (d,) + stack_shape + (rL, nU, rL)' + bad_str(LL_good) + '\n'
        shapes_string += 'right_tt_supercore.shape  = ' + str(self.right_tt_supercore.shape)    + ' =? (d,) + stack_shape + (rR, nU, rR)' + bad_str(RR_good) + '\n'
        shapes_string += 'up_mask.shape             = ' + str(up_mask.shape) + ' =? (d,) + stack_shape + (nU,)' + bad_str(UM_good) + '\n'
        shapes_string += 'down_mask.shape           = ' + str(down_mask.shape) + ' =? (d,) + stack_shape + (nD,)' + bad_str(DM_good) + '\n'
        shapes_string += 'basis_left_mask.shape     = ' + str(basis_left_mask.shape) + ' =? (d+1,) + stack_shape + (rL,)' + bad_str(LM_good) + '\n'
        shapes_string += 'basis_right_mask.shape    = ' + str(basis_right_mask.shape) + ' =? (d+1,) + stack_shape + (rR,)' + bad_str(RM_good) + '\n'
        shapes_string += 'shape                     = ' + str(self.shape) + ' =? length-d ints in [0, N]' + bad_str(SH_good)

        if not (UU_good and DD_good and LL_good and RR_good and UM_good and DM_good and LM_good and RM_good and SH_good):
            raise ValueError(
                'Inconsistent shapes for UT3Basis.\n'
                + shapes_string
            )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstack a stacked UT3Basis into an array-like tree (shaped like ``stack_shape``) of UT3Basis."""
        return stacking.apply_func_to_leaf_subtrees(
            ubv_operations.ubv_unstack(self.data, 4),
            lambda leaf: UT3Basis(leaf[0], leaf[1], leaf[2], leaf[3], leaf[4], UT3BasisMasks(*leaf[5])),
            ubv_operations.ubv_leaf_structure(self.d, 4),
        )

    @staticmethod
    def stack(xx):  # Array-like tree of UT3Basis
        """Stack an array-like tree of UT3Basis into a single stacked UT3Basis."""
        data_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda b: b.data, None)
        up, down, left, right, shape, masks = ubv_operations.ubv_stack(data_tree, 4)
        return UT3Basis(up, down, left, right, shape, UT3BasisMasks(*masks))


@dataclass(frozen=True, eq=False)  # eq=False -> the mixin's VALUE-based __hash__/__eq__ stand
class UT3VariationsMasks(common.ValueHashedMasks):
    """The static rank structure of a :py:class:`UT3Variations`: its four boolean edge masks.

    Value-hashed (the :py:class:`~t3toolbox.backend.common.ValueHashedMasks` mixin) so a rebuilt-but-
    identical holder is the same jit cache key. NOTE: the left/right masks are ``(d,)`` here (not
    ``(d+1,)`` as on :py:class:`UT3BasisMasks`) -- a variation occupies one TT slot, not a boundary edge.
    """
    variations_up_mask:    NDArray  # dtype=bool, (d,)+stack_shape+(nU,)
    variations_down_mask:  NDArray  # dtype=bool, (d,)+stack_shape+(nD,)
    variations_left_mask:  NDArray  # dtype=bool, (d,)+stack_shape+(rL,)
    variations_right_mask: NDArray  # dtype=bool, (d,)+stack_shape+(rR,)

    @property
    def data(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray]:
        """``(variations_up_mask, variations_down_mask, variations_left_mask, variations_right_mask)``."""
        return (self.variations_up_mask, self.variations_down_mask,
                self.variations_left_mask, self.variations_right_mask)


@dataclass(frozen=True)
class UT3Variations:
    """Variation cores for the basis-variations representation of uniform Tucker tensor trains.

    Uniform analog of :py:class:`~t3toolbox.basis_variations_format.T3Variations`: two padded supercores
    (``tucker_variations``, ``tt_variations``) + the static physical ``shape`` (an int tuple, shared across
    the stack) + a :py:class:`UT3VariationsMasks` holder. The variations fit in the "holes" of a
    :py:class:`UT3Basis`.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_basis_variations_format as ubcf
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

    def apply_masks(self) -> 'UT3Variations':
        """Apply masks to the variation supercores, zeroing out unmasked entries."""
        masked_tk_supercore, masked_tt_supercore = masking.apply_variations_masks(self.data)
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
        """Pack a ragged :py:class:`~t3toolbox.basis_variations_format.T3Variations` into uniform variations."""
        tkv, ttv, shape, masks = ubv_conversions.t3variations_to_ut3variations(
            variations.data, N=N, nU=nU, nD=nD, rL=rL, rR=rR)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def to_t3variations(self):  # -> bvf.T3Variations, or a nested tree (shaped like stack_shape) of them if stacked
        """Convert to ragged :py:class:`~t3toolbox.basis_variations_format.T3Variations` (or, if stacked, an
        array-like tree of them)."""
        d = self.d
        result = ubv_conversions.ut3variations_to_t3variations(self.apply_masks().data)
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
            ubv_operations.ubv_unstack(self.data, 2),
            lambda leaf: UT3Variations(leaf[0], leaf[1], leaf[2], UT3VariationsMasks(*leaf[3])),
            ubv_operations.ubv_leaf_structure(self.d, 2),
        )

    @staticmethod
    def stack(xx):  # array-like tree of UT3Variations
        """Stack an array-like tree of UT3Variations into a single stacked UT3Variations."""
        data_tree = stacking.apply_func_to_leaf_subtrees(xx, lambda v: v.data, None)
        tkv, ttv, shape, masks = ubv_operations.ubv_stack(data_tree, 2)
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
        """Corewise sum (variations form a vector space at a fixed base; the mask is unchanged)."""
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
        :py:meth:`UT3Basis.reverse` so a tangent reverses by reversing both components."""
        tkv, ttv, shape, masks = ubv_operations.ubv_reverse_variations(self.data)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def sum_stack(self, axis=None) -> 'UT3Variations':
        """Corewise sum over stack axes (a batch of variations -> their sum; the tangent sum, by linearity).
        ``axis`` indexes the stack (default: the whole stack). The mask ORs over the summed axes -- a no-op
        for a same-mask (single-base) stack; see ``docs/uniform_masks_vs_ranks.md``."""
        tkv, ttv, shape, masks = ubv_operations.ubv_variations_sum_stack(self.data, axis)
        return UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*masks))

    def allclose(self, other: 'UT3Variations', rtol: float = 1e-9, atol: float = 0.0) -> bool:
        """``True`` if ``other`` holds the same variations as ``self`` on the **real (masked)** content:
        ``||self - other|| <= atol + rtol * ||other||`` in the corewise norm. (A single bool for now; the
        per-stack-element checker semantics land in slice 2c-G.)"""
        dn = cw.corewise_norm((self - other).apply_masks().supercores)
        rn = cw.corewise_norm(other.apply_masks().supercores)
        return bool(dn <= atol + rtol * rn)

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
        a :py:class:`UT3Basis` / :py:class:`UT3Variations`.)"""
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
        # x: UT3Basis (gauge-shift its frame masks -> left[:-1], right[1:]) or UT3Variations (its own masks).
        if isinstance(x, UT3Basis):
            bm = x.masks
            return UT3VariationsMasks(bm.up_mask, bm.down_mask, bm.basis_left_mask[:-1], bm.basis_right_mask[1:])
        return x.masks

    @staticmethod
    def zeros_like(x) -> 'UT3Variations':
        """Zero variations matching the structure of ``x`` (a :py:class:`UT3Basis` or :py:class:`UT3Variations`).
        For a basis this is the zero tangent carrying the base's gauge masks."""
        return UT3Variations.zeros(x.uniform_variation_shapes, x.shape, stack_shape=x.stack_shape,
                                   masks=UT3Variations._variation_masks_of(x), use_jax=x.contains_jax)

    @staticmethod
    def randn_like(x) -> 'UT3Variations':
        """Random variations matching the structure (incl. gauge masks) of ``x`` (a UT3Basis or UT3Variations)."""
        return UT3Variations.randn(x.uniform_variation_shapes, x.shape, stack_shape=x.stack_shape,
                                   masks=UT3Variations._variation_masks_of(x), use_jax=x.contains_jax)


def check_ubv_pair(base: UT3Basis, variations: UT3Variations) -> None:
    """Check rank and shape consistency between UT3Basis and UT3Variations.

    This ensures that the variation cores (V, H) have the correct dimensions
     to interface with the base cores (U, L, R, O).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_basis_variations_format as ubcf
    >>> stack_shape = ()                                    # unstacked, for a small readable example
    >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
    >>> uc = np.random.randn(*((d,) + stack_shape + (nU, N)))
    >>> dc = np.random.randn(*((d,) + stack_shape + (rL, nD, rR)))
    >>> lc = np.random.randn(*((d,) + stack_shape + (rL, nU, rL)))
    >>> rc = np.random.randn(*((d,) + stack_shape + (rR, nU, rR)))
    >>> shape = (10, 11, 12)
    >>> up = np.arange(nU) < np.array([[2],[3],[4]])        # (d, nU)
    >>> dn = np.arange(nD) < np.array([[3],[4],[5]])        # (d, nD)
    >>> bl = np.arange(rL) < np.array([[1],[2],[3],[1]])    # (d+1, rL) basis left
    >>> br = np.arange(rR) < np.array([[1],[2],[2],[1]])    # (d+1, rR) basis right
    >>> B = ubcf.UT3Basis(uc, dc, lc, rc, shape, ubcf.UT3BasisMasks(up, dn, bl, br))
    >>> tkv = np.random.randn(*((d,) + stack_shape + (nD, N)))
    >>> ttv = np.random.randn(*((d,) + stack_shape + (rL, nU, rR)))
    >>> V = ubcf.UT3Variations(tkv, ttv, shape,
    ...                        ubcf.UT3VariationsMasks(up, dn, bl[:-1], br[1:]))
    >>> ubcf.check_ubv_pair(B, V)   # consistent base/variations pair -> no error
    """
    if base.uniform_structure != variations.uniform_structure:
        raise ValueError(
            'Inconsistent (UT3Basis, UT3Variations) pair: structures differ.\n'
            + str(base.uniform_structure) + ' (base) != ' + str(variations.uniform_structure) + ' (variations)')

    if base.shape != variations.shape:
        raise ValueError('Inconsistent (UT3Basis, UT3Variations) pair: shapes differ (%s vs %s).'
                         % (base.shape, variations.shape))

    bm, vm = base.masks, variations.masks
    for a, b, name in (
            (bm.up_mask,             vm.variations_up_mask,    'up'),
            (bm.down_mask,           vm.variations_down_mask,  'down'),
            (bm.basis_left_mask[:-1], vm.variations_left_mask,  'left'),
            (bm.basis_right_mask[1:], vm.variations_right_mask, 'right'),
    ):
        if not np.array_equal(a, b):
            raise ValueError(
                'Inconsistent (UT3Basis, UT3Variations) pair: %s rank masks differ.' % name)


def ut3_orthogonal_representations(
        x: ut3.UniformTuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash: bool = True,
) -> typ.Tuple[
    UT3Basis,  # orthogonal base
    UT3Variations,  # variations
]:
    '''Construct base-variation representations of UniformTuckerTensorTrain with orthogonal base.

    Input TuckerTensorTrain::

                  1 -- G0 -- G1 -- G2 -- G3 -- 1
        X    =         |     |     |     |
                       B0    B1    B2    B3
                       |     |     |     |

    Base-variation representation with non-orthogonal TT-backend H1::

                  1 -- L0 -- H1 -- R2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    U2    U3
                       |     |     |     |

    Base-variation representation with non-orthogonal tucker backend V2::

                  1 -- L0 -- L1 -- O2 -- R3 -- 1
        X    =         |     |     |     |
                       U0    U1    V2    U3
                       |     |     |     |

    The input tensor train x is defined by:
        - x_tucker_cores     = (B0, B1, B2, B3)
        - x_tt_cores        = (G0, G1, G2, G3)
    The "base cores" are:
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
        Orthogonal base for base-variation representations of x.
    T3Variation
        Variation for base-variation representaions of x.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_basis_variations_format as ubvf
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
    >>> base, variations = ubvf.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
    >>> type(base).__name__, type(variations).__name__
    ('UT3Basis', 'UT3Variations')
    >>> base.shape
    (4, 5, 6)
    >>> # the orthogonal frame still represents the original tensor:
    >>> bool(np.allclose(base.to_t3basis().to_dense(), x.to_dense()))
    True

    '''
    # Thin wrapper: the backend twin carries the logic (orthogonalize + build the SVD-justified prefix
    # masks); this only wraps the raw frame/variation .data into the OO classes.
    frame_data, variation_data = ubv_conversions.ut3_orthogonal_representations(
        x.data, already_left_orthogonal=already_left_orthogonal, squash=squash)
    uc, dc, lc, rc, shape, basis_masks = frame_data
    tkv, ttv, _, variation_masks = variation_data
    return (UT3Basis(uc, dc, lc, rc, shape, UT3BasisMasks(*basis_masks)),
            UT3Variations(tkv, ttv, shape, UT3VariationsMasks(*variation_masks)))


# (`ubv_to_ut3` -- the uniform analog of `bv_to_t3`, substituting one variation core into the frame -- was
# dropped: the left/right subchains become differently-shaped supercores glued by the variation, with no
# clean single uniform supercore op. Low importance. See dev/uniform_fix_plan.md "Refinements (round 2)".)



if common.has_jax:
    # UT3Basis as a jax pytree: the four supercores are the (traced) children; the static aux_data is
    # (shape, UT3BasisMasks). `shape` is a value-hashable int tuple (same shape -> same jit cache key);
    # UT3BasisMasks is eq=False (identity hash/eq), valid hashable aux even though it holds bool arrays.
    # Mirrors UniformTuckerTensorTrain. See docs/uniform_pytree_composition.md.
    import jax
    jax.tree_util.register_pytree_node(
        UT3Basis,
        lambda x: ((x.up_tucker_supercore, x.down_tt_supercore,
                    x.left_tt_supercore, x.right_tt_supercore),
                   (x.shape, x.masks)),
        lambda aux, children: UT3Basis(children[0], children[1], children[2], children[3],
                                       aux[0], aux[1]),
    )

    # UT3Variations: the two variation supercores are the (traced) children; (shape, UT3VariationsMasks)
    # is the value-keyed static aux (same pattern as UT3Basis / the plain UT3).
    jax.tree_util.register_pytree_node(
        UT3Variations,
        lambda x: ((x.tucker_variations, x.tt_variations), (x.shape, x.masks)),
        lambda aux, children: UT3Variations(children[0], children[1], aux[0], aux[1]),
    )
