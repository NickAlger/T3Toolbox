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
import t3toolbox.backend.orthogonal_representations as orth_reps
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ubv_masking as masking
import t3toolbox.backend.common as common
from t3toolbox.backend.common import *


__all__ = [
    'UT3Basis',
    'UT3Variations',
    'ubv_to_ut3',
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
        """Unstack a stacked UT3Basis into an array-like tree of unstacked UT3Basis.

        DEFERRED -- uniform-fix slice 3a increment 2. The int-tuple ``shape`` is a ``Sequence`` the
        tree-walker recurses into, so this needs the plain-layer dynamic-leaf-template treatment
        (``ut3_operations.ut3_leaf_structure`` + a first-leaf drill), not the stale ``basic_uniform_*``.
        """
        raise NotImplementedError(
            'UT3Basis.unstack: rebuild pending (uniform-fix slice 3a, increment 2).')

    @staticmethod
    def stack(xx):  # Array-like tree of UT3Basis
        """Stack an array-like tree of UT3Basis into a single UT3Basis.

        DEFERRED -- uniform-fix slice 3a increment 2 (see :py:meth:`unstack`).
        """
        raise NotImplementedError(
            'UT3Basis.stack: rebuild pending (uniform-fix slice 3a, increment 2).')


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
        """Unstack a stacked UT3Variations. DEFERRED -- uniform-fix slice 3a increment 2 (needs the
        plain-layer dynamic-leaf-template, like :py:meth:`UT3Basis.unstack`)."""
        raise NotImplementedError(
            'UT3Variations.unstack: rebuild pending (uniform-fix slice 3a, increment 2).')

    @staticmethod
    def stack(xx):  # array-like tree of UT3Variations
        """Stack an array-like tree of UT3Variations. DEFERRED -- uniform-fix slice 3a increment 2."""
        raise NotImplementedError(
            'UT3Variations.stack: rebuild pending (uniform-fix slice 3a, increment 2).')


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


def ut3basis_to_t3basis(
        x: UT3Basis,
):  # -> bvf.T3Basis, or a nested tree (shaped like stack_shape) of them if stacked
    """Convert a UT3Basis to a ragged :py:class:`~t3toolbox.basis_variations_format.T3Basis` (or, if
    stacked, an array-like tree of them)."""
    d = x.d
    result = ubv_conversions.ut3basis_to_t3basis(x.apply_masks().data)
    return stacking.apply_func_to_leaf_subtrees(
        result,
        lambda c: bvf.T3Basis(*c),
        ((None,) * d,) * 4,  # leaf_structure: 4 core-families, each a length-d tuple
    )


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
    >>> base, variations = ubvf.ut3_orthogonal_representations(ut3.t3_to_ut3(x))
    >>> type(base).__name__, type(variations).__name__
    ('UT3Basis', 'UT3Variations')
    >>> base.shape
    (4, 5, 6)
    >>> # the orthogonal frame still represents the original tensor:
    >>> bool(np.allclose(ubvf.ut3basis_to_t3basis(base).to_dense(), x.to_dense()))
    True

    '''
    x = x.apply_masks()
    utk, utt = x.data[:2]  # plain UT3 .data = (tk_sc, tt_sc, shape, (tkm, ttm)); only the supercores needed

    # orth_reps.orthogonal_representations is polymorphic (it accepts uniform supercores) and infers
    # numpy/jax from its inputs -- no use_jax. The frame masks are built from the (host int) ranks.
    (uc, dc, lc, rc), (tkv, ttv) = orth_reps.orthogonal_representations(
        (utk, utt), already_left_orthogonal=already_left_orthogonal, squash=squash,
    )

    up_ranks, down_ranks, left_ranks, right_ranks = ranks.compute_orthogonal_representation_ranks(
        x.shape, x.tucker_ranks, x.tt_ranks,
    )

    nU, nD, rL, rR = uc.shape[-2], dc.shape[-2], lc.shape[-1], rc.shape[-1]
    um, dm, lm, rm = masking.make_basis_masks(up_ranks, down_ranks, left_ranks, right_ranks, nU, nD, rL, rR)

    return (UT3Basis(uc, dc, lc, rc, x.shape, UT3BasisMasks(um, dm, lm, rm)),
            UT3Variations(tkv, ttv, x.shape, UT3VariationsMasks(um, dm, lm[:-1], rm[1:])))


if False:
    def ubv_to_ut3(
            ii: int, # index of variation
            use_tt_variation: bool, # If True, use TT variation. If False, use Tucker variation
            basis: UT3Basis,
            variations: UT3Variations,
    ) -> ut3.UniformTuckerTensorTrain:
        '''Convert basis-variations representation to TuckerTensorTrain.

        If replacement_ind=1, replace_tt=True::

            1 -- L0 --(H1)-- R2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    U2    U3
                 |     |     |     |

        If replacement_ind=2, replace_tt=False::

            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1   (V2)   U3
                 |     |     |     |

        Parameters
        ----------
        ii: int
            Index of variation. 0 <= replacement_ind < num_cores
        replace_tt: bool
            Indicates whether to use TT variation (True) or a Tucker variation (False)
        base: T3Basis
            Basis cores
        variations: T3Variations
            Variation cores

        Raises
        ------
        RuntimeError
            - Error raised if the basis and variations do not fit with each other

        Examples
        --------
    import t3toolbox.backend.basis_variations_format.bv_conversions    >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bcf
        >>> randn = np.random.randn # shorthand
        >>> (U0,U1,U2) = (randn(10, 14), randn(11, 15), randn(12, 16))
        >>> (L0,L1,L2) = (randn(1, 10, 2), randn(2, 11, 3), randn(3,12,4))
        >>> (R0,R1,R2) = (randn(2,10,4), randn(4, 11, 5), randn(5, 12, 1))
        >>> (O0,O1,O2) = (randn(1, 9, 4), randn(2, 8, 5), randn(3, 7, 1))
        >>> base = bcf.T3Basis((U0,U1,U2), (L0,L1,L2), (R0,R1,R2), (O0,O1,O2))
        >>> (V0,V1,V2) = (randn(9,14), randn(8,15), randn(7,16))
        >>> (H0,H1,H2) = (randn(1,10,4), randn(2,11,5), randn(3,12,1))
        >>> variations = bcf.T3Variations((V0,V1,V2), (H0,H1,H2))
        >>> ((B0, B1, B2), (G0, G1, G2)) = t3toolbox.backend.basis_variations_format.bv_conversions.bv_to_t3(1, True, base, variations).data # replace index-1 TT-backend
        >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,U1,U2), (L0,H1,R2)))
        True
    import t3toolbox.backend.basis_variations_format.bv_conversions    >>> ((B0, B1, B2), (G0, G1, G2)) = t3toolbox.backend.basis_variations_format.bv_conversions.bv_to_t3(1, False, base, variations).data # replace index-1 tucker backend
        >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,V1,U2), (L0,O1,R2)))
        True
        '''
        check_basis_variations_pair(basis, variations)
        return t3.TuckerTensorTrain(*t3toolbox.backend.basis_variations_format.bv_conversions.bv_to_t3(ii, use_tt_variation, basis.data, variations.data))





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
