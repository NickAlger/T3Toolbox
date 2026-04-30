# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.backend.basis_variations_format.bv_conversions
import t3toolbox.backend.uniform_basis_variations_format.ubv_conversions as ubv_conversions
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.basis_variations_format as bvf
import t3toolbox.backend.orthogonal_representations as orth_reps
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.uniform_basis_variations_format.ubv_masking as masking
from t3toolbox.backend.common import *


__all__ = [
    'UT3Basis',
    'UT3Variations',
    'ubv_to_ut3',
    'ut3_orthogonal_representations',
]


@dataclass(frozen=True)
class UT3Basis:
    """Basis for basis-variations representation of uniform Tucker tensor trains

    Uniform version of T3Basis

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_basis_variations_format as ubcf
    >>> stack_shape = (2,)
    >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
    >>> up_cores = np.random.randn(*((d,) + stack_shape + (nU, N)))
    >>> down_cores = np.random.randn(*((d,) + stack_shape + (rL, nD, rR)))
    >>> left_cores = np.random.randn(*((d,) + stack_shape + (rL, nU, rL)))
    >>> right_cores = np.random.randn(*((d,) + stack_shape + (rR, nU, rR)))
    >>> shape_mask = np.random.choice([True, False], (d,N))
    >>> up_mask = np.random.choice([True, False], (d,)+stack_shape+(nU,))
    >>> down_mask = np.random.choice([True, False], (d,)+stack_shape+(nD,))
    >>> left_mask = np.random.choice([True, False], (d+1,)+stack_shape+(rL,))
    >>> right_mask = np.random.choice([True, False], (d+1,)+stack_shape+(rR,))
    >>> basis = ubcf.UT3Basis(up_cores, down_cores, left_cores, right_cores, shape_mask, up_mask, down_mask, basis_left_mask, basis_right_mask)
    """
    up_tucker_supercore:    NDArray  # B_dxo B_dyo   = I_dxy, shape = (d,)+stack_shape+(nU, N)
    down_tt_supercore:      NDArray  # R_dixj R_diyj = I_dxy  shape = (d,)+stack_shape+(rL, nD, rR)
    left_tt_supercore:      NDArray  # P_diax P_diay = I_dxy, shape = (d,)+stack_shape+(rL, nU, rL)
    right_tt_supercore:     NDArray  # Q_dxaj Q_dyaj = I_dxy  shape = (d,)+stack_shape+(rR, nU, rR)

    shape_mask: NDArray # dtype=bool, (d,N)

    up_mask:            NDArray # dtype=bool, shape=(d,)+stack_shape+nU
    down_mask:          NDArray # dtype=bool, shape=(d,)+stack_shape+nD
    basis_left_mask:    NDArray # dtype=bool, shape=(d+1,)+stack_shape+rL
    basis_right_mask:   NDArray # dtype=bool, shape=(d+1,)+stack_shape+rR

    @ft.cached_property
    def data(self) -> typ.Tuple[
        NDArray, # up_tucker_supercore
        NDArray, # down_tt_supercore
        NDArray, # left_tt_supercore
        NDArray, # right_tt_supercore
        NDArray, # shape_mask
        NDArray, # up_mask
        NDArray, # down_mask
        NDArray, # left_mask
        NDArray, # right_mask
    ]:
        return (
            self.up_tucker_supercore, self.down_tt_supercore, self.left_tt_supercore, self.right_tt_supercore,
            self.shape_mask,
            self.up_mask, self.down_mask, self.basis_left_mask, self.basis_right_mask,
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

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple(list(self.shape_mask.sum(axis=-1)))

    @ft.cached_property
    def up_ranks(self) -> NDArray:
        return self.up_mask.sum(axis=-1)

    @ft.cached_property
    def down_ranks(self) -> NDArray:
        return self.down_mask.sum(axis=-1)

    @ft.cached_property
    def left_ranks(self) -> NDArray:
        return self.basis_left_mask.sum(axis=-1)

    @ft.cached_property
    def right_ranks(self) -> NDArray:
        return self.basis_right_mask.sum(axis=-1)

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
        up_sc, down_sc, left_sc, right_sc = masking.apply_basis_masks(*self.data)
        return UT3Basis(
            up_sc, down_sc, left_sc, right_sc,
            self.shape_mask, self.up_mask, self.down_mask,
            self.basis_left_mask, self.basis_right_mask,
        )

    def validate(self) -> None:
        '''Check rank and shape consistency of uniform Tucker tensor train basis (`UT3Basis`).

        Parameters
        ----------
        x : UT3Basis

        Raises
        ------
        ValueError
            Error raised if the cores of the UT3Basis have inconsistent shapes.

        See Also
        --------
        UT3Basis
        UT3Variations
        '''
        UU_good = self.up_tucker_supercore.shape  == (self.d,) + self.stack_shape + (self.nU, self.N)
        DD_good = self.down_tt_supercore.shape    == (self.d,) + self.stack_shape + (self.rL, self.nD, self.rR)
        LL_good = self.left_tt_supercore.shape    == (self.d,) + self.stack_shape + (self.rL, self.nU, self.rL)
        RR_good = self.right_tt_supercore.shape   == (self.d,) + self.stack_shape + (self.rR, self.nU, self.rR)

        SM_good = self.shape_mask.shape           == (self.d, self.N)

        UM_good = self.up_mask.shape == (self.d,) + self.stack_shape + (self.nU,)
        DM_good = self.down_mask.shape == (self.d,) + self.stack_shape + (self.nD,)
        LM_good = self.basis_left_mask.shape == (self.d + 1,) + self.stack_shape + (self.rL,)
        RM_good = self.basis_right_mask.shape == (self.d + 1,) + self.stack_shape + (self.rR,)

        bad_str = lambda x: ' <-- Bad' if not x else ''

        shapes_string = ''
        shapes_string += 'up_tucker_supercore.shape = ' + str(self.up_tucker_supercore.shape)   + ' =? (d,) + stack_shape + (nU, N)' + bad_str(UU_good) + '\n'
        shapes_string += 'down_tt_supercore.shape   = ' + str(self.down_tt_supercore.shape)     + ' =? (d,) + stack_shape + (rL, nD, rR)' + bad_str(DD_good) + '\n'
        shapes_string += 'left_tt_supercore.shape   = ' + str(self.left_tt_supercore.shape)     + ' =? (d,) + stack_shape + (rL, nU, rL)' + bad_str(LL_good) + '\n'
        shapes_string += 'right_tt_supercore.shape  = ' + str(self.right_tt_supercore.shape)    + ' =? (d,) + stack_shape + (rR, nU, rR)' + bad_str(RR_good) + '\n'

        shapes_string += 'shape_mask.shape          = ' + str(self.shape_mask.shape)            + ' =? (d, N)' + bad_str(SM_good) + '\n'

        shapes_string += 'up_mask.shape             = ' + str(self.up_mask.shape) + ' =? (d,) + stack_shape + (nU,)' + bad_str(UM_good) + '\n'
        shapes_string += 'down_mask.shape           = ' + str(self.down_mask.shape) + ' =? (d,) + stack_shape + (nD,)' + bad_str(DM_good) + '\n'
        shapes_string += 'left_mask.shape           = ' + str(self.basis_left_mask.shape) + ' =? (d+1,) + stack_shape + (rL,)' + bad_str(LM_good) + '\n'
        shapes_string += 'right_mask.shape          = ' + str(self.basis_right_mask.shape) + ' =? (d+1,) + stack_shape + (rR,)' + bad_str(RM_good)

        if not (UU_good and DD_good and LL_good and RR_good and SM_good and UM_good and DM_good and LM_good and RM_good):
            raise ValueError(
                'Inconsistent shapes for UT3Basis.\n'
                + shapes_string
            )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstacks stacked UT3Basis into into array-like tree of unstacked UT3Basis.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_basis_variations_format as ubcf
        >>> import t3toolbox.corewise as cw
        >>> stack_shape = (2,3)
        >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
        >>> uc = np.random.randn(*((d,) + stack_shape + (nU, N)))
        >>> dc = np.random.randn(*((d,) + stack_shape + (rL, nD, rR)))
        >>> lc = np.random.randn(*((d,) + stack_shape + (rL, nU, rL)))
        >>> rc = np.random.randn(*((d,) + stack_shape + (rR, nU, rR)))
        >>> sm = np.random.choice([True, False], (d,N))
        >>> um = np.random.choice([True, False], (d,)+stack_shape+(nU,))
        >>> dm = np.random.choice([True, False], (d,)+stack_shape+(nD,))
        >>> lm = np.random.choice([True, False], (d+1,)+stack_shape+(rL,))
        >>> rm = np.random.choice([True, False], (d+1,)+stack_shape+(rR,))
        >>> B = ubcf.UT3Basis(uc, dc, lc, rc, sm, um, dm, lm, rm)
        >>> BB = B.unstack()
        >>> ii, jj = 1, 2
        >>> B_ij = BB[ii][jj]
        >>> cores_ij = tuple(c[:,ii,jj] for c in (uc, dc, lc, rc))
        >>> masks_ij = tuple(m[:,ii,jj] for m in (um, dm, lm, rm))
        >>> B_ij2 = ubcf.UT3Basis(*(cores_ij + (sm,) + masks_ij))
        >>> print(cw.corewise_norm(cw.corewise_sub(B_ij.data[:4], B_ij2.data[:4])))
        0.0
        >>> print([np.all(x == x2) for x, x2 in zip(B_ij.data[4:], B_ij2.data[4:])])
        [True, True, True, True, True]
        """
        stacked_data = self.data[:4] + self.data[5:] # no shape_mask
        unstacked_data = stacking.basic_uniform_unstack(stacked_data, 3)
        return stacking.apply_func_to_leaf_subtrees(
            unstacked_data,
            lambda x: UT3Basis(*(x[:4] + (self.shape_mask,) + x[4:])),
            (None,)*8, # leaf_structure
        )

    @staticmethod
    def stack(
            xx, # Array-like tree of UT3Basis
            use_jax: bool = False,
    ):
        """Stack array-like tree of UT3Basis into a single UT3Basis.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_basis_variations_format as ubcf
        >>> import t3toolbox.corewise as cw
        >>> stack_shape = (2,3)
        >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
        >>> uc = np.random.randn(*((d,) + stack_shape + (nU, N)))
        >>> dc = np.random.randn(*((d,) + stack_shape + (rL, nD, rR)))
        >>> lc = np.random.randn(*((d,) + stack_shape + (rL, nU, rL)))
        >>> rc = np.random.randn(*((d,) + stack_shape + (rR, nU, rR)))
        >>> sm = np.random.choice([True, False], (d,N))
        >>> um = np.random.choice([True, False], (d,)+stack_shape+(nU,))
        >>> dm = np.random.choice([True, False], (d,)+stack_shape+(nD,))
        >>> lm = np.random.choice([True, False], (d+1,)+stack_shape+(rL,))
        >>> rm = np.random.choice([True, False], (d+1,)+stack_shape+(rR,))
        >>> B = ubcf.UT3Basis(uc, dc, lc, rc, sm, um, dm, lm, rm)
        >>> BB = B.unstack()
        >>> B2 = ubcf.UT3Basis.stack(BB)
        >>> print(cw.corewise_norm(cw.corewise_sub(B.data[:4], B2.data[:4])))
        0.0
        >>> print([np.all(x == x2) for x, x2 in zip(B.data[4:], B2.data[4:])])
        [True, True, True, True, True]
        """
        stacked_data = stacking.apply_func_to_leaf_subtrees(
            xx,
            lambda x: x.data[:4] + x.data[5:],
            None,  # leaf_structure
        )
        sm = stacking.get_first_leaf(xx).shape_mask
        unstacked_data = stacking.basic_uniform_stack(stacked_data, use_jax=use_jax)
        return UT3Basis(*(unstacked_data[:4] + (sm,) + unstacked_data[4:]))


@dataclass(frozen=True)
class UT3Variations:
    """
    Tuple containing variation cores for basis-variation representations of TuckerTensorTrains.

    *Components*
        - tucker_variations    = (V0, ..., V(d-1)), elm_shape=(nDi, Ni)
        - tt_variations        = (H0, ..., H(d-1)), elm_shape=(rLi, nUi, rRi)

    The variations should fit in the "holes" of a T3Basis.

    See Also
    --------
    UT3Basis

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_basis_variations_format as ubcf
    >>> import t3toolbox.corewise as cw
    >>> stack_shape = (2,3)
    >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
    >>> tkv = np.random.randn(*((d,) + stack_shape + (nD, N)))
    >>> ttv = np.random.randn(*((d,) + stack_shape + (rL, nU, rR)))
    >>> sm = np.random.choice([True, False], (d,N))
    >>> um = np.random.choice([True, False], (d,)+stack_shape+(nU,))
    >>> dm = np.random.choice([True, False], (d,)+stack_shape+(nD,))
    >>> lm = np.random.choice([True, False], (d,)+stack_shape+(rL,))
    >>> rm = np.random.choice([True, False], (d,)+stack_shape+(rR,))
    >>> V = ubcf.UT3Variations(tkv, ttv, sm, um, dm, lm, rm)
    >>> print(V.uniform_structure)
    (3, 12, 7, 8, 5, 4, (2, 3))
    >>> print(V.uniform_variation_shapes)
    ((3, 8, 12), (3, 5, 7, 4))
    """
    tucker_variations: NDArray  # shape=(d,) + stack_shape + (nD,N)
    tt_variations:     NDArray  # shape=(d,) + stack_shape + (rL,nU, rR)

    shape_mask: NDArray # dtype=bool, (d,N)

    variations_up_mask:    NDArray # dtype=bool, shape=(d,)+stack_shape+nU
    variations_down_mask:  NDArray # dtype=bool, shape=(d,)+stack_shape+nD
    variations_left_mask:  NDArray # dtype=bool, shape=(d,)+stack_shape+rL # Note: d = variations_left_mask.shape[0] != basis_left_mask.shape[0] = d+1
    variations_right_mask: NDArray # dtype=bool, shape=(d,)+stack_shape+rR # Note: d = variations_right_mask.shape[0] != basis_right_mask.shape[0] = d+1

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

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple(list(self.shape_mask.sum(axis=-1)))

    @ft.cached_property
    def up_ranks(self) -> NDArray:
        return self.variations_up_mask.sum(axis=-1)

    @ft.cached_property
    def down_ranks(self) -> NDArray:
        return self.variations_down_mask.sum(axis=-1)

    @ft.cached_property
    def variation_left_ranks(self) -> NDArray:
        return self.variations_left_mask.sum(axis=-1)

    @ft.cached_property
    def variation_right_ranks(self) -> NDArray:
        return self.variations_right_mask.sum(axis=-1)

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
        NDArray, # tucker_variations
        NDArray, # tt_variations
        NDArray, # shape_mask
        NDArray, # variations_up_mask
        NDArray, # variations_down_mask
        NDArray, # variations_left_mask
        NDArray, # variations_right_mask
    ]:
        return (
            self.tucker_variations, self.tt_variations, self.shape_mask,
            self.variations_up_mask, self.variations_down_mask,
            self.variations_left_mask, self.variations_right_mask,
        )

    def apply_masks(self) -> 'UT3Variations':
        """Apply masks to the variation supercores, zeroing out unmasked entries.
        """
        masked_tk_supercore, masked_tt_supercore = masking.apply_variations_masks(*self.data)
        return UT3Variations(
            masked_tk_supercore, masked_tt_supercore,
            self.shape_mask, self.variations_up_mask, self.variations_down_mask,
            self.variations_left_mask, self.variations_right_mask,
        )

    def validate(self) -> None:
        '''Check rank and shape consistency of uniform Tucker tensor train basis (`UT3Basis`).

        Raises
        ------
        ValueError
            Error raised if the cores of the UT3Basis have inconsistent shapes.

        See Also
        --------
        UT3Basis
        '''
        TK_good = self.tucker_variations.shape  == (self.d,) + self.stack_shape + (self.nD, self.N)
        TT_good = self.tt_variations.shape      == (self.d,) + self.stack_shape + (self.rL, self.nU, self.rR)

        SM_good = self.shape_mask.shape         == (self.d, self.N)

        UM_good = self.variations_up_mask.shape     == (self.d,) + self.stack_shape + (self.nU,)
        DM_good = self.variations_down_mask.shape   == (self.d,) + self.stack_shape + (self.nD,)
        LM_good = self.variations_left_mask.shape   == (self.d,) + self.stack_shape + (self.rL,)
        RM_good = self.variations_right_mask.shape  == (self.d,) + self.stack_shape + (self.rR,)

        bad_str = lambda x: ' <-- Bad' if not x else ''

        shapes_string = ''
        shapes_string += 'tucker_variations.shape     = '  + str(self.tucker_variations.shape)      + ' =? (d,) + stack_shape + (nD, N)'        + bad_str(TK_good) + '\n'
        shapes_string += 'tt_variations.shape         = '  + str(self.tt_variations.shape)          + ' =? (d,) + stack_shape + (rL, nU, rL)'   + bad_str(TT_good) + '\n'

        shapes_string += 'shape_mask.shape            = ' + str(self.shape_mask.shape)              + ' =? (d, N)' + bad_str(SM_good) + '\n'

        shapes_string += 'variations_up_mask.shape    = ' + str(self.variations_up_mask.shape)      + ' =? (d,) + stack_shape + (nU,)' + bad_str(UM_good) + '\n'
        shapes_string += 'variations_down_mask.shape  = ' + str(self.variations_down_mask.shape)    + ' =? (d,) + stack_shape + (nD,)' + bad_str(DM_good) + '\n'
        shapes_string += 'variations_left_mask.shape  = ' + str(self.variations_left_mask.shape)    + ' =? (d,) + stack_shape + (rL,)' + bad_str(LM_good) + '\n'
        shapes_string += 'variations_right_mask.shape = ' + str(self.variations_right_mask.shape)   + ' =? (d,) + stack_shape + (rR,)' + bad_str(RM_good)

        if not (TK_good and TT_good and SM_good and UM_good and DM_good and LM_good and RM_good):
            raise ValueError(
                'Inconsistent shapes for UT3Variations.\n'
                + shapes_string
            )

    def __post_init__(self):
        self.validate()

    def unstack(self):
        """Unstacks stacked UT3Basis into into array-like tree of unstacked UT3Basis.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_basis_variations_format as ubcf
        >>> import t3toolbox.corewise as cw
        >>> stack_shape = (2,3)
        >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
        >>> tkv = np.random.randn(*((d,) + stack_shape + (nD, N)))
        >>> ttv = np.random.randn(*((d,) + stack_shape + (rL, nU, rR)))
        >>> sm = np.random.choice([True, False], (d,N))
        >>> um = np.random.choice([True, False], (d,)+stack_shape+(nU,))
        >>> dm = np.random.choice([True, False], (d,)+stack_shape+(nD,))
        >>> lm = np.random.choice([True, False], (d,)+stack_shape+(rL,))
        >>> rm = np.random.choice([True, False], (d,)+stack_shape+(rR,))
        >>> V = ubcf.UT3Variations(tkv, ttv, sm, um, dm, lm, rm)
        >>> VV = V.unstack()
        >>> ii, jj = 1, 2
        >>> V_ij = VV[ii][jj]
        >>> V_ij2 = ubcf.UT3Variations(tkv[:,ii,jj], ttv[:,ii,jj], sm, um[:,ii,jj], dm[:,ii,jj], lm[:,ii,jj], rm[:,ii,jj])
        >>> print(cw.corewise_norm(cw.corewise_sub(V_ij.data[:2], V_ij2.data[:2])))
        0.0
        >>> print([np.all(x == x2) for x, x2 in zip(V_ij.data[2:], V_ij2.data[2:])])
        [True, True, True, True, True]
        """
        stacked_data = self.data[:2] + self.data[3:] # no shape_mask
        unstacked_data = stacking.basic_uniform_unstack(stacked_data, 3)
        return stacking.apply_func_to_leaf_subtrees(
            unstacked_data,
            lambda x: UT3Variations(*(x[:2] + (self.shape_mask,) + x[2:])),
            (None,)*6, # leaf_structure
        )

    @staticmethod
    def stack(
            xx, # Array-like tree of UT3Basis
            use_jax: bool = False,
    ):
        """Stack array-like tree of UT3Basis into a single UT3Basis.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_basis_variations_format as ubcf
        >>> import t3toolbox.corewise as cw
        >>> stack_shape = (2,3)
        >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
        >>> tkv = np.random.randn(*((d,) + stack_shape + (nD, N)))
        >>> ttv = np.random.randn(*((d,) + stack_shape + (rL, nU, rR)))
        >>> sm = np.random.choice([True, False], (d,N))
        >>> um = np.random.choice([True, False], (d,)+stack_shape+(nU,))
        >>> dm = np.random.choice([True, False], (d,)+stack_shape+(nD,))
        >>> lm = np.random.choice([True, False], (d,)+stack_shape+(rL,))
        >>> rm = np.random.choice([True, False], (d,)+stack_shape+(rR,))
        >>> V = ubcf.UT3Variations(tkv, ttv, sm, um, dm, lm, rm)
        >>> VV = V.unstack()
        >>> V2 = ubcf.UT3Variations.stack(VV)
        >>> print(cw.corewise_norm(cw.corewise_sub(V.data[:2], V2.data[:2])))
        0.0
        >>> print([np.all(x == x2) for x, x2 in zip(V.data[2:], V2.data[2:])])
        [True, True, True, True, True]
        """
        stacked_data = stacking.apply_func_to_leaf_subtrees(
            xx,
            lambda x: x.data[:2] + x.data[3:],
            None,  # leaf_structure
        )
        sm = stacking.get_first_leaf(xx).shape_mask
        unstacked_data = stacking.basic_uniform_stack(stacked_data, use_jax=use_jax)
        return UT3Variations(*(unstacked_data[:2] + (sm,) + unstacked_data[2:]))


def check_ubv_pair(base: UT3Basis, variations: UT3Variations) -> None:
    """Check rank and shape consistency between UT3Basis and UT3Variations.

    This ensures that the variation cores (V, H) have the correct dimensions
     to interface with the base cores (U, L, R, O).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform_basis_variations_format as ubcf
    >>> import t3toolbox.corewise as cw
    >>> stack_shape = (2,3)
    >>> d, N, nU, nD, rL, rR = 3, 12, 7, 8, 5, 4
    >>> uc = np.random.randn(*((d,) + stack_shape + (nU, N)))
    >>> dc = np.random.randn(*((d,) + stack_shape + (rL, nD, rR)))
    >>> lc = np.random.randn(*((d,) + stack_shape + (rL, nU, rL)))
    >>> rc = np.random.randn(*((d,) + stack_shape + (rR, nU, rR)))
    >>> sm = np.random.choice([True, False], (d,N))
    >>> um = np.random.choice([True, False], (d,)+stack_shape+(nU,))
    >>> dm = np.random.choice([True, False], (d,)+stack_shape+(nD,))
    >>> lm = np.random.choice([True, False], (d+1,)+stack_shape+(rL,))
    >>> rm = np.random.choice([True, False], (d+1,)+stack_shape+(rR,))
    >>> B = ubcf.UT3Basis(uc, dc, lc, rc, sm, um, dm, lm, rm)
    >>> tkv = np.random.randn(*((d,) + stack_shape + (nD, N)))
    >>> ttv = np.random.randn(*((d,) + stack_shape + (rL, nU, rR)))
    >>> V = ubcf.UT3Variations(tkv, ttv, sm, um, dm, lm[:-1], rm[1:])
    >>> ubcf.check_ubv_pair(B, V) # Does nothing since base and variations are consistent
    """
    if base.uniform_structure != variations.uniform_structure:
        raise ValueError(
            'Inconsistent (UT3Basis, UT3Variations) pair.\n'
            + str(base.uniform_structure) + ' = base.uniform_structure != variations.uniform_structure = ' + str(variations.uniform_structure)
        )

    if (base.up_mask != variations.variations_up_mask).all():
        raise ValueError(
            'Inconsistent (UT3Basis, UT3Variations) pair.\n'
            + str(base.up_mask) + ' = base.up_mask != variations.variations_up_mask = ' + str(variations.variations_up_mask)
        )

    if (base.down_mask != variations.variations_down_mask).all():
        raise ValueError(
            'Inconsistent (UT3Basis, UT3Variations) pair.\n'
            + str(base.down_mask) + ' = base.down_mask != variations.variations_down_mask = ' + str(variations.variations_down_mask)
        )

    if (base.basis_left_mask[:-1] != variations.variations_left_mask).all():
        raise ValueError(
            'Inconsistent (UT3Basis, UT3Variations) pair.\n'
            + str(base.basis_left_mask[:-1]) + ' = base.basis_left_mask[:-1] != variations.variations_left_mask = ' + str(variations.variations_left_mask)
        )

    if (base.basis_right_mask[1:] != variations.variations_right_mask).all():
        raise ValueError(
            'Inconsistent (UT3Basis, UT3Variations) pair.\n'
            + str(base.basis_right_mask[:-1]) + ' = base.basis_right_mask[:-1] != variations.variations_right_mask = ' + str(variations.variations_right_mask)
        )


def ut3basis_to_t3basis(
        x: UT3Basis,
        use_jax: bool = False,
) -> bvf.T3Basis:
    """Convert UT3Basis to array-like tree of T3Basis.
    """
    x = x.apply_masks()

    result = ubv_conversions.ut3basis_to_t3basis(x.data, use_jax=use_jax)

    return stacking.apply_func_to_leaf_subtrees(
        result,
        lambda x: bvf.T3Basis(*x),
        ((None,)*x.d,)*4, # leaf_structure
    )


def ut3_orthogonal_representations(
        x: ut3.UniformTuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash: bool = True,
        use_jax: bool = False,
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
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.uniform_basis_variations_format as ubvf
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> import t3toolbox.corewise as cw
    >>> d, N, n, r = 3, 6, 5, 4
    >>> stack_shape = (2,3)
    >>> tucker_supercore = np.random.randn(*((d,)+stack_shape+(n,N)))
    >>> tt_supercore = np.random.randn(*((d,)+stack_shape+(r,n,r)))
    >>> shape_mask = np.random.choice([True, False], (d,N))
    >>> tucker_edge_mask = np.random.choice([True, False], (d,)+stack_shape+(n,))
    >>> tt_edge_mask = np.random.choice([True, False], (d+1,)+stack_shape+(r,))
    >>> ux = ut3.UniformTuckerTensorTrain(tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask)
    >>> ubase, uvar = ubvf.ut3_orthogonal_representations(ux)
    >>> all_base = ubvf.ut3basis_to_t3basis(ubase)
    >>> ii, jj = 1, 2
    >>> base_ij = all_base[ii][jj]
    >>> xx = ut3.ut3_to_t3(ux)
    >>> x_ij = xx[ii][jj]
    >>> base_ij2, var_ij2 = bvf.t3_orthogonal_representations(x_ij)
    >>> print(cw.corewise_norm(cw.corewise_sub(base_ij.data, base_ij2.data)))


    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.basis_variations_format as bvf
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (3,3,2,1), stack_shape=(2,3))
    >>> base, variations = bvf.t3_orthogonal_representations(x) # Compute orthogonal representations
    >>> up_tucker_cores, down_tt_cores, left_tt_cores, right_tt_cores = base.data
    >>> tucker_variations, tt_variations = variations.data
    >>> (U0,U1,U2) = up_tucker_cores
    >>> (D0,D1,D2) = down_tt_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (V0,V1,V2) = tucker_variations
    >>> (H0,H1,H2) = tt_variations
    >>> x2 = t3.TuckerTensorTrain((U0,U1,U2), (L0,H1,R2)) # representation with TT-core variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = t3.TuckerTensorTrain((U0,V1,U2), (L0,D1,R2)) # representation with tucker core variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x3.to_dense())) # Still represents origional tensor
    5.4355175448533146e-12
    >>> print(np.linalg.norm(np.einsum('...io,...jo', U1, U1) - np.eye(U1.shape[-2]))) # U: orthogonal
    1.1915111872574236e-15
    >>> print(np.linalg.norm(np.einsum('...iaj,...iak', L1, L1) - np.eye(L1.shape[-1]))) # L: left orthogonal
    9.733823879665448e-16
    >>> print(np.linalg.norm(np.einsum('...iaj,...kaj', R1, R1) - np.eye(R1.shape[-3]))) # R: right orthogonal
    8.027553546330097e-16
    >>> print(np.linalg.norm(np.einsum('...iaj,...ibj', D1, D1) - np.eye(D1.shape[-2]))) # O: outer orthogonal
    1.3870474292323159e-15
    '''
    x = x.apply_masks()
    utk, utt, sm, tkm, ttm = x.data

    (uc, dc, lc, rc), (tkv, ttv) = orth_reps.orthogonal_representations(
        (utk, utt), already_left_orthogonal=already_left_orthogonal, squash=squash, use_jax=use_jax,
    )

    # up_ranks, down_ranks, left_ranks, right_ranks = ranks.compute_orthogonal_representation_ranks(
    #     x.shape, x.tucker_ranks, x.tt_ranks, use_jax=use_jax,
    # )

    return UT3Basis(uc, dc, lc, rc, sm, tkm, tkm, ttm, ttm), UT3Variations(tkv, ttv, sm, tkm, tkm, ttm[:-1], ttm[1:])


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



