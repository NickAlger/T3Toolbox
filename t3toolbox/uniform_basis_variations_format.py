# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.bvf_conversions as bvf_ops
import t3toolbox.backend.orthogonal_representations as orth_reps
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

    @ft.cached_property
    def variation_shapes(
            self,
    ) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int, ...], ...],  # tucker_variation_shapes. len=d. elm_len=2
        typ.Tuple[typ.Tuple[int, ...], ...],  # tt_variation_shapes. len=d. elm_len=3
    ]:
        '''T3Variations shapes that fit with this T3Basis.

        Shapes of the "holes" in the following tensor diagrams::

            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |

            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |

        Examples
        --------
        #### EXAMPLE IS WORK IN PROGRESS
        >>> import numpy as np
        >>> import t3toolbox.basis_variations_format as bcf
        >>> ss = (2,3) # not included in variation_shapes.
        >>> tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
        >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
        >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
        >>> outer_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
        >>> basis = bcf.T3Basis(tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
        >>> print(basis.variation_shapes)
        (((9, 14), (8, 15), (7, 16)), ((1, 10, 4), (2, 11, 5), (3, 12, 1)))
        '''
        tucker_variation_shapes = tuple([(nD, N) for nD, N in zip(self.down_ranks, self.shape)])
        tt_variation_shapes = tuple([
            (rL, nU, rR) for rL, nU, rR
            in zip(self.left_ranks[:-1], self.up_ranks, self.right_ranks[1:])])

        return tucker_variation_shapes, tt_variation_shapes

    def apply_masks(self) -> 'UT3Basis':
        """Apply masks to the basis supercores, zeroing out unmasked entries.

        Examples
        --------
        >>> import numpy as np

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
                'Inconsistent shapes for T3Basis.\n'
                + shapes_string
            )

    def __post_init__(self):
        self.validate()


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
    T3Basis

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.basis_variations_format as bvf
    >>> ss = (2,3) # stack shape
    >>> tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> outer_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> base = bvf.T3Basis(tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> print(base.structure)
    ((14, 15, 16), (10, 11, 12), (1, 2, 3, 5), (2, 4, 5, 1), (9, 8, 7), (2, 3))
    >>> tucker_variations = tuple([np.ones(ss + B_shape) for B_shape in base.variation_shapes[0]])
    >>> tt_variations = tuple([np.ones(ss + G_shape) for G_shape in base.variation_shapes[1]])
    >>> variations = bvf.T3Variations(tucker_variations, tt_variations) # variation that fits with base
    >>> print(variations.structure) # same as base, except first right tt rank and last left tt rank, which are None
    ((14, 15, 16), (9, 8, 7), (1, 2, 3, None), (None, 4, 5, 1), (10, 11, 12), (2, 3))
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
        return len(self.tucker_variations)

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]:
        return tuple([U.shape[-1] for U in self.tucker_variations])

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        return self.tucker_variations[0].shape[:-2]

    @ft.cached_property
    def variation_shapes(
            self,
    ) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int, ...], ...],  # tucker_variation_shapes. len=d. elm_len=2
        typ.Tuple[typ.Tuple[int, ...], ...],  # tt_variation_shapes. len=d. elm_len=3
    ]:
        '''T3Variations shapes that fit with this T3Basis.

        Shapes of the "holes" in the following tensor diagrams::

            1 -- L0 -- ( ) -- R2 -- R3 -- 1
                 |      |      |      |
                 U0     U1     U2     U3
                 |      |      |      |

            1 -- L0 -- L1 -- O2 -- R3 -- 1
                 |     |     |     |
                 U0    U1    ( )   U3
                 |     |     |     |
        '''
        tucker_variation_shapes = tuple([B.shape[-2:] for B in self.tucker_variations])
        tt_variation_shapes = tuple([G.shape[-3:] for G in self.tt_variations])
        return tucker_variation_shapes, tt_variation_shapes

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray,...], # tucker_variations
        typ.Tuple[NDArray,...], # tt_variations
    ]:
        return self.tucker_variations, self.tt_variations

    def validate(self) -> None:
        '''Check rank and shape consistency of Tucker tensor train variations (`T3Variations`).

        Parameters
        ----------
        self : T3Variations

        Raises
        ------
        ValueError
            Error raised if the cores of the T3Variations have inconsistent shapes.

        See Also
        --------
        T3Basis
        T3Variations
        '''
        VV, HH = self.data

        d = len(VV)
        if len(HH) != d:
            raise ValueError(
                'Inconsistent T3Variations.\n'
                + 'All backend sequences must have length d=' + str(d) + '.\n'
                + 'len(VV)=' + str(len(VV))
                + ', len(HH)=' + str(len(HH))
            )

        for ii, V in enumerate(VV):
            if len(V.shape) < 2:
                raise ValueError(
                    'Inconsistent T3Variations.\n'
                    + 'tucker_cores[' + str(ii) + '] is not a (stacked) matrix. shape=' + str(V.shape)
                )

        for ii, H in enumerate(HH):
            if len(H.shape) < 3:
                raise ValueError(
                    'Inconsistent T3Variations.\n'
                    + 'tt_cores[' + str(ii) + '] is not a (stacked) 3-tensor. '
                    + 'shape=' + str(H.shape)
                )

        tucker_stack_shapes = tuple([B.shape[:-2] for B in self.tucker_variations])
        tt_stack_shapes = tuple([G.shape[:-3] for G in self.tt_variations])

        if not (tucker_stack_shapes == tt_stack_shapes == (self.stack_shape,)*self.d):
            raise ValueError(
                'Inconsistent T3Basis.\n'
                + str(tucker_stack_shapes) + ' = tucker_stack_shapes.\n'
                + str(tt_stack_shapes) + ' = tt_stack_shapes.\n'
            )

    def __post_init__(self):
        self.validate()


def check_ubv_pair(base: UT3Basis, variations: UT3Variations) -> None:
    """Check rank and shape consistency between T3Basis and T3Variations.

    This ensures that the variation cores (V, H) have the correct dimensions
     to interface with the base cores (U, L, R, O).

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.basis_variations_format as bcf
    >>> ss = (2,3) # stack shape
    >>> tucker_cores = (np.ones(ss+(10, 14)), np.ones(ss+(11, 15)), np.ones(ss+(12, 16)))
    >>> left_tt_cores = (np.ones(ss+(1, 10, 2)), np.ones(ss+(2, 11, 3)), np.ones(ss+(3,12,5)))
    >>> right_tt_cores = (np.ones(ss+(2, 10, 4)), np.ones(ss+(4, 11, 5)), np.ones(ss+(5, 12, 1)))
    >>> outer_tt_cores = (np.ones(ss+(1, 9, 4)), np.ones(ss+(2, 8, 5)), np.ones(ss+(3, 7, 1)))
    >>> base = bcf.T3Basis(tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)
    >>> tucker_variations = tuple([np.ones(ss + B_shape) for B_shape in base.variation_shapes[0]])
    >>> tt_variations = tuple([np.ones(ss + G_shape) for G_shape in base.variation_shapes[1]])
    >>> variations = bcf.T3Variations(tucker_variations, tt_variations)
    >>> bcf.check_bv_pair(base, variations) # does nothing since these are consistent
    """
    if base.stack_shape != variations.stack_shape:
        raise ValueError(
            'Inconsistent (T3Basis, T3Variations) pair.\n'
            + str(base.stack_shape) + ' = base.stack_shape != variations.stack_shape = ' + str(variations.stack_shape)
        )

    xVV, xHH = base.variation_shapes
    yVV, yHH = variations.variation_shapes

    for ii, (xV, yV) in enumerate(zip(xVV, yVV)):
        if xV != yV:
            raise ValueError(
                'Inconsistent T3Base - T3Variation pair.\n'
                + str(ii) + '-th Tucker variation shape' + str(yV)
                + ' does not fit base hole ' + str(xV)
            )

    for ii, (xH, yH) in enumerate(zip(xHH, yHH)):
        if xH != yH:
            raise ValueError(
                'Inconsistent T3Base - T3Variation pair.\n'
                + str(ii) + '-th tensor train variation shape' + str(yH)
                + ' does not fit base hole ' + str(xH)
            )


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
    >>> import numpy as np
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
    >>> ((B0, B1, B2), (G0, G1, G2)) = bcf.bv_to_t3(1, True, base, variations).data # replace index-1 TT-backend
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,U1,U2), (L0,H1,R2)))
    True
    >>> ((B0, B1, B2), (G0, G1, G2)) = bcf.bv_to_t3(1, False, base, variations).data # replace index-1 tucker backend
    >>> print(((B0,B1,B2), (G0,G1,G2)) == ((U0,V1,U2), (L0,O1,R2)))
    True
    '''
    check_basis_variations_pair(basis, variations)
    return t3.TuckerTensorTrain(*bvf_ops.bv_to_t3(ii, use_tt_variation, basis.data, variations.data))


def ut3_orthogonal_representations(
        x: t3.TuckerTensorTrain,
        already_left_orthogonal: bool = False,
        squash: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    UT3Basis,  # orthogonal base
    T3Variations,  # variations
]:
    '''Construct base-variation representations of TuckerTensorTrain with orthogonal base.

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
    >>> import t3toolbox.basis_variations_format as bcf
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (3,3,2,1), stack_shape=(2,3))
    >>> base, variations = bcf.t3_orthogonal_representations(x) # Compute orthogonal representations
    >>> up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base.data
    >>> tucker_variations, tt_variations = variations.data
    >>> (U0,U1,U2) = up_tucker_cores
    >>> (L0,L1,L2) = left_tt_cores
    >>> (R0,R1,R2) = right_tt_cores
    >>> (O0,O1,O2) = outer_tt_cores
    >>> (V0,V1,V2) = tucker_variations
    >>> (H0,H1,H2) = tt_variations
    >>> x2 = t3.TuckerTensorTrain((U0,U1,U2), (L0,H1,R2)) # representation with TT-backend variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Still represents origional tensor
    4.978421562425667e-12
    >>> x3 = t3.TuckerTensorTrain((U0,V1,U2), (L0,O1,R2)) # representation with tucker backend variation in index 1
    >>> print(np.linalg.norm(x.to_dense() - x3.to_dense())) # Still represents origional tensor
    5.4355175448533146e-12
    >>> print(np.linalg.norm(np.einsum('...io,...jo', U1, U1) - np.eye(U1.shape[-2]))) # U: orthogonal
    1.1915111872574236e-15
    >>> print(np.linalg.norm(np.einsum('...iaj,...iak', L1, L1) - np.eye(L1.shape[-1]))) # L: left orthogonal
    9.733823879665448e-16
    >>> print(np.linalg.norm(np.einsum('...iaj,...kaj', R1, R1) - np.eye(R1.shape[-3]))) # R: right orthogonal
    8.027553546330097e-16
    >>> print(np.linalg.norm(np.einsum('...iaj,...ibj', O1, O1) - np.eye(O1.shape[-2]))) # O: outer orthogonal
    1.3870474292323159e-15
    '''
    result = bvf_ops.orthogonal_representations(
        x.data, already_left_orthogonal=already_left_orthogonal, squash=squash, use_jax=use_jax,
    )
    return T3Basis(*result[0]), T3Variations(*result[1])
