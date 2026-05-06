# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
"""
Basic Tucker tensor trains with non-uniform (ragged) shape and ranks.

Other
=====

Operations here are defined with respect to the dense N0 x ... x N(d-1) tensors that
are *represented* by the Tucker tensor train, even though these dense tensors
are not formed during computations.

For corewise backend, see :mod:`t3toolbox.corewise`
"""
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.tucker_tensor_train.dense_t3svd as dense_t3svd
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.tucker_tensor_train.t3_operations as ragged_operations
import t3toolbox.backend.tucker_tensor_train.t3_orthogonalization as ragged_orthogonalization
import t3toolbox.backend.tucker_tensor_train.t3_linalg as ragged_linalg
import t3toolbox.backend.tucker_tensor_train.t3_svd as ragged_t3svd

from t3toolbox.backend.common import *

jax = None
if has_jax:
    import jax

__all__ = [
    'TuckerTensorTrain',
    #
    't3_core_shapes',
    'compute_minimal_t3_ranks',
    #
    't3_apply',
    't3_entries',
    't3_probe',
    #
    't3_add',
    't3_sub',
    't3_inner_product',
    #
    't3svd',
    't3svd_dense',
]


###########################################
########    Tucker Tensor Train    ########
###########################################


@dataclass(frozen=True)
class TuckerTensorTrain:
    """
    Tucker tensor train with variable ranks.

    Tensor network diagram for a dth order Tucker tensor train::

            r0        r1        r2       r(d-1)          rd
        1 ------ G0 ------ G1 ------ ... ------ G(d-1) ------ 1
                 |         |                    |
                 | n0      | n1                 | nd
                 |         |                    |
                 B0        B1                   Bd
                 |         |                    |
                 | N0      | N1                 | Nd
                 |         |                    |

    Attributes:
    -----------
    tucker_cores : Tuple[NDArray]
        Tucker cores: (B0, ..., B(d-1)), len=d, elm_shape=VS+(ni, Ni).

    tt_cores : Tuple[NDArray]
        Tensor train cores: (G0, ..., G(d-1)), len=d, elm_shape=VS+(ri, ni, r(i+1)).

    d: int
        Number of indices of the tensor

    stack_shape: typ.Tuple[int, ...]
        The stack shape, VS. Non-empty if this object stores many different Tucker tensor trains with the same structure.
        Shape of the leading parts of tucker_cores[ii].shape and tt_cores[ii].shape.

    shape: typ.Tuple[int,...]
        Tensor shape: (N0, N1, ..., N(d-1))

    tucker_ranks: typ.Tuple[int,...]
        Tucker ranks: (n0, r1, ..., n(d-1))

    tt_ranks: typ.Tuple[int, ...]
        TT ranks: (r0, r1, ..., rd)

    structure: typ.Tuple[typ.Tuple[int,...], typ.Tuple[int,...], typ.Tuple[int,...]]
        Structure of the Tucker tensor train: (shape, tucker_ranks, tt_ranks)

    data: typ.Tuple[Tuple[NDArray], Tuple[NDArray]]
        The cores defining the Tucker tensor train

    minimal_ranks: typ.Tuple[typ.Tuple[int,...], typ.Tuple[int, ...]]
        Tucker and tensor train ranks of the smallest possible Tucker tensor train that represents the same tensor.
        Tucker tensor trains may be made to have minimal ranks using T3-SVD.

    has_minimal_ranks: bool
        True if this Tucker tensor train's ranks equal the minimal ranks, False otherwise.

    Notes:
    ------

    The structure of a Tucker tensor train is defined by:

    - Tensor shape: (N0, N1, ..., N(d-1))
    - Tucker ranks: (n0, r1, ..., n(d-1))
    - TT ranks: (r0, r1, ..., rd)

    Typically, the first and last TT-ranks satisfy r0=rd=1, and "1" in the diagram
    is the number 1. However, it is allowed for these ranks to not be 1, in which case
    the "1"s in the diagram are vectors of ones.


    Many stacked Tucker tensor trains with the same structure may be stored in this object for vectorization.
    In this case,
        - tucker_cores[ii].shape = stack_shape + (ni,Ni)
        - tt_cores[ii].shape = stack_shape + (ri, ni, r(i+1))

    .. seealso::


    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = (np.ones((4,14)),np.ones((5,15)),np.ones((6,16)))
    >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with ones
    >>> print(x.d)
    3
    >>> print(x.shape)
    (14, 15, 16)
    >>> print(x.tucker_ranks)
    (4, 5, 6)
    >>> print(x.tt_ranks)
    (1, 3, 2, 1)
    >>> print(x.uniform_structure)
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), ())

    Example with stacking:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = [np.ones((6,7, 4,14)),np.ones((6,7, 5,15)),np.ones((6,7, 6,16))]
    >>> tt_cores = [np.ones((6,7, 1,4,3)), np.ones((6,7, 3,5,2)), np.ones((6,7, 2,6,1))]
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with ones
    >>> print(x.uniform_structure)
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), (6, 7))

    Minimal ranks

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((13,14,15,16), (4,5,6,7), (1,4,9,7,1))
    >>> print(x.has_minimal_ranks)
    True

    Using T3-SVD to make equivalent T3 with minimal ranks:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((13,14,15,16), (4,5,6,7), (1,99,9,7,1))
    >>> print(x.has_minimal_ranks)
    False
    >>> x2 = t3.t3svd(x)[0]
    >>> print(x2.has_minimal_ranks)
    True
    """
    tucker_cores:   typ.Tuple[NDArray,...] # len=d, elm_shape=VS+(ni, Ni)
    tt_cores:       typ.Tuple[NDArray,...] # len=d, elm_shape=VS+(ri, ni, r(i+1))

    @ft.cached_property
    def data(self) -> typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]]:
        return tuple(self.tucker_cores), tuple(self.tt_cores)

    @ft.cached_property
    def d(self) -> int:
        return len(self.tucker_cores)

    @ft.cached_property
    def is_empty(self) -> bool:
        return self.d == 0

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int, ...]:
        """If this object contains multiple stacked T3s with the same structure, this is the shape of the stack.
        """
        return self.tucker_cores[0].shape[:-2] if not self.is_empty else ()

    @ft.cached_property
    def shape(self) -> typ.Tuple[int, ...]: # len=d
        return tuple([B.shape[-1] for B in self.tucker_cores]) if not self.is_empty else ()

    @ft.cached_property
    def tucker_ranks(self) -> typ.Tuple[int, ...]: # len=d
        return tuple([B.shape[-2] for B in self.tucker_cores]) if not self.is_empty else ()

    @ft.cached_property
    def tt_ranks(self) -> typ.Tuple[int, ...]: # len=d+1
        rr = tuple([G.shape[-3] for G in self.tt_cores]) + (self.tt_cores[-1].shape[-1],)
        return rr if not self.is_empty else ()

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int,...], # shape
        typ.Tuple[int,...], # tucker_ranks
        typ.Tuple[int,...], # tt_ranks
        typ.Tuple[int,...], # stack_shape
    ]:
        return self.shape, self.tucker_ranks, self.tt_ranks, self.stack_shape

    @ft.cached_property
    def core_shapes(self) -> typ.Tuple[
        typ.Tuple[typ.Tuple[int,...],...], # tucker core shapes
        typ.Tuple[typ.Tuple[int,...],...], # tt core shapes
    ]:
        return (
            tuple([B.shape[len(self.stack_shape):] for B in self.tucker_cores]),
            tuple([G.shape[len(self.stack_shape):] for G in self.tt_cores]),
        )

    @ft.cached_property
    def size(self) -> int:
        return sum([x.size for x in self.tucker_cores]) + sum([x.size for x in self.tt_cores])

    @ft.cached_property
    def minimal_ranks(self) -> typ.Tuple[typ.Tuple[int,...], typ.Tuple[int,...]]:
        minimal_tucker_ranks, minimal_tt_ranks = compute_minimal_t3_ranks(
            self.shape, self.tucker_ranks, self.tt_ranks,
        )
        return minimal_tucker_ranks, minimal_tt_ranks

    @ft.cached_property
    def has_minimal_ranks(self) -> bool:
        return (self.tucker_ranks, self.tt_ranks) == self.minimal_ranks

    def validate(self):
        """Check internal consistency of the Tucker tensor train.
        """
        if len(self.tucker_cores) != len(self.tt_cores):
            raise ValueError(
                'Inconsistent TuckerTensorTrain.\n'
                + str(len(self.tucker_cores)) + ' = len(tucker_cores) != len(tt_cores) = ' + str(len(self.tt_cores))
            )

        if len(self.tucker_cores) < 1:
            raise ValueError(
                'Empty TuckerTensorTrain not supported.\n'
                + str(len(self.tucker_cores)) + ' = len(tucker_cores)'
            )

        for ii, G in enumerate(self.tt_cores):
            if len(G.shape) < 3:
                raise ValueError(
                    'Inconsistent TuckerTensorTrain.\n'
                    + 'tt_cores[' + str(ii) + '] has less than 3 indices. shape=' + str(G.shape)
                )

        right_tt_ranks = tuple([int(self.tt_cores[0].shape[-3])] + [int(G.shape[-1]) for G in self.tt_cores])
        left_tt_ranks = tuple([int(G.shape[-3]) for G in self.tt_cores] + [int(self.tt_cores[-1].shape[-1])])
        if left_tt_ranks != right_tt_ranks:
            raise ValueError(
                'Inconsistent TuckerTensorTrain.\n'
                + str(left_tt_ranks) + ' = left_tt_ranks != right_tt_ranks = ' + str(right_tt_ranks)
            )

        for ii, B in enumerate(self.tucker_cores):
            if len(B.shape) < 2:
                raise ValueError(
                    'Inconsistent TuckerTensorTrain.\n'
                    + 'tucker_cores[' + str(ii) + '] has less than 2 indices. shape=' + str(B.shape)
                )

        for ii, (B, G) in enumerate(zip(self.tucker_cores, self.tt_cores)):
            if B.shape[-2] != G.shape[-2]:
                raise ValueError(
                    'Inconsistent TuckerTensorTrain.\n'
                    + str(B.shape[-2]) + ' = tucker_cores[' + str(ii) + '].shape[-2]'
                    + ' != '
                    + 'tt_cores[' + str(ii) + '].shape[-2] = ' + str(G.shape[-2])
                )

        desired_stack_shapes = tuple(self.stack_shape for _ in range(self.d))
        tt_stack_shapes = tuple(G.shape[:-3] for G in self.tt_cores)
        tucker_stack_shapes = tuple(B.shape[:-2] for B in self.tucker_cores)
        if ((tt_stack_shapes) != (desired_stack_shapes)
                or (tucker_stack_shapes != desired_stack_shapes)):
            raise ValueError(
                'Inconsistent TuckerTensorTrain.\n'
                + str(tt_stack_shapes) + ' = tt_stack_shapes'
                + '\n'
                + str(tucker_stack_shapes) + ' = tucker_stack_shapes'
            )

    def __post_init__(self):
        self.validate()

    ############################################
    ##########    Basic operations    ##########
    ############################################

    def to_dense(
            self,
            squash_tails: bool = True,
            use_jax: bool = False,
    ) -> NDArray:
        """Contract a Tucker tensor train to a dense tensor.

        Parameters
        ----------
        x : TuckerTensorTrain
            Tucker tensor train which will be contracted to a dense tensor.

        squash_tails: bool, defaults to True
            Whether to contract the leading and trailing 1s with the first and last TT indices.

        use_jax: bool, defaults to False
            Whether to use Jax for linear algebra. Default: False (use numpy).

        Returns
        -------
        dense_x: NDArray
            Dense tensor represented by x,
            which has shape (N0, ..., N(d-1)) if squash_tails=True,
            or (r0,N0,...,N(d-1),rd) if squash_tails=False.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(4,14),randn(5,15),randn(6,16))
        >>> tt_cores = (randn(2,4,3), randn(3,5,2), randn(2,6,5))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x_dense = x.to_dense() # Convert TuckerTensorTrain to dense tensor
        >>> ((B0,B1,B2), (G0,G1,G2)) = tucker_cores, tt_cores
        >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
        >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
        7.48952547844518e-16

        Example where leading and trailing ones are not contracted

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(4,14),randn(5,15),randn(6,16))
        >>> tt_cores = (randn(2,4,3), randn(3,5,2), randn(2,6,2))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x_dense = x.to_dense(squash_tails=False) # Convert TuckerTensorTrain to dense tensor
        >>> print(x_dense.shape)
        (2, 14, 15, 16, 2)
        >>> ((B0,B1,B2), (G0,G1,G2)) = tucker_cores, tt_cores
        >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
        >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
        1.1217675019342066e-15

        Example with stacking

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(2,3, 4,10), randn(2,3, 5,11), randn(2,3, 6,12))
        >>> tt_cores = (randn(2,3, 2,4,3), randn(2,3, 3,5,2), randn(2,3, 2,6,5))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x_dense = x.to_dense() # Convert TuckerTensorTrain to dense tensor
        >>> ((B0,B1,B2), (G0,G1,G2)) = tucker_cores, tt_cores
        >>> x_dense2 = np.einsum('uvxi,uvyj,uvzk,uvaxb,uvbyc,uvczd->uvijk', B0, B1, B2, G0, G1, G2)
        >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
        1.3614138244072514e-15
        """
        return ragged_operations.to_dense(
            self.data, squash_tails=squash_tails, use_jax=use_jax,
        )

    def segment(self, start: int, stop: int) -> 'TuckerTensorTrain':
        """Extract contiguous segment of TuckerTensorTrain. Requires stop > start.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(4,14), randn(5,15), randn(6,16), randn(7,17))
        >>> tt_cores = (randn(2,4,3), randn(3,5,2), randn(2,6,2), randn(2,7,4))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x01 = x.segment(1,3)
        >>> print(x01.core_shapes)
        (((5, 15), (6, 16)), ((3, 5, 2), (2, 6, 2)))
        """
        if start is None:
            start = 0

        if stop is None:
            stop = self.d

        if start < 0:
            start = self.d + start

        if stop < 0:
            stop = self.d + stop

        if stop <= start:
            raise ValueError(
                "Attempted to extract segment with length < 1.\n"
                + str(start) + ' = start >= stop = ' + str(stop)
            )

        return TuckerTensorTrain(
            self.tucker_cores[start:stop],
            self.tt_cores[start:stop],
        )

    @staticmethod
    def concatenate(
            xx: typ.Sequence['TuckerTensorTrain'],
    ) -> 'TuckerTensorTrain':
        """Concatenates TuckerTensorTrains.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tk = (randn(4,14), randn(5,15), randn(6,16), randn(7,17), randn(8,18), randn(9,19))
        >>> tt = (randn(2,4,3), randn(3,5,2), randn(2,6,2), randn(2,7,3), (randn(3,8,4)), (randn(4,9,1)))
        >>> x = t3.TuckerTensorTrain(tk[:3], tt[:3])
        >>> y = t3.TuckerTensorTrain(tk[3:4], tt[3:4])
        >>> z = t3.TuckerTensorTrain(tk[4:], tt[4:])
        >>> xyz = t3.TuckerTensorTrain.concatenate([x, y, z])
        >>> xyz2 = t3.TuckerTensorTrain(tk, tt)
        >>> print((xyz-xyz2).norm() / xyz.norm())
        1.959150523916366e-15
        """
        if len(xx) < 1:
            raise ValueError(
                'Empty TuckerTensorTrain not supported.\n'
                + str(len(xx)) + ' = len(xx)'
            )
        elif len(xx) == 1:
            return xx[0]
        elif len(xx) == 2:
            x, y = xx[0], xx[1]
            if x.tt_ranks[-1] != y.tt_ranks[0]:
                raise ValueError(
                    'First and last TT-ranks inconsistent for concatenation.\n'
                    + str(x.tt_ranks[-1]) + ' = x.tt_ranks[-1] != y.tt_ranks[0] = ' + str(y.tt_ranks[0])
                )

            return TuckerTensorTrain(
                x.tucker_cores + y.tucker_cores,
                x.tt_cores + y.tt_cores
            )
        else:
            return TuckerTensorTrain.concatenate(
                [TuckerTensorTrain.concatenate(xx[:2])] + xx[2:]
            )

    def squash_tails(
            self,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Make leading and trailing TT ranks equal to 1 (r0=rd=1), without changing tensor being represented.

        Parameters
        ----------
        x : TuckerTensorTrain
            Tucker tensor train with tt_ranks=(r0,r1,...,r(d-1),rd).

        use_jax: bool, defaults to False
            Whether to use Jax for linear algebra. Default: False (use numpy).

        Returns
        -------
        squashed_x: TuckerTensorTrain
            Tucker tensor train with tt_ranks=(1,r1,...,r(d-1),1).

        See Also:
        ---------
        TuckerTensorTrain
        T3Structure

        Examples
        ________
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(2,3, 4,10), randn(2,3, 5,11), randn(2,3, 6,12))
        >>> tt_cores = (randn(2,3, 2,4,3), randn(2,3, 3,5,2), randn(2,3, 2,6,5))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.tt_ranks)
        (2, 3, 2, 5)
        >>> x2 = x.squash_tails()
        >>> print(x2.tt_ranks)
        (1, 3, 2, 1)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense()))
        5.805155892491438e-12
        """
        return TuckerTensorTrain(self.tucker_cores, ragged_operations.squash_tt_tails(self.tt_cores, use_jax=use_jax))

    def reverse(self) -> 'TuckerTensorTrain':
        """Reverse Tucker tensor train.

        Parameters
        ----------
        x : TuckerTensorTrain
            Tucker tensor train with:

                shape=(N0, ..., N(d-1)),

                tucker_ranks=(n0,...,n(d-1)),

                tt_ranks=(1,r1,...,r(d-1),1).

        Returns
        -------
        reversed_x : TuckerTensorTrain
            Tucker tensor train with index order reversed.

                shape=(N(d-1), ..., N0),

                tucker_ranks=(n(d-1),...,n0),

                tt_ranks=(1,r(d-1),...,r1,1).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(2,3, 4,10), randn(2,3, 5,11), randn(2,3, 6,12))
        >>> tt_cores = (randn(2,3, 1,4,2), randn(2,3, 2,5,3), randn(2,3, 3,6,4))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.uniform_structure)
        ((10, 11, 12), (4, 5, 6), (1, 2, 3, 4), (2,3))
        >>> reversed_x = x.reverse()
        >>> print(reversed_x.uniform_structure)
        ((12, 11, 10), (6, 5, 4), (4, 3, 2, 1), (2,3))
        >>> x_dense = x.to_dense()
        >>> reversed_x_dense = reversed_x.to_dense()
        >>> x_dense2 = reversed_x_dense.transpose([0,1, 4,3,2])
        >>> print(np.linalg.norm(x_dense - x_dense2))
        1.859018050214056e-13
        """
        reversed_tucker_cores = tuple([B.copy() for B in self.tucker_cores[::-1]])
        reversed_tt_cores = ragged_operations.reverse_tt(self.tt_cores)
        return TuckerTensorTrain(reversed_tucker_cores, reversed_tt_cores)

    def resize_cores(
            self,
            new_shape: typ.Sequence[int], # len=d
            new_tucker_ranks: typ.Sequence[int], # len=d
            new_tt_ranks: typ.Sequence[int], # len=d+1
    ) -> 'TuckerTensorTrain':
        '''Change cores shapes via zero padding to make cores bigger or truncation to make cores smaller.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,6,5), (1,3,2,1))
        >>> padded_x = x.resize_cores((17,18,17), (8,8,8), (1,5,6,1))
        >>> print(padded_x.structure)
        ((17, 18, 17), (8, 8, 8), (1, 5, 6, 1), ())

        Example where first and last ranks are nonzero:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,6,5), (3,3,2,4))
        >>> padded_x = x.resize_cores((17,18,17), (8,8,8), (5,5,6,7))
        >>> print(padded_x.structure)
        ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7), ())
        '''
        tucker_cores, tt_cores = self.data

        new_tucker_cores = ragged_operations.change_tucker_core_shapes(tucker_cores, new_shape, new_tucker_ranks)
        new_tt_cores = ragged_operations.change_tt_core_shapes(tt_cores, new_tucker_ranks, new_tt_ranks)

        return TuckerTensorTrain(tuple(new_tucker_cores), tuple(new_tt_cores))

    def to_jax(self) -> 'TuckerTensorTrain':
        """Convert arrays defining TuckerTensorTrain into Jax arrays.
        """
        return TuckerTensorTrain(
            tuple(to_jax(B) for B in self.tucker_cores),
            tuple(to_jax(G) for G in self.tt_cores)
        )

    def to_numpy(self) -> 'TuckerTensorTrain':
        """Convert arrays defining TuckerTensorTrain into Numpy arrays.
        """
        return TuckerTensorTrain(
            tuple(to_numpy(B) for B in self.tucker_cores),
            tuple(to_numpy(G) for G in self.tt_cores)
        )

    def copy(self):
        """Copy TuckerTensorTrain.
        """
        return TuckerTensorTrain(
            tuple(B.copy() for B in self.tucker_cores),
            tuple(G.copy() for G in self.tt_cores)
        )

    #### Vectorization / stacking ####

    def unstack(self): # returns an array-like structure of nested tuples containing TuckerTensorTrains
        """If this object contains multiple stacked T3s, this unstacks them
        into an array-like structure of nested tuples with the same "shape" as self.stack_shape.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(3,5))
        >>> unstacked_x = x.unstack()
        >>> print([len(s) for s in unstacked_x])
        [5, 5, 5]
        >>> tucker13 = tuple([B[1,3] for B in x.tucker_cores])
        >>> tt13 = tuple([G[1,3] for G in x.tt_cores])
        >>> x13 = t3.TuckerTensorTrain(tucker13, tt13)
        >>> print((x13 - unstacked_x[1][3]).norm())
        0.0
        """
        def _dfs(xx):
            if is_ndarray(xx[0][0]):
                return TuckerTensorTrain(*xx)
            return tuple([_dfs(x) for x in xx])

        return _dfs(ragged_operations.t3_unstack(self.data))

    @staticmethod
    def stack(
            xx, # array-like structure of nested tuples containing TuckerTensorTrains
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':  # (stacked_tucker_cores, stacked_tt_cores)
        """Stacks an array-like structure of TuckerTensorTrains into one stacked TuckerTensorTrain.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(3,5))
        >>> xx = x.unstack()
        >>> x2 = t3.TuckerTensorTrain.stack(xx)
        >>> print(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)))
        0.0
        """
        def _data(xs):
            if isinstance(xs, TuckerTensorTrain):
                return xs.data
            return tuple([_data(x) for x in xs])
        xx_data = _data(xx)

        stacked_tucker_cores, stacked_tt_cores = ragged_operations.t3_stack(xx_data, use_jax=use_jax)
        return TuckerTensorTrain(stacked_tucker_cores, stacked_tt_cores)

    ############################################################################
    ##########    Constructing specific types of TuckerTensorTrain    ##########
    ############################################################################

    @staticmethod
    def zeros(
            shape: typ.Tuple[int, ...],
            tucker_ranks: typ.Tuple[int, ...] = None,
            tt_ranks: typ.Tuple[int, ...] = None,
            stack_shape: typ.Tuple[int, ...] = (),
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Construct a Tucker tensor train of zeros.

        Parameters
        ----------
        structure:  T3Structure
            Tucker tensor train structure, (shape, tucker_ranks, tt_ranks)=((N0,...,N(d-1)), (n0,...,n(d-1)), (1,r1,...,r(d-1),1))).
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        NDArray
            Dense tensor represented by x, which has shape (N0, ..., N(d-1))

        See Also
        --------
        TuckerTensorTrain
        T3Structure

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> shape = (14, 15, 16)
        >>> tucker_ranks = (4, 5, 6)
        >>> tt_ranks = (1, 3, 2, 1)
        >>> stack_shape = (2,3)
        >>> z = t3.TuckerTensorTrain.zeros(shape, tucker_ranks, tt_ranks, stack_shape)
        >>> print(np.linalg.norm(z.to_dense()))
        0.0
        """
        tucker_ranks = (1,)*len(shape) if tucker_ranks is None else tucker_ranks
        tt_ranks = (1,)*(len(shape)+1) if tt_ranks is None else tt_ranks
        return TuckerTensorTrain(*ragged_operations.t3_zeros(
            shape, tucker_ranks, tt_ranks, stack_shape, use_jax=use_jax,
        ))

    @staticmethod
    def ones(
            shape: typ.Tuple[int, ...],
            stack_shape: typ.Tuple[int, ...] = (),
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Construct the rank-1 Tucker tensor train which represents the dense tensor filled with ones.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> shape = (14, 15, 16)
        >>> stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.ones(shape, stack_shape=stack_shape)
        >>> print(np.linalg.norm(x.to_dense() - np.ones(stack_shape+shape)))
        0.0
        >>> print(x.tucker_ranks)
        (1, 1, 1)
        >>> print(x.tt_ranks)
        (1, 1, 1, 1)
        """
        return TuckerTensorTrain(*ragged_operations.t3_ones(
            shape, stack_shape, use_jax=use_jax,
        ))

    @staticmethod
    def corewise_randn(
            shape: typ.Tuple[int, ...],
            tucker_ranks: typ.Tuple[int, ...],
            tt_ranks: typ.Tuple[int, ...],
            stack_shape: typ.Tuple[int, ...] = (),
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Construct a Tucker tensor train with random cores.

        Parameters
        ----------
        structure:  T3Structure
            Tucker tensor train structure
            (shape, tucker_ranks, tt_ranks)=((N0,...,N(d-1)), (n0,...,n(d-1)), (1,r1,...,r(d-1),1))).
        randn: typ.Callable[[..., NDArray]
            Function for creating random arrays. Arguments are a sequence of ints defining the shape of the array.
            Default: np.random.randn (numpy)

        Returns
        -------
        NDArray
            Dense tensor represented by x, which has shape (N0, ..., N(d-1))

        See Also
        --------
        TuckerTensorTrain
        T3Structure

        Examples
        --------
        >>> from t3toolbox import *
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> shape = (14, 15, 16)
        >>> tucker_ranks = (4, 5, 6)
        >>> tt_ranks = (1, 3, 2, 1)
        >>> stack_shape = (2,3)
        >>> x = t3.t3_corewise_randn(shape, tucker_ranks, tt_ranks, stack_shape=stack_shape) # TuckerTensorTrain with random cores
        >>> x.uniform_structure == (shape, tucker_ranks, tt_ranks, stack_shape)
        True
        >>> print(x.tucker_cores[0][0,0,0,0]) # should be random N(0,1)
        0.0331003310807162
        >>> print(x.tt_cores[0][0,0,0,0,0]) # should be random N(0,1)
        -0.10778923886039414
        """
        return TuckerTensorTrain(*ragged_operations.t3_corewise_randn(
            shape, tucker_ranks, tt_ranks, stack_shape, use_jax=use_jax,
        ))

    #############################################################
    ##########    Converting data to/from 1D vector    ##########
    #############################################################

    def to_vector(
            self,
    ) -> NDArray:
        """Converts a TuckerTensorTrain into a 1D vector containing the core entries.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(2,3))
        >>> x_flat = x.to_vector()
        >>> x2 = t3.TuckerTensorTrain.from_vector(x_flat, x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape)
        >>> print(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)))
        0.0
        """
        return ragged_operations.t3_to_vector(self.data)

    @staticmethod
    def from_vector(
            x_flat: NDArray,
            shape: typ.Sequence[int],
            tucker_ranks: typ.Sequence[int],
            tt_ranks: typ.Sequence[int],
            stack_shape: typ.Sequence[int] = (),
    ) -> 'TuckerTensorTrain':
        """Constructs a TuckerTensorTrain from a 1D vector containing the core entries.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(2,3))
        >>> x_flat = x.to_vector()
        >>> x2 = t3.TuckerTensorTrain.from_vector(x_flat, x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape)
        >>> print(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)))
        0.0
        """
        return TuckerTensorTrain(*ragged_operations.t3_from_vector(
            x_flat, shape, tucker_ranks, tt_ranks, stack_shape=stack_shape,
        ))


    ###############################################################
    ##########    Saving to file and loading from file   ##########
    ###############################################################

    def save(
            self,
            file,
    ) -> None:
        """Save a Tucker tensor train to a file.

        Parameters
        ----------
        file:  str or file
            Either the filename (string) or an open file (file-like object)
            where the data will be saved. If file is a string or a Path, the
            ``.npz`` extension will be appended to the filename if it is not
            already there.
        x: TuckerTensorTrain
            The Tucker tensor train to save

        Raises
        ------
        ValueError
            Error raised if the Tucker tensor train is inconsistent
        RuntimeError
            Error raised if the Tucker tensor train fails to save.

        See Also
        --------
        TuckerTensorTrain
        t3_load
        check_t3

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> fname = 't3_file'
        >>> t3.t3_save(fname, x) # Save to file 't3_file.npz'
        >>> x2 = t3.t3_load(fname) # Load from file
        >>> tucker_cores, tt_cores = x.data
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> print([np.linalg.norm(B - B2) for B, B2 in zip(tucker_cores, tucker_cores2)])
        [0.0, 0.0, 0.0]
        >>> print([np.linalg.norm(G - G2) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
        """
        tucker_cores, tt_cores = self.data
        cores_dict = {'tucker_cores_' + str(ii): tucker_cores[ii] for ii in range(len(tucker_cores))}
        cores_dict.update({'tt_cores_' + str(ii): tt_cores[ii] for ii in range(len(tt_cores))})

        try:
            np.savez(file, **cores_dict)
        except RuntimeError:
            print('Failed to save TuckerTensorTrain to file')

    @staticmethod
    def load(
            file,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Load a Tucker tensor train from a file.

        Parameters
        ----------
        file:  str or file
            Either the filename (string) or an open file (file-like object)
            where the data will be saved. If file is a string or a Path, the
            ``.npz`` extension will be appended to the filename if it is not
            already there.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train loaded from the file

        Raises
        ------
        RuntimeError
            Error raised if the Tucker tensor train fails to load.
        ValueError
            Error raised if the Tucker tensor train fails is inconsistent.

        See Also
        --------
        TuckerTensorTrain
        t3_save
        check_t3

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> fname = 't3_file'
        >>> t3.t3_save(fname, x) # Save to file 't3_file.npz'
        >>> x2 = t3.t3_load(fname) # Load from file
        >>> tucker_cores, tt_cores = x.data
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> print([np.linalg.norm(B - B2) for B, B2 in zip(tucker_cores, tucker_cores2)])
        [0.0, 0.0, 0.0]
        >>> print([np.linalg.norm(G - G2) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
        """
        xnp, _, _ = get_backend(False, use_jax)

        #
        if isinstance(file, str):
            if not file.endswith('.npz'):
                file = file + '.npz'

        try:
            d = np.load(file)
        except RuntimeError:
            print('Failed to load TuckerTensorTrain from file')

        assert (len(d.files) % 2 == 0)
        num_cores = len(d.files) // 2
        tucker_cores = [d['tucker_cores_' + str(ii)] for ii in range(num_cores)]
        tt_cores = [d['tt_cores_' + str(ii)] for ii in range(num_cores)]

        tucker_cores = [xnp.array(B) for B in tucker_cores]  # in case we are using jax or some other linalg backend
        tt_cores = [xnp.array(G) for G in tt_cores]

        return TuckerTensorTrain(tuple(tucker_cores), tuple(tt_cores))

    ##########################################
    ##########    Linear Algebra    ##########
    ##########################################

    def __add__(
            self,
            other,
    ):
        """Add Tucker tensor trains x and y, yielding a Tucker tensor train x+y with summed ranks.

        Addition is defined with respect to the dense N0 x ... x N(d-1) tensors that
        are *represented* by the Tucker tensor trains.

        For corewise addition, see :func:`t3toolbox.corewise.corewise_add`

        T3 + T3 -> T3
        T3 + dense -> dense
        T3 + scalar -> T3

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> z = x + y
        >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - z.to_dense()))
        6.524094086845177e-13
        >>> print(z.structure)
        ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1), ())

        Adding T3 + dense

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = np.random.randn(14,15,16)
        >>> z = x + y
        >>> print(np.linalg.norm(x.to_dense() + y - z))
        0.0
        >>> print(type(z))
        <class 'numpy.ndarray'>

        Adding T3 + scalar

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> s = 3.5
        >>> z = x + s
        >>> print(np.linalg.norm(x.to_dense() + s - z.to_dense()))
        0.0
        >>> print(z.structure)
        ((14, 15, 16), (5, 6, 7), (2, 4, 3, 2), ())
        """
        if isinstance(other, TuckerTensorTrain):
            if self.shape != other.shape:
                raise ValueError(
                    'Attempted to add TuckerTensorTrains self+other with inconsistent shapes.'
                    + str(self.shape) + ' = self.shape != other.shape = ' + str(other.shape)
                )
            if self.stack_shape != other.stack_shape:
                raise NotImplementedError(
                    'Cannot add TuckerTensorTrains with different stack shapes.\n'
                    + str(self.stack_shape)
                    + ' = self.stack_shape != other.stack_shape = '
                    + str(other.stack_shape)
                )
            return TuckerTensorTrain(*ragged_linalg.t3_add(self.data, other.data))

        elif is_ndarray(other):
            if other.shape == (): # scalar "array"
                return TuckerTensorTrain(*ragged_linalg.t3_plus_scalar(self.data, other))

            if self.stack_shape + self.shape != other.shape:
                raise ValueError(
                    'Attempted to add TuckerTensorTrain self to array other with inconsistent shapes.'
                    + str(self.stack_shape + self.shape) + ' = self.stack_shape + self.shape != other.shape = ' + str(other.shape)
                )
            return self.to_dense() + other

        else: # assume other is a scalar
            return TuckerTensorTrain(*ragged_linalg.t3_plus_scalar(self.data, other))

    def __mul__(
            self,
            other,  # scalar
            use_jax: bool = None, # None: automatically decide based on input types
    ):
        """Pointwise multiplication of a Tucker tensor train.

        Scaling is defined with respect to the dense N0 x ... x N(d-1) tensor that
        is *represented* by the Tucker tensor trains.

        For corewise scaling, see :func:`t3toolbox.corewise.corewise_scale`

        Parameters
        ----------
        x: TuckerTensorTrain
            Tucker tensor train
        s: scalar
            scaling factor

        Returns
        -------
        TuckerTensorTrain
            Scaled TuckerTensorTrain s*x, with the same structure as x.

        Raises
        ------
        ValueError
            - Error raised if the TuckerTensorTrains are internally inconsistent

        See Also
        --------
        TuckerTensorTrain
        t3_add
        t3_neg
        t3_sub
        :func:`~t3toolbox.corewise.corewise_scale`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> s = 3.2
        >>> sx = x * s
        >>> print(np.linalg.norm(s*x.to_dense() - sx.to_dense()))
        1.6268482531988893e-13

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> y = np.random.randn(*(x.stack_shape + x.shape))
        >>> xy = x * y
        >>> print(np.linalg.norm(x.to_dense()*y - xy))
        0.0

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (2,3,4), (3,2,3,2), stack_shape=(2,3))
        >>> xy = x * y
        >>> print(np.linalg.norm(x.to_dense()*y.to_dense() - xy.to_dense()))
        8.556292929330887e-11
        >>> print(xy.tucker_ranks) # ranks get multiplied!
        (8, 15, 24)
        >>> print(xy.tt_ranks)
        (1, 6, 6, 1)
        """
        if is_ndarray(other):
            if other.shape == ():
                return TuckerTensorTrain(*ragged_linalg.t3_scale(self.data, other))
            else:
                assert(other.shape == self.stack_shape + self.shape)
                return self.to_dense() * other

        elif isinstance(other, TuckerTensorTrain):
            assert(self.shape == other.shape)
            assert(self.stack_shape == other.stack_shape)
            return TuckerTensorTrain(*ragged_linalg.t3_mult(self.data, other.data))

        else: # assume scalar
            return TuckerTensorTrain(*ragged_linalg.t3_scale(self.data, other))

    def __neg__(
            self,
    ) -> 'TuckerTensorTrain':
        """Scale a Tucker tensor train by -1.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> neg_x = -x
        >>> print(np.linalg.norm(x.to_dense() + neg_x.to_dense()))
        0.0
        """
        return self * (-1.0)

    def __sub__(
            self,
            other,
    ) -> 'TuckerTensorTrain':
        """Subtract Tucker tensor trains, x - y, yielding a Tucker tensor train with summed ranks.

        Subtraction is defined with respect to the dense N0 x ... x N(d-1) tensors that
        are *represented* by the Tucker tensor trains.

        For corewise subtraction, see :func:`t3toolbox.corewise.corewise_sub`
        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> z = x - y
        >>> print(np.linalg.norm(x.to_dense() - y.to_dense() - z.to_dense()))
        0.0
        >>> print(z.structure)
        ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2), ())

        T3 - dense

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = np.random.randn(14,15,16)
        >>> z = x - y
        >>> print(np.linalg.norm(x.to_dense() - y - z))
        0.0
        >>> print(type(z))
        <class 'numpy.ndarray'>

        T3 - scalar

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> s = 3.5
        >>> z = x - s
        >>> print(np.linalg.norm(x.to_dense() - s - z.to_dense()))
        0.0
        >>> print(z.structure)
        ((14, 15, 16), (5, 6, 7), (2, 4, 3, 2), ())
        """
        return self + (-other)

    @staticmethod
    def inner_product(
            x,
            y,
            use_orthogonalization: bool = True,  # for numerical stability
    ):
        """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.

        The Hilbert-Schmidt inner product is defined with respect to the dense N0 x ... x N(d-1)
        tensors that are *represented* by the Tucker tensor trains.

        For corewise dot product, see :func:`t3toolbox.corewise.corewise_dot`

        Parameters
        ----------
        x: TuckerTensorTrain
            First Tucker tensor train. shape=(N0,...,N(d-1))
        y: TuckerTensorTrain
            Second Tucker tensor train. shape=(N0,...,N(d-1))
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        scalar
            Hilbert-Schmidt inner product of Tucker tensor trains, (x, y)_HS.

        Raises
        ------
        ValueError
            - Error raised if either of the TuckerTensorTrains are internally inconsistent
            - Error raised if the TuckerTensorTrains have different shapes.

        See Also
        --------
        TuckerTensorTrain
        t3_shape
        t3_add
        t3_scale
        :func:`~t3toolbox.corewise.corewise_dot`

        Notes
        -----
        Algorithm contracts the TuckerTensorTrains in a zippering fashion from left to right.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (3,7,2), (3,5,6,3))
        >>> x_dot_y = t3.TuckerTensorTrain.inner_product(x, y)
        >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense())
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        1.3096723705530167e-10

        (T3, T3) using stacking:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (3,7,2), (3,5,6,3), stack_shape=(2,3))
        >>> x_dot_y = t3.TuckerTensorTrain.inner_product(x, y)
        >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense(), axis=(2,3,4))
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        2.7761383858792984e-09

        Inner product of T3 with dense:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = np.random.randn(14,15,16)
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (3,7,2), (3,5,6,3))
        >>> x_dot_y = t3.TuckerTensorTrain.inner_product(x, y)
        >>> x_dot_y2 = np.sum(x * y.to_dense())
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        0.0

        Inner product of T3 with dense including stacking:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = np.random.randn(2,3, 14,15,16)
        >>> y = t3.TuckerTensorTrain.corewise_randn((14,15,16), (3,7,2), (3,5,6,3), stack_shape=(2,3))
        >>> x_dot_y = t3.TuckerTensorTrain.inner_product(x, y)
        >>> x_dot_y2 = np.einsum('ijxyz,ijxyz->ij', x, y.to_dense())
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        1.2014283869232628e-11
        """
        # If anything has jax in it, we use jax.
        use_jax = False
        if is_ndarray(x):
            use_jax = use_jax or is_jax_ndarray(x)
        else:
            use_jax = use_jax or any([is_jax_ndarray(c) for c in x.tucker_cores + x.tt_cores])

        if is_ndarray(y):
            use_jax = use_jax or is_jax_ndarray(y)
        else:
            use_jax = use_jax or any([is_jax_ndarray(c) for c in y.tucker_cores + y.tt_cores])

        xnp, _, _ = get_backend(False, use_jax)

        #
        if isinstance(x, TuckerTensorTrain) and isinstance(y, TuckerTensorTrain):
            if x.shape != y.shape:
                raise ValueError(
                    'Attempted to take inner product of TuckerTensorTrains (x,y) with inconsistent shapes.'
                    + str(x.shape) + ' = x.shape != y.shape = ' + str(y.shape)
                )

            if x.stack_shape != y.stack_shape:
                raise NotImplementedError(
                    'Cannot take inner product of TuckerTensorTrains with different stack shapes.\n'
                    + str(x.stack_shape)
                    + ' = x.stack_shape != y.stack_shape = '
                    + str(y.stack_shape)
                )

            return ragged_linalg.t3_inner_product_t3(
                x.data, y.data, use_orthogonalization=use_orthogonalization,
            )

        elif is_ndarray(x) and isinstance(y, TuckerTensorTrain):  # Could be done better with zippering
            if x.shape != y.stack_shape + y.shape:
                raise ValueError(
                    'Attempted to take inner product of array x with TuckerTensorTrain y with inconsistent shapes.'
                    + str(x.shape) + ' = x.shape != y.stack_shape + y.shape = ' + str(y.stack_shape + y.shape)
                )
            contraction_inds = tuple(range(len(y.stack_shape), len(x.shape)))
            contraction_inds = contraction_inds if contraction_inds else None

            return xnp.sum(x * y.to_dense(), axis=contraction_inds)

        elif isinstance(x, TuckerTensorTrain) and is_ndarray(y):
            return TuckerTensorTrain.inner_product(y, x)

        else:
            raise NotImplementedError(
                'T3 inner product only implemented for (x, y) in: {(T3, T3), (T3, dense), (dense, T3)}.\n'
                + 'type(x) = ' + str(type(x)) + '\n'
                + 'type(y) = ' + str(type(y))
            )


    def norm(
            self,
            use_orthogonalization: bool = True, # for numerical stability
    ):
        """Compute Hilbert-Schmidt (Frobenius) norm of a Tucker tensor train.

        The Hilbert-Schmidt norm is defined with respect to the dense N0 x ... x N(d-1) tensor
        that is *represented* by the Tucker tensor trains, even though this dense tensor
        is not formed during computations.

        For corewise norm, see :func:`t3toolbox.corewise.corewise_norm`

        Parameters
        ----------
        x: TuckerTensorTrain
            First Tucker tensor train. shape=(N0,...,N(d-1))
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        scalar
            Hilbert-Schmidt (Frobenius) norm of Tucker tensor trains, ||x||_HS

        Raises
        ------
        ValueError
            - Error raised if the TuckerTensorTrain is internally inconsistent

        See Also
        --------
        TuckerTensorTrain
        t3_dot_t3
        :func:`t3toolbox.corewise.corewise_norm`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> print(x.norm() - np.linalg.norm(x.to_dense()))
        9.094947017729282e-13

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
        >>> norms_x = x.norm(use_orthogonalization=True)
        >>> x_dense = x.to_dense()
        >>> norms_x_dense = np.sqrt(np.sum(x_dense**2, axis=(-3,-2,-1)))
        >>> print(norms_x - norms_x_dense)
        [[-1.36424205e-12 -2.50111043e-12  1.36424205e-12]
         [ 1.59161573e-12  4.09272616e-12  2.72848411e-12]]
        """
        return ragged_linalg.t3_norm(
            self.data, use_orthogonalization=use_orthogonalization,
        )

    ##########################################
    ########    Orthogonalization    #########
    ##########################################

    def down_svd_tucker_core(
            self,
            ii: int,  # which Tucker core to orthogonalize
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
    ) -> typ.Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # ss_x. singular values
    ]:
        '''Compute SVD of ith tucker core and contract non-orthogonal factor into the TT-core above.

        Parameters
        ----------
        ii: int
            index of tucker core to SVD
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (r0,r1,...r(d-1),rd))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor, but with ith tucker core orthogonal.
            new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
            new_tucker_cores[ii].shape = (new_ni, Ni)
            new_tucker_cores[ii] @ new_tucker_cores[ii].T = identity matrix
        ss_x: NDArray
            Singular values of prior ith tucker core. shape=(new_ni,).

        See Also
        --------
        truncated_svd
        left_svd_ith_tt_core
        right_svd_ith_tt_core
        up_svd_ith_tt_core
        down_svd_ith_tt_core
        t3_svd

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.down_svd_tucker_core(ind)
        >>> print(np.linalg.norm(x.to_dense() - x.to_dense())) # Tensor unchanged
        0.0
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> rank = len(ss)
        >>> B = tucker_cores2[ind]
        >>> print(np.linalg.norm(B @ B.T - np.eye(rank))) # Tucker core is orthogonal
        8.456498415401757e-16
        '''
        result = ragged_orthogonalization.down_svd_tucker_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1]

    def left_svd_tt_core(
            self,
            ii: int,  # which tt core to orthogonalize
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
    ) -> typ.Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(r(i+1),)
    ]:
        '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.

        Stacking not supported: the truncated ranks vary based on this T3's numerical properties.

        Parameters
        ----------
        ii: int
            index of TT-core to SVD
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
            new_tt_cores[ii].shape = (ri, ni, new_r(i+1))
            new_tt_cores[ii+1].shape = (new_r(i+1), n(i+1), r(i+2))
            einsum('iaj,iak->jk', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
        ss_x: NDArray
            Singular values of prior ith TT-core left unfolding. shape=(new_r(i+1),).

        See Also
        --------
        truncated_svd
        left_svd_3tensor
        up_svd_ith_tucker_core
        right_svd_ith_tt_core
        up_svd_ith_tt_core
        down_svd_ith_tt_core
        t3_svd

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.left_svd_tt_core(ind)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
            5.186463661974644e-13
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', G, G) - np.eye(G.shape[2]))) # TT-core is left-orthogonal
            4.453244025338311e-16
        '''
        result = ragged_orthogonalization.left_svd_tt_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1]

    def right_svd_tt_core(
            self,
            ii: int,  # which tt core to orthogonalize
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
    ) -> typ.Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(new_ri,)
    ]:
        '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.

        Stacking not supported: the truncated ranks vary based on this T3's numerical properties.

        Parameters
        ----------
        ii: int
            index of TT-core to SVD
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra core. Default: np (numpy)

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
            new_tt_cores[ii].shape = (new_ri, ni, r(i+1))
            new_tt_cores[ii-1].shape = (r(i-1), n(i-1), new_ri)
            einsum('iaj,kaj->ik', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
        ss_x: NDArray
            Singular values of prior ith TT-core right unfolding. shape=(new_ri,).

        See Also
        --------
        truncated_svd
        left_svd_3tensor
        up_svd_ith_tucker_core
        left_svd_ith_tt_core
        up_svd_ith_tt_core
        down_svd_ith_tt_core
        t3_svd

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.right_svd_tt_core(ind)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        5.304678679078675e-13
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', G, G) - np.eye(G.shape[0]))) # TT-core is right orthogonal
        4.207841813173725e-16
        '''
        result = ragged_orthogonalization.right_svd_tt_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1]

    def down_svd_tt_core(
            self,
            ii: int,  # which tt core to orthogonalize
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
    ) -> typ.Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(new_ni,)
    ]:
        '''Compute SVD of ith TT-core down unfolding and keep non-orthogonal factor with this core.

        Stacking not supported: the truncated ranks vary based on this T3's numerical properties.

        Parameters
        ----------
        ii: int
            index of TT-core to SVD
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor.
            new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
            new_tucker_cores[ii].shape = (new_ni, Ni)
        ss_x: NDArray
            Singular values of prior ith TT-core down unfolding. shape=(new_ri,).

        See Also
        --------
        truncated_svd
        outer_svd_3tensor
        up_svd_ith_tucker_core
        left_svd_ith_tt_core
        right_svd_ith_tt_core
        down_svd_ith_tt_core
        t3_svd

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x2, ss = x.down_svd_tt_core(1)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        1.002901486286745e-12
        '''
        result = ragged_orthogonalization.up_svd_tt_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1]

    def up_svd_tt_core(
            self,
            ii: int,  # which tt core to orthogonalize
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
    ) -> typ.Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(new_ni,)
    ]:
        '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below.

        Stacking not supported: the truncated ranks vary based on this T3's numerical properties.

        Parameters
        ----------
        ii: int
            index of TT-core to SVD
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor, but with ith TT-core down orthogonal.
            new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
            new_tucker_cores[ii].shape = (new_ni, Ni)
            einsum('iaj,ibj->ab', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
        ss_x: NDArray
            Singular values of prior ith TT-core down unfolding. shape=(new_ni,).

        See Also
        --------
        truncated_svd
        outer_svd_3tensor
        up_svd_ith_tucker_core
        left_svd_ith_tt_core
        right_svd_ith_tt_core
        up_svd_ith_tt_core
        t3_svd

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.up_svd_tt_core(ind)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        4.367311712704942e-12
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', G, G) - np.eye(G.shape[1]))) # TT-core is down orthogonal
        1.0643458053135608e-15
        '''
        result = ragged_orthogonalization.down_svd_tt_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1]

    ####

    def orthogonalize_relative_to_ith_tucker_core(
            self,
            ii: int,
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        '''Orthogonalize all cores in the TuckerTensorTrain except for the ith tucker core.

        Stacking not supported: the truncated ranks vary based on this T3's numerical properties.

        Orthogonal is done relative to the ith tucker core:
            - ith tucker core is not orthogonalized
            - All other tucker cores are orthogonalized.
            - TT-cores to the left are left orthogonalized.
            - TT-core directly above is outer orthogonalized.
            - TT-cores to the right are right orthogonalized.

        Parameters
        ----------
        ii: int
            index of tucker core that is not orthogonalized
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith tucker core.

        See Also
        --------
        up_svd_ith_tucker_core
        left_svd_ith_tt_core
        right_svd_ith_tt_core
        up_svd_ith_tt_core
        down_svd_ith_tt_core

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x2 = x.orthogonalize_relative_to_tucker_core(1)
        >>> print(np.linalg.norm(x.to_dense(x) - x2.to_dense(x2))) # Tensor unchanged
        8.800032152216517e-13
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> X = np.einsum('xi,axb,byc,czd,zk->iyk', B0, G0, G1, G2, B2) # Contraction of everything except B1
        >>> print(np.linalg.norm(np.einsum('iyk,iwk->yw', X, X) - np.eye(B1.shape[0]))) # Complement of B1 is orthogonal
        1.7116160385376214e-15

        Example where first and last TT-ranks are not 1:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> x2 = x.orthogonalize_relative_to_tucker_core(0)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        5.152424496985265e-12
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> X = np.einsum('yj,zk,axb,byc,czd->axjkd', B1, B2, G0, G1, G2) # Contraction of everything except B0
        >>> print(np.linalg.norm(np.einsum('axjkd,ayjkd->xy', X, X) - np.eye(B0.shape[0]))) # Complement of B1 is orthogonal
        2.3594586449868743e-15
        '''
        return TuckerTensorTrain(*ragged_orthogonalization.orthogonalize_relative_to_tucker_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol, use_jax=use_jax,
        ))

    def orthogonalize_relative_to_ith_tt_core(
            self,
            ii: int,
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.

        Stacking not supported: the truncated ranks vary based on this T3's numerical properties.

        Orthogonal is done relative to the ith TT-core:
            - All Tucker cores are orthogonalized.
            - TT-cores to the left are left orthogonalized.
            - ith TT-core is not orthogonalized.
            - TT-cores to the right are right orthogonalized.

        Parameters
        ----------
        ii: int
            index of TT-core that is not orthogonalized
        x: TuckerTensorTrain
            The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
        min_rank: int
            Minimum rank for truncation.
        min_rank: int
            Maximum rank for truncation.
        rtol: float
            Relative tolerance for truncation.
        atol: float
            Absolute tolerance for truncation.
        xnp:
            Linear algebra backend. Default: np (numpy)

        See Also
        --------
        up_svd_ith_tucker_core
        left_svd_ith_tt_core
        right_svd_ith_tt_core
        up_svd_ith_tt_core
        down_svd_ith_tt_core

        Returns
        -------
        new_x: NDArray
            New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith TT-core.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x2 = x.orthogonalize_relative_to_tt_core(1)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        8.800032152216517e-13
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> XL = np.einsum('axb,xi -> aib', G0, B0) # Everything to the left of G1
        >>> print(np.linalg.norm(np.einsum('aib,aic->bc', XL, XL) - np.eye(G1.shape[0]))) # Left subtree is left orthogonal
        9.820411604510197e-16
        >>> print(np.linalg.norm(np.einsum('xi,yi->xy', B1, B1) - np.eye(G1.shape[1]))) # Core below G1 is up orthogonal
        2.1875310121178e-15
        >>> XR = np.einsum('axb,xi->aib', G2, B2) # Everything to the right of G1
        >>> print(np.linalg.norm(np.einsum('aib,cib->ac', XR, XR) - np.eye(G1.shape[2]))) # Right subtree is right orthogonal
        1.180550381921849e-15

        Example where first and last TT-ranks are not 1:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> x2 = x.orthogonalize_relative_to_tt_core(0)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        5.4708999671349535e-12
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> XR = np.einsum('yi,zj,byc,czd->bijd', B1, B2, G1, G2) # Everything to the right of G0
        >>> print(np.linalg.norm(np.einsum('bijd,cijd->bc', XR, XR) - np.eye(G0.shape[2]))) # Right subtree is right orthogonal
        8.816596607002667e-16
        '''
        return TuckerTensorTrain(*ragged_orthogonalization.orthogonalize_relative_to_tt_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol, use_jax=use_jax,
        ))

    def up_orthogonalize_tucker_cores(
            self,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.up_orthogonalize_tucker_cores()
        >>> print((x - x_orth).norm())
        4.420285752780219e-12
        >>> ind = 1
        >>> B = x_orth.data[0][ind]
        >>> print(np.linalg.norm(B @ B.T - np.eye(B.shape[0])))
        1.2059032102772812e-15

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.up_orthogonalize_tucker_cores()
        >>> print((x - x_orth).norm())
        [[2.27267321e-12 1.92787570e-12 1.60830015e-12]
         [9.54262022e-13 1.45211899e-12 3.27867574e-12]]
        >>> ind = 1
        >>> B = x_orth.data[0][ind]
        >>> BtB = np.einsum('abio,abjo->abij',B,B)
        >>> errs = [[np.linalg.norm(BtB[ii,jj] - np.eye(BtB.shape[-1])) for jj in range(3)] for ii in range(2)]
        >>> print(np.linalg.norm(errs))
        4.118375471407983e-15
        """
        return TuckerTensorTrain(*ragged_orthogonalization.up_orthogonalize_tucker_cores(self.data, use_jax=use_jax))

    def down_orthogonalize_tt_cores(
            self,
            use_jax: bool = False,
    ):
        """Outer orthogonalize TT cores, pushing remainders downward onto tucker cores below.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.down_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        1.927414448489825e-12
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab',G,G)-np.eye(G.shape[1])))
        1.9491561709929213e-15

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.down_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        [[1.65714673e-12 1.52503536e-12 2.94647811e-12]
         [1.56839190e-12 2.61963262e-12 8.78269349e-12]]
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> GdG = np.einsum('xyaib,xyajb->xyij',G,G)
        >>> errs = [[np.linalg.norm(GdG[ii,jj] - np.eye(GdG.shape[-1])) for jj in range(3)] for ii in range(2)]
        >>> print(np.linalg.norm(errs))
        4.0492695830155885e-15
        """
        return TuckerTensorTrain(
            *ragged_orthogonalization.down_orthogonalize_tt_cores(self.data, use_jax=use_jax),
        )

    def left_orthogonalize_tt_cores(
            self,
            return_variation_cores: bool = False,
            use_jax: bool = False,
    ):
        """Left orthogonalize the TT cores, possibly returning variation cores as well.

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.left_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        2.9839379127106095e-12
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('iaj,iak->jk',G,G)-np.eye(G.shape[2])))
        1.3526950544911367e-16

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.left_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        [[1.46128743e-12 1.25202737e-12 5.60494449e-13]
         [9.77331695e-13 2.50200307e-12 3.07559340e-12]]
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('xyiaj,xyiak->xyjk',G,G)-np.eye(G.shape[-1])))
        9.02970295614302e-16
        """
        result = orth.left_orthogonalize_tt_cores(
            self.tt_cores, return_variation_cores=return_variation_cores, use_jax=use_jax,
        )
        if return_variation_cores:
            return TuckerTensorTrain(self.tucker_cores, result[0]), result[1]
        else:
            return TuckerTensorTrain(self.tucker_cores, result)

    def right_orthogonalize_tt_cores(
            self,
            return_variation_cores: bool = False,
            use_jax: bool = False,
    ):
        """Right orthogonalize the TT cores, possibly returning variation cores as well.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.right_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        2.9839379127106095e-12
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('iaj,kaj->jk',G,G)-np.eye(G.shape[0])))
        1.3526950544911367e-16

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.left_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        [[1.33512640e-12 1.84518324e-12 6.79235325e-13]
         [1.34334400e-12 3.38154895e-12 2.93760867e-12]]
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('xyiaj,xyiak->xyjk',G,G)-np.eye(G.shape[-1])))
        1.3585381944466237e-15
        """
        result = orth.right_orthogonalize_tt_cores(
            self.tt_cores, return_variation_cores=return_variation_cores, use_jax=use_jax,
        )
        if return_variation_cores:
            return TuckerTensorTrain(self.tucker_cores, result[0]), result[1]
        else:
            return TuckerTensorTrain(self.tucker_cores, result)

    #######################################################
    ##########    Entries, Apply, and Probing    ##########
    #######################################################

    def get_entries(
            self,
            index: NDArray,  # or convertible to NDArray. dtype=int
            use_jax: bool = False,
    ) -> NDArray:
        '''Compute an entry (or multiple entries) of a Tucker tensor train.

        See Also:
        ---------
        t3_apply

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> index = [[9,8], [4,10], [7,13]] # get entries (9,4,7) and (8,10,13)
        >>> entries = x.get_entries(index)
        >>> x_dense = x.to_dense(x)
        >>> entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
        >>> print(np.linalg.norm(entries - entries2))
        1.7763568394002505e-15
        '''
        return t3_entries(self, index, use_jax=use_jax)

    def apply(
            self,
            vecs: typ.Sequence[NDArray], # len=d, elm_shape=(Ni,) or (num_applies, Ni)
            use_jax: bool = False,
    ) -> NDArray:
        '''Contract a Tucker tensor train with vectors in all indices.

        See Also
        --------
        t3_get_entries

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
        >>> result = x.apply(vecs)
        >>> result2 = np.einsum('ijk,ni,nj,nk->n', x.to_dense(), vecs[0], vecs[1], vecs[2])
        >>> print(np.linalg.norm(result - result2))
        3.1271953680324864e-12
        '''
        return t3_apply(self, vecs, use_jax=use_jax)

    def probe(
            self,
            ww: typ.Sequence[NDArray], # len=d, elm_shape=W+(Ni,)
            use_jax: bool=False,
    ) -> typ.Sequence[NDArray]: # zz, len=d, elm_shape=X+W+(Ni,)
        """Probe a TuckerTensorTrain.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.backend.probing as probing
        >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zz = x.probe(ww)
        >>> x_dense = x.to_dense()
        >>> zz2 = probing.probe_dense(ww, x_dense)
        >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
        [1.0259410400851746e-12, 1.0909087370186656e-12, 3.620283224238675e-13]
        """
        return probing.probe_t3(ww, self.data, use_jax=use_jax)



if has_jax:
    jax.tree_util.register_pytree_node(
        TuckerTensorTrain,
        lambda x: (x.data, None),
        lambda aux_data, children: TuckerTensorTrain(*children),
    )


####


def t3_core_shapes(
        shape: typ.Sequence[int],
        tucker_ranks: typ.Sequence[int],
        tt_ranks: typ.Sequence[int],
        stack_shape: typ.Sequence[int] = (),
) -> typ.Tuple[
    typ.Tuple[int,...], # tucker_core_shapes
    typ.Tuple[int,...], # tt_core_shapes
]:
    """Compute the tucker and TT core shapes for a Tucker tensor train.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.corewise as cw
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(9,))
    >>> print(t3.t3_core_shapes(x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape))
    (((9, 4, 14), (9, 5, 15), (9, 6, 16)), ((9, 1, 4, 3), (9, 3, 5, 4), (9, 4, 6, 5)))
    >>> print(x.core_shapes)
    (((9, 4, 14), (9, 5, 15), (9, 6, 16)), ((9, 1, 4, 3), (9, 3, 5, 4), (9, 4, 6, 5)))
    """
    return ragged_operations.t3_core_shapes(
        shape, tucker_ranks, tt_ranks, stack_shape,
    )


def compute_minimal_t3_ranks(
        shape:          typ.Sequence[int],
        tucker_ranks:   typ.Sequence[int],
        tt_ranks:       typ.Sequence[int],
) -> typ.Tuple[
    typ.Tuple[int,...], # new_tucker_ranks
    typ.Tuple[int,...], # new_tt_ranks
]:
    '''Find minimal ranks for a generic Tucker tensor train with a given structure.

    Minimal ranks satisfy:
        - Left TT core unfoldings are full rank: r(i+1) <= (ri*ni)
        - Right TT core unfoldings are full rank: ri <= (ni*r(i+1))
        - Outer TT core unfoldings are full rank: ni <= (ri*r(i+1))
        - Basis matrices have full row rank: ni <= Ni

    In this function, minimal ranks are defined with respect to a
    generic Tucker tensor train of the given form based on its structure.
    We do not account for possible additional rank deficiency due to
    the numerical values within the cores.

    Minimal ranks always exist and are unique.
        - Minimal TT ranks are equal to the ranks of (N*...*Ni) x (N(i+1)*...*N(d-1)) matrix unfoldings.
        - Minimal Tucker ranks are equal to the ranks of Ni x (N1*...*N(i-1)*N(i+1)*...*N(d-1)) matricizations.

    Examples
    --------
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> print(t3.compute_minimal_t3_ranks((10,11,12,13), (14,15,16,17), (98,99,100,101,102)))
    ((10, 11, 12, 13), (1, 10, 100, 13, 1))
    '''
    return ranks.compute_minimal_ranks(shape, tucker_ranks, tt_ranks)

#


###########################################################################
########    Scalar valued M.L.F. applies, entries, and probing    #########
###########################################################################

def t3_apply(
        x: TuckerTensorTrain, # shape=(N0,...,N(d-1))
        vecs: typ.Sequence[NDArray], # len=d, elm_shape=V+(Ni,)
        use_jax: bool = False,
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train. shape=(N0,...,N(d-1))
    vecs: typ.Sequence[NDArray]
        Vectors to contract with indices of x. len=d, elm_shape=stack_shape+(Ni,)
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    NDArray or scalar
        Result of contracting x with the vectors in all indices.
        scalar if vecs elements are vectors, NDArray with shape (num_applies,) if vecs elements are matrices.

    Raises
    ------
    ValueError
        Error raised if the provided vectors in vecs are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_entry

    Examples
    --------

    Apply to one set of vectors:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> vecs = [np.random.randn(14), np.random.randn(15), np.random.randn(16)]
    >>> result = t3.apply(x, vecs) # <-- contract x with vecs in all indices
    >>> result2 = np.einsum('ijk,i,j,k', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.abs(result - result2))
    5.229594535194337e-12

    Apply to multiple sets of vectors (vectorized):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    3.1271953680324864e-12

    Apply to tensor sets of vectors and tensor sets of T3s (supervectorized)

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
    >>> vecs = [np.random.randn(4, 14), np.random.randn(4, 15), np.random.randn(4, 16)]
    >>> result = t3.apply(x, vecs)
    >>> result2 = np.einsum('uvijk,xi,xj,xk->uvx', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    3.1271953680324864e-12

    First and last TT-ranks are not ones:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,4))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    6.481396196459234e-12

    Example using jax automatic differentiation:

	>>> import numpy as np
    >>> import jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> jax.config.update("jax_enable_x64", True)
    >>> A = t3.t3_corewise_randn((10,10,10),(5,5,5),(1,4,4,1)) # random 10x10x10 Tucker tensor train
    >>> apply_A_sym = lambda u: t3.apply(A, (u,u,u), use_jax=True) # symmetric apply function
    >>> u0 = np.random.randn(10)
    >>> Auuu0 = apply_A_sym(u0)
    >>> g0 = jax.grad(apply_A_sym)(u0) # gradient using automatic differentiation
    >>> du = np.random.randn(10)
    >>> dAuuu = np.dot(g0, du) # derivative in direction du
    >>> print(dAuuu)
    766.5390335764645
    >>> s = 1e-7
    >>> u1 = u0 + s*du
    >>> Auuu1 = apply_A_sym(u1)
    >>> dAuuu_diff = (Auuu1 - Auuu0) / s # finite difference approximation
    >>> print(dAuuu_diff) #ths same as dAuuu
    766.5390504030256
    '''
    xnp, _, _ = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x.data
    shape, _, _, _ = x.structure

    if len(vecs) != len(shape):
        raise ValueError(
            'Attempted to apply TuckerTensorTrain to wrong number of vectors.'
            + str(str(len(shape)) + ' = num_indices != len(vecs) = ' + str(len(vecs)))
        )

    result = apply.t3_apply(x.data, vecs, use_jax=use_jax)

    return result


def t3_entries(
        x: TuckerTensorTrain, # shape=(N0,...,N(d-1))
        index: NDArray, # or convertible to NDArray. dtype=int
        use_jax: bool = False,
) -> NDArray:
    '''Compute an entry (or multiple entries) of a Tucker tensor train.

    This is the entry of the N0 x ... x N(d-1) tensor *represented* by the
    Tucker tensor train, even though this dense tensor is never formed.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train. shape=(N0,...,N(d-1))
    index: NDArray or convertible to NDArray, dtype=int
        Indices of desired entries. shape=(d,)+vsi
        len(index)=d. If many entries: elm_size=num_entries
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    scalar or NDArray
        Desired entries.shape=(vsi,)

    Raises
    ------
    ValueError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_apply

    Examples
    --------

    Compute one entry:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> index = [9, 4, 7] # get entry (9,4,7)
    >>> result = t3.t3_entries(x, index)
    >>> result2 = x.to_dense()[9, 4, 7]
    >>> print(np.abs(result - result2))
    1.3322676295501878e-15

    Compute multiple entries (vectorized):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> index = [[9,8], [4,10], [7,13]] # get entries (9,4,7) and (8,10,13)
    >>> entries = t3.t3_entries(x, index)
    >>> x_dense = x.to_dense(x)
    >>> entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
    >>> print(np.linalg.norm(entries - entries2))
    1.7763568394002505e-15

    Vectorize over entries AND T3s

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
    >>> index = [[9,0], [4,0], [7,0]] # get entries (9,4,7) and (0,0,0)
    >>> entries = t3.t3_entries(x, index)
    >>> x_dense = x.to_dense()
    >>> entries2 = np.moveaxis(np.array([x_dense[:,:, 9,4,7], x_dense[:,:, 0,0,0]]), 0,2)
    >>> print(np.linalg.norm(entries - entries2))
    9.22170384909466e-15

    Example using jax jit compiling:

	>>> import numpy as np
    >>> import jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> get_entry_123 = lambda x: t3.t3_entries(x, (1,2,3), use_jax=True)
    >>> A = t3.t3_corewise_randn((10,10,10),(5,5,5),(1,4,4,1)) # random 10x10x10 Tucker tensor train
    >>> a123 = get_entry_123(A)
    >>> print(a123)
    -1.3764521
    >>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
    >>> a123_jit = get_entry_123_jit(A)
    >>> print(a123_jit)
    -1.3764523

    Example using jax automatic differentiation

    >>> import numpy as np
    >>> import jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.common as common
    >>> import t3toolbox.corewise as cw
    >>> jax.config.update("jax_enable_x64", True) # enable double precision for finite difference
    >>> get_entry_123 = lambda x: t3.t3_entries(x, (1,2,3), use_jax=True)
    >>> A0 = t3.t3_corewise_randn((10,10,10),(5,5,5),(1,4,4,1)) # random 10x10x10 Tucker tensor train
    >>> f0 = get_entry_123(A0)
    >>> G0 = jax.grad(get_entry_123)(A0) # gradient using automatic differentiation
    >>> dA = t3.t3_corewise_randn((10,10,10),(5,5,5),(1,4,4,1))
    >>> df = cw.corewise_dot(dA.data, G0.data) # sensitivity in direction dA
    >>> print(df)
    -7.418801772515241
    >>> s = 1e-7
    >>> A1 = cw.corewise_add(A0.data, cw.corewise_scale(dA.data, s)) # A1 = A0 + s*dA
    >>> f1 = get_entry_123(t3.TuckerTensorTrain(*A1))
    >>> df_diff = (f1 - f0) / s # finite difference
    >>> print(df_diff)
    -7.418812309825662
    '''
    xnp, _, _ = get_backend(False, use_jax)

    index = xnp.array(index)

    if index.shape[0] != x.d:
        raise ValueError(
            'Wrong number of indices for Tucker tensor train.\n'
            + str(x.d) + ' = num tensor indices != num provided indices = ' + str(index.shape[0])
        )

    return entries.get_tucker_tensor_train_entries(x.data, index, use_jax=use_jax)


def t3_probe(
        ww: typ.Sequence[NDArray], # len=d, elm_shape=W+(Ni,)
        x: TuckerTensorTrain, # x.stack_shape=X
        use_jax: bool=False,
) -> typ.Sequence[NDArray]: # zz, len=d, elm_shape=X+W+(Ni,)
    """Probe a TuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as probing
    >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2))
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3.t3_probe(ww, x)
    >>> x_dense = x.to_dense()
    >>> zz2 = probing.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.0259410400851746e-12, 1.0909087370186656e-12, 3.620283224238675e-13]

    Vectorize over probes:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as probing
    >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2))
    >>> ww = (np.random.randn(2,3, 10), np.random.randn(2,3, 11), np.random.randn(2,3, 12))
    >>> zz = t3.t3_probe(ww, x)
    >>> x_dense = x.to_dense()
    >>> zz2 = probing.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [2.9290244450205316e-12, 2.0347746956505754e-12, 1.7784156096697445e-12]

    Vectorize over probes and T3s:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.backend.probing as probing
    >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2), stack_shape=(4,5))
    >>> ww = (np.random.randn(2,3, 10), np.random.randn(2,3, 11), np.random.randn(2,3, 12))
    >>> zz = t3.t3_probe(ww, x)
    >>> x_dense = x.to_dense()
    >>> zz2 = probing.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.4471391818397927e-11, 1.0485601346346092e-11, 1.437623640611662e-11]
    """
    return probing.probe_t3(ww, x.data, use_jax=use_jax)


###########################################################
##################    Linear algebra    ###################
###########################################################



# def t3_inner_product(
#         x: typ.Union[TuckerTensorTrain, NDArray],
#         y: typ.Union[TuckerTensorTrain, NDArray],
#         use_orthogonalization: bool = True, # for numerical stability
#         use_jax: bool = False,
# ):
#     """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.
#
#     The Hilbert-Schmidt inner product is defined with respect to the dense N0 x ... x N(d-1)
#     tensors that are *represented* by the Tucker tensor trains, even though these dense tensors
#     are not formed during computations.
#
#     For corewise dot product, see :func:`t3toolbox.corewise.corewise_dot`
#
#     Parameters
#     ----------
#     x: TuckerTensorTrain
#         First Tucker tensor train. shape=(N0,...,N(d-1))
#     y: TuckerTensorTrain
#         Second Tucker tensor train. shape=(N0,...,N(d-1))
#     xnp:
#         Linear algebra backend. Default: np (numpy)
#
#     Returns
#     -------
#     scalar
#         Hilbert-Schmidt inner product of Tucker tensor trains, (x, y)_HS.
#
#     Raises
#     ------
#     ValueError
#         - Error raised if either of the TuckerTensorTrains are internally inconsistent
#         - Error raised if the TuckerTensorTrains have different shapes.
#
#     See Also
#     --------
#     TuckerTensorTrain
#     t3_shape
#     t3_add
#     t3_scale
#     :func:`~t3toolbox.corewise.corewise_dot`
#
#     Notes
#     -----
#     Algorithm contracts the TuckerTensorTrains in a zippering fashion from left to right.
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
#     >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (1,5,6,1))
#     >>> x_dot_y = t3.t3_inner_product(x, y)
#     >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense())
#     >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
#     8.731149137020111e-11
#
#     Example where leading and trailing TT-ranks are not 1:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
#     >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (3,5,6,3))
#     >>> x_dot_y = t3.t3_inner_product(x, y)
#     >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense())
#     >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
#     1.3096723705530167e-10
#
#     (T3, T3) using stacking:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
#     >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (3,5,6,3), stack_shape=(2,3))
#     >>> x_dot_y = t3.t3_inner_product(x, y)
#     >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense(), axis=(2,3,4))
#     >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
#     2.7761383858792984e-09
#
#     Inner product of T3 with dense:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = np.random.randn(14,15,16)
#     >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (3,5,6,3))
#     >>> x_dot_y = t3.t3_inner_product(x, y)
#     >>> x_dot_y2 = np.sum(x * y.to_dense())
#     >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
#     0.0
#
#     Inner product of T3 with dense including stacking:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = np.random.randn(2,3, 14,15,16)
#     >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (3,5,6,3), stack_shape=(2,3))
#     >>> x_dot_y = t3.t3_inner_product(x, y)
#     >>> x_dot_y2 = np.einsum('ijxyz,ijxyz->ij', x, y.to_dense())
#     >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
#     1.2014283869232628e-11
#     """
#     xnp, _, _ = get_backend(False, use_jax)
#
#     if isinstance(x, TuckerTensorTrain) and isinstance(y, TuckerTensorTrain):
#         if x.shape != y.shape:
#             raise ValueError(
#                 'Attempted to take inner product of TuckerTensorTrains (x,y) with inconsistent shapes.'
#                 + str(x.shape) + ' = x.shape != y.shape = ' + str(y.shape)
#             )
#
#         vsx = x.stack_shape
#         vsy = y.stack_shape
#         if vsx != vsy:
#             raise NotImplementedError(
#                 'Cannot take inner product of TuckerTensorTrains with different stack shapes.\n'
#                 + str(x.stack_shape)
#                 + ' = x.stack_shape != y.stack_shape = '
#                 + str(y.stack_shape)
#             )
#
#         return ragged_linalg.t3_inner_product_t3(
#             x.data, y.data, use_orthogonalization=use_orthogonalization, use_jax=use_jax,
#         )
#
#     elif is_ndarray(x) and isinstance(y, TuckerTensorTrain): # Could be done better with zippering
#         vsy = y.stack_shape
#         if x.shape != vsy + y.shape:
#             raise ValueError(
#                 'Attempted to take inner product of array x with TuckerTensorTrain y with inconsistent shapes.'
#                 + str(x.shape) + ' = x.shape != y.stack_shape + y.shape = ' + str(vsy + y.shape)
#             )
#         contraction_inds = tuple(range(len(vsy), len(x.shape)))
#         contraction_inds = contraction_inds if contraction_inds else None
#
#         return xnp.sum(x * y.to_dense(), axis=contraction_inds)
#
#     elif isinstance(x, TuckerTensorTrain) and is_ndarray(y): # Could be done better with zippering
#         vsx = x.stack_shape
#         if vsx + x.shape != y.shape:
#             raise ValueError(
#                 'Attempted to take inner product of TuckerTensorTrain x with array y with inconsistent shapes.'
#                 + str(vsx + x.shape) + ' = x.stack_shape + x.shape != y.shape = ' + str(y.shape)
#             )
#         contraction_inds = tuple(range(len(vsx), len(y.shape)))
#         contraction_inds = contraction_inds if contraction_inds else None
#
#         return xnp.sum(x.to_dense() * y, axis=contraction_inds)
#
#     elif is_ndarray(x) and is_ndarray(y):
#         return (x * y).sum()
#
#     else:
#         raise NotImplementedError(
#             't3_add only implements x+y for T3+T3, T3+dense, dense+T3, dense+dense.\n'
#             + 'type(x) = ' + str(type(x)) + '\n'
#             + 'type(y) = ' + str(type(y))
#         )


##############################################################
########################    T3-SVD    ########################
##############################################################

def t3svd_dense(
        T: NDArray, # shape=(N1, N2, .., Nd)
        min_tucker_ranks:  typ.Sequence[int] = None, # len=d
        max_tucker_ranks:  typ.Sequence[int] = None,  # len=d
        min_tt_ranks:  typ.Sequence[int] = None, # len=d+1
        max_tt_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # Approximation of T by Tucker tensor train
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
]:
    '''Compute TuckerTensorTrain and edge singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation. len=d. e.g., (3,3,3)
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation. len=d. e.g., (5,5,5)
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation. len=d+1. e.g., (1,3,3,3,1)
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation. len=d+1. e.g., (1,5,5,5,1)
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train approxiamtion of T
    typ.Tuple[NDArray,...]
        Singular values of matricizations. len=d. elm_shape=(ni,)
    typ.Tuple[NDArray,...]
        Singular values of unfoldings. len=d+1. elm_shape=(ri,)

    See Also
    --------
    truncated_svd
    tucker_svd_dense
    tt_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> x, ss_tucker, ss_tt = t3.t3svd_dense(T, rtol=1e-3) # Truncate T3-SVD to reduce rank
    >>> print(x.uniform_structure)
    ((40, 50, 60), (11, 10, 12), (1, 11, 12, 1), ())
    >>> T2 = x.to_dense()
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.002920893302364434
    '''
    result = dense_t3svd.t3svd_dense(
        T,
        min_tucker_ranks=min_tucker_ranks, max_tucker_ranks=max_tucker_ranks,
        min_tt_ranks=min_tt_ranks, max_tt_ranks=max_tt_ranks,
        rtol=rtol, atol=atol,
        use_jax=use_jax,
    )
    return TuckerTensorTrain(*result[0]), result[1], result[2]


def t3svd(
        x: TuckerTensorTrain,
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # Tucker singular values, len=d
    typ.Tuple[NDArray,...], # TT singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.

    Parameters
    ----------
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation.
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation.
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation.
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    NDArray
        New TuckerTensorTrain representing the same tensor (or a truncated version), but with modified cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between Tucker cores and TT-cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between adjacent TT-cores

    See Also
    --------
    left_svd_3tensor
    right_svd_3tensor
    outer_svd_3tensor
    up_svd_ith_tucker_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    truncated_svd

    Examples
    --------

    T3-SVD with no truncation:
    (ranks may decrease to minimal values, but no approximation error)

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((5,6,3), (4,4,3), (1,3,2,1))
    >>> x2, ss_tucker, ss_tt = t3.t3svd(x) # Compute T3-SVD
    >>> x_dense = x.to_dense()
    >>> x2_dense = x2.to_dense()
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    7.556835759880194e-13
    >>> ss_tt1 = np.linalg.svd(x_dense.reshape((5, 6*3)))[1] # Singular values of unfolding 1
    >>> print(ss_tt1); print(ss_tt[1])
    [1.75326490e+02 3.41363029e+01 9.31164204e+00 1.33610061e-14 4.11601708e-15]
    [175.32648969  34.13630287   9.31164204]
    >>> ss_tucker2 = np.linalg.svd(x_dense.transpose([2,0,1]).reshape((3,5*6)))[1] # Singular values of matricization 2
    >>> print(ss_tucker2); print(ss_tucker[2])
    [1.71350937e+02 5.12857505e+01 1.36927051e-14]
    [171.35093708  51.28575045]

    T3-SVD with truncation based on relative tolerance:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> B0 = np.random.randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2) # preconditioned indices
    >>> B1 = np.random.randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
    >>> B2 = np.random.randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
    >>> G0 = np.random.randn(1,35,30)
    >>> G1 = np.random.randn(30,45,40)
    >>> G2 = np.random.randn(40,55,1)
    >>> tucker_cores_x = (B0, B1, B2)
    >>> tt_cores_x = (G0, G1, G2)
    >>> x = t3.TuckerTensorTrain(tucker_cores_x, tt_cores_x) # Tensor has spectral decay due to preconditioning
    >>> x2, ss_tucker, ss_tt = t3.t3svd(x, rtol=1e-2) # Truncate singular values to reduce rank
    >>> print(x.uniform_structure)
    ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1), ())
    >>> print(x2.uniform_structure)
    ((40, 50, 60), (6, 6, 5), (1, 6, 5, 1), ())
    >>> x_dense = x.to_dense()
    >>> x2_dense = x2.to_dense()
    >>> print(np.linalg.norm(x_dense - x2_dense)/np.linalg.norm(x_dense)) # Should be near rtol=1e-2
    0.01919726997372989

    T3-SVD with truncation based on absolute tolerance:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (10,11,12), (1,8,9,1))
    >>> x2, ss_tucker, ss_tt = t3.t3svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1)) # Truncate based on ranks
    >>> print(x.uniform_structure)
    ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1), ())
    >>> print(x2.uniform_structure)
    ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1), ())

    Example where first and last ranks are not ones:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((5,6,3), (4,4,3), (2,3,2,2))
    >>> x2, ss_tucker, ss_tt = t3.t3svd(x, squash_tails_first=False) # Compute T3-SVD
    >>> x_dense = x.to_dense(squash_tails=False)
    >>> x2_dense = x2.to_dense(squash_tails=False)
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    5.486408687260824e-13
    >>> ss_tt0 = np.linalg.svd(x_dense.reshape((2,5*6*3*2)))[1] # Singular values of leading unfolding
    >>> print(ss_tt0); print(ss_tt[0])
    [303.0474449   88.85034392]
    [303.0474449   88.85034392]
    >>> ss_tt3 = np.linalg.svd(x_dense.reshape((2*5*6*3,2)))[1] # Singular values of trailing unfolding
    >>> print(ss_tt3); print(ss_tt[3])
    [299.45433768 100.29574828]
    [299.45433768 100.29574828]
    '''
    result = ragged_t3svd.t3svd(
        x.data,
        min_tt_ranks=min_tt_ranks, min_tucker_ranks=min_tucker_ranks,
        max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks,
        rtol=rtol, atol=atol,
        squash_tails_first=squash_tails_first,
        use_jax=use_jax,
    )
    return TuckerTensorTrain(*result[0]), result[1], result[2]

