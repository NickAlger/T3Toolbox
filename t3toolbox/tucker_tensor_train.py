# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
"""
This module contains type aliases and basic operations for Tucker tensor trains.

Tucker tensor trains:
---------------------

* :py:type:`TuckerTensorTrain`
    | The cores of a Tucker tensor train with varied ranks
    | (tucker_cores, tt_cores)

* :py:type:`T3Structure`
    | The structure of a Tucker tensor train.
    | (shape, tucker_ranks, tt_ranks)

* :py:type:`EdgeWeights`
    | Weights for edges of a Tucker tensor train
    | (tucker_weights, tt_weights)

* :py:type:`T3Base`
    | Base cores for Tucker tensor trains in base-variation format
    | (up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores)

* :py:type:`T3Variation`
    | Variation cores for Tucker tensor trains in base-variation format
    | (tucker_variations, tt_variations)

* :py:type:`BVStructure`
    | The structure of a Tucker tensor train in base-variation format
    | (shape, up_ranks, outer_ranks, left_ranks, right_ranks)

* :py:type:`BVEdgeWeights`
    | Weights for the edges of a Tucker tensor train in base-variation format
    | (shape_weights, up_weights, outer_weights, left_weights, right_weights)

Uniform Tucker tensor trains:
-----------------------------

* :py:type:`UniformTuckerTensorTrain`
    | Supercores of a uniform Tucker tensor train with uniform ranks
    | (tucker_supercore, tt_supercore)

* :py:type:`UniformT3Structure`
    | The structure of a uniform Tucker tensor train.
    | (num indices d, index size N, tucker rank nU, TT rank r)

* :py:type:`UniformEdgeWeights`
    | Weights for the edges of a uniform Tucker tensor train
    | (tucker_weights, tt_weights)

* :py:type:`UniformT3Base`
    | Base cores for Tucker tensor trains in base-variation format
    | (up_tucker_supercore, left_tt_supercore, right_tt_supercore, outer_tt_supercore)

* :py:type:`UniformT3Variation`
    | Variation cores for Tucker tensor trains in base-variation format
    | (tucker_variations_supercore, tt_variations_supercore)

* :py:type:`BVStructure`
    | The structure of a Tucker tensor train in base-variation format
    | (num indices d, index size N, up rank nU, outer rank nO, left rank rL, right rank rR)

* :py:type:`UniformBVEdgeWeights`
    | Weights for the edges of a Tucker tensor train in base-variation format
    | (shape_weights, up_weights, outer_weights, left_weights, right_weights)






Other
=====

Operations here are defined with respect to the dense N0 x ... x N(d-1) tensors that
are *represented* by the Tucker tensor train, even though these dense tensors
are not formed during computations.

For corewise operations, see :mod:`t3toolbox.corewise`
"""
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

from t3toolbox.common import *

jax = None
if has_jax:
    import jax

__all__ = [
    # Tucker tensor train
    'TuckerTensorTrain',
    # 'T3Structure',
    'EdgeWeights',
    # 'get_structure',
    't3_apply',
    't3_entry',
    # 't3_to_dense',
    # 'squash_tails',
    'reverse_tt',
    'reverse_t3',
    # 'check_t3',
    'absorb_edge_weights_into_cores',
    't3_zeros',
    't3_corewise_randn',
    'compute_minimal_ranks',
    'are_t3_ranks_minimal',
    'change_tucker_core_shapes',
    'change_tt_core_shapes',
    'change_structure',
    't3_save',
    't3_load',
    # Linear algebra
    't3_add',
    't3_scale',
    't3_neg',
    't3_sub',
    't3_inner_product_t3',
    't3_norm',
]


###########################################
########    Tucker Tensor Train    ########
###########################################

# TuckerTensorTrain = typ.Tuple[
#     typ.Sequence[NDArray], # tucker_cores, len=d, elm_shape=(ni, Ni)
#     typ.Sequence[NDArray], # tt_cores,     len=d, elm_shape=(ri, ni, r(i+1))
# ]

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

    vectorization_shape: typ.Tuple[int, ...]
        The vectorization shape, VS. Non-empty if this object stores many different Tucker tensor trains with the same structure.
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


    Many Tucker tensor trains with the same structure may be stored in this object for vectorization.
    In this case,
        - tucker_cores[ii].shape = vectorization_shape + (ni,Ni)
        - tt_cores[ii].shape = vectorization_shape + (ri, ni, r(i+1))

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
    >>> print(x.structure)
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))

    Example with vectorization:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = [np.ones((6,7, 4,14)),np.ones((6,7, 5,15)),np.ones((6,7, 6,16))]
    >>> tt_cores = [np.ones((6,7, 1,4,3)), np.ones((6,7, 3,5,2)), np.ones((6,7, 2,6,1))]
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with ones
    >>> print(x.structure)
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
    >>> print(x.vectorization_shape)
    (6, 7)

    Minimal ranks

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((13,14,15,16), (4,5,6,7), (1,4,9,7,1))
    >>> print(x.has_minimal_ranks)
    True

    Using T3-SVD to make equivalent T3 with minimal ranks:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.t3svd as t3svd
    >>> x = t3.t3_corewise_randn((13,14,15,16), (4,5,6,7), (1,99,9,7,1))
    >>> print(x.has_minimal_ranks)
    False
    >>> x2 = t3svd.t3_svd(x)[0]
    >>> print(x2.has_minimal_ranks)
    True
    """
    tucker_cores:   typ.Tuple[NDArray] # len=d, elm_shape=VS+(ni, Ni)
    tt_cores:       typ.Tuple[NDArray] # len=d, elm_shape=VS+(ri, ni, r(i+1))

    def __post_init__(self):
        self.check()

    @ft.cached_property
    def data(self) -> typ.Tuple[typ.Tuple[NDArray], typ.Tuple[NDArray]]:
        return tuple(self.tucker_cores), tuple(self.tt_cores)

    @ft.cached_property
    def d(self) -> int:
        return len(self.tucker_cores)

    @ft.cached_property
    def is_empty(self) -> bool:
        return self.d == 0

    @ft.cached_property
    def vectorization_shape(self) -> typ.Tuple[int, ...]:
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
    def structure(self) -> typ.Tuple[typ.Tuple[int,...], typ.Tuple[int,...], typ.Tuple[int,...]]:
        return self.shape, self.tucker_ranks, self.tt_ranks

    @ft.cached_property
    def minimal_ranks(self) -> typ.Tuple[typ.Tuple[int,...], typ.Tuple[int,...]]:
        minimal_tucker_ranks, minimal_tt_ranks = compute_minimal_ranks(*self.structure)
        return minimal_tucker_ranks, minimal_tt_ranks

    @ft.cached_property
    def has_minimal_ranks(self) -> bool:
        return (self.tucker_ranks, self.tt_ranks) == self.minimal_ranks

    def check(self):
        """Check internal consistency of the Tucker tensor train.
        """
        if len(self.tucker_cores) != len(self.tt_cores):
            raise ValueError(
                'Inconsistent TuckerTensorTrain.\n'
                + str(len(self.tucker_cores)) + ' = len(tucker_cores) != len(tt_cores) = ' + str(len(self.tt_cores))
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

        desired_vectorization_shapes = tuple([self.vectorization_shape for _ in range(self.d)])
        tt_vectorization_shapes = tuple([G.shape[:-3] for G in self.tt_cores])
        tucker_vectorization_shapes = tuple([B.shape[:-2] for B in self.tucker_cores])
        if ((tt_vectorization_shapes) != (desired_vectorization_shapes)
                or (tucker_vectorization_shapes != desired_vectorization_shapes)):
            raise ValueError(
                'Inconsistent TuckerTensorTrain.\n'
                + str(tt_vectorization_shapes) + ' = tt_vectorization_shapes'
                + '\n'
                + str(tt_vectorization_shapes) + ' = tucker_vectorization_shapes'
            )

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

        Example with vectorization

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
        xnp, _, _ = get_backend(False, use_jax)

        #
        big_tt_cores = [xnp.einsum('...iaj,...ab->...ibj', G, U) for G, U in zip(self.tt_cores, self.tucker_cores)]

        T = big_tt_cores[0]
        for G in big_tt_cores[1:]:
            vs = self.vectorization_shape
            ts = T.shape[len(vs):-1]
            cs = (T.shape[-1],)
            T_a_b_c_xyz_r = T.reshape(vs + (xnp.prod(ts, dtype=int),) + cs)

            ts2 = G.shape[-2:]
            G_a_b_c_r_lm = G.reshape(vs + cs + (xnp.prod(ts2, dtype=int),))
            T_a_b_c_xyzlm = T_a_b_c_xyz_r @ G_a_b_c_r_lm
            T = T_a_b_c_xyzlm.reshape(vs + ts + ts2)

        if squash_tails:
            mu_L = xnp.ones(big_tt_cores[0].shape[-3])
            mu_R = xnp.ones(big_tt_cores[-1].shape[-1])

            T = xnp.tensordot(T, mu_R, axes=1)
            T = xnp.tensordot(mu_L, T, axes=((0,), (len(self.vectorization_shape),)))

        return T

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
        xnp, _, _ = get_backend(False, use_jax)

        #
        G0 = self.tt_cores[0]
        G0 = xnp.einsum('az,...aib->...zib', xnp.ones((G0.shape[-3],1)), G0)

        Gf = self.tt_cores[-1]
        Gf = xnp.einsum('...aib,bz->...aiz', Gf, xnp.ones((Gf.shape[-1],1)))

        return TuckerTensorTrain(tuple(self.tucker_cores), (G0,) + tuple(self.tt_cores[1:-1]) + (Gf,))

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
        >>> print(x.structure)
        ((10, 11, 12), (4, 5, 6), (1, 2, 3, 4))
        >>> reversed_x = x.reverse()
        >>> print(reversed_x.structure)
        ((12, 11, 10), (6, 5, 4), (4, 3, 2, 1))
        >>> x_dense = x.to_dense()
        >>> reversed_x_dense = reversed_x.to_dense()
        >>> x_dense2 = reversed_x_dense.transpose([0,1, 4,3,2])
        >>> print(np.linalg.norm(x_dense - x_dense2))
        1.859018050214056e-13
        """
        is_ragged = isinstance(self.tucker_cores, typ.Sequence)

        if is_ragged:
            reversed_tucker_cores = tuple([B.copy() for B in self.tucker_cores[::-1]])
        else:
            reversed_tucker_cores = self.tucker_cores[::-1,:,:]

        reversed_tt_cores = tuple([G.swapaxes(-3, -1) for G in self.tt_cores[::-1]])
        return TuckerTensorTrain(reversed_tucker_cores, reversed_tt_cores)

    def change_structure(
            self,
            new_structure: typ.Tuple[typ.Sequence[int], typ.Sequence[int]],
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        '''Increase Tucker tensor train ranks and/or shape via zero padding.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (1,3,2,1))
        >>> new_structure = ((17,18,17), (8,8,8), (1,5,6,1))
        >>> padded_x = x.change_structure(new_structure)
        >>> print(padded_x.structure)
        ((17, 18, 17), (8, 8, 8), (1, 5, 6, 1))

        Example where first and last ranks are nonzero:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (3,3,2,4))
        >>> new_structure = ((17,18,17), (8,8,8), (5,5,6,7))
        >>> padded_x = x.change_structure(new_structure)
        >>> print(padded_x.structure)
        ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7))
        '''
        new_shape, new_tucker_ranks, new_tt_ranks = new_structure
        tucker_cores, tt_cores = self.data

        new_tucker_cores = change_tucker_core_shapes(tucker_cores, new_shape, new_tucker_ranks, use_jax=use_jax)
        new_tt_cores = change_tt_core_shapes(tt_cores, new_tucker_ranks, new_tt_ranks, use_jax=use_jax)

        return TuckerTensorTrain(tuple(new_tucker_cores), tuple(new_tt_cores))

    def __add__(
            self: 'TuckerTensorTrain',
            other: 'TuckerTensorTrain',
            squash: bool = True,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Add this Tucker tensor train to another one, yielding a Tucker tensor train with summed ranks.

        Addition is defined with respect to the dense N0 x ... x N(d-1) tensors that
        are *represented* by the Tucker tensor trains, even though these dense tensors
        are not formed during computations.

        For corewise addition, see :func:`t3toolbox.corewise.corewise_add`

        Parameters
        ----------
        other: TuckerTensorTrain
            The other Tucker tensor train to add to this one.
            structure=((N0,...,N(d-1)), (m0,...,m(d-1)), (q0, q1,...,qd))

        squash: bool
            Squash the first and last TT cores so that r0=rd=1 in the result. Default: True.

        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        TuckerTensorTrain
            Sum of Tucker tensor trains, x+y.
                | shape=(N0,...,N(d-1),
                | tucker_ranks=(n0+m0,...,n(d-1)+m(d-1),
                | TT ranks=(1, r1+q1,...,r(d-1)+q(d-1),1)) if squash=True,
                | or (r0+q0, r1+q1,...,r(d-1)+q(d-1),rd+qd)) if squash=False.

        Raises
        ------
        ValueError
            - Error raised if either of the TuckerTensorTrains are internally inconsistent
            - Error raised if the TuckerTensorTrains have different shapes.

        See Also
        --------
        TuckerTensorTrain
        __scale__
        __sub__
        __neg__
        squash_tails
        :func:`~t3toolbox.corewise.corewise_add`


        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> z = x + y
        >>> print(z.structure)
        ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1))
        >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - z.to_dense()))
        6.524094086845177e-13

        With vectorized TuckerTensorTrains

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), vectorization_shape=(2,3))
        >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (1,5,6,1), vectorization_shape=(2,3))
        >>> z = x + y
        >>> print(z.structure)
        ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1))
        >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - z.to_dense()))
        """
        xnp, _, _ = get_backend(False, use_jax)

        if not isinstance(other, TuckerTensorTrain):
            raise NotImplementedError(
                'Can only add TuckerTensorTrain to another TuckerTensorTrain.'
            )

        #

        if self.shape != other.shape:
            raise ValueError(
                'Attempted to add TuckerTensorTrains x+y with inconsistent shapes.'
                + str(self.shape) + ' = x.shape != y.shape = ' + str(other.shape)
            )

        vsx = self.vectorization_shape
        vsy = other.vectorization_shape

        # vs = (xnp.zeros(vsx) + xnp.zeros(vsy)).shape # attempt at broadcasting (doesn't work)

        if vsx != vsy:
            raise NotImplementedError(
                'Cannot add TuckerTensorTrains with different vectorization shapes.\n'
                + str(self.vectorization_shape)
                + ' = x.vectorization_shape != y.vectorization_shape = '
                + str(other.vectorization_shape)
            )


        tucker_cores_x, tt_cores_x = self.data
        tucker_cores_y, tt_cores_y = other.data
        tucker_cores_z = [xnp.concatenate([Bx, By], axis=-2) for Bx, By in zip(tucker_cores_x, tucker_cores_y)]

        tt_cores_z = []

        for Gx, Gy in zip(tt_cores_x, tt_cores_y):
            G000 = Gx
            G001 = xnp.zeros(vsx + (Gx.shape[-3], Gx.shape[-2], Gy.shape[-1]))
            G010 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gx.shape[-1]))
            G011 = xnp.zeros(vsx + (Gx.shape[-3], Gy.shape[-2], Gy.shape[-1]))
            G100 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gx.shape[-1]))
            G101 = xnp.zeros(vsx + (Gy.shape[-3], Gx.shape[-2], Gy.shape[-1]))
            G110 = xnp.zeros(vsx + (Gy.shape[-3], Gy.shape[-2], Gx.shape[-1]))
            G111 = Gy
            Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
            tt_cores_z.append(Gz)

        z = TuckerTensorTrain(tuple(tucker_cores_z), tuple(tt_cores_z))
        if squash:
            z = z.squash_tails()
        return z

    def __mul__(
            self,
            s,  # scalar
    ) -> 'TuckerTensorTrain':
        """Multipy a Tucker tensor train by a scaling factor.

        Scaling is defined with respect to the dense N0 x ... x N(d-1) tensor that
        is *represented* by the Tucker tensor trains, even though this dense tensor
        is not formed during computations.

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
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> s = 3.2
        >>> sx = x * s
        >>> print(np.linalg.norm(s*x.to_dense() - sx.to_dense()))
        1.6268482531988893e-13
        """
        tucker_cores, tt_cores = self.data

        scaled_tucker_cores = [B.copy() for B in tucker_cores]
        scaled_tucker_cores[-1] = s * scaled_tucker_cores[-1]

        copied_tt_cores = [G.copy() for G in tt_cores]

        return TuckerTensorTrain(tuple(scaled_tucker_cores), tuple(copied_tt_cores))

    def __neg__(
            self,
    ) -> 'TuckerTensorTrain':
        """Scale a Tucker tensor train by -1.

        Negation is defined with respect to the dense N0 x ... x N(d-1) tensor that
        is *represented* by the Tucker tensor trains, even though this dense tensor
        is not formed during computations.

        For corewise negation, see :func:`t3toolbox.corewise.corewise_neg`

        Parameters
        ----------
        x: TuckerTensorTrain
            Tucker tensor train

        Returns
        -------
        TuckerTensorTrain
            Negated TuckerTensorTrain -x, with the same structure as x.

        Raises
        ------
        ValueError
            - Error raised if the TuckerTensorTrains is internally inconsistent

        See Also
        --------
        TuckerTensorTrain
        t3_add
        t3_scale
        t3_sub
        :func:`~t3toolbox.corewise.corewise_neg`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> neg_x = -x
        >>> print(np.linalg.norm(x.to_dense() + neg_x.to_dense()))
        0.0
        """
        return self * (-1.0)

    def __sub__(
            self,
            other: 'TuckerTensorTrain',
            squash: bool = True,
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Subtract two Tucker tensor trains, yielding a Tucker tensor train with summed ranks.

        Subtraction is defined with respect to the dense N0 x ... x N(d-1) tensors that
        are *represented* by the Tucker tensor trains, even though these dense tensors
        are not formed during computations.

        For corewise subtraction, see :func:`t3toolbox.corewise.corewise_sub`

        Parameters
        ----------
        x: TuckerTensorTrain
            First summand. structure=((N0,...,N(d-1)), (n1,...,nd), (r0, r1,...,rd))
        y: TuckerTensorTrain
            Second summand. structure=((N0,...,N(d-1)), (m1,...,md), (q0, q1,...,qd))
        squash: bool
            Squash the first and last TT cores so that r0=rd=1 in the result. Default: True.
        xnp:
            Linear algebra backend. Default: np (numpy)

        Returns
        -------
        TuckerTensorTrain
            Difference of Tucker tensor trains, x-y.
                - shape=(N0,...,N(d-1),
                - tucker_ranks=(n0+m0,...,n(d-1)+m(d-1),
                - TT ranks=(1, r1+q1,...,r(d-1)+q(d-1),1)) if squash=True,
                or (r0+q0, r1+q1,...,r(d-1)+q(d-1),rd+qd)) if squash=False.

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
        t3_neg
        :func:`~t3toolbox.corewise.corewise_neg`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> x_minus_y = x - y
        >>> print(x_minus_y.structure)
        ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2))
        >>> dense_x = x.to_dense()
        >>> dense_y = y.to_dense()
        >>> dense_x_minus_y = x_minus_y.to_dense()
        >>> print(np.linalg.norm(dense_x - dense_y - dense_x_minus_y))
        3.5875705233607603e-13
        """
        return self.__add__(-other, squash=squash, use_jax=use_jax)


    def flatten(self):
        return (self.data, None)

    @classmethod
    def unflatten(cls, aux_data, children):
        return cls(*children)



if has_jax:
    jax.tree_util.register_pytree_node(TuckerTensorTrain, TuckerTensorTrain.flatten, TuckerTensorTrain.unflatten)



# T3Structure = typ.Tuple[
#     typ.Sequence[int], # shape,        len=d
#     typ.Sequence[int], # tucker_ranks, len=d
#     typ.Sequence[int], # tt_ranks,     len=d+1
# ]
# """
# Tuple (shape, tucker_ranks, tt_ranks) containing the structure of a Tucker tensor train.
#
# Components:
# -----------
#
# - **shape** : *Sequence[int]*
#     Shape of the represented tensor: (N0, ..., N(d-1)), len=d.
#
# - **tucker_ranks** : *Sequence[int]*
#     Tucker ranks: (n0, ..., n(d-1)), len=d.
#
# - **tt_ranks** : *Sequence[int]*
#     TT-ranks: (1, r1, ..., r(d-1), 1), len=d+1.
#
#
# .. seealso::
#     :py:type:`T3Structure`
#     :py:func:`get_structure`
#     :py:func:`check_t3`
#
#
# Examples
# --------
# >>> import numpy as np
# >>> import t3toolbox.tucker_tensor_train as t3
# >>> tucker_cores = [np.ones((4,14)),np.ones((5,15)),np.ones((6,16))]
# >>> tt_cores = [np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))]
# >>> x = (tucker_cores, tt_cores)
# >>> shape, tucker_ranks, tt_ranks = t3.get_structure(x)
# >>> print(shape)
# (14, 15, 16)
# >>> print(tucker_ranks)
# (4, 5, 6)
# >>> print(tt_ranks)
# (1, 3, 2, 1)
# """


EdgeWeights = typ.Tuple[
    typ.Sequence[NDArray],  # shape_weights, len=d, elm_shape=(Ni,)
    typ.Sequence[NDArray],  # tucker_weights, len=d, elm_shape=(ni,)
    typ.Sequence[NDArray],  # tt_weights, len=d+1, elm_shape=(ri,)
]

# @dataclass(frozen=True)
# class EdgeWeights:
#     """Weighting vectors for the edges of a Tucker tensor train.
#
#     Attributes:
#     -----------
#     shape_weights: Tuple[int]
#         Weights for externally facing edges. len=d, elm_shape=(Ni,)
#     tucker_weights: Sequence[int]
#         Weights for edges between Tucker cores and TT cores. len=d, elm_shape=(ni,)
#     tt_weights: Sequence[int]
#         Weights for edges between adjacent TT cores. len=d+1, elm_shape=(ri,)
#     """
#     shape_weights:  typ.Sequence[NDArray] # len=d,   elm_shape=(Ni,)
#     tucker_weights: typ.Sequence[NDArray] # len=d,   elm_shape=(ni,)
#     tt_weights:     typ.Sequence[NDArray] # len=d+1, elm_shape=(ri,)




#####################################################################
########    Structural properties and consistency checks    #########
#####################################################################

# def get_structure(
#         x: TuckerTensorTrain,
# ) -> T3Structure:
#     """Get the structure of a Tucker tensor train.
#
#     Parameters
#     ----------
#     x : TuckerTensorTrain
#         Tucker tensor train with:
#
#             shape=(N0, ..., N(d-1)),
#
#             tucker_ranks=(n0, ..., n(d-1)),
#
#             tt_ranks=(1, r1, ..., r(d-1), 1)).
#
#     Returns
#     -------
#     structure: T3Structure
#         The structure of the Tucker tensor train:
#
#             ((N1, ..., Nd), (n1, ..., nd), (1, r1, ... r(d-1), 1))
#
#     See Also
#     --------
#     TuckerTensorTrain
#     T3Structure
#     check_t3
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
#     >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
#     >>> x = (tucker_cores, tt_cores)
#     >>> shape, tucker_ranks, tt_ranks = t3.get_structure(x)
#     >>> print(shape)
#     (14, 15, 16)
#     >>> print(tucker_ranks)
#     (4, 5, 6)
#     >>> print(tt_ranks)
#     (1, 3, 2, 1)
#     """
#     tucker_cores, tt_cores = x
#     shape = tuple([B.shape[1] for B in tucker_cores])
#     tucker_ranks = tuple([B.shape[0] for B in tucker_cores])
#     tt_ranks = tuple([int(tt_cores[0].shape[0])] + [int(G.shape[2]) for G in tt_cores])
#     return shape, tucker_ranks, tt_ranks


# def check_t3(
#         x: TuckerTensorTrain,
# ) -> None:
#     '''Check rank and shape consistency of Tucker tensor train.
#
#     Parameters
#     ----------
#     x : TuckerTensorTrain
#
#     Raises
#     ------
#     ValueError
#         Error raised if the cores of the Tucker tensor train have inconsistent shapes.
#
#     See Also
#     --------
#     TuckerTensorTrain
#     T3Structure
#     get_structure
#
#     Examples
#     --------
#
#     (Good) Consistent Tucker tensor train:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = [np.ones((4,14)),np.ones((5,15)),np.ones((6,16))]
#     >>> tt_cores = [np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))]
#     >>> x = (tucker_cores, tt_cores)
#     >>> t3.check_t3(x) # Nothing happens because T3 is consistent
#
#     (Bad) Mismatch between number of Tucker cores and number of TT-cores:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = (np.ones((4,14)), np.ones((5,15))) # one too few Tucker cores
#     >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
#     >>> x = (tucker_cores, tt_cores)
#     >>> t3.check_t3(x)
#     ValueError: Inconsistent TuckerTensorTrain.
#     2 = len(tucker_cores) != len(tt_cores) = 3
#
#     (Bad) One of the TT-cores is not a 3-tensor:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
#     >>> tt_cores = (np.ones((4,3)), np.ones((3,5,2)), np.ones((2,6,1))) # first TT-core is not a 3-tensor
#     >>> x = (tucker_cores, tt_cores)
#     >>> t3.check_t3(x)
#     ValueError: Inconsistent TuckerTensorTrain.
#     tt_cores[0] is not a 3-tensor. shape=(4, 3)
#
#     (Bad) TT-core shapes inconsistent with each other:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
#     >>> tt_cores = (np.ones((1,4,9)), np.ones((3,5,2)), np.ones((2,6,1))) # Inconsistent TT-core shapes
#     >>> x = (tucker_cores, tt_cores)
#     >>> t3.check_t3(x)
#     ValueError: Inconsistent TuckerTensorTrain.
#     (1, 3, 2, 1) = left_tt_ranks != right_tt_ranks = (1, 9, 2, 1)
#
#     (Bad) Basis core is not a matrix:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = (np.ones((4,14)), np.ones((5,15,3)), np.ones((6,16))) # Basis core 2 is not a matrix
#     >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
#     >>> x = (tucker_cores, tt_cores)
#     >>> t3.check_t3(x)
#     ValueError: Inconsistent TuckerTensorTrain.
#     tucker_cores[1] is not a matrix. shape=(5, 15, 3)
#
#     (Bad) Inconsistent shapes for tucker core and adjacent TT-core
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> tucker_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((9,16)))
#     >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))) # Last Tucker and TT-cores inconsistent
#     >>> x = (tucker_cores, tt_cores)
#     >>> t3.check_t3(x)
#     ValueError: Inconsistent TuckerTensorTrain.
#     9 = tucker_cores[2].shape[0] != tt_cores[2].shape[1] = 6
#     '''
#     tucker_cores, tt_cores = x
#     if len(tucker_cores) != len(tt_cores):
#         raise ValueError(
#             'Inconsistent TuckerTensorTrain.\n'
#             + str(len(tucker_cores)) + ' = len(tucker_cores) != len(tt_cores) = ' + str(len(tt_cores))
#         )
#
#     for ii, G in enumerate(tt_cores):
#         if len(G.shape) != 3:
#             raise ValueError(
#                 'Inconsistent TuckerTensorTrain.\n'
#                 + 'tt_cores[' + str(ii) + '] is not a 3-tensor. shape=' + str(G.shape)
#             )
#
#     right_tt_ranks = tuple([int(tt_cores[0].shape[0])] + [int(G.shape[2]) for G in tt_cores])
#     left_tt_ranks = tuple([int(G.shape[0]) for G in tt_cores] + [int(tt_cores[-1].shape[2])])
#     if left_tt_ranks != right_tt_ranks:
#         raise ValueError(
#             'Inconsistent TuckerTensorTrain.\n'
#             + str(left_tt_ranks) + ' = left_tt_ranks != right_tt_ranks = ' + str(right_tt_ranks)
#         )
#
#     for ii, B in enumerate(tucker_cores):
#         if len(B.shape) != 2:
#             raise ValueError(
#                 'Inconsistent TuckerTensorTrain.\n'
#                 + 'tucker_cores['+str(ii)+'] is not a matrix. shape='+str(B.shape)
#             )
#
#     for ii, (B, G) in enumerate(zip(tucker_cores, tt_cores)):
#         if B.shape[0] != G.shape[1]:
#             raise ValueError(
#                 'Inconsistent TuckerTensorTrain.\n'
#                 + str(B.shape[0]) + ' = tucker_cores[' + str(ii) + '].shape[0]'
#                 + ' != '
#                 + 'tt_cores[' + str(ii) + '].shape[1] = ' + str(G.shape[1])
#             )


###########################################################
################    Basic T3 functions    #################
###########################################################

# def t3_to_dense(
#         x: TuckerTensorTrain,
#         squash_tails: bool = True,
#         use_jax: bool = False,
# ) -> NDArray:
#     """Contract a Tucker tensor train to a dense tensor.
#
#     Parameters
#     ----------
#     x : TuckerTensorTrain
#         Tucker tensor train which will be contracted to a dense tensor.
#
#     squash_tails: bool, defaults to True
#         Whether to contract the leading and trailing 1s with the first and last TT indices.
#
#     use_jax: bool, defaults to False
#         Whether to use Jax for linear algebra. Default: False (use numpy).
#
#     Returns
#     -------
#     dense_x: NDArray
#         Dense tensor represented by x,
#         which has shape (N0, ..., N(d-1)) if squash_tails=True,
#         or (r0,N0,...,N(d-1),rd) if squash_tails=False.
#
#     See Also
#     --------
#     TuckerTensorTrain
#     squash_tails
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16),(4,5,6),(1,3,2,1))) # make TuckerTensorTrain
#     >>> x_dense = t3.t3_to_dense(x) # Convert TuckerTensorTrain to dense tensor
#     >>> ((B0,B1,B2), (G0,G1,G2)) = x
#     >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
#     >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
#     7.48952547844518e-16
#
#     Case where the first and last TT-ranks are not 1:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16),(4,5,6),(2,3,2,4))) # make TuckerTensorTrain
#     >>> x_dense = t3.t3_to_dense(x) # Convert TuckerTensorTrain to dense tensor
#     >>> ((B0,B1,B2), (G0,G1,G2)) = x
#     >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
#     >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
#     1.4635914598284152e-15
#
#     Example where leading and trailing ones are not contracted
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16),(4,5,6),(2,3,4,2))) # make TuckerTensorTrain
#     >>> x_dense = t3.t3_to_dense(x, squash_tails=False) # Convert TuckerTensorTrain to dense tensor
#     >>> print(x_dense.shape)
#     (2, 14, 15, 16, 2)
#     >>> ((B0,B1,B2), (G0,G1,G2)) = x
#     >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
#     >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
#     1.1217675019342066e-15
#     """
#     xnp, _, _ = get_backend(False, use_jax)
#
#     #
#     tucker_cores, tt_cores = x
#     big_tt_cores = [xnp.einsum('iaj,ab->ibj', G, U) for G, U in zip(tt_cores, tucker_cores)]
#
#     T = big_tt_cores[0]
#     for G in big_tt_cores[1:]:
#         T = xnp.tensordot(T, G, axes=1)
#
#     if squash_tails:
#         mu_L = xnp.ones(big_tt_cores[0].shape[0])
#         mu_R = xnp.ones(big_tt_cores[-1].shape[2])
#
#         T = xnp.tensordot(mu_L, T, axes=1)
#         T = xnp.tensordot(T, mu_R, axes=1)
#
#     # T = xnp.tensordot(T, xnp.ones(big_tt_cores[-1].shape[-1]), axes=1)
#
#     return T


# def squash_tails(
#         x: TuckerTensorTrain,
#         use_jax: bool = False,
# ) -> TuckerTensorTrain:
#     """Make leading and trailing TT ranks equal to 1 (r0=rd=1), without changing tensor being represented.
#
#     Parameters
#     ----------
#     x : TuckerTensorTrain
#         Tucker tensor train with tt_ranks=(r0,r1,...,r(d-1),rd).
#
#     use_jax: bool, defaults to False
#         Whether to use Jax for linear algebra. Default: False (use numpy).
#
#     Returns
#     -------
#     squashed_x: TuckerTensorTrain
#         Tucker tensor train with tt_ranks=(1,r1,...,r(d-1),1).
#
#     See Also:
#     ---------
#     TuckerTensorTrain
#     T3Structure
#
#     Examples
#     ________
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((10,10,10), (5,5,5), (9,3,3,9)))
#     >>> x2 = t3.squash_tails(x)
#     >>> print(t3.get_structure(x2))
#     ((10, 10, 10), (5, 5, 5), (1, 3, 3, 1))
#     >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2)))
#     5.805155892491438e-12
#     """
#     xnp, _, _ = get_backend(False, use_jax)
#
#     #
#     G0 = x[1][0]
#     G0 = xnp.tensordot(xnp.ones((1, G0.shape[0])), G0, axes=1)
#
#     Gf = x[1][-1]
#     Gf = xnp.tensordot(Gf, xnp.ones((Gf.shape[2], 1)), axes=1)
#
#     x = (tuple(x[0]), (G0,) + tuple(x[1][1:-1]) + (Gf,))
#     return x


def reverse_tt(
        tt_cores: typ.Union[typ.Sequence[NDArray], NDArray]
) -> typ.Union[typ.Sequence[NDArray], NDArray]:
    """Reverse a tensor train (no Tucker).

    Parameters
    ----------
    x : typ.Sequence[NDArray] or NDArray
        Either: tensor train (no Tucker) with:

            tucker_ranks=(n0,...,n(d-1)),

            tt_ranks=(1,r1,...,r(d-1),1).

        Or: uniform tensor train (no Tucker).


    Returns
    -------
    reversed_x : typ.Sequence[NDArray] or NDArray
        Either: tensor train (no Tucker) with index order reversed, and

            tucker_ranks=(n(d-1),...,n0),

            tt_ranks=(1,r(d-1),...,r1,1).

        Or: uniform tensor train (no Tucker) with index order reversed, and same structure as x.

    See Also
    --------
    reverse_t3
    """
    is_uniform = not isinstance(tt_cores, typ.Sequence)
    if is_uniform:
        return tt_cores[::-1, :, :, :].swapaxes(1, 3)
    else:
        return tuple([G.swapaxes(0, 2) for G in tt_cores[::-1]])



def absorb_edge_weights_into_cores(
        x0, # Tucker tensor train data. Should also work for uniform. Maybe move this
        weights: EdgeWeights,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    """Contract each edge weight into a neighboring core.

    Tensor network diagram illustrating groupings::

             ____     ____     ________
            /    \   /    \   /        \
        1---w---G0---w---G1---w---G2---w---1
                |        |        |
              / w      / w      / w
              | |      | |      | |
              | B0     | B1     | B2
              | |      | |      | |
              \ w      \ w      \ w
                |        |        |

    """
    is_uniform = not isinstance(x0[0], typ.Sequence)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    #
    tucker_cores0, tt_cores0 = x0
    shape_weights, tucker_weights, tt_weights = weights

    if is_uniform:
        tucker_cores = xnp.einsum('di,dio,do->dio', tucker_weights, tucker_cores0, shape_weights)
        first_tt_cores = xnp.einsum('di,diaj->diaj', tt_weights[:-2], tt_cores0[:-1])

        Gf = xnp.einsum('i,iaj,j->iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
        tt_cores = xnp.concatenate([first_tt_cores, Gf.reshape((1,) + Gf.shape)], axis=0)
    else:
        (tucker_cores,) = xmap(
            lambda tw_B_sw: (xnp.einsum('i,io,o->io', tw_B_sw[0], tw_B_sw[1], tw_B_sw[2]),),
            (tucker_weights, tucker_cores0, shape_weights)
        )
        (first_tt_cores,) = xmap(
            lambda lw_G: (xnp.einsum('i,iaj->iaj', lw_G[0], lw_G[1]),),
            (tt_weights[:-2], tt_cores0[:-1])
        )
        Gf = xnp.einsum('i,iaj,j->iaj', tt_weights[-2], tt_cores0[-1], tt_weights[-1])
        tt_cores = tuple(first_tt_cores) + (Gf,)

    return tucker_cores, tt_cores


def t3_zeros(
        shape:                  typ.Tuple[int,...],
        tucker_ranks:           typ.Tuple[int,...],
        tt_ranks:               typ.Tuple[int,...],
        vectorization_shape:    typ.Tuple[int,...] = (),
        use_jax: bool = False,
) -> TuckerTensorTrain:
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
    >>> vs = (2,3)
    >>> z = t3.t3_zeros(shape, tucker_ranks, tt_ranks, vectorization_shape=vs)
    >>> print(np.linalg.norm(z.to_dense()))
    0.0
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    vs = vectorization_shape

    tt_cores = tuple([xnp.zeros(vs+(tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])) for ii in range(len(tucker_ranks))])
    tucker_cores = tuple([xnp.zeros(vs+(n, N)) for n, N  in zip(tucker_ranks, shape)])
    return TuckerTensorTrain(tucker_cores, tt_cores)


def t3_corewise_randn(
        shape:                  typ.Tuple[int, ...],
        tucker_ranks:           typ.Tuple[int, ...],
        tt_ranks:               typ.Tuple[int, ...],
        vectorization_shape:    typ.Tuple[int, ...] = (),
        use_jax: bool = False,
) -> TuckerTensorTrain:
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
    >>> vectorization_shape = (2,3)
    >>> x = t3.t3_corewise_randn(shape, tucker_ranks, tt_ranks, vectorization_shape=vectorization_shape) # TuckerTensorTrain with random cores
    >>> x.structure == (shape, tucker_ranks, tt_ranks)
    True
    >>> print(x.vectorization_shape == vectorization_shape)
    True
    >>> print(x.tucker_cores[0][0,0,0,0]) # should be random N(0,1)
    0.0331003310807162
    >>> print(x.tt_cores[0][0,0,0,0,0]) # should be random N(0,1)
    -0.10778923886039414
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    d = len(tucker_ranks)
    vs = vectorization_shape

    tt_cores = []
    for ii in range(d):
        shape_G = vs + (tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])
        G = randn(*shape_G, use_jax=use_jax)
        tt_cores.append(G)

    tucker_cores = []
    for ii in range(d):
        shape_B = vs + (tucker_ranks[ii], shape[ii])
        B = randn(*shape_B, use_jax=use_jax)
        tucker_cores.append(B)

    return TuckerTensorTrain(tuple(tucker_cores), tuple(tt_cores))


def t3_save(
        file,
        x: TuckerTensorTrain,
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
    tucker_cores, tt_cores = x.data
    cores_dict = {'tucker_cores_'+str(ii): tucker_cores[ii] for ii in range(len(tucker_cores))}
    cores_dict.update({'tt_cores_'+str(ii): tt_cores[ii] for ii in range(len(tt_cores))})

    try:
        np.savez(file, **cores_dict)
    except RuntimeError:
        print('Failed to save TuckerTensorTrain to file')


def t3_load(
        file,
        use_jax: bool = False,
) -> TuckerTensorTrain:
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

    tucker_cores = [xnp.array(B) for B in tucker_cores] # in case we are using jax or some other linalg backend
    tt_cores = [xnp.array(G) for G in tt_cores]

    return TuckerTensorTrain(tuple(tucker_cores), tuple(tt_cores))


###############################################################################
########    Scalar valued multilinear function applies and entries    #########
###############################################################################

def t3_apply(
        x: TuckerTensorTrain, # shape=(N0,...,N(d-1))
        vecs: typ.Sequence[NDArray], # len=d, elm_shape=(Ni,) or (num_applies, Ni)
        use_jax: bool = False,
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train. shape=(N0,...,N(d-1))
    vecs: typ.Sequence[NDArray]
        Vectors to contract with indices of x. len=d, elm_shape=(Ni,) or (num_applies, Ni) if vectorization is desired.
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
    >>> result = t3.t3_apply(x, vecs) # <-- contract x with vecs in all indices
    >>> result2 = np.einsum('ijk,i,j,k', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.abs(result - result2))
    5.229594535194337e-12

    Apply to multiple sets of vectors (vectorized):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    3.1271953680324864e-12

    First and last TT-ranks are not ones:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,4))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    6.481396196459234e-12

    Example using jax automatic differentiation:

	>>> import numpy as np
    >>> import jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> jax.config.update("jax_enable_x64", True)
    >>> A = t3.t3_corewise_randn((10,10,10),(5,5,5),(1,4,4,1)) # random 10x10x10 Tucker tensor train
    >>> apply_A_sym = lambda u: t3.t3_apply(A, (u,u,u), use_jax=True) # symmetric apply function
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
    shape, tucker_ranks, tt_ranks = x.structure

    if x.vectorization_shape != ():
        raise NotImplementedError(
            'Apply with vectorized TuckerTensorTrain is not implemented yet. Can only vectorize over input vectors.'
        )

    if len(vecs) != len(shape):
        raise ValueError(
            'Attempted to apply TuckerTensorTrain to wrong number of vectors.'
            + str(str(len(shape)) + ' = num_indices != len(vecs) = ' + str(len(vecs)))
        )

    vecs_dims = [len(v.shape) for v in vecs]
    if vecs_dims != [vecs_dims[0]]*len(vecs_dims):
        raise ValueError(
            'Inconsistent array dimensions for vecs.'
            + '[len(v.shape) for v in vecs]=' + str([len(v.shape) for v in vecs])
        )

    vectorized = True
    if vecs_dims[0] == 1:
        vectorized = False
        vecs = [v.reshape((1,-1)) for v in vecs]

    num_applies = vecs[0].shape[0]
    if [v.shape[0] for v in vecs] != [num_applies] * len(vecs):
        raise ValueError(
            'Inconsistent numbers of applies per index.'
            + '[v.shape[0] for v in vecs]=' + str([v.shape[0] for v in vecs])
        )

    vector_sizes = tuple([v.shape[1] for v in vecs])
    if vector_sizes != shape:
        raise ValueError(
            'Input vector sizes to not match tensor shape.'
            + str(vector_sizes) + ' = vector_sizes != x_shape = ' + str(shape)
        )

    mu_na = xnp.ones((num_applies, tt_cores[0].shape[0]))
    for V_ni, B_xi, G_axb in zip(vecs, tucker_cores, tt_cores):
        v_nx = xnp.einsum('ni,xi->nx', V_ni, B_xi)
        g_anb = xnp.einsum('axb,nx->anb', G_axb, v_nx)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        mu_na = mu_nb
    result = xnp.einsum('na->n', mu_na)

    if not vectorized:
        result = result[0]

    return result


def t3_entry(
        x: TuckerTensorTrain, # shape=(N0,...,N(d-1))
        index: typ.Union[typ.Sequence[int], typ.Sequence[typ.Sequence[int]]], # len=d. one entry: typ.Sequence[int]. many entries: typ.Sequence[typ.Sequence[int]], elm_size=num_entries
        use_jax: bool = False,
) -> NDArray:
    '''Compute an entry (or multiple entries) of a Tucker tensor train.

    This is the entry of the N0 x ... x N(d-1) tensor *represented* by the
    Tucker tensor train, even though this dense tensor is never formed.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train. shape=(N0,...,N(d-1))
    index: typ.Union[typ.Sequence[int], typ.Sequence[typ.Sequence[int]]]
        Index of the desired entry (typ.Sequence[int]), or indices of desired entries (typ.Sequence[typ.Sequence[int]])
        len(index)=d. If many entries: elm_size=num_entries
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    scalar or NDArray
        Desired entry or entries.
        Scalar if one entry, or NDArray with shape (num_entries,) if many entries.

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
    >>> result = t3.t3_entry(x, index)
    >>> result2 = x.to_dense()[9, 4, 7]
    >>> print(np.abs(result - result2))
    1.3322676295501878e-15

    Compute multiple entries (vectorized):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> index = [[9,8], [4,10], [7,13]] # get entries (9,4,7) and (8,10,13)
    >>> entries = t3.t3_entry(x, index)
    >>> x_dense = x.to_dense(x)
    >>> entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
    >>> print(np.linalg.norm(entries - entries2))
    1.7763568394002505e-15

    Example using jax jit compiling:

	>>> import numpy as np
    >>> import jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> get_entry_123 = lambda x: t3.t3_entry(x, (1,2,3), use_jax=True)
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
    >>> get_entry_123 = lambda x: t3.t3_entry(x, (1,2,3), use_jax=True)
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

    #
    tucker_cores, tt_cores = x.data
    shape, tucker_ranks, tt_ranks = x.structure

    if x.vectorization_shape != ():
        raise NotImplementedError(
            'Apply with vectorized TuckerTensorTrain is not implemented yet. Can only vectorize over input vectors.'
        )

    if len(index) != len(shape):
        raise ValueError(
            'Wrong number of indices for TuckerTensorTrain.'
            + str(str(len(shape)) + ' = num tensor indices != num provided indices = ' + str(len(index)))
        )

    vectorized = True
    if isinstance(index[0], int):
        vectorized = False
        index = [[ind] for ind in index]
    else:
        index = [list(ind) for ind in index]

    num_entries = len(index[0])
    if [len(ind) for ind in index] != [num_entries] * len(shape):
        raise ValueError(
            'Inconsistent numbers of index entries across different dimensions. The following should be all the same:'
            + '[len(ind) for ind in index]=' + str([len(ind) for ind in index])
        )

    mu_na = xnp.ones((num_entries, tt_cores[0].shape[0]))
    for ind, B_xi, G_axb in zip(index, tucker_cores, tt_cores):
        v_xn = B_xi[:, ind]
        g_anb = xnp.einsum('axb,xn->anb', G_axb, v_xn)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        mu_na = mu_nb
    result = xnp.einsum('na->n', mu_na)

    if not vectorized:
        result = result[0]

    return result


########################################################################
########################    Rank adjustment    #########################
########################################################################

def compute_minimal_ranks(
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
    >>> print(t3.compute_minimal_ranks((10,11,12,13), (14,15,16,17), (98,99,100,101,102)))
    ((10, 11, 12, 13), (1, 10, 100, 13, 1))
    '''
    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = int(np.minimum(new_tucker_ranks[ii], shape[ii]))

    new_tt_ranks[-1] = 1
    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = int(np.minimum(rL, n*rR))

    new_tt_ranks[0] = 1
    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = int(np.minimum(n, rL*rR))
        rR =int(np.minimum(rR, rL*n))
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    return tuple(new_tucker_ranks), tuple(new_tt_ranks)


# def are_t3_ranks_minimal(
#         x: TuckerTensorTrain,
# ) -> bool:
#     """Check if the ranks of a Tucker tensor train are minimal.
#
#     See Also:
#         compute_minimal_ranks
#
#     Example
#     -------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,4,9,7,1)))
#     >>> print(t3.are_t3_ranks_minimal(x))
#     True
#
#     Using T3-SVD to make equivalent T3 with minimal ranks:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> import t3toolbox.t3svd as t3svd
#     >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,99,9,7,1)))
#     >>> print(t3.are_t3_ranks_minimal(x))
#     False
#     >>> x2 = t3svd.t3_svd(x)[0]
#     >>> print(t3.are_t3_ranks_minimal(x2))
#     True
#     """
#     minimal_tucker_ranks, minimal_tt_ranks = compute_minimal_ranks(*x.structure)
#     return (x.tucker_ranks == minimal_tucker_ranks) and (x.tt_ranks == minimal_tt_ranks)


def change_tucker_core_shapes(
        tucker_cores: typ.Sequence[NDArray],
        new_shape: typ.Sequence[int], # len=d
        new_tucker_ranks: typ.Sequence[int], # len=d
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]:
    """Increase Tucker and/or TT ranks for TT cores using zero padding.
    """
    xnp, _, _ = get_backend(False, use_jax)

    old_shape = [B.shape[1] for B in tucker_cores]
    old_tucker_ranks = [B.shape[0] for B in tucker_cores]

    num_cores = len(tucker_cores)

    delta_shape         = [N_new - N_old for N_new, N_old in zip(new_shape, old_shape)]
    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]

    new_tucker_cores = []
    for ii in range(num_cores):
        new_tucker_cores.append(xnp.pad(
            tucker_cores[ii],
            (
                (0,delta_tucker_ranks[ii]),
                (0,delta_shape[ii]),
            ),
        ))

    return tuple(new_tucker_cores)


def change_tt_core_shapes(
        tt_cores: typ.Sequence[NDArray],
        new_tucker_ranks: typ.Sequence[int], # len=d
        new_tt_ranks: typ.Sequence[int], # len=d+1
        use_jax: bool = False,
) -> typ.Tuple[NDArray,...]:
    """Increase Tucker and/or TT ranks for TT cores using zero padding.
    """
    xnp, _, _ = get_backend(False, use_jax)

    old_tucker_ranks = [G.shape[1] for G in tt_cores]
    old_tt_ranks = [G.shape[0] for G in tt_cores] + [tt_cores[-1].shape[2]]

    num_cores = len(tt_cores)

    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    new_tt_cores = []
    for ii in range(num_cores):
        new_tt_cores.append(xnp.pad(
            tt_cores[ii],
            (
                (0,delta_tt_ranks[ii]),
                (0,delta_tucker_ranks[ii]),
                (0,delta_tt_ranks[ii+1]),
            ),
        ))

    return tuple(new_tt_cores)


# def change_structure(
#         x:                  TuckerTensorTrain,
#         new_structure, #:      T3Structure,
#         use_jax: bool = False,
# ) -> TuckerTensorTrain:
#     '''Increase Tucker tensor train ranks and/or shape via zero padding.
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
#     >>> new_structure = ((17,18,17), (8,8,8), (1,5,6,1))
#     >>> padded_x = t3.change_structure(x, new_structure)
#     >>> print(t3.get_structure(padded_x))
#     ((17, 18, 17), (8, 8, 8), (1, 5, 6, 1))
#
#     Example where first and last ranks are nonzero:
#
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (3,3,2,4)))
#     >>> new_structure = ((17,18,17), (8,8,8), (5,5,6,7))
#     >>> padded_x = t3.change_structure(x, new_structure)
#     >>> print(t3.get_structure(padded_x))
#     ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7))
#     '''
#     new_shape, new_tucker_ranks, new_tt_ranks = new_structure
#     tucker_cores, tt_cores = x
#
#     new_tucker_cores = change_tucker_core_shapes(tucker_cores, new_shape, new_tucker_ranks, use_jax=use_jax)
#     new_tt_cores = change_tt_core_shapes(tt_cores, new_tucker_ranks, new_tt_ranks, use_jax=use_jax)
#
#     return tuple(new_tucker_cores), tuple(new_tt_cores)


###########################################################
##################    Linear algebra    ###################
###########################################################

# def t3_add(
#         x: TuckerTensorTrain,
#         y: TuckerTensorTrain,
#         squash: bool = True,
#         use_jax: bool = False,
# ) -> TuckerTensorTrain:
#     """Add two Tucker tensor trains, yielding a Tucker tensor train with summed ranks.
#
#     Addition is defined with respect to the dense N0 x ... x N(d-1) tensors that
#     are *represented* by the Tucker tensor trains, even though these dense tensors
#     are not formed during computations.
#
#     For corewise addition, see :func:`t3toolbox.corewise.corewise_add`
#
#     Parameters
#     ----------
#     x: TuckerTensorTrain
#         First summand. structure=((N0,...,N(d-1)), (n0,...,n(d-1)), (r0, r1,...,rd))
#     y: TuckerTensorTrain
#         Second summand. structure=((N0,...,N(d-1)), (m0,...,m(d-1)), (q0, q1,...,qd))
#     squash: bool
#         Squash the first and last TT cores so that r0=rd=1 in the result. Default: True.
#     xnp:
#         Linear algebra backend. Default: np (numpy)
#
#     Returns
#     -------
#     TuckerTensorTrain
#         Sum of Tucker tensor trains, x+y.
#             - shape=(N0,...,N(d-1),
#             - tucker_ranks=(n0+m0,...,n(d-1)+m(d-1),
#             - TT ranks=(1, r1+q1,...,r(d-1)+q(d-1),1)) if squash=True,
#             or (r0+q0, r1+q1,...,r(d-1)+q(d-1),rd+qd)) if squash=False.
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
#     t3_scale
#     t3_sub
#     t3_neg
#     squash_tails
#     :func:`~t3toolbox.corewise.corewise_add`
#
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
#     >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
#     >>> z = t3.t3_add(x, y)
#     >>> print(t3.get_structure(z))
#     ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2))
#     >>> print(np.linalg.norm(t3.t3_to_dense(x) + t3.t3_to_dense(y) - t3.t3_to_dense(z)))
#     6.524094086845177e-13
#     """
#     xnp, _, _ = get_backend(False, use_jax)
#
#     #
#     check_t3(x)
#     check_t3(y)
#
#     x_shape = get_structure(x)[0]
#     y_shape = get_structure(y)[0]
#     if x_shape != y_shape:
#         raise ValueError(
#             'Attempted to add TuckerTensorTrains x+y with inconsistent shapes.'
#             + str(x_shape) + ' = x_shape != y_shape = ' + str(y_shape)
#         )
#
#     tucker_cores_x, tt_cores_x = x
#     tucker_cores_y, tt_cores_y = y
#     tucker_cores_z = [xnp.concatenate([Bx, By], axis=0) for Bx, By in zip(tucker_cores_x, tucker_cores_y)]
#
#     tt_cores_z = []
#
#     for Gx, Gy in zip(tt_cores_x, tt_cores_y):
#         G000 = Gx
#         G001 = xnp.zeros((Gx.shape[0], Gx.shape[1], Gy.shape[2]))
#         G010 = xnp.zeros((Gx.shape[0], Gy.shape[1], Gx.shape[2]))
#         G011 = xnp.zeros((Gx.shape[0], Gy.shape[1], Gy.shape[2]))
#         G100 = xnp.zeros((Gy.shape[0], Gx.shape[1], Gx.shape[2]))
#         G101 = xnp.zeros((Gy.shape[0], Gx.shape[1], Gy.shape[2]))
#         G110 = xnp.zeros((Gy.shape[0], Gy.shape[1], Gx.shape[2]))
#         G111 = Gy
#         Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
#         tt_cores_z.append(Gz)
#
#     z = (tuple(tucker_cores_z), tuple(tt_cores_z))
#     if squash:
#         z = squash_tails(z)
#     return z


# def t3_scale(
#         x: TuckerTensorTrain,
#         s, # scalar
# ) -> TuckerTensorTrain:
#     """Multipy a Tucker tensor train by a scaling factor.
#
#     Scaling is defined with respect to the dense N0 x ... x N(d-1) tensor that
#     is *represented* by the Tucker tensor trains, even though this dense tensor
#     is not formed during computations.
#
#     For corewise scaling, see :func:`t3toolbox.corewise.corewise_scale`
#
#     Parameters
#     ----------
#     x: TuckerTensorTrain
#         Tucker tensor train
#     s: scalar
#         scaling factor
#
#     Returns
#     -------
#     TuckerTensorTrain
#         Scaled TuckerTensorTrain s*x, with the same structure as x.
#
#     Raises
#     ------
#     ValueError
#         - Error raised if the TuckerTensorTrains are internally inconsistent
#
#     See Also
#     --------
#     TuckerTensorTrain
#     t3_add
#     t3_neg
#     t3_sub
#     :func:`~t3toolbox.corewise.corewise_scale`
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
#     >>> s = 3.2
#     >>> z = t3.t3_scale(x, s)
#     >>> print(np.linalg.norm(s*t3.t3_to_dense(x) - t3.t3_to_dense(z)))
#     1.6268482531988893e-13
#     """
#     check_t3(x)
#
#     tucker_cores, tt_cores = x
#
#     scaled_tucker_cores = [B.copy() for B in tucker_cores]
#     scaled_tucker_cores[-1] = s*scaled_tucker_cores[-1]
#
#     copied_tt_cores = [G.copy() for G in tt_cores]
#
#     z = (tuple(scaled_tucker_cores), tuple(copied_tt_cores))
#     return z


# def t3_neg(
#         x: TuckerTensorTrain,
# ) -> TuckerTensorTrain:
#     """Scale a Tucker tensor train by -1.
#
#     Negation is defined with respect to the dense N0 x ... x N(d-1) tensor that
#     is *represented* by the Tucker tensor trains, even though this dense tensor
#     is not formed during computations.
#
#     For corewise negation, see :func:`t3toolbox.corewise.corewise_neg`
#
#     Parameters
#     ----------
#     x: TuckerTensorTrain
#         Tucker tensor train
#
#     Returns
#     -------
#     TuckerTensorTrain
#         Negated TuckerTensorTrain -x, with the same structure as x.
#
#     Raises
#     ------
#     ValueError
#         - Error raised if the TuckerTensorTrains is internally inconsistent
#
#     See Also
#     --------
#     TuckerTensorTrain
#     t3_add
#     t3_scale
#     t3_sub
#     :func:`~t3toolbox.corewise.corewise_neg`
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
#     >>> neg_x = t3.t3_neg(x)
#     >>> print(np.linalg.norm(t3.t3_to_dense(x) + t3.t3_to_dense(neg_x)))
#     0.0
#     """
#     return t3_scale(x, -1.0)


# def t3_sub(
#         x: TuckerTensorTrain,
#         y: TuckerTensorTrain,
#         squash: bool = True,
#         use_jax: bool = False,
# ) -> TuckerTensorTrain:
#     """Subtract two Tucker tensor trains, yielding a Tucker tensor train with summed ranks.
#
#     Subtraction is defined with respect to the dense N0 x ... x N(d-1) tensors that
#     are *represented* by the Tucker tensor trains, even though these dense tensors
#     are not formed during computations.
#
#     For corewise subtraction, see :func:`t3toolbox.corewise.corewise_sub`
#
#     Parameters
#     ----------
#     x: TuckerTensorTrain
#         First summand. structure=((N0,...,N(d-1)), (n1,...,nd), (r0, r1,...,rd))
#     y: TuckerTensorTrain
#         Second summand. structure=((N0,...,N(d-1)), (m1,...,md), (q0, q1,...,qd))
#     squash: bool
#         Squash the first and last TT cores so that r0=rd=1 in the result. Default: True.
#     xnp:
#         Linear algebra backend. Default: np (numpy)
#
#     Returns
#     -------
#     TuckerTensorTrain
#         Difference of Tucker tensor trains, x-y.
#             - shape=(N0,...,N(d-1),
#             - tucker_ranks=(n0+m0,...,n(d-1)+m(d-1),
#             - TT ranks=(1, r1+q1,...,r(d-1)+q(d-1),1)) if squash=True,
#             or (r0+q0, r1+q1,...,r(d-1)+q(d-1),rd+qd)) if squash=False.
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
#     t3_neg
#     :func:`~t3toolbox.corewise.corewise_neg`
#
#     Examples
#     --------
#     >>> import numpy as np
#     >>> import t3toolbox.tucker_tensor_train as t3
#     >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
#     >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
#     >>> x_minus_y = t3.t3_sub(x, y)
#     >>> print(t3.get_structure(x_minus_y))
#     ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2))
#     >>> dense_x = t3.t3_to_dense(x)
#     >>> dense_y = t3.t3_to_dense(y)
#     >>> dense_x_minus_y = t3.t3_to_dense(x_minus_y)
#     >>> print(np.linalg.norm(dense_x - dense_y - dense_x_minus_y))
#     3.5875705233607603e-13
#     """
#     return t3_add(x, t3_neg(y), squash=squash, use_jax=use_jax)


def t3_inner_product_t3(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt inner product of two Tucker tensor trains.

    The Hilbert-Schmidt inner product is defined with respect to the dense N0 x ... x N(d-1)
    tensors that are *represented* by the Tucker tensor trains, even though these dense tensors
    are not formed during computations.

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
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (1,5,6,1))
    >>> x_dot_y = t3.t3_inner_product_t3(x, y)
    >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense())
    >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
    8.731149137020111e-11

    Example where leading and trailing TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2))
    >>> y = t3.t3_corewise_randn((14,15,16), (3,7,2), (3,5,6,3))
    >>> x_dot_y = t3.t3_inner_product_t3(x, y)
    >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense())
    >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
    1.3096723705530167e-10
    """
    xnp, _, _ = get_backend(False, use_jax)

    #

    if x.vectorization_shape != () or y.vectorization_shape != ():
        raise NotImplementedError(
            'T3 inner product not implemented for vectorized TuckerTensorTrains.'
        )

    if x.shape != y.shape:
        raise ValueError(
            'Attempted to dot TuckerTensorTrains (x,y)_HS with inconsistent shapes.'
            + str(x.shape) + ' = x_shape != y_shape = ' + str(y.shape)
        )

    x = x.squash_tails(use_jax=use_jax)
    y = y.squash_tails(use_jax=use_jax)

    tucker_cores_x, tt_cores_x = x.data
    tucker_cores_y, tt_cores_y = y.data

    r0_x = tt_cores_x[0].shape[0]
    r0_y = tt_cores_y[0].shape[0]

    M_sp = xnp.ones((r0_x, r0_y))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(tucker_cores_x, tt_cores_x, tucker_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('ai,bi->ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('sat,ab->sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('sp,sbt->pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('pbt,pbq->tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    rd_x = tt_cores_x[-1].shape[2]
    rd_y = tt_cores_y[-1].shape[2]

    result = xnp.einsum('tq,t,q', M_sp, np.ones(rd_x), np.ones(rd_y))
    return result


def t3_norm(
        x: TuckerTensorTrain,
        use_jax: bool = False,
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
    >>> norm_x = t3.t3_norm(x)
    >>> print(np.abs(norm_x - np.linalg.norm(x.to_dense())))
    1.3642420526593924e-12
    """
    xnp, _, _ = get_backend(False, use_jax)

    #
    return xnp.sqrt(t3_inner_product_t3(x, x, use_jax=use_jax))


