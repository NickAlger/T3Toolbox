# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""
Basic Tucker tensor trains with non-uniform (ragged) shape and ranks.
"""
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass
from functools import cached_property

import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.dense_t3svd as dense_t3svd
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.t3_orthogonalization as ragged_orthogonalization
import t3toolbox.backend.t3_linalg as ragged_linalg
import t3toolbox.backend.t3_svd as ragged_t3svd
import t3toolbox.corewise as corewise

import t3toolbox.backend.common as common
from t3toolbox.backend.common import NDArray
from collections.abc import Sequence
from typing import Tuple

jax = None
if common.has_jax:
    import jax

__all__ = [
    'TuckerTensorTrain',
]


###########################################
########    Tucker Tensor Train    ########
###########################################


@dataclass(frozen=True)
class TuckerTensorTrain:
    """
    Tucker tensor train with non-uniform (ragged) shape and ranks.

    Tensor network diagram for a TuckerTensorTrain with ``d`` free indices::

            r0        r1        r2       r(d-1)          rd
        1 ------ G0 ------ G1 ------ ... ------ G(d-1) ------ 1
                 |         |                    |
                 | n0      | n1                 | nd
                 |         |                    |
                 B0        B1                   B(d-1)
                 |         |                    |
                 | N0      | N1                 | Nd
                 |         |                    |

    Cores:
    ------
    The TuckerTensorTrain is defined by its cores:

    - :py:attr:`~tucker_cores`: Tuple[NDArray,...]
        ``tucker_cores = (B0, ..., B(d-1))``, ``Bi.shape=stack_shape+(ni, Ni)``
    - :py:attr:`~tt_cores`: Tuple[NDArray,...]
        ``tt_cores = (G0, ..., G(d-1))``, ``Gi.shape=stack_shape+(ri, ni, r(i+1))``

    Example:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
    >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with zeros
    >>> print(x.core_shapes)
    (((4, 14), (5, 15), (6, 16)), ((1, 4, 3), (3, 5, 2), (2, 6, 1)))
    >>> print(x.data == (tucker_cores, tt_cores))
    True

    Shape and ranks:
    ----------------
    The structure of a Tucker tensor train is defined by its shape and ranks:

    - :py:attr:`~shape`: Tuple[int,...]
        ``shape = (N0, N1, ..., N(d-1))``
    - :py:attr:`~tucker_ranks`: Tuple[int,...]
        ``tucker_ranks = (n0, r1, ..., n(d-1))``
    - :py:attr:`~tt_ranks`: Tuple[int,...]
        ``tt_ranks = (r0, r1, ..., rd)``
    - :py:attr:`~stack_shape`: Tuple[int,...]
        (optional, more on this below)

    Often, the first and last TT-ranks satisfy ``r0=rd=1``, and "1" in the diagram
    is the number 1. However, it is allowed for these ranks to not be 1, in which case
    the "1"s in the diagram are vectors of ones. You can make ``r0=rd=1`` using :py:meth:`~squash`.

    Example:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
    >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with zeros
    >>> print(x.d)
    3
    >>> print(x.shape)
    (14, 15, 16)
    >>> print(x.tucker_ranks)
    (4, 5, 6)
    >>> print(x.tt_ranks)
    (1, 3, 2, 1)


    Stacking:
    ---------
    Many stacked Tucker tensor trains with the same shape and ranks may be stored in this object for vectorized operations.
    In this case,
        - ``tucker_cores[ii].shape=stack_shape+(ni,Ni)``
        - ``tt_cores[ii].shape=stack_shape+(ri, ni, r(i+1))``

    If no stacking is used, then ``stack_shape=()``.

    Operations that use a numerical tolerance (``rtol`` or ``atol``) cannot be used with stacked TuckerTensorTrains
    because the shape of the results could vary between different elements of the stack.

    Examples:

    Create a stacked TuckerTensorTrain from stacked core arrays:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = [np.ones((6,7, 4,14)),np.ones((6,7, 5,15)),np.ones((6,7, 6,16))]
    >>> tt_cores = [np.ones((6,7, 1,4,3)), np.ones((6,7, 3,5,2)), np.ones((6,7, 2,6,1))]
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with ones
    >>> print(x.stack_shape)
    (6, 7)
    >>> print(x.structure)
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), (6, 7))

    Create a stacked TuckerTensorTrain by stacking several TuckerTensorTrains:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x00 = t3.TuckerTensorTrain.randn((13,14,15), (4,5,6), (2,8,9,3))
    >>> x01 = t3.TuckerTensorTrain.randn((13,14,15), (4,5,6), (2,8,9,3))
    >>> x10 = t3.TuckerTensorTrain.randn((13,14,15), (4,5,6), (2,8,9,3))
    >>> x11 = t3.TuckerTensorTrain.randn((13,14,15), (4,5,6), (2,8,9,3))
    >>> print([B.shape for B in x00.tucker_cores])
    [(4, 13), (5, 14), (6, 15)]
    >>> print([G.shape for G in x00.tt_cores])
    [(2, 4, 8), (8, 5, 9), (9, 6, 3)]
    >>> print(x00.stack_shape)
    ()
    >>> x_stacked = t3.TuckerTensorTrain.stack([[x00, x01], [x10, x11]])
    >>> print([B.shape for B in x_stacked.tucker_cores])
    [(2, 2, 4, 13), (2, 2, 5, 14), (2, 2, 6, 15)]
    >>> print([G.shape for G in x_stacked.tt_cores])
    [(2, 2, 2, 4, 8), (2, 2, 8, 5, 9), (2, 2, 9, 6, 3)]
    >>> print(x_stacked.stack_shape)
    (2, 2)

    Using ``rtol`` option in :py:meth:`~t3svd` yields an error for stacked TuckerTensorTrains

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.TuckerTensorTrain.randn((13,14,15), (4,5,6), (2,8,9,3))
    >>> result = x.t3svd() # OK
    >>> result = x.t3svd(rtol=1e-2) # OK
    >>> x = t3.TuckerTensorTrain.randn((13,14,15), (4,5,6), (2,8,9,3), stack_shape=(2,3))
    >>> result = x.t3svd() # OK
    >>> result = x.t3svd(rtol=1e-2) # Error!
    ValueError: Cannot use rtol or atol with t3svd for stacked Tucker tensor train.
    Different elements of the stack could end out having different shapes.
    First unstack, then call t3svd for each unstacked Tucker tensor train.


    Minimal ranks:
    --------------
    Tucker tensor train ranks are minimal if they satisfy the following conditions,
        - ``r(i+1) <= (ri*ni)`` for ``i=1,...,d``
        - ``ri <= (ni*r(i+1))`` for ``i=1,...,d``
        - ``ni <= (ri*r(i+1))`` for ``i=1,...,d``
        - ``ni <= Ni`` for ``i=1,...,d``

    The first three conditions say that the product of any two dimensions of a TT core
    is at least as large as the other dimension. The last condition says that the Tucker ranks
    are less than the tensor shape.

    Here, minimal ranks are defined with respect to a generic Tucker tensor train
    with the given shape and rank structure. We do not account for numerical
    rank deficiency.

    Minimal ranks always exist and are unique.
        - Minimal TT ranks are equal to the ranks of ``(N0*...*Ni) x (N(i+1)*...*N(d-1))`` matrix unfoldings.
        - Minimal Tucker ranks are equal to the ranks of ``Ni x (N0*...*N(i-1)*N(i+1)*...*N(d-1))`` matricizations.

    More details on the connection between minimal ranks and unfoldings/matricizations are given in Section 2.3 of [1]_.

    Example:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,99,6,7), (1,4,9,7,1)) # random T3
    >>> print(x.ranks)
    ((4, 99, 6, 7), (1, 4, 9, 7, 1))
    >>> print(x.minimal_ranks)
    ((4, 14, 6, 7), (1, 4, 9, 7, 1))
    >>> print(x.has_minimal_ranks)
    False
    >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (1,4,9,7,1))
    >>> print(x.has_minimal_ranks)
    True
    >>> x = t3.TuckerTensorTrain.zeros((13,14,15,16), (4,5,6,7), (1,4,9,7,1)) # T3 filled with zeros
    >>> print(x.has_minimal_ranks) # minimal ranks depends on structural ranks, not numerical ranks
    True

    Making a TuckerTensorTrain have minimal ranks using :py:meth:`~t3svd`:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (1,99,9,7,1))
    >>> print(x.has_minimal_ranks)
    False
    >>> print(x.minimal_ranks)
    >>> x2, _, _ = x.t3svd()
    >>> print(x2.has_minimal_ranks)
    True


    Tensor linear algebra:
    ----------------------
    Linear algebra operations (:py:meth:`addition <__add__>`, :py:meth:`subtraction <__sub__>`, :py:meth:`multiplication <__mul__>`,
    :py:meth:`negation <__neg__>`, :py:meth:`inner products <inner>`, :py:meth:`norms <norm>`, :py:meth:`summing over axes <sum>`)
    are mathematically defined with respect to the ``N0 x ... x N(d-1)`` dense tensors represented by the Tucker tensor trains.
    These operations are performed implicitly using Tucker tensor train cores as a computational device,
    because the dense tensors can be extremely large.
    The results faithfully represent what one would have gotten if one performed the operations on the dense tensors.
    E.g.:
    .. math:: (x + y).to_dense() = x.to_dense() + y.to_dense()

    Adding Tucker tensor trains adds their ranks, and multiplication multiplies their ranks.
    To prevent ranks growing too large when many linear algebra operations are performed in sequence,
    it may be useful to perform truncated T3SVDs between operations
    (using either ``max_tucker_ranks``, ``rtol``, or ``atol`` as parameters in :py:meth:`t3svd`).

    For corewise operations, see :py:mod:`t3toolbox.corewise`

    Examples:

    Add two TuckerTensorTrains

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (2,8,9,7,3))
    >>> y = t3.TuckerTensorTrain.randn((13,14,15,16), (9,8,7,6), (1,2,3,4,5))
    >>> print(np.linalg.norm((x + y).to_dense() - (x.to_dense() + y.to_dense())))
    3.8159914295689006e-11

    A more complicated linear algebra operation with three TuckerTensorTrains

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (2,3,4,3), (1,2,4,3,2))
    >>> y = t3.TuckerTensorTrain.randn((13,14,15,16), (4,3,5,1), (4,3,2,1,2))
    >>> z = t3.TuckerTensorTrain.randn((13,14,15,16), (1,2,3,4), (1,2,3,4,5))
    >>> result = (x * (y * 2.4 + z)).inner(z) + (x - y).norm() + z.sum()
    >>> X, Y, Z = x.to_dense(), y.to_dense(), z.to_dense()
    >>> result2 = np.einsum('ijkl,ijkl', (X * (Y * 2.4 + Z)), Z) + np.linalg.norm(X - Y) + Z.sum()
    >>> print(np.linalg.norm(result - result2) / np.linalg.norm(result2))
    1.1486488440369942e-15

    References
    ----------
    .. [1] Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
           Tucker Tensor Train Taylor Series.
           arXiv preprint arXiv:2603.21141.
           .. __: https://arxiv.org/abs/2603.21141
    """

    tucker_cores:   Tuple[NDArray,...] # len=d, elm_shape=stack_shape+(ni, Ni)
    """Tucker cores for the TuckerTensorTrain.
    
    - ``tucker_cores=(B0, ..., B(d-1))``. 
    - ``len(tucker_cores)=d``, 
    - ``tucker_cores[ii]=stack_shape+(ni, Ni)``.
    
    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
    >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
    >>> print(x.tucker_cores == tucker_cores)
    True
    """

    tt_cores:       Tuple[NDArray,...] # len=d, elm_shape=stack_shape+(ri, ni, r(i+1))
    """TT cores for the TuckerTensorTrain.

    - ``tt_cores=(G0, ..., G(d-1))``. 
    - ``len(tt_cores)=d``, 
    - ``tt_cores[ii]=stack_shape+(ri, ni, r(i+1))``.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
    >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
    >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
    >>> print(x.tt_cores == tt_cores)
    True
    """

    @cached_property
    def data(self) -> Tuple[Tuple[NDArray,...], Tuple[NDArray,...]]:
        """Tuple containing the Tucker cores and TT cores. ``data=(tucker_cores, tt_cores)``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.ones((4,14)),np.ones((5,15)),np.ones((6,16)))
        >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.data == (tucker_cores, tt_cores))
        True
        """
        return tuple(self.tucker_cores), tuple(self.tt_cores)

    @cached_property
    def d(self) -> int:
        """Number of indices of the tensor. ``d=len(tucker_cores)=len(tt_cores)``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.d)
        3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.d)
        2
        """
        return len(self.tucker_cores)

    @cached_property
    def stack_shape(self) -> Tuple[int, ...]:
        """If this object contains multiple stacked T3s with the same structure, this is the shape of the stack.
        If no stacking is used then ``stack_shape=()``.

        - ``tucker_cores[ii].shape  = stack_shape+(ni, Ni)``
        - ``tt_cores[ii].shape      = stack_shape+(ri, ni, r(i+1))``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = [np.zeros((4,14)),np.zeros((5,15)), np.zeros((6,16))]
        >>> tt_cores = [np.zeros((1,4,3)), np.zeros((3,5,2)), np.ones((2,6,1))]
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.stack_shape)
        ()
        >>> tucker_cores = [np.zeros((6, 4,14)),np.zeros((6, 5,15)), np.zeros((6, 6,16))]
        >>> tt_cores = [np.zeros((6, 1,4,3)), np.zeros((6, 3,5,2)), np.ones((6, 2,6,1))]
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.stack_shape)
        (6,)
        >>> tucker_cores = [np.zeros((6,7, 4,14)),np.zeros((6,7, 5,15)), np.zeros((6,7, 6,16))]
        >>> tt_cores = [np.zeros((6,7, 1,4,3)), np.zeros((6,7, 3,5,2)), np.ones((6,7, 2,6,1))]
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.stack_shape)
        (6, 7)
        """
        return self.tucker_cores[0].shape[:-2]

    @cached_property
    def shape(self) -> Tuple[int, ...]: # len=d
        """Shape of the represented dense tensor. ``shape=(N0,...,N(d-1))``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.shape)
        (14, 15, 16)
        """
        return tuple([B.shape[-1] for B in self.tucker_cores])

    @cached_property
    def tucker_ranks(self) -> Tuple[int, ...]: # len=d
        """Tucker ranks. ``tucker_ranks=(n0,...,n(d-1))``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.tucker_ranks)
        (4, 5, 6)
        """
        return tuple([B.shape[-2] for B in self.tucker_cores])

    @cached_property
    def tt_ranks(self) -> Tuple[int, ...]: # len=d+1
        """TT ranks. ``tt_ranks=(r0,...,rd)``

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.tt_ranks)
        (1, 3, 2, 1)
        """
        rr = tuple([G.shape[-3] for G in self.tt_cores]) + (self.tt_cores[-1].shape[-1],)
        return rr

    @cached_property
    def ranks(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Tuple containing Tucker ranks and TT ranks.

        - ``ranks           = (tucker_ranks, tt_ranks)``
        - ``tucker_ranks    = (n0,...,n(d-1))``
        - ``tt_ranks        = (r0,...,rd)``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.ranks)
        ((4, 5, 6), (1, 3, 2, 1))
        """
        return self.tucker_ranks, self.tt_ranks

    @cached_property
    def structure(self) -> Tuple[
        Tuple[int,...], # shape
        Tuple[int,...], # tucker_ranks
        Tuple[int,...], # tt_ranks
        Tuple[int,...], # stack_shape
    ]:
        """Tuple containing tensor shape, Tucker ranks, TT ranks, and stack shape.

        - ``structure = (shape, tucker_ranks, tt_ranks, stack_shape)``
        - ``shape           = (N0,...,N(d-1))``
        - ``tucker_ranks    = (n0,...,n(d-1))``
        - ``tt_ranks        = (r0,...,rd)``
        - ``stack_shape`` (optional, default: ``stack_shape=()``)

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.structure)
        ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), ())
        """
        return self.shape, self.tucker_ranks, self.tt_ranks, self.stack_shape

    @staticmethod
    def get_core_shapes(
            shape: Sequence[int],
            tucker_ranks: Sequence[int],
            tt_ranks: Sequence[int],
            stack_shape: Sequence[int] = (),
    ) -> Tuple[
        Tuple[int, ...],  # tucker_core_shapes
        Tuple[int, ...],  # tt_core_shapes
    ]:
        """Compute the Tucker and TT core shapes for a Tucker tensor train.

        Parameters
        ----------
        shape: Sequence[int]
            Shape of hypothetical TuckerTensorTrain. ``len(shape)=d``.
        tucker_ranks: Sequence[int]
            Tucker ranks of hypothetical TuckerTensorTrain. ``len(tucker_ranks)=d``.
        tt_ranks: Sequence[int]
            TT ranks of hypothetical TuckerTensorTrain. ``len(tt_ranks)=d+1``

        Returns
        -------
        (tucker_core_shapes, t_core_shapes): Tuple[Tuple[int,...], Tuple[int,...]]
            Tucker and TT core shapes for hypothetical TuckerTensorTrain with given shape and ranks.


        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(9,))
        >>> print(t3.TuckerTensorTrain.get_core_shapes(x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape))
        (((9, 4, 14), (9, 5, 15), (9, 6, 16)), ((9, 1, 4, 3), (9, 3, 5, 4), (9, 4, 6, 5)))
        >>> print(x.core_shapes)
        (((9, 4, 14), (9, 5, 15), (9, 6, 16)), ((9, 1, 4, 3), (9, 3, 5, 4), (9, 4, 6, 5)))
        """
        return ragged_operations.t3_core_shapes(
            shape, tucker_ranks, tt_ranks, stack_shape,
        )

    @cached_property
    def core_shapes(self) -> Tuple[
        Tuple[Tuple[int,...],...], # tucker core shapes
        Tuple[Tuple[int,...],...], # tt core shapes
    ]:
        """Shapes of the Tucker and TT cores.

        - ``cores_shapes            = (tucker_core_shapes, tt_core_shapes)``.
        - ``len(tucker_core_shapes) = len(tt_core_shapes) = d``
        - ``tucker_core_shapes[ii]  = (ni, Ni)``
        - ``tt_core_shapes[ii]      = (ri, ni, r(i+1))``

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with zeros
        >>> print(x.core_shapes)
        (((4, 14), (5, 15), (6, 16)), ((1, 4, 3), (3, 5, 2), (2, 6, 1)))
        """
        return (
            tuple([B.shape[len(self.stack_shape):] for B in self.tucker_cores]),
            tuple([G.shape[len(self.stack_shape):] for G in self.tt_cores]),
        )

    @cached_property
    def size(self) -> int:
        """Size of the dense tensor represented by this TuckerTensorTrain. ``size=N0*...*N(d-1)``.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with zeros
        >>> print(x.size == 14*15*16)
        True
        """
        return np.prod(self.shape)

    @cached_property
    def data_size(self) -> int:
        """Sum of the sizes of all Tucker and TT cores.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> tucker_cores = (np.zeros((4,14)), np.zeros((5,15)), np.zeros((6,16)))
        >>> tt_cores = (np.zeros((1,4,3)), np.zeros((3,5,2)), np.zeros((2,6,1)))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores) # TuckerTensorTrain, cores filled with zeros
        >>> print(x.data_size == 4*14 + 5*15 + 6*16 + 1*4*3 + 3*5*2 + 2*6*1)
        True
        """
        return sum([x.size for x in self.tucker_cores]) + sum([x.size for x in self.tt_cores])

    @staticmethod
    def get_minimal_ranks(
            shape: Sequence[int],
            tucker_ranks: Sequence[int],
            tt_ranks: Sequence[int],
    ) -> Tuple[
        Tuple[int, ...],  # new_tucker_ranks
        Tuple[int, ...],  # new_tt_ranks
    ]:
        '''Find minimal ranks for a hypothetical TuckerTensorTrain with given shape and ranks.

        Minimal ranks satisfy:
            - Left TT core unfoldings are full rank: ``r(i+1) <= (ri*ni)``
            - Right TT core unfoldings are full rank: ``ri <= (ni*r(i+1))``
            - Down TT core unfoldings are full rank: ``ni <= (ri*r(i+1))``
            - Tucker ranks do not exceed shape: ``ni <= Ni``

        In this function, minimal ranks are defined with respect to a
        generic Tucker tensor train of the given form based on its structure.
        We do not account for possible additional rank deficiency due to
        the numerical values within the cores.

        Minimal ranks always exist and are unique.
            - Minimal TT ranks are equal to the ranks of ``(N*...*Ni) x (N(i+1)*...*N(d-1))`` matrix unfoldings.
            - Minimal Tucker ranks are equal to the ranks of ``Ni x (N1*...*N(i-1)*N(i+1)*...*N(d-1))`` matricizations.

        Examples
        --------
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> print(t3.TuckerTensorTrain.get_minimal_ranks((10,11,12,13), (14,15,16,17), (98,99,100,101,102)))
        ((10, 11, 12, 13), (1, 10, 100, 13, 1))
        '''
        return ranks.compute_minimal_ranks(shape, tucker_ranks, tt_ranks)

    @cached_property
    def minimal_ranks(self) -> Tuple[Tuple[int,...], Tuple[int,...]]:
        """Ranks of the smallest possible TuckerTensorTrain that could represent 
        the same dense tensor as this TuckerTensorTrain. 
        TuckerTensorTrains ranks may be made minimal using T3-SVD.

        - ``minimal_ranks = (minimal_tucker_ranks, minimal_tt_ranks)``
        - ``len(minimal_tucker_ranks) = d``
        - ``len(minimal_tt_ranks) = d+1``

        Examples
        --------

        A Tucker rank is not minimal:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,99,6,7), (1,4,9,7,1))
        >>> print(x.ranks)
        ((4, 99, 6, 7), (1, 4, 9, 7, 1))
        >>> print(x.minimal_ranks)
        ((4, 14, 6, 7), (1, 4, 9, 7, 1))

        A TT-rank is not minimal:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (1,4,99,7,1))
        >>> print(x.ranks)
        ((4, 5, 6, 7), (1, 4, 99, 7, 1))
        >>> print(x.minimal_ranks)
        ((4, 5, 6, 7), (1, 4, 20, 7, 1))
        """
        minimal_tucker_ranks, minimal_tt_ranks = TuckerTensorTrain.get_minimal_ranks(
            self.shape, self.tucker_ranks, self.tt_ranks,
        )
        return minimal_tucker_ranks, minimal_tt_ranks

    @cached_property
    def has_minimal_ranks(self) -> bool:
        """True if this Tucker tensor train's ranks are minimal, False otherwise.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,99,6,7), (1,4,9,7,1))
        >>> print(x.has_minimal_ranks) # Tucker rank too big
        False
        >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (1,99,9,7,1))
        >>> print(x.has_minimal_ranks) # TT rank too big
        False
        >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (1,4,9,7,1))
        >>> print(x.has_minimal_ranks)
        True

        Make ranks minimal with t3svd:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (4,5,6,7), (1,99,9,7,1))
        >>> print(x.has_minimal_ranks)
        False
        >>> print(x.minimal_ranks)
        >>> x2 = x.t3svd()[0]
        >>> print(x2.has_minimal_ranks)
        True
        """
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
    ) -> NDArray:
        """Form dense tensor from this TuckerTensorTrain.

        Parameters
        ----------
        squash_tails: bool, optional
            Whether to contract the leading and trailing 1s with the first and last TT indices. (Default: True)

        Returns
        -------
        NDArray
            Dense tensor represented by this TuckerTensorTrain,
            which has ``shape=stack_shape+(N0, ..., N(d-1))`` if ``squash_tails=True``,
            or ``shape=stack_shape+(r0,N0,...,N(d-1),rd)`` if ``squash_tails=False``.

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
            self.data, squash_tails=squash_tails,
        )

    def segment(self, start: int, stop: int) -> 'TuckerTensorTrain':
        """Extract contiguous segment of this TuckerTensorTrain. Segments must have length at least one.

        Parameters
        ----------
        start: int
            Starting index for segment. Requires ``stop > start``.
        stop: int
            Stopping index for segment. Requires ``stop > start``.

        Returns
        -------
        TuckerTensorTrain
            Segment of this TuckerTensorTrain, with ``shape=(N(start), ..., N(stop-1))``.

        Raises
        ------
        ValueError
            If ``stop <= start``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.concatenate`

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
            xx: Sequence['TuckerTensorTrain'],
    ) -> 'TuckerTensorTrain':
        """Concatenates TuckerTensorTrain segments.

        Parameters
        ----------
        xx: Sequence[TuckerTensorTrain]
            TuckerTensorTrain segments to be concatenated

        Returns
        -------
        TuckerTensorTrain
            Concatenated TuckerTensorTrain.

        Raises
        ------
        ValueError
            If segments have incompatible leading and trailing TT ranks.
            I.e., if ``x[ii].tt_ranks[-1] != x[ii+1].tt_ranks[0]``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.segment`

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

    def squash(
            self,
    ) -> 'TuckerTensorTrain':
        """Make leading and trailing TT ranks equal to 1 (``r0=rd=1``), without changing represented dense tensor.

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train with ``tt_ranks=(1,r1,...,r(d-1),1)``.

        See Also:
        ---------
        :py:attr:`.TuckerTensorTrain.tt_ranks`

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
        >>> x2 = x.squash()
        >>> print(x2.tt_ranks)
        (1, 3, 2, 1)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense()))
        5.805155892491438e-12
        """
        return TuckerTensorTrain(self.tucker_cores, ragged_operations.squash_tt_tails(self.tt_cores))

    def reverse(self) -> 'TuckerTensorTrain':
        """Reverse Tucker tensor train.

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train with index order reversed.
            ``shape=(N(d-1), ..., N0)``,
            ``tucker_ranks=(n(d-1),...,n0)``,
            ``tt_ranks=(1,r(d-1),...,r1,1)``.

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
        ((10, 11, 12), (4, 5, 6), (1, 2, 3, 4), (2,3))
        >>> reversed_x = x.reverse()
        >>> print(reversed_x.structure)
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

    def resize(
            self,
            new_shape: Sequence[int], # len=d
            new_tucker_ranks: Sequence[int], # len=d
            new_tt_ranks: Sequence[int], # len=d+1
    ) -> 'TuckerTensorTrain':
        '''Change shape and ranks by resizing cores. Makes cores bigger via zero padding. Makes cores smaller via truncation.

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train with cores resized so that
            ``shape=new_shape``,
            ``tucker_ranks=new_tucker_ranks``,
            ``tt_ranks=new_tt_ranks``.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,6,5), (1,3,2,1))
        >>> padded_x = x.resize((17,18,17), (8,8,8), (1,5,6,1))
        >>> print(padded_x.structure)
        ((17, 18, 17), (8, 8, 8), (1, 5, 6, 1), ())

        Example where first and last ranks are nonzero:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,6,5), (3,3,2,4))
        >>> padded_x = x.resize((17,18,17), (8,8,8), (5,5,6,7))
        >>> print(padded_x.structure)
        ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7), ())
        '''
        tucker_cores, tt_cores = self.data

        new_tucker_cores = ragged_operations.change_tucker_core_shapes(tucker_cores, new_shape, new_tucker_ranks)
        new_tt_cores = ragged_operations.change_tt_core_shapes(tt_cores, new_tucker_ranks, new_tt_ranks)

        return TuckerTensorTrain(tuple(new_tucker_cores), tuple(new_tt_cores))

    def to_jax(self) -> 'TuckerTensorTrain':
        """Convert core arrays defining TuckerTensorTrain to Jax arrays.

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train where ``tucker_cores`` and ``tt_cores`` are jax arrays.

        See Also:
        ---------
        :py:meth:`.TuckerTensorTrain.to_numpy`
        """
        return TuckerTensorTrain(
            tuple(common.to_jax(B) for B in self.tucker_cores),
            tuple(common.to_jax(G) for G in self.tt_cores)
        )

    def to_numpy(self) -> 'TuckerTensorTrain':
        """Convert arrays defining TuckerTensorTrain into Numpy arrays.

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train where ``tucker_cores`` and ``tt_cores`` are numpy arrays.

        See Also:
        ---------
        :py:meth:`.TuckerTensorTrain.to_jax`
        """
        return TuckerTensorTrain(
            tuple(common.to_numpy(B) for B in self.tucker_cores),
            tuple(common.to_numpy(G) for G in self.tt_cores)
        )

    @cached_property
    def contains_jax(self) -> bool:
        """True if any Tucker or TT cores are jax arrays, False if all Tucker and TT cores are numpy arrays.

        See Also:
        ---------
        :py:meth:`.TuckerTensorTrain.to_jax`
        :py:meth:`.TuckerTensorTrain.to_numpy`
        """
        return common.tree_contains_jax(self.data)

    def copy(self):
        """Copy TuckerTensorTrain.

        Returns
        -------
        TuckerTensorTrain
            Deep copy.
        """
        return TuckerTensorTrain(
            tuple(B.copy() for B in self.tucker_cores),
            tuple(G.copy() for G in self.tt_cores)
        )

    ####################################################
    ##########    Vectorization / stacking    ##########
    ####################################################

    def unstack(self): # returns an array-like structure of nested tuples containing TuckerTensorTrains
        """If this object contains multiple stacked T3s, this unstacks them
        into an array-like tree of nested tuples with the same "tree shape" as self.stack_shape.

        Returns
        -------
        Array-like tree of nested tuples with TuckerTensorTrain leafs
            Unstacked TuckerTensorTrain.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.stack`
        :py:attr:`.TuckerTensorTrain.stack_shape`

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
            if common.is_ndarray(xx[0][0]):
                return TuckerTensorTrain(*xx)
            return tuple([_dfs(x) for x in xx])

        return _dfs(ragged_operations.t3_unstack(self.data))

    @staticmethod
    def stack(
            xx, # array-like structure of nested tuples containing TuckerTensorTrains
    ) -> 'TuckerTensorTrain':  # (stacked_tucker_cores, stacked_tt_cores)
        """Stacks an array-like tree of TuckerTensorTrains into one stacked TuckerTensorTrain.

        Parameters
        ----------
        xx: Array-like tree of nested tuples with TuckerTensorTrain leafs
            TuckerTensorTrains to be stacked. All TuckerTensorTrains must have the same shape and ranks.

        Returns
        -------
        TuckerTensorTrain
            Stacked TuckerTensorTrain.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.unstack`
        :py:attr:`.TuckerTensorTrain.stack_shape`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,6,2), (1,4,2,1), stack_shape=(3,5))
        >>> xx = x.unstack()
        >>> print(len(xx))
        3
        >>> print(len(xx[0]))
        5
        >>> x2 = t3.TuckerTensorTrain.stack(xx)
        >>> print(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)))
        0.0
        """
        def _data(xs):
            if isinstance(xs, TuckerTensorTrain):
                return xs.data
            return tuple([_data(x) for x in xs])
        xx_data = _data(xx)

        stacked_tucker_cores, stacked_tt_cores = ragged_operations.t3_stack(xx_data)
        return TuckerTensorTrain(stacked_tucker_cores, stacked_tt_cores)

    ############################################################################
    ##########    Constructing specific types of TuckerTensorTrain    ##########
    ############################################################################

    @staticmethod
    def zeros(
            shape:          Sequence[int],
            tucker_ranks:   Sequence[int] = None,
            tt_ranks:       Sequence[int] = None,
            stack_shape:    Sequence[int] = (),
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Construct a Tucker tensor train of zeros.

        Parameters
        ----------
        shape: Sequence[int]
            Shape of the TuckerTensorTrain. ``len(shape)=d``.
        tucker_ranks: Sequence[int], optional
            Tucker ranks. ``len(tucker_ranks)=d``. Default (``tucker_ranks=None``): all Tucker ranks equal 1 .
        tt_ranks: Sequence[int], optional
            TT ranks. ``len(tt_ranks)=d+1``. Default (``tt_ranks=None``): all TT ranks equal 1.
        stack_shape: Sequence[int], optional
            Stack shape. Default (``stack_shape=()``): No stacking.
        use_jax: bool, optional
            Cores are jax arrays if True, and numpy arrays if False. (default: ``use_jax=False``)

        Returns
        -------
        TuckerTensorTrain
            Zero TuckerTensorTrain with the desired shape and ranks.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.ones`
        :py:meth:`.TuckerTensorTrain.randn`

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
        d = len(shape)

        tucker_ranks = (1,)*d if tucker_ranks is None else tucker_ranks
        tt_ranks = (1,)*(d+1) if tt_ranks is None else tt_ranks

        if len(tucker_ranks) != d:
            raise ValueError(
                'Wrong number of Tucker ranks.\n' +
                str(len(tucker_ranks)) + ' = len(tucker_ranks) != len(shape) = ' + str(len(shape))
            )

        if len(tt_ranks) != d+1:
            raise ValueError(
                'Wrong number of TT ranks.\n' +
                str(len(tt_ranks)) + ' = len(tt_ranks) != len(shape)+1 = ' + str(len(shape)+1)
            )

        return TuckerTensorTrain(*ragged_operations.t3_zeros(
            shape, tucker_ranks, tt_ranks, stack_shape, use_jax=use_jax,
        ))

    @staticmethod
    def ones(
            shape: Tuple[int, ...],
            stack_shape: Tuple[int, ...] = (),
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Construct TuckerTensorTrain representation of dense tensor filled with ones.
        Has Tucker and TT ranks equal to 1.

        Parameters
        ----------
        shape: Sequence[int]
            Shape of the TuckerTensorTrain. ``len(shape)=d``.
        stack_shape: Sequence[int], optional
            Stack shape. Default (``stack_shape=()``): No stacking.
        use_jax: bool, optional
            Cores are jax arrays if True, and numpy arrays if False. (default: ``use_jax=False``)

        Returns
        -------
        TuckerTensorTrain
            Ones TuckerTensorTrain with the desired shape. ``tucker_ranks=(1,...,1)`` and ``tt_ranks=(1,...,1)``

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.zeros`
        :py:meth:`.TuckerTensorTrain.randn`

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
    def randn(
            shape: Tuple[int, ...],
            tucker_ranks: Tuple[int, ...],
            tt_ranks: Tuple[int, ...],
            stack_shape: Tuple[int, ...] = (),
            use_jax: bool = False,
    ) -> 'TuckerTensorTrain':
        """Construct a Tucker tensor train with random cores. Core entries are i.i.d. draws from N(0,1).

        Parameters
        ----------
        shape: Sequence[int]
            Shape of the TuckerTensorTrain. ``len(shape)=d``.
        tucker_ranks: Sequence[int], optional
            Tucker ranks. ``len(tucker_ranks)=d``. Default (``tucker_ranks=None``): all Tucker ranks equal 1 .
        tt_ranks: Sequence[int], optional
            TT ranks. ``len(tt_ranks)=d+1``. Default (``tt_ranks=None``): all TT ranks equal 1.
        stack_shape: Sequence[int], optional
            Stack shape. Default (``stack_shape=()``): No stacking.
        use_jax: bool, optional
            Cores are jax arrays if True, and numpy arrays if False. (default: ``use_jax=False``)

        Returns
        -------
        TuckerTensorTrain
            Random TuckerTensorTrain with the desired shape and ranks.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.zeros`
        :py:meth:`.TuckerTensorTrain.ones`

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

    ###################################################################################
    ##########    Coverting TuckerTensorTrain to/from other tensor formats   ##########
    ###################################################################################

    @staticmethod
    def from_canonical(
            factors: Sequence[NDArray], # elm_shape = stack_shape + (canonical_rank, Ni)
    ) -> 'TuckerTensorTrain':
        """Constructs TuckerTensorTrain from Canonical decomposition.

        Canonical decomposition represents a tensor X as a sum of rank-1 tensors of the form

            X[i1, ..., id] = sum_j F0[j,i1] * ... * F(d-1)[j,id],

        where F0,...,F(d-1) are the canonical factor matrices.

        Parameters
        ----------
        factors: Sequence[NDArray]
            Canonical factors. ``len(factors)=d``, ``factors[ii].shape=stack_shape+(canonical_rank, Ni)``.

        Returns
        -------
        T: TuckerTensorTrain
            TuckerTensorTrain representation of dense tensor which is represented by provided canonical decomposition.
            ``T.to_dense()[S,i1,...,id] = sum(factors[S,:,i1]*...*factors[S,:,id])``. Here ``S`` is a stack index.

        Raises
        ------
        ValueError
            If factor matrices in factors have inconsistent shapes.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> rank = 3
        >>> shape = (5,6,7)
        >>> stack_shape = (2,3)
        >>> FF = [np.random.randn(*(stack_shape+(rank, N))) for N in shape]
        >>> x = t3.TuckerTensorTrain.from_canonical(FF)
        >>> x_dense = x.to_dense()
        >>> x_dense2 = np.einsum('abri,abrj,abrk->abijk', FF[0], FF[1], FF[2])
        >>> print(np.linalg.norm(x_dense - x_dense2))
        0.0
        >>> print(x.tucker_ranks)
        (3, 3, 3)
        >>> print(x.tt_ranks)
        (3, 3, 3, 3)
        """
        shape           = tuple(F.shape[-1] for F in factors)
        canonical_ranks = tuple(F.shape[-2] for F in factors)
        stack_shapes    = tuple(F.shape[:-2] for F in factors)

        n = canonical_ranks[0]
        ss = stack_shapes[0]

        if canonical_ranks != (n,)*len(shape):
            raise ValueError(
                'Inconsistent ranks in Canonical decomposition.\n'
                + 'canonical_ranks = ' + str(canonical_ranks)
            )

        if stack_shapes != (ss,)*len(shape):
            raise ValueError(
                'Inconsistent stack_shapes in Canonical decomposition.\n'
                + 'stack_shapes = ' + str(stack_shapes)
            )

        return TuckerTensorTrain(*ragged_operations.t3_from_canonical(factors))

    @staticmethod
    def from_tensor_train(
            tt_cores: Sequence[NDArray], # elm_shape=stack_shape+(ri, N, r(i+1))
    ) -> 'TuckerTensorTrain':
        """Convert tensor train into Tucker tensor train by using identity matrices for Tucker bases.

        Parameters
        ----------
        tt_cores: Sequence[NDArray]
            Tensor train cores. ``len(tt_cores)=d``, ``tt_cores[ii].shape=stack_shape+(ri, Ni, r(i+1))``.

        Returns
        -------
        T: TuckerTensorTrain
            Input tensor train, converted to TuckerTensorTrain format.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.to_tensor_train`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> tt_cores = [randn(4,14,5), randn(5,15,3), randn(3,16,2)]
        >>> x = t3.TuckerTensorTrain.from_tensor_train(tt_cores)
        >>> x_dense = x.to_dense()
        >>> x_dense2 = np.einsum('...aib,...bjc,...ckd->...ijk', *tt_cores)
        >>> print(np.linalg.norm(x_dense - x_dense2))
        1.8303194206478734e-13
        """
        return TuckerTensorTrain(*ragged_operations.t3_from_tensor_train(tt_cores))

    def to_tensor_train(
            self,
    ) -> Tuple[NDArray,...]: # tt_cores
        """Convert this TuckerTensorTrain to a tensor train by contracting Tucker bases with TT cores.

        Returns
        -------
        tt_cores: Sequence[NDArray]
            Tensor train cores. ``len(tt_cores)=d``, ``tt_cores[ii].shape=stack_shape+(ri, Ni, r(i+1))``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.from_tensor_train`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (5,6,7), (2,3,4,1), (2,3))
        >>> big_tt_cores = x.to_tensor_train()
        >>> x_dense = np.einsum('...aib,...bjc,...ckd->...ijk', *big_tt_cores)
        >>> x_dense2 = x.to_dense()
        >>> print(np.linalg.norm(x_dense - x_dense2))
        2.337172789566996e-12
        """
        return ragged_operations.t3_to_tensor_train(self.data)

    #############################################################
    ##########    Converting data to/from 1D vector    ##########
    #############################################################

    def to_vector(
            self,
    ) -> NDArray:
        """Converts a TuckerTensorTrain into a 1D vector containing the core entries.

        Returns
        -------
        NDArray
            The vector of all core entries. ``shape=(self.data_size,)``

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.from_vector`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(2,3))
        >>> x_flat = x.to_vector()
        >>> x2 = t3.TuckerTensorTrain.from_vector(x_flat, x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape)
        >>> print(cw.corewise_norm(cw.corewise_sub(x.data, x2.data)))
        0.0
        """
        return ragged_operations.t3_to_vector(self.data)

    @staticmethod
    def from_vector(
            x_flat: NDArray,
            shape: Sequence[int],
            tucker_ranks: Sequence[int],
            tt_ranks: Sequence[int],
            stack_shape: Sequence[int] = (),
    ) -> 'TuckerTensorTrain':
        """Constructs a TuckerTensorTrain from a 1D vector containing the core entries.

        Parameters
        ----------
        x_flat: NDArray
            The flattened vector of core entries. ``x_flat.shape=(data_size,)``
        shape: Sequence[int]
            Shape of the tensor.
        tucker_ranks: Sequence[int]
            Tucker ranks of the tensor.
        tt_ranks: Sequence[int]
            TT ranks.
        stack_shape: Sequence[int], optional
            Stack shape. Default (``stack_shape=()``): No stacking.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.to_vector`

        Returns
        -------
        T: TuckerTensorTrain
            TuckerTensorTrain constructed from the vector of all core entries.
            ``T.data_size=len(x_flat)``,
            ``T.shape=shape``, ``T.tucker_ranks=tucker_ranks``, ``T.tt_ranks=tt_ranks``,
            ``T.stack_shape=stack_shape``.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(2,3))
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

        Raises
        ------
        ValueError
            If the Tucker tensor train is inconsistent
        RuntimeError
            If the Tucker tensor train fails to save.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.load`


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
        use_jax: bool, optional
            If True, TuckerTensorTrain cores are jax arrays. If False, they are numpy arrays.
            Default (``use_jax=False``): use numpy arrays.

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
        :py:meth:`.TuckerTensorTrain.save`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> fname = 't3_file'
        >>> x.save(fname) # Save to file 't3_file.npz'
        >>> x2 = t3.TuckerTensorTrain.load(fname) # Load from file
        >>> tucker_cores, tt_cores = x.data
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> print([np.linalg.norm(B - B2) for B, B2 in zip(tucker_cores, tucker_cores2)])
        [0.0, 0.0, 0.0]
        >>> print([np.linalg.norm(G - G2) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
        """
        xnp, _, _ = common.get_backend(False, use_jax)

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
        """Add this TuckerTensorTrains self to other tensor, yielding a tensor ``result = self + other`` with summed ranks.

        Addition is defined with respect to the dense ``N0 x ... x N(d-1)`` tensor that
        is *represented* by the TuckerTensorTrain.

        For corewise addition, see :func:`t3toolbox.corewise.corewise_add`

        Allowed types are as follows:

        - ``TuckerTensorTrain + TuckerTensorTrain -> TuckerTensorTrain``
            (self + other).to_dense() = self.to_dense() + other.to_dense()

        - ``TuckerTensorTrain + NDArray -> NDArray``
            self + other = self.to_dense() + other

        - ``TuckerTensorTrain + scalar -> TuckerTensorTrain``
            (self + other).to_dense() = self.to_dense() + other * np.ones(self.stack_shape + self.shape)

        Parameters
        ----------
        other: TuckerTensorTrain or NDArray or scalar
            Other tensor or scalar to add to this TuckerTensorTrain.
            If ``other`` is TuckerTensorTrain, requires ``other.shape=self.shape`` and ``other.stack_shape=self.stack_shape``.
            If ``other`` is NDArray, requires ``other.shape=self.stack_shape+self.shape``.

        Returns
        -------
        result: TuckerTensorTrain or NDArray
            Sum of tensors self and other.
            If ``other`` is TuckerTensorTrain or scalar, ``result.shape=self.shape``, ``result.stack_shape=self.stack_shape``.
            If other is ``NDArray``, ``result.shape=self.stack_shape+self.shape``.

        Raises
        ------
        ValueError
            If shapes and/or stack shapes of self and other are inconsistent.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__sub__`
        :py:meth:`.TuckerTensorTrain.__neg__`
        :py:meth:`.TuckerTensorTrain.__mul__`
        :py:meth:`.TuckerTensorTrain.inner`
        :py:meth:`.TuckerTensorTrain.norm`
        :py:meth:`.TuckerTensorTrain.sum`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> z = x + y
        >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - z.to_dense()))
        6.524094086845177e-13
        >>> print(z.structure)
        ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1), ())

        Adding T3 + dense

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = np.random.randn(14,15,16)
        >>> z = x + y
        >>> print(np.linalg.norm(x.to_dense() + y - z))
        0.0
        >>> print(type(z))
        <class 'numpy.ndarray'>

        Adding T3 + scalar

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
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

        elif common.is_ndarray(other):
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
        """Elementwise multiplication of a Tucker tensor train by another tensor, ``result = self * other``.

        Multiplication is defined with respect to the dense ``N0 x ... x N(d-1)`` tensor that
        is *represented* by the TuckerTensorTrain.

        For corewise scaling, see :func:`t3toolbox.corewise.corewise_scale`

        Allowed types are as follows:

        - ``TuckerTensorTrain * TuckerTensorTrain -> TuckerTensorTrain``
            (self * other).to_dense() = self.to_dense() * other.to_dense()

        - ``TuckerTensorTrain * NDArray -> NDArray``
            self * other = self.to_dense() * other

        - ``TuckerTensorTrain * scalar -> TuckerTensorTrain``
            (self * other).to_dense() = self.to_dense() * other

        Parameters
        ----------
        other: TuckerTensorTrain or NDArray or scalar
            Other tensor or scalar to be multiplied this TuckerTensorTrain with.
            If ``other`` is TuckerTensorTrain, requires ``other.shape=self.shape`` and ``other.stack_shape=self.stack_shape``.
            If ``other`` is NDArray, requires ``other.shape=self.stack_shape+self.shape``.

        Returns
        -------
        result: TuckerTensorTrain or NDArray
            Elementwise multiplication of tensors ``self`` and ``other``.
            If ``other`` is TuckerTensorTrain or scalar, ``result.shape=self.shape``, ``result.stack_shape=self.stack_shape``.
            If ``other`` is NDArray, ``result.shape=self.stack_shape+self.shape``.

        Raises
        ------
        ValueError
            If shapes and/or stack shapes of self and other are inconsistent.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.__sub__`
        :py:meth:`.TuckerTensorTrain.__neg__`
        :py:meth:`.TuckerTensorTrain.inner`
        :py:meth:`.TuckerTensorTrain.norm`
        :py:meth:`.TuckerTensorTrain.sum`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> s = 3.2
        >>> sx = x * s
        >>> print(np.linalg.norm(s*x.to_dense() - sx.to_dense()))
        1.6268482531988893e-13

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> y = np.random.randn(*(x.stack_shape + x.shape))
        >>> xy = x * y
        >>> print(np.linalg.norm(x.to_dense()*y - xy))
        0.0

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> y = t3.TuckerTensorTrain.randn((14,15,16), (2,3,4), (3,2,3,2), stack_shape=(2,3))
        >>> xy = x * y
        >>> print(np.linalg.norm(x.to_dense()*y.to_dense() - xy.to_dense()))
        8.556292929330887e-11
        >>> print(xy.tucker_ranks) # ranks get multiplied!
        (8, 15, 24)
        >>> print(xy.tt_ranks)
        (1, 6, 6, 1)
        """
        if common.is_ndarray(other):
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
        """Scale a TuckerTensorTrain by -1. ``result=-self``.

        Negation is defined with respect to the dense ``N0 x ... x N(d-1)`` tensor that
        is *represented* by the TuckerTensorTrains.

        Returns
        -------
        result: TuckerTensorTrain or NDArray
            Negative of this TuckerTensorTrain satisfying ``result.to_dense() = -self.to_dense()``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.__sub__`
        :py:meth:`.TuckerTensorTrain.__mul__`
        :py:meth:`.TuckerTensorTrain.inner`
        :py:meth:`.TuckerTensorTrain.norm`
        :py:meth:`.TuckerTensorTrain.sum`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> neg_x = -x
        >>> print(np.linalg.norm(x.to_dense() + neg_x.to_dense()))
        0.0
        """
        return self * (-1.0)

    def __sub__(
            self: 'TuckerTensorTrain',
            other: 'TuckerTensorTrain',
    ) -> 'TuckerTensorTrain':
        """Subtract Tucker tensor trains, ``result = self - other``, yielding a Tucker tensor train with summed ranks.

        Subtraction is defined with respect to the dense ``N0 x ... x N(d-1)`` tensor that
        is *represented* by this TuckerTensorTrains.

        For corewise subtraction, see :func:`t3toolbox.corewise.corewise_sub`

        Allowed types are as follows:

        - ``TuckerTensorTrain - TuckerTensorTrain -> TuckerTensorTrain``
            (self - other).to_dense() = self.to_dense() - other.to_dense()

        - ``TuckerTensorTrain - NDArray -> NDArray``
            self - other = self.to_dense() - other

        - ``TuckerTensorTrain - scalar -> TuckerTensorTrain``
            (self - other).to_dense() = self.to_dense() - other

        Parameters
        ----------
        other: TuckerTensorTrain or NDArray or scalar
            Other tensor or scalar to be subtracted from this TuckerTensorTrain.
            If ``other`` is TuckerTensorTrain, requires ``other.shape=self.shape`` and ``other.stack_shape=self.stack_shape``.
            If ``other`` is NDArray, requires ``other.shape=self.stack_shape+self.shape``.

        Returns
        -------
        result: TuckerTensorTrain or NDArray
            Difference, ``result = self - other``.
            If ``other` is TuckerTensorTrain or scalar, ``result.shape=self.shape``, ``result.stack_shape=self.stack_shape``.
            If ``other`` is NDArray, ``result.shape=self.stack_shape+self.shape``.

        Raises
        ------
        ValueError
            If shapes and/or stack shapes of self and other are inconsistent.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.__neg__`
        :py:meth:`.TuckerTensorTrain.__mul__`
        :py:meth:`.TuckerTensorTrain.inner`
        :py:meth:`.TuckerTensorTrain.norm`
        :py:meth:`.TuckerTensorTrain.sum`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> z = x - y
        >>> print(np.linalg.norm(x.to_dense() - y.to_dense() - z.to_dense()))
        0.0
        >>> print(z.structure)
        ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2), ())

        T3 - dense

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = np.random.randn(14,15,16)
        >>> z = x - y
        >>> print(np.linalg.norm(x.to_dense() - y - z))
        0.0
        >>> print(type(z))
        <class 'numpy.ndarray'>

        T3 - scalar

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> s = 3.5
        >>> z = x - s
        >>> print(np.linalg.norm(x.to_dense() - s - z.to_dense()))
        0.0
        >>> print(z.structure)
        ((14, 15, 16), (5, 6, 7), (2, 4, 3, 2), ())
        """
        return self + (-other)

    def inner(
            self,
            other,
            use_orthogonalization: bool = True,  # for numerical stability
    ):
        """Compute Hilbert-Schmidt inner product of this TuckerTensorTrain with other tensor, ``result=(self, other)_HS``.

        The Hilbert-Schmidt inner product is defined with respect to the dense ``N0 x ... x N(d-1)``
        tensor *represented* by the TuckerTensorTrain.

        For corewise dot product, see :func:`t3toolbox.corewise.corewise_dot`

        Allowed types are as follows:

        - ``other: TuckerTensorTrain``
            ``self.inner(other) = np.sum(self.to_dense() * other.to_dense())``

        - ``other: NDArray``
            ``self.inner(other) = np.sum(self.to_dense() * other)``

        Parameters
        ----------
        other: TuckerTensorTrain
            Other tensor to take the inner product with. Requires ``other.shape=(N0,...,N(d-1))``.
        use_orthogonalization: bool, optional
            If True, orthogonalize tensors before computing inner product (more stable).
            If False, use simple zippering without orthogonalization (faster, better for automatic differentiation).
            Default: ``use_orthogonalization=True``.

        Returns
        -------
        scalar or NDArray
            Hilbert-Schmidt inner product of Tucker tensor trains, (self, other)_HS.
            If stacked, ``result.shape=self.stack_shape``. Otherwise, result is scalar.

        Raises
        ------
        ValueError
            - Error raised if the TuckerTensorTrains have different shapes and/or stack shapes.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.__sub__`
        :py:meth:`.TuckerTensorTrain.__neg__`
        :py:meth:`.TuckerTensorTrain.__mul__`
        :py:meth:`.TuckerTensorTrain.norm`
        :py:meth:`.TuckerTensorTrain.sum`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> y = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (3,5,6,3))
        >>> x_dot_y = x.inner(y)
        >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense())
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        1.3096723705530167e-10

        (T3, T3) using stacking:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
        >>> y = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (3,5,6,3), stack_shape=(2,3))
        >>> x_dot_y = x.inner(y)
        >>> x_dot_y2 = np.sum(x.to_dense() * y.to_dense(), axis=(2,3,4))
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        2.7761383858792984e-09

        Inner product of T3 with dense:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (3,5,6,3))
        >>> y = np.random.randn(14,15,16)
        >>> x_dot_y = x.inner(y)
        >>> x_dot_y2 = np.sum(x.to_dense() * y)
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        0.0

        Inner product of T3 with dense including stacking:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (3,5,6,3), stack_shape=(2,3))
        >>> y = np.random.randn(2,3, 14,15,16)
        >>> x_dot_y = x.inner(y)
        >>> x_dot_y2 = np.einsum('ijxyz,ijxyz->ij', x.to_dense(), y)
        >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        1.2014283869232628e-11
        """
        if isinstance(other, TuckerTensorTrain):
            if self.shape != other.shape:
                raise ValueError(
                    'Attempted to take inner product of TuckerTensorTrains (x,y) with inconsistent shapes.'
                    + str(self.shape) + ' = x.shape != y.shape = ' + str(other.shape)
                )

            if self.stack_shape != other.stack_shape:
                raise NotImplementedError(
                    'Cannot take inner product of TuckerTensorTrains with different stack shapes.\n'
                    + str(self.stack_shape)
                    + ' = x.stack_shape != y.stack_shape = '
                    + str(other.stack_shape)
                )

            return ragged_linalg.t3_inner_product_t3(
                self.data, other.data, use_orthogonalization=use_orthogonalization,
            )

        elif common.is_ndarray(other):
            if self.stack_shape + self.shape != other.shape:
                raise ValueError(
                    'Attempted to take inner product of array x with TuckerTensorTrain y with inconsistent shapes.'
                    + str(self.stack_shape + self.shape) + ' = self.stack_shape + self.shape != other.shape = ' + str(other.shape)
                )
            contraction_inds = tuple(range(len(self.stack_shape), len(other.shape)))
            contraction_inds = contraction_inds if contraction_inds else None

            return (self.to_dense() * other).sum(axis=tuple(contraction_inds))

        else:
            raise NotImplementedError(
                'T3 inner product only implemented for other in: {T3, dense}.\n'
                + 'type(other) = ' + str(type(other))
            )

    def norm(
            self,
            use_orthogonalization: bool = True, # for numerical stability
    ):
        """Compute Hilbert-Schmidt (Frobenius) norm of this TuckerTensorTrain.

        The Hilbert-Schmidt norm is defined with respect to the dense ``N0 x ... x N(d-1)`` tensor
        that is *represented* by the TuckerTensorTrain.

        ``x.norm() = np.linalg.norm(x.to_dense())``

        For corewise norm, see :func:`t3toolbox.corewise.corewise_norm`

        Parameters
        ----------
        use_orthogonalization: bool, optional
            If True, compute norm by orthogonalizing (more stable).
            If False, compute norm with conventional zippering (faster, more suited for automatic differentiation).
            Default: ``use_orthogonalization=True``.

        Returns
        -------
        result: scalar or NDArray
            Hilbert-Schmidt (Frobenius) norm of Tucker tensor train, ||x||_HS.
            If stacked, ``result.shape=self.stack_shape``. Otherwise, result is scalar.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.__sub__`
        :py:meth:`.TuckerTensorTrain.__neg__`
        :py:meth:`.TuckerTensorTrain.__mul__`
        :py:meth:`.TuckerTensorTrain.inner`
        :py:meth:`.TuckerTensorTrain.sum`

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

    def sum(
            self,
            axis=None,
    ):
        """Sum over one or more axes of TuckerTensorTrain.

        The sum is defined with respect to the dense ``N0 x ... x N(d-1)`` tensor
        that is *represented* by the TuckerTensorTrain.

        For corewise norm, see :func:`t3toolbox.corewise.corewise_norm`

        If all axes are summed over, returns NDArray or scalar, depending on whether or not self is stacked.
        If at least one axis is not summed over, returns TuckerTensorTrain.

        Parameters
        ----------
        axis: int or Sequence[int], optional
            If ``int``, sum over index specified by ``axis`.
            If ``Sequence[int]``, sum over all indices in ``axis``.
            If None (default), sum over all axes.

        Returns
        -------
        result: scalar or NDArray or TuckerTensorTrain
            Sum of tensor over specified axes.
            Case 1a: ``axis`` is None or ``axis`` contains all indices ``1,dots,d`` and self is not stacked: ``result`` is scalar.
            Case 1b: ``axis`` is None or ``axis`` contains all axes ``1,\dots,d`` and self is stacked: ``result`` is NDArray and ``result.shape=self.stack_shape``.
            Case 2: ``axis`` is ``int``, or ``axis`` is ``Sequence[int]``, and ``axis`` is missing at least ine index from ``1,...,d``: ``result`` is TuckerTensorTrain.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.__sub__`
        :py:meth:`.TuckerTensorTrain.__neg__`
        :py:meth:`.TuckerTensorTrain.__mul__`
        :py:meth:`.TuckerTensorTrain.inner`
        :py:meth:`.TuckerTensorTrain.norm`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((10,11,12,13), (7,8,9,10), (2,3,4,3,1), (2,3))
        >>> S = x.sum()
        >>> dense_x = x.to_dense()
        >>> non_stack_axes = (2,3,4,5)
        >>> print(np.linalg.norm(S - dense_x.sum(axis=non_stack_axes)))
        1.4038073554965914e-10
        >>> print(type(S))
        <class 'numpy.ndarray'>
        >>> print(S.shape)
        (2, 3)

        Axis is a tuple of ints:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((10,11,12,13), (7,8,9,10), (2,3,4,3,1), (2,3))
        >>> axis = (1,3)
        >>> S = x.sum(axis=axis)
        >>> dense_x = x.to_dense()
        >>> shifted_axis = tuple(ii + len(x.stack_shape) for ii in axis)
        >>> print(np.linalg.norm(S.to_dense() - dense_x.sum(axis=shifted_axis)))
        8.457133031493982e-11
        >>> print(type(S))
        <class 't3toolbox.tucker_tensor_train.TuckerTensorTrain'>
        >>> print(S.shape)
        (10, 12)
        >>> print(S.stack_shape)
        (2, 3)

        Axis is int:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((10,11,12,13), (7,8,9,10), (2,3,4,3,1), (2,3))
        >>> axis = 1
        >>> S = x.sum(axis=axis)
        >>> dense_x = x.to_dense()
        >>> shifted_axis = axis + len(x.stack_shape)
        >>> print(np.linalg.norm(S.to_dense() - dense_x.sum(axis=shifted_axis)))
        4.906645592301091e-11
        >>> print(type(S))
        <class 't3toolbox.tucker_tensor_train.TuckerTensorTrain'>
        >>> print(S.shape)
        (10, 12, 13)
        >>> print(S.stack_shape)
        (2, 3)
        """
        result = ragged_operations.t3_sum(self.data, axis=axis)
        if isinstance(result, Sequence):
            result = TuckerTensorTrain(*result)
        return result

    def sum_stack(
            self,
            axis = None, # stack axis or axes to sum over. None: sum over all stack axes
    ) -> 'TuckerTensorTrain':
        '''Sum the tensors represented by a stacked TuckerTensorTrain over one or more stack axes.

        This is the genuine *tensor* sum: the result represents the sum of the dense tensors over
        the chosen stack axes,
            ``result.to_dense() = self.to_dense().sum(axis=stack axes)``.
        The summed-over stack axes are removed; any remaining stack axes are kept.
        For a corewise sum of the core arrays instead, see :py:meth:`.sum_stack_corewise`.

        .. warning::
            Ranks grow. Summing over stack axes whose sizes multiply to ``S`` multiplies every
            Tucker and TT rank by ``S`` (this is the ``S``-fold generalization of
            :py:meth:`__add__`, which is the ``S=2`` case). Follow with :py:meth:`t3svd` to
            truncate the ranks if needed.

        Parameters
        ----------
        axis: int or Sequence[int], optional
            Stack axis or axes to sum over, indexed within ``stack_shape``.
            Default (``axis=None``): sum over all stack axes (the result is unstacked).

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train representing the sum over the chosen stack axes.
            Its ``stack_shape`` consists of the un-summed stack axes.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.sum`
        :py:meth:`.TuckerTensorTrain.sum_stack_corewise`
        :py:meth:`.TuckerTensorTrain.__add__`
        :py:meth:`.TuckerTensorTrain.t3svd`

        Examples
        --------
        Sum over all stack axes (the result is unstacked):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((4,5,6), (2,3,2), (1,2,2,1), stack_shape=(3,))
        >>> y = x.sum_stack()
        >>> print(y.stack_shape)
        ()
        >>> print(np.linalg.norm(y.to_dense() - x.to_dense().sum(axis=0)))
        1.3344301227427602e-14

        Ranks grow by the summed stack size (here ``S=3``):

        >>> print(x.tucker_ranks, '->', y.tucker_ranks)
        (2, 3, 2) -> (6, 9, 6)
        >>> print(x.tt_ranks, '->', y.tt_ranks)
        (1, 2, 2, 1) -> (1, 6, 6, 1)

        Sum over one of several stack axes (the rest are kept):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((4,5,6), (2,3,2), (1,2,2,1), stack_shape=(2,3))
        >>> y = x.sum_stack(axis=0)
        >>> print(y.stack_shape)
        (3,)
        >>> print(np.linalg.norm(y.to_dense() - x.to_dense().sum(axis=0)))
        1.7956338470588144e-14
        '''
        return TuckerTensorTrain(*ragged_linalg.t3_sum_stack(self.data, axis=axis))

    def sum_stack_corewise(
            self,
            axis = None, # stack axis or axes to sum over. None: sum over all stack axes
    ) -> 'TuckerTensorTrain':
        '''Sum the *core arrays* of a stacked TuckerTensorTrain over one or more stack axes.

        This is a corewise sum: the Tucker and TT core arrays are summed directly along the chosen
        stack axes (see :py:func:`t3toolbox.corewise.corewise_sum`). Because a Tucker tensor train
        is multilinear in its cores, this is generally **not** the tensor sum of the represented
        tensors (for that, use :py:meth:`.sum_stack`). It is useful when the cores carry an additive
        structure of their own (e.g. stacked tangent-vector variations), and it leaves ranks unchanged.

        Parameters
        ----------
        axis: int or Sequence[int], optional
            Stack axis or axes to sum over, indexed within ``stack_shape``.
            Default (``axis=None``): sum over all stack axes (the result is unstacked).

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train whose cores are the corewise sums over the chosen stack axes.
            Tucker and TT ranks are unchanged; its ``stack_shape`` consists of the un-summed stack axes.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.sum_stack`
        :py:func:`t3toolbox.corewise.corewise_sum`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> x = t3.TuckerTensorTrain.randn((4,5,6), (2,3,2), (1,2,2,1), stack_shape=(2,3))
        >>> y = x.sum_stack_corewise(axis=0)
        >>> print(y.stack_shape)
        (3,)
        >>> print(y.ranks) # ranks are unchanged
        ((2, 3, 2), (1, 2, 2, 1))
        >>> tucker_cores2 = tuple(B.sum(axis=0) for B in x.tucker_cores)
        >>> tt_cores2 = tuple(G.sum(axis=0) for G in x.tt_cores)
        >>> print(cw.corewise_norm(cw.corewise_sub((tucker_cores2, tt_cores2), y.data)))
        0.0
        '''
        m = len(self.stack_shape)

        if axis is None:
            stack_axes = tuple(range(m))
        elif not isinstance(axis, Sequence):
            stack_axes = ((axis + m) if axis < 0 else axis,)
        else:
            stack_axes = tuple((ax + m) if ax < 0 else ax for ax in axis)

        return TuckerTensorTrain(*corewise.corewise_sum(self.data, axis=stack_axes))


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
    ) -> Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # ss_x. singular values
    ]:
        '''Compute SVD of ith tucker core and contract non-orthogonal factor into the TT-core above.

        Parameters
        ----------
        ii: int
            index of TT core to SVD
        min_rank: int, optional
            Minimum rank for truncation. Default (``None``): no minimum rank.
        max_rank: int, optional
            Maximum rank for truncation. Default (``None``): no maximum rank.
        rtol: float, optional
            Relative tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``rtol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``rtol=None``.
        atol: float, optional
            Absolute tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``atol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``atol=None``.

        Returns
        -------
        new_x: TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with ith tucker core down orthogonal.
            I.e., ``np.einsum('...io,jo->...ij', B, B) = (stacked) identity matrix``, where ``B=new_x.tucker_cores[ii]``.
            May have different ith Tucker rank.
        ss: NDArray
            Singular values of ith Tucker core.
            ``ss[ii].shape = new_x.stack_shape + new_x.tucker_ranks[ii]``.

        Raises
        ------
        ValueError
            If this TuckerTensorTrain is stacked and ``rtol`` or ``atol`` are not ``None.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.left_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.right_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
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
    ) -> Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(r(i+1),)
    ]:
        '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.

        Parameters
        ----------
        ii: int
            index of TT core to SVD
        min_rank: int, optional
            Minimum rank for truncation. Default (``None``): no minimum rank.
        max_rank: int, optional
            Maximum rank for truncation. Default (``None``): no maximum rank.
        rtol: float, optional
            Relative tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``rtol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``rtol=None``.
        atol: float, optional
            Absolute tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``atol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``atol=None``.

        Returns
        -------
        new_x: TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
            I.e., ``einsum('...iaj,...iak->...jk', G, G) = (stacked) identity matrix``, where ``G=new_x.tt_cores[ii]``.
            May have different (i+1)th TT rank.
        ss: NDArray
            Singular values of prior ith TT-core left unfolding.
            ``ss.shape = new_x.stack_shape + (new_x.tt_ranks[ii+1],)``.

        Raises
        ------
        ValueError
            If this TuckerTensorTrain is stacked and ``rtol`` or ``atol`` are not ``None.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_svd_tucker_core`
        :py:meth:`.TuckerTensorTrain.right_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

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
    ) -> Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(new_ri,)
    ]:
        '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.

        Parameters
        ----------
        ii: int
            index of TT core to SVD
        min_rank: int, optional
            Minimum rank for truncation. Default (``None``): no minimum rank.
        max_rank: int, optional
            Maximum rank for truncation. Default (``None``): no maximum rank.
        rtol: float, optional
            Relative tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``rtol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``rtol=None``.
        atol: float, optional
            Absolute tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``atol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``atol=None``.

        Returns
        -------
        new_x: TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
            I.e., ``einsum('...iaj,...kaj->...ik', G, G) = (stacked) identity matrix``, where ``G=new_tt_cores[ii]``.
            May have different ith TT rank.
        ss: NDArray
            Singular values of prior ith TT-core right unfolding.
            ``ss.shape = new_x.stack_shape + (new_x.tt_ranks[ii],)``.

        Raises
        ------
        ValueError
            If this TuckerTensorTrain is stacked and ``rtol`` or ``atol`` are not ``None.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_svd_tucker_core`
        :py:meth:`.TuckerTensorTrain.left_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

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

    def up_svd_tt_core(
            self,
            ii: int,  # which tt core to orthogonalize
            min_rank: int = None,
            max_rank: int = None,
            rtol: float = None,
            atol: float = None,
    ) -> Tuple[
        'TuckerTensorTrain',  # new_x
        NDArray,  # singular values, shape=(new_ni,)
    ]:
        '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the tucker core below.

        Parameters
        ----------
        ii: int
            index of TT core to SVD
        min_rank: int, optional
            Minimum rank for truncation. Default (``None``): no minimum rank.
        max_rank: int, optional
            Maximum rank for truncation. Default (``None``): no maximum rank.
        rtol: float, optional
            Relative tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``rtol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``rtol=None``.
        atol: float, optional
            Absolute tolerance for truncation (in Hilbert-Schmidt/Frobenius norm).
            Default (``None``): no ``atol`` truncation.
            If this TuckerTensorTrain is stacked, requires ``atol=None``.

        Returns
        -------
        new_x: TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with ith TT-core down orthogonal.
            I.e., ``einsum('...iaj,...ibj->...ab', G, G) = (stacked) identity matrix``, where ``G=new_tt_cores[ii]``.
            May have different ith Tucker rank.
        ss: NDArray
            ``ss.shape = new_x.stack_shape + (new_x.tucker_ranks[ii],)``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_svd_tucker_core`
        :py:meth:`.TuckerTensorTrain.left_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.orthogonalization as orth
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.down_svd_tt_core(ind)
        >>> print(np.linalg.norm(x.to_dense() - x2.to_dense())) # Tensor unchanged
        4.367311712704942e-12
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', G, G) - np.eye(G.shape[1]))) # TT-core is down orthogonal
        1.0643458053135608e-15
        '''
        result = ragged_orthogonalization.up_svd_tt_core(
            self.data, ii, min_rank=min_rank, max_rank=max_rank, rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1]

    ####

    def orthogonalize_relative_to_tucker_core(
            self,
            ii: int,
    ) -> 'TuckerTensorTrain':
        '''Orthogonalize cores in the TuckerTensorTrain relative to the ith Tucker core.

        - ith Tucker core is not orthogonalized
        - All other Tucker cores are down orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - TT-core directly above is up orthogonalized.
        - TT-cores to the right are right orthogonalized.

        Parameters
        ----------
        ii: int
            index of tucker core that is not orthogonalized

        Returns
        -------
        TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith Tucker core.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.orthogonalize_relative_to_tt_core`

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
            self.data, ii,
        ))

    def orthogonalize_relative_to_tt_core(
            self,
            ii: int,
    ) -> 'TuckerTensorTrain':
        '''Orthogonalize cores in the TuckerTensorTrain relative to the ith TT-core.

        - All Tucker cores are down orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - ith TT-core is not orthogonalized.
        - TT-cores to the right are right orthogonalized.

        Parameters
        ----------
        ii: int
            index of TT-core that is not orthogonalized

        Returns
        -------
        TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith TT-core.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.orthogonalize_relative_to_tucker_core`

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
            self.data, ii,
        ))

    def down_orthogonalize_tucker_cores(
        self,
    ) -> 'TuckerTensorTrain':
        """Orthogonalize Tucker cores downwards, pushing remainders onto TT cores above.

        Returns
        -------
        TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with all Tucker cores down orthogonal.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.up_orthogonalize_tt_cores`
        :py:meth:`.TuckerTensorTrain.left_orthogonalize_tt_cores`
        :py:meth:`.TuckerTensorTrain.right_orthogonalize_tt_cores`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.down_orthogonalize_tucker_cores()
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
        >>> x_orth = x.down_orthogonalize_tucker_cores()
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
        return TuckerTensorTrain(*ragged_orthogonalization.down_orthogonalize_tucker_cores(self.data))

    def up_orthogonalize_tt_cores(
        self,
    ) -> 'TuckerTensorTrain':
        """Up orthogonalize TT cores, pushing remainders onto Tucker cores below.

        Returns
        -------
        TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with all TT cores up orthogonal.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_orthogonalize_tucker_cores`
        :py:meth:`.TuckerTensorTrain.left_orthogonalize_tt_cores`
        :py:meth:`.TuckerTensorTrain.right_orthogonalize_tt_cores`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.up_orthogonalize_tt_cores()
        >>> print((x - x_orth).norm())
        1.927414448489825e-12
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab',G,G)-np.eye(G.shape[1])))
        1.9491561709929213e-15

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.up_orthogonalize_tt_cores()
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
            *ragged_orthogonalization.up_orthogonalize_tt_cores(self.data),
        )

    def left_orthogonalize_tt_cores(
        self,
        return_variation_cores: bool = False,
    ) -> 'TuckerTensorTrain':
        """Left orthogonalize the TT cores, possibly returning variation cores as well.

        Parameters
        ----------
        return_variation_cores: bool, optional
            If True, also return each TT core just before it is orthogonalized. Default: ``return_variation_cores=False``.
            Used to construct variation cores when converting a TuckerTensorTrain to basis-variation format.

        Returns
        -------
        new_x: TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with all TT cores left orthogonal.
        var_cores: Tuple[NDArray,...], optional
            TT cores just before they are orthogonalized. Only returned if ``return_variation_cores=True``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_orthogonalize_tucker_cores`
        :py:meth:`.TuckerTensorTrain.up_orthogonalize_tt_cores`
        :py:meth:`.TuckerTensorTrain.right_orthogonalize_tt_cores`

        Examples
        --------

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
            self.tt_cores, return_variation_cores=return_variation_cores,
        )
        if return_variation_cores:
            return TuckerTensorTrain(self.tucker_cores, result[0]), result[1]
        else:
            return TuckerTensorTrain(self.tucker_cores, result)

    def right_orthogonalize_tt_cores(
        self,
        return_variation_cores: bool = False,
    ) -> 'TuckerTensorTrain':
        """Right orthogonalize the TT cores, possibly returning variation cores as well.

        Parameters
        ----------
        return_variation_cores: bool, optional
            If True, also return each TT core just before it is orthogonalized. Default: ``return_variation_cores=False``.
            Used to construct variation cores when converting a TuckerTensorTrain to basis-variation format.

        Returns
        -------
        new_x: TuckerTensorTrain
            New TuckerTensorTrain representing the same tensor, but with all TT cores right orthogonal.
        var_cores: Tuple[NDArray,...], optional
            TT cores just before they are orthogonalized. Only returned if ``return_variation_cores=True``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_orthogonalize_tucker_cores`
        :py:meth:`.TuckerTensorTrain.up_orthogonalize_tt_cores`
        :py:meth:`.TuckerTensorTrain.left_orthogonalize_tt_cores`

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
            self.tt_cores, return_variation_cores=return_variation_cores,
        )
        if return_variation_cores:
            return TuckerTensorTrain(self.tucker_cores, result[0]), result[1]
        else:
            return TuckerTensorTrain(self.tucker_cores, result)

    #######################################################
    ##########    Entries, Apply, and Probing    ##########
    #######################################################

    def entries(
            self,           # shape=(N0,...,N(d-1))
            index: NDArray, # shape=(d,)+idx_stack_shape, dtype=int 
    ) -> NDArray:
        '''Compute an entry (or multiple entries) of a Tucker tensor train.

        This is the entry of the ``N0 x ... x N(d-1)`` tensor *represented* by the
        Tucker tensor train, even though this dense tensor is never formed.

        Parameters
        ----------
        self: TuckerTensorTrain
            Tucker tensor train with ``shape=(N0,...,N(d-1))``
        index: NDArray
            Index array or convertible to ``NDArray`` with ``dtype=int`` and 
            ``shape=(d,)+idx_stack_shape``

        Returns
        -------
        :py:class:`.NDArray`
            Array of selected entry or multiple entries with ``shape=idx_stack_shape``

        Raises
        ------
        ValueError
            If ``len(index)`` is not equal to Tucker tensor train dimension

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.apply`
        :py:meth:`.TuckerTensorTrain.probe`

        Examples
        --------

        Compute one entry:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> index = [9, 4, 7]
        >>> result = x.entries(index)
        >>> result2 = x.to_dense()[9, 4, 7]
        >>> print(np.abs(result - result2))
        1.3322676295501878e-15

        With stacked index and stacked T3s:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> choice = np.random.choice
        >>> t3_stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2), t3_stack_shape)
        >>> idx_stack_shape = (4,5,1)
        >>> index = [choice(14, size=idx_stack_shape), choice(15, size=idx_stack_shape), choice(16, size=idx_stack_shape)]
        >>> entries = x.entries(index)
        >>> ii, jj = 1, 2
        >>> ll, mm, nn =  3, 2, 0
        >>> entry_ij_lmn = entries[ii,jj, ll,mm,nn]
        >>> x_ij_dense = x.to_dense()[ii,jj]
        >>> index_lmk = (index[0][ll,mm,nn], index[1][ll,mm,nn], index[2][ll,mm,nn])
        >>> entry_ij_lmn_true = x_ij_dense[index_lmk]
        >>> print(np.abs(entry_ij_lmn - entry_ij_lmn_true))
        0.0

        Example using jax jit compiling:

    	>>> import numpy as np
        >>> import jax
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> get_entry_123 = lambda x: x.entries((1,2,3))
        >>> A = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1)).to_jax() # Random 10x10x10 Tucker tensor train
        >>> a123 = get_entry_123(A)
        >>> print(a123)
        -1.3764521
        >>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
        >>> a123_jit = get_entry_123_jit(A)
        >>> print(a123_jit)
        -1.3764523

        .. Example using jax automatic differentiation
           
           >>> import numpy as np
           >>> import jax
           >>> import t3toolbox.tucker_tensor_train as t3
           >>> import t3toolbox.corewise as cw
           >>> jax.config.update("jax_enable_x64", True) # Enable double precision for finite difference
           >>> get_entry_123 = lambda x: x.entries((1,2,3))
           >>> A0 = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1), use_jax=True) # Random 10x10x10 Tucker tensor train
           >>> f0 = get_entry_123(A0)
           >>> G0 = jax.grad(get_entry_123)(A0) # Gradient using automatic differentiation
           >>> dA = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1), use_jax=True)
           >>> df = cw.corewise_dot(dA.data, G0.data) # Sensitivity in direction dA
           >>> print(df)
           -7.418801772515241
           >>> s = 1e-7
           >>> A1 = cw.corewise_add(A0.data, cw.corewise_scale(dA.data, s)) # A1 = A0 + s*dA
           >>> f1 = get_entry_123(t3.TuckerTensorTrain(*A1))
           >>> df_diff = (f1 - f0) / s # Finite difference
           >>> print(df_diff)
           -7.418812309825662
        '''
        if len(index) != self.d:
            raise ValueError(
                'Wrong number of indices for Tucker tensor train.\n'
                + str(self.d) + ' = num tensor indices != num provided indices = ' + str(index.shape[0])
            )

        return entries.tucker_tensor_train_entries(self.data, index)

    def apply(
        self,                     # shape=(N0,...,N(d-1))
        vecs: Sequence[NDArray],  # len=d, elm_shape=vecs_stack_shape+(Ni,)
    ) -> NDArray:
        '''Contract a Tucker tensor train with vectors in all indices.

        Parameters
        ----------
        self: TuckerTensorTrain
            Tucker tensor train with ``shape=(N0,...,N(d-1))``
        vecs: Sequence[NDArray]
            Vectors to contract with indices of ``self``. ``len=d``, ``elm_shape=vec_stack_shape+(Ni,)``

        Returns
        -------
        NDArray or scalar
            Result of contracting ``self`` with the vectors in all indices.
            Scalar if ``vecs`` elements are vectors, ``NDArray`` with shape ``vec_stack_shape`` if ``vecs`` elements are matrices.

        Raises
        ------
        ValueError
            Error raised if the provided vectors in ``vecs`` are inconsistent with each other or the Tucker tensor train.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.entries`
        :py:meth:`.TuckerTensorTrain.probe`

        Examples
        --------

        Apply to one set of vectors:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,1))
        >>> vecs = [np.random.randn(14), np.random.randn(15), np.random.randn(16)]
        >>> result = x.apply(vecs) # Contract x with vecs in all indices
        >>> result2 = np.einsum('ijk,i,j,k', x.to_dense(), vecs[0], vecs[1], vecs[2])
        >>> print(np.abs(result - result2))
        5.229594535194337e-12

        Apply to stacked vectors and stacked T3s (vectorized)

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,1), stack_shape)
        >>> vec_stack_shape = (4,5,1)
        >>> vecs = [randn(*(vec_stack_shape+(14,))), randn(*(vec_stack_shape+(15,))), randn(*(vec_stack_shape+(16,)))]
        >>> result = x.apply(vecs)
        >>> ii, jj = 1, 2 # T3 stack index
        >>> ll, mm, nn =  3, 2, 0 # Vectors stack index
        >>> result_ij_lmn = result[ii,jj, ll,mm,nn]
        >>> x_ij_dense = x.to_dense()[ii,jj]
        >>> vecs_lmn = [vecs[0][ll,mm,nn], vecs[1][ll,mm,nn], vecs[2][ll,mm,nn]]
        >>> result_ij_lmn_true = np.einsum('abc,a,b,c', x_ij_dense, *vecs_lmn)
        >>> print(np.abs(result_ij_lmn - result_ij_lmn_true))
        6.252776074688882e-13

        Example using jax automatic differentiation:

    	>>> import numpy as np
        >>> import jax
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> jax.config.update("jax_enable_x64", True)
        >>> A = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1)).to_jax() # Random 10x10x10 Tucker tensor train
        >>> apply_A_sym = lambda u: A.apply((u,u,u), use_jax=True) # Symmetric apply function
        >>> u0 = np.random.randn(10)
        >>> Auuu0 = apply_A_sym(u0)
        >>> g0 = jax.grad(apply_A_sym)(u0) # Gradient using automatic differentiation
        >>> du = np.random.randn(10)
        >>> dAuuu = np.dot(g0, du) # Derivative in direction du
        >>> print(dAuuu)
        766.5390335764645
        >>> s = 1e-7
        >>> u1 = u0 + s*du
        >>> Auuu1 = apply_A_sym(u1)
        >>> dAuuu_diff = (Auuu1 - Auuu0) / s # Finite difference approximation
        >>> print(dAuuu_diff)
        766.5390504030256
        '''
        if len(vecs) != len(self.shape):
            raise ValueError(
                'Attempted to apply TuckerTensorTrain to wrong number of vectors.'
                + str(str(len(self.shape)) + ' = num_indices != len(vecs) = ' + str(len(vecs)))
            )
        return apply.tucker_tensor_train_apply(self.data, vecs)

    def probe(
        self,
        ww: Sequence[NDArray],  # len=d, elm_shape=W+(Ni,)
    ) -> Sequence[NDArray]:     # zz, len=d, elm_shape=X+W+(Ni,)
        """Probe a TuckerTensorTrain.

        Parameters
        ----------
        self: TuckerTensorTrain
            Tucker tensor train with ``shape=(N0,...,N(d-1))``
        ww: Sequence[NDArray]
            Vectors to probe ``self`` with ``len=d``, ``elm_shape=W+(Ni,)``

        Returns
        -------
        Sequence[:py:class:`NDArray`]
            Results of contracting ``self`` with the vectors in all but one index for all indices.
            Sequence of vectors if ``ww`` elements are vectors, and sequence of ``NDArray``s each
            with ``elm_shape=W+(Ni,)`` if ``ww`` elements are matrices.
        
        See Also
        --------
        :py:meth:`.TuckerTensorTrain.entries`
        :py:meth:`.TuckerTensorTrain.apply`

        Examples
        --------

        Basic probing example:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.backend.probing as probing
        >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(2,3,4,2))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zz = x.probe(ww)
        >>> x_dense = x.to_dense()
        >>> zz0_true = np.einsum('abc,b,c', x_dense, ww[1], ww[2])
        >>> zz1_true = np.einsum('abc,a,c', x_dense, ww[0], ww[2])
        >>> zz2_true = np.einsum('abc,a,b', x_dense, ww[0], ww[1])
        >>> print(np.linalg.norm(zz[0] - zz0_true))
        1.5071547731580326e-12
        >>> print(np.linalg.norm(zz[1] - zz1_true))
        4.945327672021522e-13
        >>> print(np.linalg.norm(zz[2] - zz2_true))
        1.8042504599894852e-12

        Probe with stacked vectors and stacked T3s:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> randn = np.random.randn
        >>> stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,1), stack_shape)
        >>> vstack_shape = (4,5,1)
        >>> ww = [randn(*(vstack_shape+(14,))), randn(*(vstack_shape+(15,))), randn(*(vstack_shape+(16,)))]
        >>> result = x.probe(ww)
        >>> ii, jj = 1, 2
        >>> ll, mm, nn =  3, 2, 0
        >>> result_ij_lmn_0 = result[0][ii,jj, ll,mm,nn]
        >>> result_ij_lmn_1 = result[1][ii,jj, ll,mm,nn]
        >>> result_ij_lmn_2 = result[2][ii,jj, ll,mm,nn]
        >>> x_ij_dense = x.to_dense()[ii,jj]
        >>> result_ij_lmn_0_true = np.einsum('abc,b,c', x_ij_dense, ww[1][ll,mm,nn], ww[2][ll,mm,nn])
        >>> result_ij_lmn_1_true = np.einsum('abc,a,c', x_ij_dense, ww[0][ll,mm,nn], ww[2][ll,mm,nn])
        >>> result_ij_lmn_2_true = np.einsum('abc,a,b', x_ij_dense, ww[0][ll,mm,nn], ww[1][ll,mm,nn])
        >>> print(np.linalg.norm(result_ij_lmn_0 - result_ij_lmn_0_true))
        1.7836179565776773e-12
        >>> print(np.linalg.norm(result_ij_lmn_1 - result_ij_lmn_1_true))
        1.0522031983404444e-12
        >>> print(np.linalg.norm(result_ij_lmn_2 - result_ij_lmn_2_true))
        1.3936060000339696e-12
        """
        return probing.probe_t3(ww, self.data)

    ##############################################################
    ########################    T3-SVD    ########################
    ##############################################################

    def t3svd(
            self: 'TuckerTensorTrain',
            max_tt_ranks: Sequence[int] = None,     # len=d+1
            max_tucker_ranks: Sequence[int] = None, # len=d
            rtol: float = None,
            atol: float = None,
    ) -> Tuple[
        'TuckerTensorTrain', # new_x
        Tuple[NDArray,...],  # Tucker singular values, len=d
        Tuple[NDArray,...],  # TT singular values, len=d+1
    ]:
        '''Compute (truncated) T3-SVD of Tucker tensor train.
        
        Parameters
        ----------
        self: TuckerTensorTrain
            The Tucker tensor train. ``structure=((N0,...,N(d-1)), (n0,...,n(d-1)), (1,r1,...r(d-1),1))``
        max_tt_ranks: Sequence[int], optional
            Maximum TT-ranks ``ri``, e.g., ``(1,5,5,5,1)``. ``len(max_tt_ranks)=d+1``.
            Default: no max TT rank truncation (``None``).
        max_tucker_ranks: Sequence[int], optional
            Maximum Tucker ranks ``ni``, e.g., ``(5,5,5)``. ``len(max_tucker_ranks)=d``.
            Default: no max Tucker rank truncation (``None``).
        rtol: float, optional
            Relative tolerance for truncation (in the Frobenius norm).
            Default: no ``rtol`` rank truncation (``None``).
            Requires ``stack_shape=()``.
        atol: float, optional
            Absolute tolerance for truncation (in the Frobenius norm).
            Default: no ``atol`` rank truncation (``None``).
            Requires ``stack_shape=()``.

        Returns
        -------
        :py:class:`TuckerTensorTrain`
            New Tucker tensor train representing the same tensor (or a truncated version), but with modified cores
        Tuple[:py:class:`NDArray`,...]
            Singular values associated with edges between Tucker cores and TT cores
        Tuple[:py:class:`NDArray,...]
            Singular values associated with edges between adjacent TT cores

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.t3svd_dense`
        :py:meth:`.TuckerTensorTrain.get_minimal_ranks`

        Examples
        --------
        T3-SVD with no truncation (NOTE: ranks may decrease to minimal values):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((5,6,3), (4,4,3), (1,3,2,1))
        >>> x2, ss_tucker, ss_tt = x.t3svd() # Compute T3-SVD
        >>> x_dense = x.to_dense()
        >>> x2_dense = x2.to_dense()
        >>> print(np.linalg.norm(x_dense - x2_dense)) # Check error: Tensor unchanged
        7.556835759880194e-13
        >>> _, full_ss_tt0, _ = np.linalg.svd(x_dense.reshape((1, 5*6*3))) # SVDs of matrix unfoldings
        >>> _, full_ss_tt1, _ = np.linalg.svd(x_dense.reshape((5, 6*3)))
        >>> _, full_ss_tt2, _ = np.linalg.svd(x_dense.reshape((5*6, 3)))
        >>> _, full_ss_tt3, _ = np.linalg.svd(x_dense.reshape((5*6*3, 1)))
        >>> print(full_ss_tt0)
        [206.50692417]
        >>> print(ss_tt[0])
        [206.50692417]
        >>> print(full_ss_tt1)
        [1.86019457e+02 8.77024408e+01 1.87123807e+01 1.26952234e-14 5.91156202e-15]
        >>> print(ss_tt[1])
        [186.01945711  87.70244078  18.7123807 ]
        >>> print(full_ss_tt2)
        [2.06139857e+02 1.23072708e+01 3.57067836e-15]
        >>> print(ss_tt[2])
        [206.13985742  12.30727078]
        >>> print(full_ss_tt3)
        [206.50692417]
        >>> print(ss_tt[3])
        [206.50692417]
        >>> _, full_ss_tucker0, _ = np.linalg.svd(x_dense.transpose([0,1,2]).reshape((5,6*3))) # SVDs of matricizations
        >>> _, full_ss_tucker1, _ = np.linalg.svd(x_dense.transpose([1,0,2]).reshape((6,5*3)))
        >>> _, full_ss_tucker2, _ = np.linalg.svd(x_dense.transpose([2,1,0]).reshape((3,6*5)))
        >>> print(full_ss_tucker0)
        [1.86019457e+02 8.77024408e+01 1.87123807e+01 1.26952234e-14 5.91156202e-15]
        >>> print(ss_tucker[0])
        [186.01945711  87.70244078  18.7123807 ]
        >>> print(full_ss_tucker1)
        [1.85973158e+02 8.79127342e+01 1.74592922e+01 5.06146904e+00 1.67807237e-14 1.30640174e-15]
        >>> print(ss_tucker[1])
        [185.97315811  87.91273424  17.4592922    5.06146904]
        >>> print(full_ss_tucker2)
        [2.06139857e+02 1.23072708e+01 1.33340996e-14]
        >>> print(ss_tucker[2])
        [206.13985742  12.30727078]

        Stacked T3s:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((5,6,3), (4,4,3), (1,3,2,1), stack_shape=(2,3))
        >>> x2, ss_tucker, ss_tt = x.t3svd() # Compute T3-SVD
        >>> x_dense = x.to_dense()
        >>> x2_dense = x2.to_dense()
        >>> print(np.linalg.norm(x_dense - x2_dense)) # Check error: Tensor unchanged

        T3-SVD with truncation based on relative tolerance:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> B0 = np.random.randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2) # Preconditioned indices
        >>> B1 = np.random.randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
        >>> B2 = np.random.randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
        >>> G0 = np.random.randn(1,35,30)
        >>> G1 = np.random.randn(30,45,40)
        >>> G2 = np.random.randn(40,55,1)
        >>> tucker_cores_x = (B0, B1, B2)
        >>> tt_cores_x = (G0, G1, G2)
        >>> x = t3.TuckerTensorTrain(tucker_cores_x, tt_cores_x) # Tensor has spectral decay due to preconditioning
        >>> x2, ss_tucker, ss_tt = x.t3svd(rtol=1e-2) # Truncate singular values to reduce rank
        >>> print(x.structure)
        ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1), ())
        >>> print(x2.structure)
        ((40, 50, 60), (4, 3, 4), (1, 4, 4, 1), ())
        >>> x_dense = x.to_dense()
        >>> x2_dense = x2.to_dense()
        >>> print(np.linalg.norm(x_dense - x2_dense)/np.linalg.norm(x_dense)) # Should be near rtol=1e-2
        0.016157030535557206

        T3-SVD with truncation based on maximum ranks:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.t3_corewise_randn((14,15,16), (10,11,12), (1,8,9,1))
        >>> x2, ss_tucker, ss_tt = t3.t3svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1)) # Truncate based on ranks
        >>> print(x.uniform_structure)
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1), ())
        >>> print(x2.uniform_structure)
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1), ())
        '''
        if len(self.stack_shape) > 0 and ((rtol is not None) or (atol is not None)):
            raise ValueError(
                'Cannot use rtol or atol with t3svd for stacked Tucker tensor train.\n' +
                'Different elements of the stack could end out having different shapes.\n' +
                'First unstack, then call t3svd for each unstacked Tucker tensor train.'
            )

        result = ragged_t3svd.t3svd(
            self.data,
            max_tt_ranks=max_tt_ranks, max_tucker_ranks=max_tucker_ranks,
            rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1], result[2]

    @staticmethod
    def t3svd_dense(
            T: NDArray,                              # shape=stack_shape+(N1, N2, .., Nd)
            stack_shape: Sequence[int] = (),
            max_tucker_ranks: Sequence[int] = None,  # len=d
            max_tt_ranks: Sequence[int] = None,      # len=d+1
            rtol: float = None,
            atol: float = None,
    ) -> Tuple[
        'TuckerTensorTrain',  # Approximation of T by Tucker tensor train
        Tuple[NDArray, ...],  # Tucker singular values, len=d
        Tuple[NDArray, ...],  # TT singular values, len=d+1
    ]:
        '''Compute :py:class:`.TuckerTensorTrain` representation or approximation of a dense tensor.

        Parameters
        ----------
        T: NDArray
            The dense tensor. ``shape = stack_shape + (N0, ..., N(d-1))``
        stack_shape: Sequence[int], optional
            The stack shape. Default: no stacking (``stack_shape=()``)
        max_tucker_ranks: Sequence[int], optional
            Maximum Tucker ranks ``ni``, e.g., ``(5,5,5)``. ``len(max_tucker_ranks)=d``.
            Default: no max Tucker rank truncation (``None``).
        max_tt_ranks: Sequence[int], optional
            Maximum TT-ranks ``ri``, e.g., ``(1,5,5,5,1)``. ``len(max_tt_ranks)=d+1``.
            Default: no max TT rank truncation (``None``).
        rtol: float, optional
            Relative tolerance for truncation (in the Frobenius norm).
            Default: no ``rtol`` rank truncation (``None``).
            Requires ``stack_shape=()``.
        atol: float, optional
            Absolute tolerance for truncation (in the Frobenius norm).
            Default: no ``atol`` rank truncation (``None``).
            Requires ``stack_shape=()``.

        Returns
        -------
        TuckerTensorTrain
            Tucker tensor train approximation of ``T``
        Tuple[NDArray,...]
            Singular values of matricizations. ``len=d``. ``elm_shape=(ni,)``
        Tuple[NDArray,...]
            Singular values of unfoldings. ``len=d+1``. ``elm_shape=(ri,)``

        Raises
        ------
        ValueError
            If ``stack_shape`` is not empty and ``rtol`` or ``atol`` are supplied.
            (Cannot use tolerances with stacking)

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.t3svd`
        :py:meth:`.TuckerTensorTrain.get_minimal_ranks`

        Notes
        -----
        See Algorithm 3 in Appendix A of [1]_.

        References
        ----------
        .. [1] Alger, N., Christierson, B., Chen, P., & Ghattas, O. (2026).
           Tucker Tensor Train Taylor Series.
           arXiv preprint arXiv:2603.21141.
           .. __: https://arxiv.org/abs/2603.21141

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> T0 = np.random.randn(40, 50, 60)
        >>> c0 = 1.0 / np.arange(1, 41)**2
        >>> c1 = 1.0 / np.arange(1, 51)**2
        >>> c2 = 1.0 / np.arange(1, 61)**2
        >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
        >>> x, ss_tucker, ss_tt = t3.TuckerTensorTrain.t3svd_dense(T, rtol=1e-3) # Truncate T3-SVD to reduce rank
        >>> print(x.tucker_ranks) # Different random values may lead to different ranks
        (12, 12, 11)
        >>> print(x.tt_ranks)
        (1, 11, 12, 1)
        >>> T2 = x.to_dense()
        >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # Should be slightly more than rtol=1e-3
        0.001985061012010537
        '''
        if stack_shape and ((rtol is not None) or (atol is not None)):
            raise ValueError(
                'Cannot use t3svd_dense with rtol or atol for stacked tensor T.\n' +
                'Different elements of the stack could end out having different shapes.\n' +
                'First unstack, then call t3svd_dense for each unstacked tensor.\n' +
                'stack_shape = ' + str(stack_shape)
            )

        result = dense_t3svd.t3svd_dense(
            T,
            stack_shape=stack_shape,
            max_tucker_ranks=max_tucker_ranks, max_tt_ranks=max_tt_ranks,
            rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1], result[2]


if common.has_jax:
    jax.tree_util.register_pytree_node(
        TuckerTensorTrain,
        lambda x: (x.data, None),
        lambda aux_data, children: TuckerTensorTrain(*children),
    )