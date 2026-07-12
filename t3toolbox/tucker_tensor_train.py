# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""
Basic Tucker tensor trains with non-uniform (ragged) shape and ranks.
"""
import math
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass
from functools import cached_property

import t3toolbox.backend.t3_conversions as t3_conversions
import t3toolbox.backend.t3_constructors as t3_constructors
import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.probing as probing
import t3toolbox.backend.sampling_derivatives as sampling_derivatives
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.tt_orthogonalization as orth
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
if common.jax_available:
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
    the "1"s in the diagram are vectors of ones. You can make ``r0=rd=1`` using :py:meth:`~squash_tails`.

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
    >>> result = x.t3svd(rtol=1e-2) # Error!   # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError


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
    >>> print(x.minimal_ranks)            # the inflated TT bond 99 is structurally minimal at 4
    ((4, 5, 6, 7), (1, 4, 9, 7, 1))
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
    >>> print(np.allclose((x + y).to_dense(), x.to_dense() + y.to_dense()))
    True

    A more complicated linear algebra operation with three TuckerTensorTrains

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.TuckerTensorTrain.randn((13,14,15,16), (2,3,4,3), (1,2,4,3,2))
    >>> y = t3.TuckerTensorTrain.randn((13,14,15,16), (4,3,5,1), (4,3,2,1,2))
    >>> z = t3.TuckerTensorTrain.randn((13,14,15,16), (1,2,3,4), (1,2,3,4,5))
    >>> result = (x * (y * 2.4 + z)).inner(z) + (x - y).norm() + z.sum()
    >>> X, Y, Z = x.to_dense(), y.to_dense(), z.to_dense()
    >>> result2 = np.einsum('ijkl,ijkl', (X * (Y * 2.4 + Z)), Z) + np.linalg.norm(X - Y) + Z.sum()
    >>> print(np.allclose(result, result2))
    True

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

    def __repr__(self) -> str:
        ss = f", stack_shape={self.stack_shape}" if self.stack_shape else ""
        return (f"TuckerTensorTrain(shape={self.shape}, tucker_ranks={self.tucker_ranks}, "
                f"tt_ranks={self.tt_ranks}{ss})")

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,4,5), stack_shape=(9,))
        >>> print(t3.TuckerTensorTrain.get_core_shapes(x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape))
        (((9, 4, 14), (9, 5, 15), (9, 6, 16)), ((9, 1, 4, 3), (9, 3, 5, 4), (9, 4, 6, 5)))
        >>> print(x.core_shapes)    # the core_shapes property reports the per-slice shapes (stack stripped)
        (((4, 14), (5, 15), (6, 16)), ((1, 4, 3), (3, 5, 4), (4, 6, 5)))
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
        return math.prod(self.shape)

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
        ((4, 5, 6, 7), (1, 4, 9, 7, 1))
        >>> x2 = x.t3svd()[0]
        >>> print(x2.has_minimal_ranks)
        True
        """
        return (self.tucker_ranks, self.tt_ranks) == self.minimal_ranks

    def has_numerically_minimal_ranks(self, rtol: float = 1e-9) -> bool:
        """True if the ranks are **numerically** minimal: no stored rank is numerically redundant.

        Distinct from :py:attr:`has_minimal_ranks`, which is **structural** (ranks equal the structural
        minimum). A tensor can be structurally minimal yet have a near-zero singular value at some rank
        boundary (numerically redundant); this catches that. Algorithm: the cheap **structural** check
        first (structural redundancy implies numerical redundancy), then -- only if structurally minimal
        -- an :py:meth:`t3svd` at relative tolerance ``rtol`` and a comparison of the truncated ranks to
        the stored ranks. The ``t3svd`` makes this O(tensor) -- a diagnostic, not a hot-path check.

        For an **orthonormal frame** prefer :py:meth:`T3Frame.has_numerically_minimal_ranks`, which needs
        no SVD (orthonormal cores are full-rank, so structurally-minimal => numerically-minimal).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((6, 7, 5), (2, 2, 2), (1, 2, 2, 1))
        >>> print(x.has_minimal_ranks, x.has_numerically_minimal_ranks())   # full-rank random tensor
        True True
        >>> xbig = x.resize((6, 7, 5), (3, 2, 2), (1, 2, 2, 1))  # tucker_0 padded -> a redundant rank
        >>> print(xbig.has_minimal_ranks, xbig.has_numerically_minimal_ranks())
        False False
        """
        if not self.has_minimal_ranks:
            return False                                # structural redundancy => numerical redundancy
        truncated = self.t3svd(rtol=rtol)[0]
        return (self.tucker_ranks, self.tt_ranks) == (truncated.tucker_ranks, truncated.tt_ranks)

    def is_left_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        """True (per stack element) if this T3 is in **left-orthogonal form**: every Tucker core
        down-orthogonal and every TT core except the last left-orthogonal (the last TT core is the center
        remainder).

        Non-enforcing convenience checker (max-abs deviation from the identities ``<= atol``; see
        :py:func:`~t3toolbox.backend.t3_orthogonalization.t3_orthogonality_residual`). A
        :py:meth:`t3svd` result is left-orthogonal, as is the result of
        ``rank_adjustment_sweep('left_to_right')``. **Per-stack-element bool array** (scalar when
        unstacked); reduce with ``.all()`` for a single verdict.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (5, 6, 7), (1, 4, 3, 1))
        >>> print(x.is_left_orthogonal())     # a random T3 is not in any orthogonal form
        False
        >>> x2, _, _ = x.t3svd()              # a t3svd result is left-orthogonal
        >>> print(x2.is_left_orthogonal())
        True
        >>> print(x2.is_right_orthogonal())   # ...but not right-orthogonal
        False

        Stacked: a per-element bool array. Stack a left-orthogonal element with a non-orthogonal one:

        >>> m = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1))  # minimal -> t3svd keeps ranks
        >>> stacked = t3.TuckerTensorTrain.stack([m.t3svd()[0], m])
        >>> print(stacked.is_left_orthogonal().shape, stacked.is_left_orthogonal())
        (2,) [ True False]
        """
        return ragged_orthogonalization.t3_orthogonality_residual(self.data, 'left') <= atol

    def is_right_orthogonal(self, atol: float = 1e-9) -> NDArray:  # bool array, shape = stack_shape (scalar unstacked)
        """True (per stack element) if this T3 is in **right-orthogonal form**: every Tucker core
        down-orthogonal and every TT core except the first right-orthogonal (the first TT core is the
        center remainder).

        Non-enforcing convenience checker (see :py:meth:`is_left_orthogonal`). The result of
        ``rank_adjustment_sweep('right_to_left')`` is right-orthogonal. Use this to verify before
        asserting ``t3svd(..., assume_orthogonal=True)``, which is **not** checked. **Per-stack-element
        bool array** (scalar when unstacked); reduce with ``.all()``.
        """
        return ragged_orthogonalization.t3_orthogonality_residual(self.data, 'right') <= atol

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
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(4,14),randn(5,15),randn(6,16))
        >>> tt_cores = (randn(2,4,3), randn(3,5,2), randn(2,6,5))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x_dense = x.to_dense() # Convert TuckerTensorTrain to dense tensor
        >>> ((B0,B1,B2), (G0,G1,G2)) = tucker_cores, tt_cores
        >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
        >>> print(np.allclose(x_dense, x_dense2))
        True

        Example where leading and trailing ones are not contracted

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(4,14),randn(5,15),randn(6,16))
        >>> tt_cores = (randn(2,4,3), randn(3,5,2), randn(2,6,2))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x_dense = x.to_dense(squash_tails=False) # Convert TuckerTensorTrain to dense tensor
        >>> print(x_dense.shape)                    # keeps the outer TT bonds r0=rd=2
        (2, 14, 15, 16, 2)
        >>> ((B0,B1,B2), (G0,G1,G2)) = tucker_cores, tt_cores
        >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
        >>> print(np.allclose(x_dense, x_dense2))
        True

        Example with stacking

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(2,3, 4,10), randn(2,3, 5,11), randn(2,3, 6,12))
        >>> tt_cores = (randn(2,3, 2,4,3), randn(2,3, 3,5,2), randn(2,3, 2,6,5))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> x_dense = x.to_dense() # Convert TuckerTensorTrain to dense tensor
        >>> print(x_dense.shape)                    # leading (2,3) is the stack_shape
        (2, 3, 10, 11, 12)
        >>> ((B0,B1,B2), (G0,G1,G2)) = tucker_cores, tt_cores
        >>> x_dense2 = np.einsum('uvxi,uvyj,uvzk,uvaxb,uvbyc,uvczd->uvijk', B0, B1, B2, G0, G1, G2)
        >>> print(np.allclose(x_dense, x_dense2))
        True
        """
        return t3_conversions.t3_to_dense(
            self.data, squash_tails=squash_tails,
        )

    def segment(
            self,
            start: int,  # requires stop > start
            stop:  int,  # requires stop > start
    ) -> 'TuckerTensorTrain':
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
        return TuckerTensorTrain(*ragged_operations.t3_segment(self.data, start, stop))

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
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tk = (randn(4,14), randn(5,15), randn(6,16), randn(7,17), randn(8,18), randn(9,19))
        >>> tt = (randn(2,4,3), randn(3,5,2), randn(2,6,2), randn(2,7,3), (randn(3,8,4)), (randn(4,9,1)))
        >>> x = t3.TuckerTensorTrain(tk[:3], tt[:3])
        >>> y = t3.TuckerTensorTrain(tk[3:4], tt[3:4])
        >>> z = t3.TuckerTensorTrain(tk[4:], tt[4:])
        >>> xyz = t3.TuckerTensorTrain.concatenate([x, y, z])
        >>> xyz2 = t3.TuckerTensorTrain(tk, tt)              # the same train, built in one piece
        >>> print(np.allclose(xyz.to_dense(), xyz2.to_dense()))
        True
        """
        return TuckerTensorTrain(*ragged_operations.t3_concatenate([x.data for x in xx]))

    def squash_tails(
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
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(2,3, 4,10), randn(2,3, 5,11), randn(2,3, 6,12))
        >>> tt_cores = (randn(2,3, 2,4,3), randn(2,3, 3,5,2), randn(2,3, 2,6,5))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.tt_ranks)
        (2, 3, 2, 5)
        >>> x2 = x.squash_tails()
        >>> print(x2.tt_ranks)                  # leading/trailing bonds forced to 1
        (1, 3, 2, 1)
        >>> print(np.allclose(x.to_dense(), x2.to_dense()))   # same dense tensor
        True
        """
        return TuckerTensorTrain(*ragged_operations.t3_squash_tails(self.data))

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
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tucker_cores = (randn(2,3, 4,10), randn(2,3, 5,11), randn(2,3, 6,12))
        >>> tt_cores = (randn(2,3, 1,4,2), randn(2,3, 2,5,3), randn(2,3, 3,6,4))
        >>> x = t3.TuckerTensorTrain(tucker_cores, tt_cores)
        >>> print(x.structure)
        ((10, 11, 12), (4, 5, 6), (1, 2, 3, 4), (2, 3))
        >>> reversed_x = x.reverse()
        >>> print(reversed_x.structure)             # shape, Tucker and TT ranks all reversed
        ((12, 11, 10), (6, 5, 4), (4, 3, 2, 1), (2, 3))
        >>> x_dense = x.to_dense()
        >>> reversed_x_dense = reversed_x.to_dense()
        >>> x_dense2 = reversed_x_dense.transpose([0,1, 4,3,2])   # un-reverse the free axes
        >>> print(np.allclose(x_dense, x_dense2))
        True
        """
        reversed_tucker_cores = tuple([B.copy() for B in self.tucker_cores[::-1]])
        reversed_tt_cores = tt_operations.tt_reverse(self.tt_cores)
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

        new_tucker_cores = ragged_operations.tucker_change_core_shapes(tucker_cores, new_shape, new_tucker_ranks)
        new_tt_cores = tt_operations.tt_change_core_shapes(tt_cores, new_tucker_ranks, new_tt_ranks)

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(3,5))
        >>> unstacked_x = x.unstack()
        >>> print([len(s) for s in unstacked_x])    # nested tuples mirror stack_shape=(3,5)
        [5, 5, 5]
        >>> tucker13 = tuple([B[1,3] for B in x.tucker_cores])   # build slice (1,3) by hand
        >>> tt13 = tuple([G[1,3] for G in x.tt_cores])
        >>> x13 = t3.TuckerTensorTrain(tucker13, tt13)
        >>> print(bool((x13 - unstacked_x[1][3]).norm() < 1e-11))   # matches unstacked leaf [1][3]
        True
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

        return TuckerTensorTrain(*t3_constructors.t3_zeros(
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
        return TuckerTensorTrain(*t3_constructors.t3_ones(
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
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> shape = (14, 15, 16)
        >>> tucker_ranks = (4, 5, 6)
        >>> tt_ranks = (1, 3, 2, 1)
        >>> stack_shape = (2, 3)
        >>> x = t3.TuckerTensorTrain.randn(shape, tucker_ranks, tt_ranks, stack_shape=stack_shape) # random cores
        >>> print(x.structure == (shape, tucker_ranks, tt_ranks, stack_shape))
        True
        >>> print(np.any(x.tucker_cores[0] != 0.0))  # cores are filled with N(0,1) draws, not zeros
        True
        """
        return TuckerTensorTrain(*t3_constructors.t3_corewise_randn(
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

        return TuckerTensorTrain(*t3_conversions.t3_from_canonical(factors))

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
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> tt_cores = [randn(4,14,5), randn(5,15,3), randn(3,16,2)]
        >>> x = t3.TuckerTensorTrain.from_tensor_train(tt_cores)
        >>> x_dense = x.to_dense()
        >>> x_dense2 = np.einsum('...aib,...bjc,...ckd->...ijk', *tt_cores)
        >>> print(np.allclose(x_dense, x_dense2))
        True
        """
        return TuckerTensorTrain(*t3_conversions.t3_from_tensor_train(tt_cores))

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (5,6,7), (2,3,4,1), (2,3))
        >>> big_tt_cores = x.to_tensor_train()
        >>> x_dense = np.einsum('...aib,...bjc,...ckd->...ijk', *big_tt_cores)
        >>> x_dense2 = x.to_dense()
        >>> print(np.allclose(x_dense, x_dense2))
        True
        """
        return t3_conversions.t3_to_tensor_train(self.data)

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
        return t3_conversions.t3_to_vector(self.data)

    @staticmethod
    def from_vector(
            x_flat:         NDArray,             # shape=(data_size,)
            shape:          Sequence[int],       # len=d
            tucker_ranks:   Sequence[int],       # len=d
            tt_ranks:       Sequence[int],       # len=d+1
            stack_shape:    Sequence[int] = (),  # () if unstacked
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
        return TuckerTensorTrain(*t3_conversions.t3_from_vector(
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
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> fname = 't3_file.npz'
        >>> x.save(fname) # Save to file 't3_file.npz'
        >>> x2 = t3.TuckerTensorTrain.load(fname) # Load from file
        >>> tucker_cores, tt_cores = x.data
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> print([float(np.linalg.norm(B - B2)) for B, B2 in zip(tucker_cores, tucker_cores2)])
        [0.0, 0.0, 0.0]
        >>> print([float(np.linalg.norm(G - G2)) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
        """
        common.save_core_families(file, self.data)

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
        >>> fname = 't3_file.npz'
        >>> x.save(fname) # Save to file 't3_file.npz'
        >>> x2 = t3.TuckerTensorTrain.load(fname) # Load from file
        >>> tucker_cores, tt_cores = x.data
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> print([float(np.linalg.norm(B - B2)) for B, B2 in zip(tucker_cores, tucker_cores2)])
        [0.0, 0.0, 0.0]
        >>> print([float(np.linalg.norm(G - G2)) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
        """
        tucker_cores, tt_cores = common.load_core_families(file)
        x = TuckerTensorTrain(tucker_cores, tt_cores)
        return x.to_jax() if use_jax else x

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> y = t3.TuckerTensorTrain.randn((14,15,16), (3,7,2), (1,5,6,1))
        >>> z = x + y
        >>> print(np.allclose(x.to_dense() + y.to_dense(), z.to_dense()))
        True
        >>> print(z.structure)                  # adding T3s ADDS their ranks: Tucker 4+3,5+7,6+2; TT 1+1,3+5,2+6,1+1
        ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2), ())

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
        >>> sx = x * 3.2                                  # scale a T3 by a scalar -> T3
        >>> print(np.allclose(3.2 * x.to_dense(), sx.to_dense()))
        True

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
        >>> y = np.random.randn(*(x.stack_shape + x.shape))
        >>> xy = x * y                                    # T3 * ndarray -> dense ndarray (elementwise product)
        >>> print(xy.shape)
        (2, 3, 14, 15, 16)
        >>> print(np.allclose(x.to_dense() * y, xy))
        True

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1), stack_shape=(2, 3))
        >>> y = t3.TuckerTensorTrain.randn((14, 15, 16), (2, 3, 4), (3, 2, 3, 2), stack_shape=(2, 3))
        >>> xy = x * y                                    # elementwise product of two T3s -> T3
        >>> print(np.allclose(x.to_dense() * y.to_dense(), xy.to_dense()))
        True
        >>> print(xy.tucker_ranks)                        # Tucker ranks MULTIPLY: 4*2, 5*3, 6*4
        (8, 15, 24)
        >>> print(xy.tt_ranks)                            # and the TT bonds: 1*3, 3*2, 2*3, 1*2
        (3, 6, 6, 2)
        """
        if common.is_ndarray(other):
            if other.shape == ():
                return TuckerTensorTrain(*ragged_linalg.t3_scale(self.data, other))
            else:
                assert(other.shape == self.stack_shape + self.shape)
                return self.to_dense() * other

        elif isinstance(other, TuckerTensorTrain):
            return self.t3m(other, method='form_then_round')

        else: # assume scalar
            return TuckerTensorTrain(*ragged_linalg.t3_scale(self.data, other))

    def t3m(
            self,
            other:              'TuckerTensorTrain',    # same shape & stack_shape
            method:             str = 'inplace_fused',  # the memory-light default for large d

            max_tucker_ranks:   typ.Union[int, Sequence[int], None] = None,  # scalar (caps all) or len=d
            max_tt_ranks:       typ.Union[int, Sequence[int], None] = None,  # scalar (caps all) or len=d+1

            rtol:               typ.Optional[float] = None,  # requires unstacked
            atol:               typ.Optional[float] = None,  # requires unstacked

            oversample:         float = 1,  # method='swap' only; intermediate-rank relaxation
    ) -> 'TuckerTensorTrain':
        """Elementwise (Hadamard) product ``self ⊙ other`` with optional rank truncation.

        Like ``self * other`` (whose ranks multiply: ``n_x·n_y`` Tucker, ``r_x·r_y`` TT) but able to
        **truncate** the result. ``method`` selects the algorithm -- all give the same product, with
        different cost/memory trade-offs (see ``docs/ttm_t3m_ht_note.tex``):

        - ``'form_then_round'`` -- form the full product, then round. Parallel forming; cheapest to
          run for small bonds. This is what ``*`` uses.
        - ``'inplace_fused'`` (default) -- a fused sweep that never materializes the full product;
          the right default for large ``d``.
        - ``'swap'`` -- the swap-based TTM generalization; best when the TT bond ``r`` greatly exceeds
          the number of modes ``d`` (``O(d²·r³)`` compute, ``O(r̃²)`` memory).

        Truncation (any combination; default none ⇒ exact full product): ``max_tucker_ranks`` /
        ``max_tt_ranks`` (a scalar caps every position, or a per-position sequence) and ``rtol`` /
        ``atol`` (per-step relative/absolute tolerances).

        ``oversample`` (``method='swap'`` only, ``>= 1``, default ``1`` = off): relaxes the
        intermediate ranks/tolerances by this factor during the swaps and runs a final ``t3svd``
        cleanup at the exact targets. ``1`` is the lowest-memory / lowest-quality corner; a modest
        ``2`` is a good default for near-``form_then_round`` quality at a small memory cost; quality
        approaches ``form_then_round`` as ``oversample → ∞``. See ``docs/ttm_t3m_ht_note.tex`` for why this is needed (the Tucker leaf-frame coupling).

        .. warning::
            ``rtol``/``atol`` are **not supported for stacked** Tucker tensor trains (different stack
            elements could truncate to different ranks). Use ``max_*_ranks`` for stacked input, or
            unstack first. Max-rank truncation *is* stacking-compatible.
        """
        if not isinstance(other, TuckerTensorTrain):
            raise TypeError('t3m requires a TuckerTensorTrain, got %s' % type(other).__name__)
        if self.shape != other.shape:
            raise ValueError('t3m shape mismatch: %s vs %s' % (self.shape, other.shape))
        if self.stack_shape != other.stack_shape:
            raise ValueError('t3m stack_shape mismatch: %s vs %s' % (self.stack_shape, other.stack_shape))
        if (rtol is not None or atol is not None) and len(self.stack_shape) > 0:
            raise ValueError(
                'rtol/atol truncation is not supported for stacked Tucker tensor trains '
                '(stack elements could truncate to different ranks).\n'
                'Use max_tucker_ranks/max_tt_ranks for stacked input, or unstack first.')

        valid = ('form_then_round', 'inplace_fused', 'swap')
        if method not in valid:
            raise ValueError('t3m method must be one of %s, got %r' % (valid, method))
        if oversample < 1:
            raise ValueError('t3m oversample must be >= 1, got %r' % (oversample,))
        if oversample != 1 and method != 'swap':
            raise ValueError('t3m oversample only applies to method="swap", got method=%r' % (method,))
        backend = {
            'form_then_round': ragged_linalg.t3m_form_then_round,
            'inplace_fused': ragged_linalg.t3m_inplace_fused,
            'swap': ragged_linalg.t3m_swap,
        }[method]

        kw = dict(max_tucker_ranks=max_tucker_ranks, max_tt_ranks=max_tt_ranks, rtol=rtol, atol=atol)
        if method == 'swap':
            kw['oversample'] = oversample
        return TuckerTensorTrain(*backend(self.data, other.data, **kw))

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (2, 3, 2, 2))
        >>> y = t3.TuckerTensorTrain.randn((14, 15, 16), (3, 7, 2), (3, 5, 6, 3))
        >>> hs = x.inner(y)                               # Hilbert-Schmidt inner product (a scalar)
        >>> print(np.allclose(hs, np.sum(x.to_dense() * y.to_dense())))
        True

        (T3, T3) with stacking -- one inner product per stack element:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (2, 3, 2, 2), stack_shape=(2, 3))
        >>> y = t3.TuckerTensorTrain.randn((14, 15, 16), (3, 7, 2), (3, 5, 6, 3), stack_shape=(2, 3))
        >>> hs = x.inner(y)
        >>> print(hs.shape)                               # result carries the stack shape
        (2, 3)
        >>> print(np.allclose(hs, np.sum(x.to_dense() * y.to_dense(), axis=(2, 3, 4))))
        True

        Inner product of a T3 with a dense tensor:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (3, 7, 2), (3, 5, 6, 3))
        >>> y = np.random.randn(14, 15, 16)
        >>> print(np.allclose(x.inner(y), np.sum(x.to_dense() * y)))
        True

        ...with stacking (the dense array carries the stack axes):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (3, 7, 2), (3, 5, 6, 3), stack_shape=(2, 3))
        >>> y = np.random.randn(2, 3, 14, 15, 16)         # shape = stack_shape + shape
        >>> print(np.allclose(x.inner(y), np.einsum('ijxyz,ijxyz->ij', x.to_dense(), y)))
        True

        Gotcha -- the two tensors must have the same shape (raises otherwise):

        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((4, 5), (2, 2), (1, 2, 1))
        >>> y = t3.TuckerTensorTrain.randn((4, 6), (2, 2), (1, 2, 1))   # different shape!
        >>> x.inner(y)                                    # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError
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

            return ragged_linalg.t3_inner_product(
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> print(np.allclose(x.norm(), np.linalg.norm(x.to_dense())))
        True

        Stacked -- ``norm()`` returns an array of shape ``stack_shape``, one norm per slice:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
        >>> norms_x = x.norm(use_orthogonalization=True)
        >>> print(norms_x.shape)
        (2, 3)
        >>> x_dense = x.to_dense()
        >>> norms_x_dense = np.sqrt(np.sum(x_dense**2, axis=(-3,-2,-1)))
        >>> print(np.allclose(norms_x, norms_x_dense))
        True
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10,11,12,13), (7,8,9,10), (2,3,4,3,1), (2,3))
        >>> S = x.sum()                         # all free axes summed -> ndarray over the stack
        >>> dense_x = x.to_dense()
        >>> non_stack_axes = (2,3,4,5)
        >>> print(np.allclose(S, dense_x.sum(axis=non_stack_axes)))
        True
        >>> print(type(S))
        <class 'numpy.ndarray'>
        >>> print(S.shape)
        (2, 3)

        Axis is a tuple of ints:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10,11,12,13), (7,8,9,10), (2,3,4,3,1), (2,3))
        >>> axis = (1,3)
        >>> S = x.sum(axis=axis)                # some axes kept -> TuckerTensorTrain
        >>> dense_x = x.to_dense()
        >>> shifted_axis = tuple(ii + len(x.stack_shape) for ii in axis)
        >>> print(np.allclose(S.to_dense(), dense_x.sum(axis=shifted_axis)))
        True
        >>> print(type(S))
        <class 't3toolbox.tucker_tensor_train.TuckerTensorTrain'>
        >>> print(S.shape)
        (10, 12)
        >>> print(S.stack_shape)
        (2, 3)

        Axis is int:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10,11,12,13), (7,8,9,10), (2,3,4,3,1), (2,3))
        >>> axis = 1
        >>> S = x.sum(axis=axis)
        >>> dense_x = x.to_dense()
        >>> shifted_axis = axis + len(x.stack_shape)
        >>> print(np.allclose(S.to_dense(), dense_x.sum(axis=shifted_axis)))
        True
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((4,5,6), (2,3,2), (1,2,2,1), stack_shape=(3,))
        >>> y = x.sum_stack()
        >>> print(y.stack_shape)
        ()
        >>> print(np.allclose(y.to_dense(), x.to_dense().sum(axis=0)))
        True

        Ranks grow by the summed stack size (here ``S=3``):

        >>> print(x.tucker_ranks, '->', y.tucker_ranks)
        (2, 3, 2) -> (6, 9, 6)
        >>> print(x.tt_ranks, '->', y.tt_ranks)
        (1, 2, 2, 1) -> (1, 6, 6, 1)

        Sum over one of several stack axes (the rest are kept):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((4,5,6), (2,3,2), (1,2,2,1), stack_shape=(2,3))
        >>> y = x.sum_stack(axis=0)
        >>> print(y.stack_shape)
        (3,)
        >>> print(np.allclose(y.to_dense(), x.to_dense().sum(axis=0)))
        True
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
        tensors (for that, use :py:meth:`.TuckerTensorTrain.sum_stack`). It is useful when the cores carry an additive
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
        return TuckerTensorTrain(*corewise.corewise_stack_sum(self.data, axis, len(self.stack_shape)))


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
            If this TuckerTensorTrain is stacked and ``rtol`` or ``atol`` are not ``None``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.left_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.right_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.down_svd_tucker_core(ind)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> rank = len(ss)
        >>> B = tucker_cores2[ind]
        >>> print(bool(np.linalg.norm(B @ B.T - np.eye(rank)) < 1e-12)) # Tucker core is (down) orthogonal
        True
        '''
        result = ragged_orthogonalization.t3_down_svd_tucker_core(
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
            If this TuckerTensorTrain is stacked and ``rtol`` or ``atol`` are not ``None``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_svd_tucker_core`
        :py:meth:`.TuckerTensorTrain.right_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.left_svd_tt_core(ind)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(bool(np.linalg.norm(np.einsum('iaj,iak->jk', G, G) - np.eye(G.shape[2])) < 1e-12)) # TT-core is left-orthogonal
        True
        '''
        result = ragged_orthogonalization.t3_left_svd_tt_core(
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
            If this TuckerTensorTrain is stacked and ``rtol`` or ``atol`` are not ``None``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.down_svd_tucker_core`
        :py:meth:`.TuckerTensorTrain.left_svd_tt_core`
        :py:meth:`.TuckerTensorTrain.up_svd_tt_core`

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.right_svd_tt_core(ind)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(bool(np.linalg.norm(np.einsum('iaj,kaj->ik', G, G) - np.eye(G.shape[0])) < 1e-12)) # TT-core is right orthogonal
        True
        '''
        result = ragged_orthogonalization.t3_right_svd_tt_core(
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ind = 1
        >>> x2, ss = x.up_svd_tt_core(ind)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> tucker_cores2, tt_cores2 = x2.data
        >>> G = tt_cores2[ind]
        >>> print(bool(np.linalg.norm(np.einsum('iaj,ibj->ab', G, G) - np.eye(G.shape[1])) < 1e-12)) # TT-core is down orthogonal
        True
        '''
        result = ragged_orthogonalization.t3_up_svd_tt_core(
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x2 = x.orthogonalize_relative_to_tucker_core(1)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> X = np.einsum('xi,axb,byc,czd,zk->iyk', B0, G0, G1, G2, B2) # Contraction of everything except B1
        >>> print(bool(np.linalg.norm(np.einsum('iyk,iwk->yw', X, X) - np.eye(B1.shape[0])) < 1e-12)) # Complement of B1 is orthogonal
        True

        Example where first and last TT-ranks are not 1:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> x2 = x.orthogonalize_relative_to_tucker_core(0)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> X = np.einsum('yj,zk,axb,byc,czd->axjkd', B1, B2, G0, G1, G2) # Contraction of everything except B0
        >>> print(bool(np.linalg.norm(np.einsum('axjkd,ayjkd->xy', X, X) - np.eye(B0.shape[0])) < 1e-12)) # Complement of B0 is orthogonal
        True
        '''
        return TuckerTensorTrain(*ragged_orthogonalization.t3_orthogonalize_relative_to_tucker_core(
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x2 = x.orthogonalize_relative_to_tt_core(1)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> XL = np.einsum('axb,xi -> aib', G0, B0) # Everything to the left of G1
        >>> print(bool(np.linalg.norm(np.einsum('aib,aic->bc', XL, XL) - np.eye(G1.shape[0])) < 1e-12)) # Left subtree is left orthogonal
        True
        >>> print(bool(np.linalg.norm(np.einsum('xi,yi->xy', B1, B1) - np.eye(G1.shape[1])) < 1e-12)) # Core below G1 is up orthogonal
        True
        >>> XR = np.einsum('axb,xi->aib', G2, B2) # Everything to the right of G1
        >>> print(bool(np.linalg.norm(np.einsum('aib,cib->ac', XR, XR) - np.eye(G1.shape[2])) < 1e-12)) # Right subtree is right orthogonal
        True

        Example where first and last TT-ranks are not 1:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2))
        >>> x2 = x.orthogonalize_relative_to_tt_core(0)
        >>> print(np.allclose(x.to_dense(), x2.to_dense())) # Tensor unchanged
        True
        >>> ((B0, B1, B2), (G0, G1, G2)) = x2.data
        >>> XR = np.einsum('yi,zj,byc,czd->bijd', B1, B2, G1, G2) # Everything to the right of G0
        >>> print(bool(np.linalg.norm(np.einsum('bijd,cijd->bc', XR, XR) - np.eye(G0.shape[2])) < 1e-12)) # Right subtree is right orthogonal
        True
        '''
        return TuckerTensorTrain(*ragged_orthogonalization.t3_orthogonalize_relative_to_tt_core(
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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.down_orthogonalize_tucker_cores()
        >>> print(bool((x - x_orth).norm() < 1e-11)) # represents the same tensor
        True
        >>> ind = 1
        >>> B = x_orth.data[0][ind]
        >>> print(bool(np.linalg.norm(B @ B.T - np.eye(B.shape[0])) < 1e-12)) # Tucker core is down orthogonal
        True

        Stacked -- ``norm()`` returns a per-stack array; orthogonality holds on every slice:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.down_orthogonalize_tucker_cores()
        >>> print(bool(np.max((x - x_orth).norm()) < 1e-11)) # same tensor on every stack slice
        True
        >>> ind = 1
        >>> B = x_orth.data[0][ind]
        >>> BtB = np.einsum('abio,abjo->abij',B,B)
        >>> errs = [[np.linalg.norm(BtB[ii,jj] - np.eye(BtB.shape[-1])) for jj in range(3)] for ii in range(2)]
        >>> print(bool(np.max(errs) < 1e-12))
        True
        """
        return TuckerTensorTrain(*ragged_orthogonalization.t3_down_orthogonalize_tucker_cores(self.data))

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.up_orthogonalize_tt_cores()
        >>> print(bool((x - x_orth).norm() < 1e-11)) # represents the same tensor
        True
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(bool(np.linalg.norm(np.einsum('iaj,ibj->ab',G,G)-np.eye(G.shape[1])) < 1e-12)) # TT core is up orthogonal
        True

        Stacked -- ``norm()`` returns a per-stack array; orthogonality holds on every slice:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.up_orthogonalize_tt_cores()
        >>> print(bool(np.max((x - x_orth).norm()) < 1e-11)) # same tensor on every stack slice
        True
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> GdG = np.einsum('xyaib,xyajb->xyij',G,G)
        >>> errs = [[np.linalg.norm(GdG[ii,jj] - np.eye(GdG.shape[-1])) for jj in range(3)] for ii in range(2)]
        >>> print(bool(np.max(errs) < 1e-12))
        True
        """
        return TuckerTensorTrain(
            *ragged_orthogonalization.t3_up_orthogonalize_tt_cores(self.data),
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
            Used to construct variation cores when converting a TuckerTensorTrain to frame-variation format.

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.left_orthogonalize_tt_cores()
        >>> print(bool((x - x_orth).norm() < 1e-11)) # represents the same tensor
        True
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(bool(np.linalg.norm(np.einsum('iaj,iak->jk',G,G)-np.eye(G.shape[2])) < 1e-12)) # TT core is left orthogonal
        True

        Stacked -- ``norm()`` returns a per-stack array; orthogonality holds on every slice:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.left_orthogonalize_tt_cores()
        >>> print(bool(np.max((x - x_orth).norm()) < 1e-11)) # same tensor on every stack slice
        True
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(bool(np.linalg.norm(np.einsum('xyiaj,xyiak->xyjk',G,G)-np.eye(G.shape[-1])) < 1e-12))
        True
        """
        result = orth.tt_left_orthogonalize(
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
            Used to construct variation cores when converting a TuckerTensorTrain to frame-variation format.

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> x_orth = x.right_orthogonalize_tt_cores()
        >>> print(bool((x - x_orth).norm() < 1e-11)) # represents the same tensor
        True
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(bool(np.linalg.norm(np.einsum('iaj,kaj->ik',G,G)-np.eye(G.shape[0])) < 1e-12)) # TT core is right orthogonal
        True

        Stacked -- ``norm()`` returns a per-stack array; orthogonality holds on every slice:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> x_orth = x.right_orthogonalize_tt_cores()
        >>> print(bool(np.max((x - x_orth).norm()) < 1e-11)) # same tensor on every stack slice
        True
        >>> ind = 1
        >>> G = x_orth.data[1][ind]
        >>> print(bool(np.linalg.norm(np.einsum('xyiaj,xykaj->xyik',G,G)-np.eye(G.shape[2])) < 1e-12))
        True
        """
        result = orth.tt_right_orthogonalize(
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
            Array of selected entries with ``shape=idx_stack_shape+t3_stack_shape`` (base-inner: the
            index stack is outer, the T3 stack inner). A scalar for an unstacked T3 and a single index.

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> index = [9, 4, 7]
        >>> result = x.entries(index)
        >>> result2 = x.to_dense()[9, 4, 7]
        >>> print(np.allclose(result, result2))
        True

        With stacked index and stacked T3s -- output is base-inner ``idx_stack_shape + t3_stack_shape``:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> choice = np.random.choice
        >>> t3_stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,2), t3_stack_shape)
        >>> idx_stack_shape = (4,5,1)
        >>> index = [choice(14, size=idx_stack_shape), choice(15, size=idx_stack_shape), choice(16, size=idx_stack_shape)]
        >>> entries = x.entries(index)
        >>> print(entries.shape)               # base-inner: idx stack (4,5,1) outer, T3 stack (2,3) inner
        (4, 5, 1, 2, 3)
        >>> ii, jj = 1, 2          # T3 stack index (inner)
        >>> ll, mm, nn =  3, 2, 0  # index stack (outer)
        >>> entry_ij_lmn = entries[ll,mm,nn, ii,jj]
        >>> x_ij_dense = x.to_dense()[ii,jj]
        >>> index_lmk = (index[0][ll,mm,nn], index[1][ll,mm,nn], index[2][ll,mm,nn])
        >>> entry_ij_lmn_true = x_ij_dense[index_lmk]
        >>> print(np.allclose(entry_ij_lmn, entry_ij_lmn_true))
        True

        Differentiable / jit-able under jax -- jit gives the same value as the eager call:

        >>> import numpy as np
        >>> import jax
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> get_entry_123 = lambda x: x.entries((1,2,3))
        >>> A = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1)).to_jax()
        >>> a123 = get_entry_123(A)
        >>> a123_jit = jax.jit(get_entry_123)(A)        # jit compile, then call
        >>> print(np.allclose(a123, a123_jit))
        True

        ``jax.grad`` differentiates through the cores; the directional derivative matches a finite difference:

        >>> import numpy as np
        >>> import jax
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.corewise as cw
        >>> jax.config.update("jax_enable_x64", True)   # double precision for the finite difference
        >>> np.random.seed(0)
        >>> get_entry_123 = lambda x: x.entries((1,2,3))
        >>> A0 = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1), use_jax=True)
        >>> f0 = get_entry_123(A0)
        >>> G0 = jax.grad(get_entry_123)(A0)            # gradient w.r.t. the cores
        >>> dA = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1), use_jax=True)
        >>> df = cw.corewise_dot(dA.data, G0.data)      # sensitivity in direction dA
        >>> s = 1e-7
        >>> A1 = cw.corewise_add(A0.data, cw.corewise_scale(dA.data, s)) # A1 = A0 + s*dA
        >>> df_diff = (get_entry_123(t3.TuckerTensorTrain(*A1)) - f0) / s # finite difference
        >>> print(bool(np.allclose(df, df_diff, rtol=1e-5)))
        True
        '''
        if len(index) != self.d:
            raise ValueError(
                'Wrong number of indices for Tucker tensor train.\n'
                + str(self.d) + ' = num tensor indices != num provided indices = ' + str(index.shape[0])
            )

        return entries.t3_entries(self.data, index)

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
            Result of contracting ``self`` with the vectors in all indices. Scalar if ``vecs``
            elements are vectors; ``NDArray`` with ``shape=vec_stack_shape+t3_stack_shape`` (base-inner:
            vec stack outer, T3 stack inner) if ``vecs`` elements are matrices and/or the T3 is stacked.

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,1))
        >>> vecs = [np.random.randn(14), np.random.randn(15), np.random.randn(16)]
        >>> result = x.apply(vecs) # Contract x with vecs in all indices
        >>> result2 = np.einsum('ijk,i,j,k', x.to_dense(), vecs[0], vecs[1], vecs[2])
        >>> print(np.allclose(result, result2))
        True

        Apply to stacked vectors and stacked T3s (vectorized) -- output is base-inner
        ``vec_stack_shape + t3_stack_shape``:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,1), stack_shape)
        >>> vec_stack_shape = (4,5,1)
        >>> vecs = [randn(*(vec_stack_shape+(14,))), randn(*(vec_stack_shape+(15,))), randn(*(vec_stack_shape+(16,)))]
        >>> result = x.apply(vecs)
        >>> print(result.shape)                # base-inner: vec stack (4,5,1) outer, T3 stack (2,3) inner
        (4, 5, 1, 2, 3)
        >>> ii, jj = 1, 2 # T3 stack index (inner)
        >>> ll, mm, nn =  3, 2, 0 # vectors stack index (outer)
        >>> result_ij_lmn = result[ll,mm,nn, ii,jj]
        >>> x_ij_dense = x.to_dense()[ii,jj]
        >>> vecs_lmn = [vecs[0][ll,mm,nn], vecs[1][ll,mm,nn], vecs[2][ll,mm,nn]]
        >>> result_ij_lmn_true = np.einsum('abc,a,b,c', x_ij_dense, *vecs_lmn)
        >>> print(np.allclose(result_ij_lmn, result_ij_lmn_true))
        True

        ``apply`` is differentiable under jax -- the directional derivative of the symmetric
        contraction ``u -> A(u,u,u)`` matches a finite difference:

        >>> import numpy as np
        >>> import jax
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> jax.config.update("jax_enable_x64", True)
        >>> np.random.seed(0)
        >>> A = t3.TuckerTensorTrain.randn((10,10,10),(5,5,5),(1,4,4,1)).to_jax()
        >>> apply_A_sym = lambda u: A.apply((u,u,u)) # symmetric apply (jax dispatch inferred from A)
        >>> u0 = np.random.randn(10)
        >>> Auuu0 = apply_A_sym(u0)
        >>> g0 = jax.grad(apply_A_sym)(u0) # gradient by automatic differentiation
        >>> du = np.random.randn(10)
        >>> dAuuu = np.dot(g0, du) # derivative in direction du
        >>> s = 1e-7
        >>> dAuuu_diff = (apply_A_sym(u0 + s*du) - Auuu0) / s # finite difference
        >>> print(bool(np.allclose(dAuuu, dAuuu_diff, rtol=1e-5)))
        True
        '''
        if len(vecs) != len(self.shape):
            raise ValueError(
                'Attempted to apply TuckerTensorTrain to wrong number of vectors.'
                + str(str(len(self.shape)) + ' = num_indices != len(vecs) = ' + str(len(vecs)))
            )
        return apply.t3_apply(self.data, vecs)

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
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10,11,12),(5,6,4),(2,3,4,2))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zz = x.probe(ww)                    # contract all-but-one index, for each index
        >>> x_dense = x.to_dense()
        >>> zz0_true = np.einsum('abc,b,c', x_dense, ww[1], ww[2])
        >>> zz1_true = np.einsum('abc,a,c', x_dense, ww[0], ww[2])
        >>> zz2_true = np.einsum('abc,a,b', x_dense, ww[0], ww[1])
        >>> print(np.allclose(zz[0], zz0_true), np.allclose(zz[1], zz1_true), np.allclose(zz[2], zz2_true))
        True True True

        Probe with stacked vectors and stacked T3s -- each probe is base-inner ``W + C + (Ni,)``:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> randn = np.random.randn
        >>> stack_shape = (2,3)
        >>> x = t3.TuckerTensorTrain.randn((14,15,16), (4,5,6), (2,3,2,1), stack_shape)
        >>> vstack_shape = (4,5,1)
        >>> ww = [randn(*(vstack_shape+(14,))), randn(*(vstack_shape+(15,))), randn(*(vstack_shape+(16,)))]
        >>> result = x.probe(ww)
        >>> print(result[0].shape)             # probe stack (4,5,1) outer, T3 stack (2,3) inner, then N0=14
        (4, 5, 1, 2, 3, 14)
        >>> ii, jj = 1, 2          # T3 (frame) stack index
        >>> ll, mm, nn =  3, 2, 0  # vector (probe) stack index
        >>> result_ij_lmn_0 = result[0][ll,mm,nn, ii,jj]
        >>> result_ij_lmn_1 = result[1][ll,mm,nn, ii,jj]
        >>> result_ij_lmn_2 = result[2][ll,mm,nn, ii,jj]
        >>> x_ij_dense = x.to_dense()[ii,jj]
        >>> result_ij_lmn_0_true = np.einsum('abc,b,c', x_ij_dense, ww[1][ll,mm,nn], ww[2][ll,mm,nn])
        >>> result_ij_lmn_1_true = np.einsum('abc,a,c', x_ij_dense, ww[0][ll,mm,nn], ww[2][ll,mm,nn])
        >>> result_ij_lmn_2_true = np.einsum('abc,a,b', x_ij_dense, ww[0][ll,mm,nn], ww[1][ll,mm,nn])
        >>> print(np.allclose(result_ij_lmn_0, result_ij_lmn_0_true))
        True
        >>> print(np.allclose(result_ij_lmn_1, result_ij_lmn_1_true))
        True
        >>> print(np.allclose(result_ij_lmn_2, result_ij_lmn_2_true))
        True
        """
        return probing.t3_probe(ww, self.data)

    @staticmethod
    def probe_ambient_transpose(
            ztildes: Sequence[NDArray],  # probe residuals, len=d, elm_shape = W + C + (Ni,)
            ww:      Sequence[NDArray],  # probe vectors,   len=d, elm_shape = W + (Ni,)
            sum_over_probes: bool = False,
    ) -> Sequence[NDArray]:             # canonical (CP) factors, len=d, elm_shape = stack + (R, Ni)
        """Ambient transpose of :py:meth:`probe`: back-project probe residuals into canonical (CP) factors.

        The probe counterpart of :py:meth:`apply_ambient_transpose` -- the literal adjoint of ``probe``
        as a linear map on the full tensor space (one of **three** probe transposes; full taxonomy in
        ``docs/transposes.md``). ``probe`` returns ``d`` vectors, so the residual ``ztildes`` is ``d``
        vectors; the back-projection is the **rank-``d``** tensor
        ``sum_i (w0 (x) ... (x) ztildes_i (x) ... (x) w_{d-1})`` (residual ``ztildes_i`` in slot ``i``,
        probe vectors elsewhere), returned as **CP factors**. Frame-free. Realize a ``TuckerTensorTrain``
        with :py:meth:`from_canonical`.

        The other two probe transposes are ``probe_corewise_transpose`` (gradient w.r.t. a frame's cores,
        for Adam / L-BFGS) and ``T3Tangent.probe_transpose`` (the Riemannian gradient).

        ``sum_over_probes=False`` keeps the probe stack ``W`` (a ``W (+ C)`` stack of rank-``d`` CPs);
        ``True`` folds ``W`` into the CP rank (one rank-``d|W|`` CP, the ambient ``J^T r``), cheap as CP.

        See Also
        --------
        probe
        apply_ambient_transpose
        probe_corewise_transpose
        from_canonical

        Examples
        --------
        Adjoint identity ``<probe_ambient_transpose(z, ww), x>_F == sum_i <z_i, x.probe(ww)_i>``:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> zt = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> T = t3.TuckerTensorTrain.from_canonical(t3.TuckerTensorTrain.probe_ambient_transpose(zt, ww))
        >>> lhs = float(np.sum(T.to_dense() * x.to_dense()))
        >>> rhs = float(sum(np.sum(z * p) for z, p in zip(zt, x.probe(ww))))
        >>> print(bool(abs(lhs - rhs) < 1e-9))
        True
        """
        return probing.t3_probe_ambient_transpose(ztildes, ww, sum_over_probes=sum_over_probes)

    def probe_corewise_transpose(
            self:    'TuckerTensorTrain',
            ztildes: Sequence[NDArray],  # probe residuals, len=d, elm_shape = W + C + (Ni,)
            ww:      Sequence[NDArray],  # probe vectors,   len=d, elm_shape = W + (Ni,)
            sum_over_probes: bool = False,
    ) -> Tuple[
        Tuple[NDArray, ...],  # tucker-core gradients, same shapes as self.tucker_cores
        Tuple[NDArray, ...],  # tt-core gradients,     same shapes as self.tt_cores
    ]:
        """Corewise (non-manifold) transpose of :py:meth:`probe`: gradient w.r.t. *this tensor's cores*.

        The probe counterpart of :py:meth:`apply_corewise_transpose` -- the gradient of
        ``self.probe(ww)`` w.r.t. ``self``'s cores (for core-wise optimizers, Adam / L-BFGS, fitting
        from probes), returned as a raw ``(tucker_grads, tt_grads)`` tuple shaped like ``self.data`` (a
        gradient, not a tensor; no ``|W|`` blow-up). Computed by the Section 6.3 corewise simplification
        -- the tangent probe transpose with this tensor's own cores substituted for the orthogonal
        frames. See ``docs/transposes.md`` for the ambient/corewise/tangent distinction; the ambient and
        Riemannian probe transposes are :py:meth:`probe_ambient_transpose` and ``T3Tangent.probe_transpose``.
        ``sum_over_probes=True`` is the summed gradient ``J^T r``; ``False`` keeps the probe stack ``W``.

        See Also
        --------
        probe
        probe_ambient_transpose
        apply_corewise_transpose
        """
        return probing.t3_probe_corewise_transpose(ztildes, ww, self.data, sum_over_probes=sum_over_probes)

    @staticmethod
    def apply_ambient_transpose(
            c:      NDArray,            # residual, shape = W + C
            ww:     Sequence[NDArray],  # apply vectors, len=d, elm_shape = W + (Ni,)
            sum_over_probes: bool = False,
    ) -> Sequence[NDArray]:            # canonical (CP) factors, len=d, elm_shape = stack + (R, Ni)
        """Ambient transpose of :py:meth:`apply`: back-project ``c`` into canonical (CP) factors.

        This is **one of three** transposes of ``apply`` -- make sure it is the one you want (full
        taxonomy and costs in ``docs/transposes.md``):

        - **ambient** (this method): the literal adjoint of ``apply`` viewed as a linear map on the
          *full tensor space*. Frame-free; the back-projection ``c * (w0 (x) ... (x) w_{d-1})`` is
          rank-1, returned as **CP factors** -- the natural type, since ``apply`` consumes one vector
          per mode and its adjoint emits one scaled vector per mode. Convert to a ``TuckerTensorTrain``
          with :py:meth:`from_canonical` if you want T3 form.
        - **corewise** (``apply_corewise_transpose``): the gradient w.r.t. a base point's cores, for
          core-wise optimizers (Adam, L-BFGS). Most users who reach for "the transpose to get a
          gradient" actually want *this* -- hence the explicit names, so neither is the silent default.
        - **tangent** (``T3Tangent.apply_transpose``): the Riemannian gradient, returned as a tangent
          vector, for manifold optimization.

        ``sum_over_probes`` chooses where the probe stack ``W`` lands (both modes are cheap, ``O(d|W|N)``):

        - ``False`` (default, primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack of
          rank-1 CP tensors (CP rank ``R=1``), one back-projection per probe.
        - ``True``: ``W`` becomes the CP **rank** -- one rank-``|W|`` CP tensor
          ``sum_W c_W (w0^W (x) ...)`` (the ambient ``J^T r``). Cheap as CP; the ``|W|^2`` cost of a
          *dense* T3 is paid only if you then call :py:meth:`from_canonical`.

        Returns the CP ``factors`` (``c`` folded into the first), in the layout
        :py:meth:`from_canonical` consumes. See *Batching & stacking* §11
        (``docs/batching_and_stacking.md``) for the stacking conventions.

        See Also
        --------
        apply
        apply_corewise_transpose
        entries_ambient_transpose
        from_canonical

        Examples
        --------
        Adjoint identity ``<apply_ambient_transpose(c, ww), x>_F == c * x.apply(ww)`` -- realize the CP
        factors as a T3 with :py:meth:`from_canonical`:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> factors = t3.TuckerTensorTrain.apply_ambient_transpose(1.7, ww)
        >>> ATc = t3.TuckerTensorTrain.from_canonical(factors)     # CP factors -> a TuckerTensorTrain
        >>> lhs = float(np.sum(ATc.to_dense() * x.to_dense()))
        >>> print(bool(abs(lhs - 1.7 * float(x.apply(ww))) < 1e-9))
        True
        """
        return apply.t3_apply_ambient_transpose(
            c, ww, sum_over_probes=sum_over_probes,
        )

    @staticmethod
    def entries_ambient_transpose(
            c:      NDArray,            # residual, shape = W + C
            index:  NDArray,            # int, shape = (d,) + W
            shape:  Sequence[int],      # ambient dims (N0, ..., N(d-1))
            sum_over_probes: bool = False,
    ) -> Sequence[NDArray]:            # canonical (CP) factors, len=d, elm_shape = stack + (R, Ni)
        """Ambient transpose of :py:meth:`entries`: scatter ``c`` at ``index`` into canonical (CP) factors.

        The ``entries`` counterpart of :py:meth:`apply_ambient_transpose` -- identical, with the apply
        vectors replaced by the unit vectors ``e_{index_k}``, so the CP factors are one-hots and the
        back-projection is ``c * e_{idx_0} (x) ... (x) e_{idx_{d-1}}``. See that method (and
        ``docs/transposes.md``) for the **ambient vs corewise vs tangent** distinction -- this is the
        *ambient* one (the frame-free adjoint on the full tensor space); the gradient-for-optimizers
        versions are ``entries_corewise_transpose`` and ``T3Tangent.entries_transpose``.

        ``sum_over_probes=True`` makes ``W`` the CP rank (scatter-adding colliding indices -- the
        ``J^T r`` for entry sampling); ``False`` keeps ``W`` as a stacking axis. ``shape`` supplies the
        ambient dims ``(N0, ..., N(d-1))``, which (unlike :py:meth:`apply_ambient_transpose`, where
        ``ww`` carries them) ``c`` and ``index`` alone do not determine. Returns CP ``factors`` for
        :py:meth:`from_canonical`.

        See Also
        --------
        entries
        entries_corewise_transpose
        apply_ambient_transpose
        from_canonical

        Examples
        --------
        Back-projecting a residual scatters it at the index (realize the CP factors with
        :py:meth:`from_canonical`):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> factors = t3.TuckerTensorTrain.entries_ambient_transpose(2.0, (3, 5, 7), x.shape)
        >>> ETc = t3.TuckerTensorTrain.from_canonical(factors)
        >>> print(bool(abs(float(ETc.to_dense()[3, 5, 7]) - 2.0) < 1e-9))     # c lands at the index
        True
        >>> rest = ETc.to_dense().copy(); rest[3, 5, 7] = 0.0
        >>> print(bool(np.linalg.norm(rest) < 1e-9))                          # zero elsewhere
        True
        """
        return entries.t3_entries_ambient_transpose(
            c, index, shape, sum_over_probes=sum_over_probes,
        )

    def apply_corewise_transpose(
            self:   'TuckerTensorTrain',
            c:      NDArray,            # residual, shape = W + C
            ww:     Sequence[NDArray],  # apply vectors, len=d, elm_shape = W + (Ni,)
            sum_over_probes: bool = False,
    ) -> Tuple[
        Tuple[NDArray, ...],  # tucker-core gradients, same shapes as self.tucker_cores
        Tuple[NDArray, ...],  # tt-core gradients,     same shapes as self.tt_cores
    ]:
        """Corewise (non-manifold) transpose of :py:meth:`apply`: gradient w.r.t. *this tensor's cores*.

        One of **three** transposes of ``apply`` -- pick the one you want (full taxonomy and costs in
        ``docs/transposes.md``):

        - **corewise** (this method): the gradient of ``self.apply(ww)`` with respect to ``self``'s
          cores, treated as independent optimization variables -- what a **core-wise optimizer** (Adam,
          L-BFGS, SGD) needs. Returns a **raw tuple** ``(tucker_grads, tt_grads)`` whose arrays have the
          exact shapes of ``self.data``. It is a *gradient, not a tensor* -- do not do T3 arithmetic
          with it. No ``|W|`` blow-up: the probe stack collapses into the fixed-size cores.
        - **ambient** (:py:meth:`apply_ambient_transpose`): the frame-free adjoint, returned as the CP
          factors of ``c * (w0 (x) ... (x) w_{d-1})``.
        - **tangent** (``T3Tangent.apply_transpose``): the Riemannian gradient, a tangent vector.

        Computed by the Section 6.3 corewise simplification -- the tangent transpose with this tensor's
        own cores substituted for the orthogonal frames (orthogonality is not required).
        ``sum_over_probes=True`` is the summed gradient ``J^T r``; ``False`` keeps the probe stack ``W``
        (one gradient set per probe, e.g. to assemble ``J^T J``).

        Note: a corewise gradient is meaningful only at the base point where it is taken (the cores are
        a representation, not intrinsic), so combining them across base points (as L-BFGS history does)
        is a heuristic; for the principled version use the tangent transpose.

        See Also
        --------
        apply
        apply_ambient_transpose
        entries_corewise_transpose

        Examples
        --------
        The gradient has the same structure as the cores (not a tensor):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> x = t3.TuckerTensorTrain.randn((10, 11, 12), (5, 6, 4), (1, 2, 3, 1))
        >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
        >>> gU, gG = x.apply_corewise_transpose(np.asarray(2.0), ww, sum_over_probes=True)
        >>> print([g.shape for g in gU] == [u.shape for u in x.tucker_cores])   # matches Tucker cores
        True
        >>> print([g.shape for g in gG] == [g.shape for g in x.tt_cores])       # matches TT cores
        True
        """
        return apply.t3_apply_corewise_transpose(c, ww, self.data, sum_over_probes=sum_over_probes)

    def entries_corewise_transpose(
            self:   'TuckerTensorTrain',
            c:      NDArray,            # residual, shape = W + C
            index:  NDArray,            # int, shape = (d,) + W
            sum_over_probes: bool = False,
    ) -> Tuple[
        Tuple[NDArray, ...],  # tucker-core gradients, same shapes as self.tucker_cores
        Tuple[NDArray, ...],  # tt-core gradients,     same shapes as self.tt_cores
    ]:
        """Corewise (non-manifold) transpose of :py:meth:`entries`: gradient w.r.t. *this tensor's cores*.

        The ``entries`` counterpart of :py:meth:`apply_corewise_transpose` -- see it (and
        ``docs/transposes.md``) for the ambient-vs-corewise-vs-tangent distinction; this is the
        *corewise* gradient (w.r.t. the cores, for Adam / L-BFGS), returned as a raw
        ``(tucker_grads, tt_grads)`` tuple. Unlike :py:meth:`entries_ambient_transpose` it needs **no**
        ``shape`` argument -- the ambient dims come from ``self``. ``sum_over_probes=True`` scatter-adds
        colliding indices (the gradient ``J^T r``).

        See Also
        --------
        entries
        entries_ambient_transpose
        apply_corewise_transpose
        """
        return entries.t3_entries_corewise_transpose(c, index, self.data, sum_over_probes=sum_over_probes)

    ##############################################################
    ###############    Symmetric derivatives    ##################
    ##############################################################

    def probe_derivatives(
            self,
            ww:     Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
    ) -> Sequence[NDArray]:             # len=d, elm_shape=(order+1,)+W+C+(Ni,)
        """Symmetric directional derivatives of probing this Tucker tensor train, in one repeated direction.

        Returns, for each mode ``i``, the stack ``y_i^(t) = d^t/ds^t [probe(X + s P)]_i|_0`` for
        ``t=0..order`` -- the derivative analogue of :py:meth:`probe`, perturbing every probe vector in
        the same direction ``P``. Index ``0`` is the ordinary :py:meth:`probe`. Stacks ``order + W + C``
        (order outermost; probe stack ``W``, T3 stack ``C``). ``X`` (``ww``) and ``P`` (``pp``) must
        share the sample stack ``W``. (For the gradient w.r.t. the cores, see
        :py:meth:`probe_corewise_derivatives_transpose`; for the Riemannian Jacobian use
        ``T3Tangent.probe_derivatives``.) See ``docs/symmetric_probe_derivatives.tex``.

        See Also
        --------
        probe
        apply_derivatives
        probe_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
        >>> ww = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> pp = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> zj = x.probe_derivatives(ww, pp, 3)
        >>> print([z.shape for z in zj])           # (order+1,) + (Ni,)
        [(4, 14), (4, 15), (4, 16)]
        >>> print([bool(np.allclose(z[0], z0)) for z, z0 in zip(zj, x.probe(ww))])  # order 0 == probe
        [True, True, True]
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return sampling_derivatives.t3_probe_derivatives(ww, pp, self.data, order)

    def apply_derivatives(
            self,
            ww:     Sequence[NDArray],  # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
    ) -> NDArray:                       # shape=(order+1,)+W+C
        """Symmetric directional derivatives of applying this T3 in all modes, in one repeated direction.

        The all-modes analogue of :py:meth:`probe_derivatives` (derivative analogue of :py:meth:`apply`):
        ``y^(t) = d^t/ds^t apply(X + s P)|_0`` for ``t=0..order`` (a scalar per stack element). Stacks
        ``order + W + C``. ``X`` and ``P`` share the sample stack ``W``.

        See Also
        --------
        apply
        probe_derivatives
        apply_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
        >>> ww = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> pp = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> yj = x.apply_derivatives(ww, pp, 3)
        >>> print(yj.shape)                        # (order+1,)
        (4,)
        >>> print(bool(np.allclose(yj[0], x.apply(ww))))     # order 0 == apply
        True
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return sampling_derivatives.t3_apply_derivatives(ww, pp, self.data, order)

    def entries_derivatives(
            self,
            index:  NDArray,            # int, shape=(d,)+W -- grid points
            pp:     Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                # highest derivative order
    ) -> NDArray:                       # shape=(order+1,)+W+C
        """Symmetric directional derivatives of this T3's entries at ``index``, in direction ``P``.

        The Taylor data of the represented tensor's multilinear extension at grid corner ``index``, in
        direction ``P``: ``y^(t) = d^t/ds^t apply(e_{index} + s P)|_0``. Index ``0`` is the ordinary
        :py:meth:`entries`. Stacks ``order + W + C``. ``index`` and ``P`` share ``W``.

        See Also
        --------
        entries
        apply_derivatives
        entries_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
        >>> index = np.array([3, 5, 7])
        >>> pp = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> yj = x.entries_derivatives(index, pp, 3)
        >>> print(yj.shape)
        (4,)
        >>> print(bool(np.allclose(yj[0], x.entries(index))))   # order 0 == entries
        True
        """
        sampling_derivatives.check_perturbation_index(index, pp, self.shape)
        return sampling_derivatives.t3_entries_derivatives(index, pp, self.data, order)

    def probe_corewise_derivatives_transpose(
            self:    'TuckerTensorTrain',
            ztildes: Sequence[NDArray],  # residual jets, len=d, elm_shape=(order+1,)+W+C+(Ni,)
            ww:      Sequence[NDArray],  # probe vectors X,        len=d, elm_shape=W+(Ni,)
            pp:      Sequence[NDArray],  # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:   int,                # highest derivative order
            sum_over_probes: bool = False,
    ) -> Tuple[
        Tuple[NDArray, ...],  # tucker-core gradients, same shapes as self.tucker_cores
        Tuple[NDArray, ...],  # tt-core gradients,     same shapes as self.tt_cores
    ]:
        """Corewise (non-manifold) transpose of :py:meth:`probe_derivatives`: the core-space gradient.

        Gradient w.r.t. *this tensor's cores*, treated as independent optimization variables
        (Adam / L-BFGS). Returns a raw ``(tucker_grads, tt_grads)`` tuple shaped like ``self.data``
        -- a gradient, not a tensor. The Section 6.3 substitution ``P,Q,O -> G`` into
        ``T3Tangent.probe_derivatives_transpose`` (no orthogonality required).
        ``sum_over_probes=True`` is the summed gradient ``J^T r``.

        See Also
        --------
        probe_derivatives
        probe_corewise_transpose
        apply_corewise_derivatives_transpose
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return sampling_derivatives.t3_probe_corewise_derivatives_transpose(
            ztildes, ww, pp, self.data, order, sum_over_probes=sum_over_probes)

    def apply_corewise_derivatives_transpose(
            self:   'TuckerTensorTrain',
            c:      NDArray,             # residual jet (scalar), shape=(order+1,)+W+C
            ww:     Sequence[NDArray],   # apply vectors X,        len=d, elm_shape=W+(Ni,)
            pp:     Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                 # highest derivative order
            sum_over_probes: bool = False,
    ) -> Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]]:  # (tucker_grads, tt_grads)
        """Corewise transpose of :py:meth:`apply_derivatives`: gradient of the apply-derivative jets
        w.r.t. *this tensor's cores* (the Section 6.3 substitution). Returns raw core gradients.

        See Also
        --------
        apply_derivatives
        probe_corewise_derivatives_transpose

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
        >>> ww = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> pp = (np.random.randn(14), np.random.randn(15), np.random.randn(16))
        >>> c = np.random.randn(4)                 # residual jet, order 0..3
        >>> gU, gG = x.apply_corewise_derivatives_transpose(c, ww, pp, 3, sum_over_probes=True)
        >>> print([g.shape for g in gU] == [u.shape for u in x.tucker_cores])   # matches the cores
        True
        >>> print([g.shape for g in gG] == [g.shape for g in x.tt_cores])
        True
        """
        sampling_derivatives.check_perturbation_vectors(ww, pp)
        return sampling_derivatives.t3_apply_corewise_derivatives_transpose(
            c, ww, pp, self.data, order, sum_over_probes=sum_over_probes)

    def entries_corewise_derivatives_transpose(
            self:   'TuckerTensorTrain',
            c:      NDArray,             # residual jet (scalar), shape=(order+1,)+W+C
            index:  NDArray,             # int, shape=(d,)+W
            pp:     Sequence[NDArray],   # perturbation vectors P, len=d, elm_shape=W+(Ni,)
            order:  int,                 # highest derivative order
            sum_over_probes: bool = False,
    ) -> Tuple[Tuple[NDArray, ...], Tuple[NDArray, ...]]:  # (tucker_grads, tt_grads)
        """Corewise transpose of :py:meth:`entries_derivatives`: gradient of the entry-derivative jets
        w.r.t. *this tensor's cores* (the Section 6.3 substitution). Returns raw core gradients.

        See Also
        --------
        entries_derivatives
        apply_corewise_derivatives_transpose
        """
        sampling_derivatives.check_perturbation_index(index, pp, self.shape)
        return sampling_derivatives.t3_entries_corewise_derivatives_transpose(
            c, index, pp, self.data, order, sum_over_probes=sum_over_probes)

    ##############################################################
    ########################    T3-SVD    ########################
    ##############################################################

    def t3svd(
            self: 'TuckerTensorTrain',
            max_tt_ranks: typ.Union[int, Sequence[int]] = None,     # scalar (caps all) or len=d+1
            max_tucker_ranks: typ.Union[int, Sequence[int]] = None, # scalar (caps all) or len=d
            rtol: float = None,
            atol: float = None,
            assume_orthogonal: bool = False,
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
        max_tt_ranks: int or Sequence[int], optional
            Maximum TT-ranks ``ri``. A scalar caps every bond; a sequence is per-bond,
            e.g., ``(1,5,5,5,1)`` with ``len(max_tt_ranks)=d+1``.
            Default: no max TT rank truncation (``None``).
        max_tucker_ranks: int or Sequence[int], optional
            Maximum Tucker ranks ``ni``. A scalar caps every mode; a sequence is per-mode,
            e.g., ``(5,5,5)`` with ``len(max_tucker_ranks)=d``.
            Default: no max Tucker rank truncation (``None``).
        rtol: float, optional
            Relative tolerance for truncation (in the Frobenius norm), applied **per truncation step**
            (see Notes -- the overall error can be larger). Default: no ``rtol`` truncation (``None``).
            Requires ``stack_shape=()``.
        atol: float, optional
            Absolute tolerance for truncation (in the Frobenius norm), applied **per truncation step**
            (see Notes). Default: no ``atol`` truncation (``None``).
            Requires ``stack_shape=()``.
        assume_orthogonal: bool, optional
            If ``True``, skip the initial orthogonalization, asserting the input is already
            **right-orthogonal** (every Tucker core down-orthogonal **and** every TT core
            right-orthogonal -- the form the left-to-right sweep needs). **Not enforced** -- a wrong
            assertion silently corrupts the result; verify first with :py:meth:`is_right_orthogonal`. A
            *left*-orthogonal input (e.g. a prior :py:meth:`t3svd` result) is not the right form; reverse
            it yourself (a left-orthogonal T3 reversed is right-orthogonal) if you want to skip. Default:
            ``False`` (always orthogonalize -- safe).

        Returns
        -------
        :py:class:`TuckerTensorTrain`
            New Tucker tensor train representing the same tensor (or a truncated version), but with modified cores
        Tuple[:py:class:`NDArray`,...]
            Singular values associated with edges between Tucker cores and TT cores
        Tuple[:py:class:`NDArray`,...]
            Singular values associated with edges between adjacent TT cores

        Notes
        -----
        ``rtol``/``atol`` bound the truncation error at **each step** (each unfolding/matricization SVD
        truncation), **not** the overall error. The per-step errors accumulate in quadrature over the
        ``2d-1`` steps (``d-1`` TT unfoldings + ``d`` Tucker matricizations), so the realized error can
        exceed the requested tolerance by up to a factor ``sqrt(2d-1)`` (the generalized Oseledets bound)::

            ||x - x2||  <=  sqrt( sum of per-step truncation errors^2 )  <=  sqrt(2d-1) * (per-step tol).

        See ``docs/t3svd_verification.md`` for the bound and its proof.

        This is the basic algorithm: it does **not** guarantee minimal ranks. The result is always
        **left-orthogonal**, but under a hard rank cap it can leave a Tucker rank / bond above its
        structural minimum (``has_minimal_ranks`` may be ``False``) -- exactly as the paper's algorithm
        (and Oseledets' TT-SVD) do. To reduce to minimal ranks, follow with
        :py:meth:`rank_adjustment_sweep` (the result is left-orthogonal, so ``'right_to_left'`` minimizes
        it). See ``docs/t3svd_minimal_ranks.md``.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.rank_adjustment_sweep`
        :py:meth:`.TuckerTensorTrain.t3svd_dense`
        :py:meth:`.TuckerTensorTrain.get_minimal_ranks`

        Examples
        --------
        No truncation -- re-represents the same tensor, with ranks reduced to minimal:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((5, 6, 3), (4, 4, 3), (1, 3, 2, 1))
        >>> x2, ss_tucker, ss_tt = x.t3svd()
        >>> print(np.allclose(x.to_dense(), x2.to_dense()))        # same tensor
        True
        >>> print(x2.tucker_ranks, x2.tt_ranks)                    # reduced to minimal ranks
        (3, 4, 2) (1, 3, 2, 1)

        The singular values ARE the singular values of the dense matrix unfoldings (the
        numerically-zero tail is dropped -- which is what reduces the ranks):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((5, 6, 3), (4, 4, 3), (1, 3, 2, 1))
        >>> _, ss_tucker, ss_tt = x.t3svd()
        >>> dense_svals = np.linalg.svd(x.to_dense().reshape(5, 6 * 3), compute_uv=False)
        >>> print(np.allclose(ss_tt[1], dense_svals[:len(ss_tt[1])]))   # leading values match the unfolding's
        True
        >>> print(len(ss_tt[1]), int(np.sum(dense_svals > 1e-9)))       # kept TT rank == numerical rank of the unfolding
        3 3
        >>> # (the Tucker singular values ss_tucker[i] relate to the mode-i matricizations the same way)

        Truncation -- a smooth tensor has gradually decaying unfolding spectra, so ``rtol``
        truncates meaningfully (a sharp random spectrum would not):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> i, j, k = np.ogrid[1:9, 1:9, 1:9]
        >>> x = t3.TuckerTensorTrain.t3svd_dense(1.0 / (i + j + k))[0]   # exact T3 of a smooth tensor
        >>> _, full_tucker_ss, full_tt_ss = x.t3svd()                    # original (untruncated) spectra
        >>> xt, _, _ = x.t3svd(rtol=1e-3)                                # truncate at rtol
        >>> print(x.tt_ranks, '->', xt.tt_ranks)                        # rtol drops the small singular values
        (1, 8, 8, 1) -> (1, 3, 3, 1)
        >>> # accuracy: ||x - xt|| <= sqrt(dropped singular-value energy at the chosen ranks) [Oseledets]
        >>> dropped_sq = (sum(float(np.sum(s[r:]**2)) for s, r in zip(full_tt_ss, xt.tt_ranks))
        ...             + sum(float(np.sum(s[r:]**2)) for s, r in zip(full_tucker_ss, xt.tucker_ranks)))
        >>> print(bool(np.linalg.norm(x.to_dense() - xt.to_dense()) <= np.sqrt(dropped_sq)))
        True
        >>> # parsimony: each chosen rank <= #{ original singular values >= tau },  tau = rtol * ||xt||
        >>> tau = 1e-3 * np.linalg.norm(xt.to_dense())
        >>> tt_ok = all(r <= max(1, int(np.sum(s >= tau))) for s, r in zip(full_tt_ss, xt.tt_ranks))
        >>> tk_ok = all(n <= max(1, int(np.sum(s >= tau))) for s, n in zip(full_tucker_ss, xt.tucker_ranks))
        >>> print(tt_ok, tk_ok)
        True True

        Stacked T3s truncate vectorized over the stack -- max-rank only (``rtol``/``atol`` need a
        single T3, since different slices could truncate to different ranks):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> xs = t3.TuckerTensorTrain.randn((5, 6, 3), (4, 4, 3), (1, 3, 2, 1), stack_shape=(2,))
        >>> xs2, _, _ = xs.t3svd(max_tucker_ranks=2, max_tt_ranks=2)
        >>> print(xs2.stack_shape, xs2.tucker_ranks, xs2.tt_ranks)
        (2,) (2, 2, 2) (1, 2, 2, 1)
        >>> xs.t3svd(rtol=1e-3)   # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError

        Under truncation, ``t3svd`` does **not** guarantee minimal ranks -- a hard cap can leave a
        Tucker rank above its structural bound (here ``n_0 = 3 > rL_0*r_1 = 1*2 = 2``). The result is
        left-orthogonal; reduce it to minimal with :py:meth:`rank_adjustment_sweep`
        (``'right_to_left'``, since a ``t3svd`` result is left-orthogonal):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((5, 6, 7), (4, 5, 6), (1, 3, 2, 1))
        >>> x2, _, _ = x.t3svd(max_tt_ranks=2)               # cap TT bonds, leave Tucker uncapped
        >>> print(x2.is_left_orthogonal(), x2.has_minimal_ranks, x2.tucker_ranks)
        True False (3, 4, 2)
        >>> x3 = x2.rank_adjustment_sweep('right_to_left')   # reduce to minimal ranks (lossless)
        >>> print(x3.has_minimal_ranks, x3.tucker_ranks, np.allclose(x3.to_dense(), x2.to_dense()))
        True (2, 4, 2) True

        ``assume_orthogonal=True`` skips the initial orthogonalization, asserting the input is already
        right-orthogonal (verify with :py:meth:`is_right_orthogonal` first -- it is not checked):

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((6, 7, 8), (5, 6, 7), (1, 4, 3, 1))
        >>> xr = x.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
        >>> print(xr.is_right_orthogonal())
        True
        >>> a, _, _ = xr.t3svd(max_tt_ranks=2, assume_orthogonal=True)   # skip the redundant sweep
        >>> b, _, _ = xr.t3svd(max_tt_ranks=2)
        >>> print(np.allclose(a.to_dense(), b.to_dense()))
        True
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
            rtol=rtol, atol=atol, assume_orthogonal=assume_orthogonal,
        )
        return TuckerTensorTrain(*result[0]), result[1], result[2]

    def continuation_ranks(
            self: 'TuckerTensorTrain',
            tau:         float = 10.0,   # grow an edge only if kappa_i < kappa_max / tau  (tau > 1)
            n_chunk:     int   = 1,      # rank increment added to each grown edge
            kappa_guard: float = 1e12,   # absolute safety cap: never grow an edge with kappa_i >= this
            max_grow:    typ.Optional[int] = None,  # cap on #edges grown per call (None = all eligible)
            rtol:        float = None,
            atol:        float = None,
    ) -> Tuple[
        Tuple[int, ...],  # (n0', ..., n(d-1)')  new Tucker ranks
        Tuple[int, ...],  # (r0', ..., rd')      new TT ranks
    ]:
        '''Rank-continuation update (Section 5.4.1): the ranks to grow to next, from this iterate's spectra.

        Computes the implicit T3-SVD of ``self`` and feeds the unfolding singular values to
        :py:func:`t3toolbox.backend.ranks.compute_continuation_ranks`, which grows the well-conditioned
        edges (each edge's condition number a factor ``tau`` below the worst) so the ranks trend toward
        comparable conditioning across edges. Pair with :py:meth:`resize` for the zero-padded warm start
        of the next fit -- this is the outer loop of Riemannian fitting with rank continuation::

            new_tucker, new_tt = X.continuation_ranks()
            X0 = X.resize(X.shape, new_tucker, new_tt)   # warm start at the grown ranks (same tensor)

        The current ranks are read from the SVD: with the default (no ``rtol``/``atol``) these are the
        structural ranks of ``self`` -- which, for a converged minimal-rank iterate, are its core ranks;
        pass ``rtol``/``atol`` to continue from the numerical rank at that tolerance instead.
        ``kappa_guard`` is an absolute conditioning safety net (see the backend function): when **no**
        edge is below it, the returned ranks **equal** ``self``'s current ranks -- the caller's signal to
        stop continuation. (The current ranks are also returned when the structure is already maximal;
        both mean "stop".) Defined for a single (unstacked) T3 only.

        ``max_grow`` caps how many edges grow per call: ``None`` (default) grows every eligible edge at
        once; ``max_grow=1`` grows **one edge at a time** (the single best-conditioned edge that has
        structural room) -- pair with ``tau=1.0`` for the most conservative, finest-grained continuation.

        See Also
        --------
        :py:func:`t3toolbox.backend.ranks.compute_continuation_ranks`
        :py:func:`t3toolbox.backend.ranks.edge_condition_numbers`
        :py:meth:`.TuckerTensorTrain.resize`
        :py:meth:`.TuckerTensorTrain.t3svd`

        Examples
        --------
        Mid-continuation: at a low-rank iterate, ask which ranks to grow to next, then warm-start there
        by zero-padding (``resize``) -- the represented tensor is unchanged, ready for the next fit:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> i, j, k = np.ogrid[1:7, 1:7, 1:7]
        >>> A = 1.0 / (i + 2 * j + 4 * k)                                   # smooth, anisotropic per mode
        >>> X = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=2, max_tt_ranks=2)[0]
        >>> print(X.tucker_ranks, X.tt_ranks)                              # a rank-2 iterate
        (2, 2, 2) (1, 2, 2, 1)
        >>> new_tucker, new_tt = X.continuation_ranks()
        >>> print(new_tucker, new_tt)                                      # the grown ranks for the next fit
        (3, 3, 3) (1, 3, 3, 1)
        >>> X0 = X.resize(X.shape, new_tucker, new_tt)                     # zero-padded warm start
        >>> print(X0.tucker_ranks, bool(np.allclose(X0.to_dense(), X.to_dense())))
        (3, 3, 3) True

        The edge condition numbers that drive the choice (length-1 boundary TT bonds are 1.0). Here all
        edges are comparably conditioned (~15--24), so none is a factor ``tau=10`` below the worst and
        the fallback grows them uniformly:

        >>> import t3toolbox.backend.ranks as ranks
        >>> _, ss_tucker, ss_tt = X.t3svd()
        >>> kappa_tucker, kappa_tt = ranks.edge_condition_numbers(ss_tucker, ss_tt)
        >>> print([round(k, 2) for k in kappa_tucker])
        [23.94, 15.91, 14.91]

        A stricter ``tau`` grows only edges well below the worst -- here it holds back the stiffest
        Tucker mode and TT bond (condition number ~24) while growing the better-conditioned ones:

        >>> print(X.continuation_ranks(tau=1.5))
        ((2, 3, 3), (1, 2, 3, 1))

        ``max_grow=1`` grows only the single best-conditioned edge that has room (one edge at a time),
        instead of every eligible edge at once:

        >>> print(X.continuation_ranks(tau=1.5, max_grow=1))
        ((2, 3, 2), (1, 2, 2, 1))
        '''
        if len(self.stack_shape) > 0:
            raise ValueError(
                'continuation_ranks is defined for a single (unstacked) Tucker tensor train.\n'
                'Different stack elements could continue to different ranks. Unstack first, then call '
                'continuation_ranks for each unstacked Tucker tensor train.'
            )
        _, ss_tucker, ss_tt = self.t3svd(rtol=rtol, atol=atol)
        return ranks.compute_continuation_ranks(
            self.shape, ss_tucker, ss_tt,
            tau=tau, n_chunk=n_chunk, kappa_guard=kappa_guard, max_grow=max_grow)

    def rank_adjustment_sweep(self, direction: str = 'right_to_left') -> 'TuckerTensorTrain':
        """A single lossless directional sweep that drops structurally-redundant ranks (the separate
        rank-minimization step; :py:meth:`t3svd` itself does **not** minimize). Returns the adjusted T3.

        ``'right_to_left'`` returns a **right-orthogonal** T3; ``'left_to_right'`` a **left-orthogonal**
        one. A single sweep reaches **minimal ranks only if the input is already orthogonal in the
        opposite direction** -- e.g. a :py:meth:`t3svd` result is left-orthogonal, so
        ``result.rank_adjustment_sweep('right_to_left')`` minimizes it (check with
        :py:attr:`has_minimal_ranks`). That precondition is **not enforced**: sweeping the wrong
        direction for the input's gauge just under-minimizes (it stays lossless here -- but the uniform
        :py:meth:`~t3toolbox.uniform_tucker_tensor_train.UniformTuckerTensorTrain.rank_adjustment_sweep`
        is *lossy* in that case). Verify the gauge with :py:meth:`is_left_orthogonal` /
        :py:meth:`is_right_orthogonal` first, or compose both directions for guaranteed minimal ranks.
        The represented tensor is unchanged (when used correctly).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((5, 6, 7), (4, 5, 6), (1, 3, 2, 1))
        >>> x2, _, _ = x.t3svd(max_tt_ranks=2)        # basic T3-SVD: left-orthogonal, NOT minimal
        >>> print(x2.has_minimal_ranks, x2.tucker_ranks)
        False (3, 4, 2)
        >>> x3 = x2.rank_adjustment_sweep('right_to_left')   # x2 is left-orthogonal -> R->L minimizes
        >>> print(x3.has_minimal_ranks, x3.tucker_ranks)
        True (2, 4, 2)
        >>> print(np.allclose(x3.to_dense(), x2.to_dense()))  # same tensor, redundant rank removed
        True

        Wrong direction for the input's gauge -- the left-orthogonal ``x2`` needs ``'right_to_left'``;
        ``'left_to_right'`` here just *under-minimizes* (lossless, but still non-minimal). Use a bond
        orphan to show it clearly:

        >>> np.random.seed(0)
        >>> y = t3.TuckerTensorTrain.randn((10, 10, 10), (9, 9, 9), (1, 9, 9, 1))
        >>> y2, _, _ = y.t3svd(max_tucker_ranks=[9, 1, 9], max_tt_ranks=[1, 9, 2, 1])  # left-orth, non-minimal
        >>> print(y2.has_minimal_ranks)
        False
        >>> wrong = y2.rank_adjustment_sweep('left_to_right')    # WRONG direction for a left-orth input
        >>> print(wrong.has_minimal_ranks, np.allclose(wrong.to_dense(), y2.to_dense()))  # non-minimal, but lossless
        False True
        >>> print(y2.rank_adjustment_sweep('right_to_left').has_minimal_ranks)   # correct direction
        True
        """
        return TuckerTensorTrain(*ragged_t3svd.t3_rank_adjustment_sweep(self.data, direction))

    @staticmethod
    def t3svd_dense(
            T: NDArray,                              # shape=stack_shape+(N1, N2, .., Nd)
            stack_shape: Sequence[int] = (),
            max_tucker_ranks: typ.Union[int, Sequence[int]] = None,  # scalar (caps all) or len=d
            max_tt_ranks: typ.Union[int, Sequence[int]] = None,      # scalar (caps all) or len=d+1
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
        max_tucker_ranks: int or Sequence[int], optional
            Maximum Tucker ranks ``ni``. A scalar caps every mode; a sequence is per-mode,
            e.g., ``(5,5,5)`` with ``len(max_tucker_ranks)=d``.
            Default: no max Tucker rank truncation (``None``).
        max_tt_ranks: int or Sequence[int], optional
            Maximum TT-ranks ``ri``. A scalar caps every bond; a sequence is per-bond,
            e.g., ``(1,5,5,5,1)`` with ``len(max_tt_ranks)=d+1``.
            Default: no max TT rank truncation (``None``).
        rtol: float, optional
            Relative tolerance for truncation (in the Frobenius norm), applied **per truncation step**
            (see Notes -- the overall error can be larger). Default: no ``rtol`` truncation (``None``).
            Requires ``stack_shape=()``.
        atol: float, optional
            Absolute tolerance for truncation (in the Frobenius norm), applied **per truncation step**
            (see Notes). Default: no ``atol`` truncation (``None``).
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

        Notes
        -----
        ``rtol``/``atol`` bound the truncation error at **each step** (each unfolding/matricization SVD
        truncation), **not** the overall error. The per-step errors accumulate in quadrature over the
        ``2d-1`` steps (``d-1`` TT unfoldings + ``d`` Tucker matricizations), so the realized error can
        exceed the requested tolerance by up to a factor ``sqrt(2d-1)`` (the generalized Oseledets bound).
        See ``docs/t3svd_verification.md`` for the bound and its proof.

        See Also
        --------
        :py:meth:`.TuckerTensorTrain.t3svd`
        :py:meth:`.TuckerTensorTrain.get_minimal_ranks`

        Notes
        -----
        See the dense T3-SVD (Algorithm 9) in Appendix A of [1]_.

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
        >>> import math
        >>> np.random.seed(0)
        >>> T0 = np.random.randn(40, 50, 60)
        >>> c0 = 1.0 / np.arange(1, 41)**2
        >>> c1 = 1.0 / np.arange(1, 51)**2
        >>> c2 = 1.0 / np.arange(1, 61)**2
        >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # graded-spectrum (preconditioned) tensor
        >>> x, ss_tucker, ss_tt = t3.TuckerTensorTrain.t3svd_dense(T, rtol=1e-3) # truncated T3-SVD
        >>> print(x.tucker_ranks, x.tt_ranks)          # rtol reduces the ranks below full (40,50,60)
        (11, 11, 10) (1, 10, 11, 1)
        >>> T2 = x.to_dense()
        >>> rel_err = np.linalg.norm(T - T2) / np.linalg.norm(T)
        >>> # per-step rtol accumulates over 2d-1 steps -> realized error within sqrt(2d-1)*rtol (d=3)
        >>> print(bool(rel_err <= math.sqrt(2 * 3 - 1) * 1e-3))
        True
        '''
        if stack_shape and ((rtol is not None) or (atol is not None)):
            raise ValueError(
                'Cannot use t3svd_dense with rtol or atol for stacked tensor T.\n' +
                'Different elements of the stack could end out having different shapes.\n' +
                'First unstack, then call t3svd_dense for each unstacked tensor.\n' +
                'stack_shape = ' + str(stack_shape)
            )

        result = ragged_t3svd.dense_t3svd(
            T,
            stack_shape=stack_shape,
            max_tucker_ranks=max_tucker_ranks, max_tt_ranks=max_tt_ranks,
            rtol=rtol, atol=atol,
        )
        return TuckerTensorTrain(*result[0]), result[1], result[2]


if common.jax_available:
    jax.tree_util.register_pytree_node(
        TuckerTensorTrain,
        lambda x: (x.data, None),
        lambda aux_data, children: TuckerTensorTrain(*children),
    )