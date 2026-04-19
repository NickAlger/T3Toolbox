# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
"""
Edge vectors.
"""
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass


import t3toolbox.tucker_tensor_train as t3
import t3toolbox.core.tucker_tensor_train.ragged.ragged_t3_operations as ragged_t3_operations
import t3toolbox.core.weighted_tucker_tensor_train.ragged.ragged_wt3_operations as ragged_wt3_operations


from t3toolbox.common import *

jax = None
if has_jax:
    import jax

__all__ = [
    'EdgeVectors',
    'WeightedTuckerTensorTrain',
    'wt3_add',
    'wt3_sub',
    'wt3_inner_product',
]


@dataclass(frozen=True)
class EdgeVectors:
    """Vectors that "live" on edges in a T3 tensor network.

    Attributes:
    -----------
    shape_vectors: typ.Sequence[NDArray]
        Vectors on externally facing edges. len=d, elm_shape=stack_shape+(Ni,)
    tucker_vectors: typ.Sequence[NDArray]
        Vectors on edges between Tucker cores and TT cores. len=d, elm_shape=stack_shape+(ni,)
    tt_vectors: typ.Sequence[NDArray]
        Vectors on edges between adjacent TT cores. len=d+1, elm_shape=stack_shape+(ri,)
    """
    tucker_edge_vectors:    typ.Tuple[NDArray,...] # len=d,   elm_shape=stack_shape+(ni,)
    tt_vectors:             typ.Tuple[NDArray,...] # len=d+1, elm_shape=stack_shape+(ri,)

    @ft.cached_property
    def data(self) -> typ.Tuple[
        typ.Tuple[NDArray, ...], # tucker_vectors
        typ.Tuple[NDArray, ...], # tt_vectors
    ]:
        return self.tucker_edge_vectors, self.tt_vectors

    @ft.cached_property
    def d(self) -> int:
        return len(self.tucker_edge_vectors)

    @ft.cached_property
    def tucker_ranks(self) -> typ.Tuple[int,...]:
        return tuple([tkw.shape[-1] for tkw in self.tucker_edge_vectors])

    @ft.cached_property
    def tt_ranks(self) -> typ.Tuple[int,...]:
        return tuple([ttw.shape[-1] for ttw in self.tt_vectors])

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        return self.tucker_edge_vectors[0].shape[:-1]

    def validate(self):
        assert(len(self.tucker_edge_vectors) == self.d)
        assert(len(self.tt_vectors) == self.d + 1)

        for tkw, n in zip(self.tucker_edge_vectors, self.tucker_ranks):
            assert(tkw.shape == self.stack_shape + (n,))

        for ttw, r in zip(self.tt_vectors, self.tt_ranks):
            assert(ttw.shape == self.stack_shape + (r,))

    def __post_init__(self):
        self.validate()

    def reverse(self) -> 'EdgeVectors':
        """Reverse edge vector ordering.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.weighted_tucker_tensor_train as wt3
        >>> randn = np.random.randn
        >>> tucker_vectors = tuple([randn(9,10, 5), randn(9,10, 6), randn(9,10, 7)])
        >>> tt_vectors = tuple([randn(9,10, 1), randn(9,10, 2), randn(9,10, 3), randn(9,10, 4)])
        >>> ev = wt3.EdgeVectors(tucker_vectors, tt_vectors)
        >>> ev_rev = ev.reverse()
        >>> print(ev.tucker_ranks, ev.tt_ranks)
        (5, 6, 7) (1, 2, 3, 4)
        >>> print(ev_rev.tucker_ranks, ev_rev.tt_ranks)
        (7, 6, 5) (4, 3, 2, 1)
        """
        return EdgeVectors(*ragged_wt3_operations.reverse_edge_vectors(self.data))



@dataclass
class WeightedTuckerTensorTrain:
    x0:             t3.TuckerTensorTrain
    edge_weights:   EdgeVectors

    @ft.cached_property
    def data(self):
        return (self.x0.data, self.edge_weights.data)

    def validate(self):
        if self.x0.d != self.edge_weights.d:
            raise RuntimeError(
                'TuckerTensorTrain and edge weights do not have the same number of cores.\n' +
                str(self.x0.d) + ' = x.d != edge_vectors.d = ' + str(self.edge_weights.d)
            )

        if self.x0.stack_shape != self.edge_weights.stack_shape:
            raise RuntimeError(
                'TuckerTensorTrain and edge weights do not have the same stack_shape.\n' +
                str(self.x0.stack_shape) + ' = x.stack_shape != edge_vectors.stack_shape = ' + str(self.edge_weights.stack_shape)
            )

        assert(self.x0.tucker_ranks == self.edge_weights.tucker_ranks)
        assert (self.x0.tt_ranks == self.edge_weights.tt_ranks)

    def __post_init__(self):
        self.validate()

    def contract_edge_weights_into_cores(
            self,
            use_jax: bool = False,
    ) -> t3.TuckerTensorTrain:
        """Contract each edge vector into a neighboring core.

        Tensor network diagram illustrating groupings::

                 ____     ____     ________
                /    \   /    \   /        \
            1---w---G0---w---G1---w---G2---w---1
                    |        |        |
                  / w      / w      / w
                  | |      | |      | |
                  \ B0     \ B1     \ B2
                    |        |        |
                    w        w        w
                    |        |        |

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.weighted_tucker_tensor_train as wt3
        >>> randn = np.random.randn
        >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
        >>> tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
        >>> tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
        >>> weights = wt3.EdgeVectors(tucker_vectors, tt_vectors)
        >>> x0_w = wt3.WeightedTuckerTensorTrain(x0, weights)
        >>> x = x0_w.contract_edge_weights_into_cores()
        >>> dense_x = x.to_dense()
        >>> all_x_vars = x0.tucker_cores + x0.tt_cores + tucker_vectors + tt_vectors
        >>> einsum_str = 'qix,qjy,qkz,qaib,qbjc,qckd,qi,qj,qk,qa,qb,qc,qd->qxyz'
        >>> dense_x2 = np.einsum(einsum_str, *all_x_vars)
        >>> print(np.linalg.norm(dense_x - dense_x2))
        4.7254283984394845e-12
        """
        return t3.TuckerTensorTrain(*ragged_wt3_operations.contract_edge_vectors_into_t3(
            self.x0.data, self.edge_weights.data, use_jax=use_jax,
        ))

    def squash_tails(self, use_jax: bool = False) -> 'WeightedTuckerTensorTrain':
        result = ragged_wt3_operations.wt3_squash_tails(self.data, use_jax=use_jax)
        return WeightedTuckerTensorTrain(
            t3.TuckerTensorTrain(*result[0]),
            EdgeVectors(*result[1]),
        )

    def reverse(self) -> 'WeightedTuckerTensorTrain':
        """Reverse the weighted Tucker tensor train.
        """
        return WeightedTuckerTensorTrain(self.x0.reverse(), self.edge_weights.reverse())

    def __neg__(self) -> 'WeightedTuckerTensorTrain':
        return WeightedTuckerTensorTrain(-self.x0, self.edge_weights)

    def __mul__(self, other) -> 'WeightedTuckerTensorTrain':
        return WeightedTuckerTensorTrain(self.x0 * other, self.edge_weights)

    def __add__(self, other, squash: bool=True, use_jax: bool=False):
        return wt3_add(self, other, squash=squash, use_jax=use_jax)

    def __sub__(self, other, squash: bool=True, use_jax: bool=False):
        return wt3_sub(self, other, squash=squash, use_jax=use_jax)

    def norm(
            self,
            use_orthogonalization: bool = True,
            use_jax: bool = False,
    ):
        """Computes the Hilbert-Schmidt norm of the weighted Tucker tensor train.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.weighted_tucker_tensor_train as wt3
        >>> randn = np.random.randn
        >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
        >>> x_tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
        >>> x_tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
        >>> x_weights = wt3.EdgeVectors(x_tucker_vectors, x_tt_vectors)
        >>> x = wt3.WeightedTuckerTensorTrain(x0, x_weights)
        >>> print(x.norm())
        [ 31.94684693   0.68957189 100.53306804  35.34732966]
        >>> x_dense = x.contract_edge_weights_into_cores().to_dense()
        >>> print(np.array([np.linalg.norm(x_dense[ii]) for ii in range(4)]))
        [ 31.94684693   0.68957189 100.53306804  35.34732966]
        """
        return self.contract_edge_weights_into_cores(use_jax=use_jax).norm(
            use_orthogonalization=use_orthogonalization, use_jax=use_jax
        )

    def to_dense(self, use_jax: bool = False) -> NDArray:
        """Convert the weighted Tucker tensor train to a dense array.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.weighted_tucker_tensor_train as wt3
        >>> randn = np.random.randn
        >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
        >>> tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
        >>> tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
        >>> weights = wt3.EdgeVectors(tucker_vectors, tt_vectors)
        >>> x0_w = wt3.WeightedTuckerTensorTrain(x0, weights)
        >>> dense_x = x0_w.to_dense()
        >>> all_x_vars = x0.tucker_cores + x0.tt_cores + tucker_vectors + tt_vectors
        >>> einsum_str = 'qix,qjy,qkz,qaib,qbjc,qckd,qi,qj,qk,qa,qb,qc,qd->qxyz'
        >>> dense_x2 = np.einsum(einsum_str, *all_x_vars)
        >>> print(np.linalg.norm(dense_x - dense_x2))
        2.8199489101171104e-12
        """
        return self.contract_edge_weights_into_cores(use_jax=use_jax).to_dense(use_jax=use_jax)




if has_jax:
    jax.tree_util.register_pytree_node(
        EdgeVectors,
        lambda x: (x.data, None),
        lambda aux_data, children: EdgeVectors(*children),
    )

    jax.tree_util.register_pytree_node(
        WeightedTuckerTensorTrain,
        lambda x: (x.data, None),
        lambda aux_data, children: WeightedTuckerTensorTrain(*children),
    )


def concatenate_edge_vectors(
        evA: EdgeVectors,
        evB: EdgeVectors,
) -> EdgeVectors:
    """Concatenates edge vectors. Vectorized over stacking dimensions.
    """
    return EdgeVectors(*ragged_wt3_operations.concatenate_edge_vectors(evA.data, evB.data))


def wt3_add(
        A: WeightedTuckerTensorTrain,
        B: WeightedTuckerTensorTrain,
        squash: bool = True,
        use_jax: bool = False,
) -> WeightedTuckerTensorTrain:
    """Add two weighted Tucker tensor trains.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.weighted_tucker_tensor_train as wt3
    >>> randn = np.random.randn
    >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
    >>> x_tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
    >>> x_tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
    >>> x_weights = wt3.EdgeVectors(x_tucker_vectors, x_tt_vectors)
    >>> x = wt3.WeightedTuckerTensorTrain(x0, x_weights)
    >>> y0 = t3.t3_corewise_randn((6,7,8), (2,4,2), (1,3,1,2), stack_shape=(4,))
    >>> y_tucker_vectors = tuple([randn(4, 2), randn(4, 4), randn(4, 2)])
    >>> y_tt_vectors = tuple([randn(4, 1), randn(4, 3), randn(4, 1), randn(4, 2)])
    >>> y_weights = wt3.EdgeVectors(y_tucker_vectors, y_tt_vectors)
    >>> y = wt3.WeightedTuckerTensorTrain(y0, y_weights)
    >>> x_plus_y = wt3.wt3_add(x, y)
    >>> x_dense = x.contract_edge_weights_into_cores().to_dense()
    >>> y_dense = y.contract_edge_weights_into_cores().to_dense()
    >>> x_plus_y_dense = x_plus_y.contract_edge_weights_into_cores().to_dense()
    >>> print(np.linalg.norm(x_dense + y_dense - x_plus_y_dense))
    1.3834329073101016e-13
    """
    assert(A.x0.shape == B.x0.shape)
    assert(A.x0.stack_shape == B.x0.stack_shape)

    C0 = t3.t3_add(A.x0, B.x0, squash=False, use_jax=use_jax)
    c_weights = concatenate_edge_vectors(A.edge_weights, B.edge_weights)

    C = WeightedTuckerTensorTrain(C0, c_weights)

    if squash:
        C = C.squash_tails()

    return C


def wt3_sub(
        A: WeightedTuckerTensorTrain,
        B: WeightedTuckerTensorTrain,
        squash: bool = True,
        use_jax: bool = False,
) -> WeightedTuckerTensorTrain:
    """Subtract two weighted Tucker tensor trains, A-B.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.weighted_tucker_tensor_train as wt3
    >>> randn = np.random.randn
    >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
    >>> x_tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
    >>> x_tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
    >>> x_weights = wt3.EdgeVectors(x_tucker_vectors, x_tt_vectors)
    >>> x = wt3.WeightedTuckerTensorTrain(x0, x_weights)
    >>> y0 = t3.t3_corewise_randn((6,7,8), (2,4,2), (1,3,1,2), stack_shape=(4,))
    >>> y_tucker_vectors = tuple([randn(4, 2), randn(4, 4), randn(4, 2)])
    >>> y_tt_vectors = tuple([randn(4, 1), randn(4, 3), randn(4, 1), randn(4, 2)])
    >>> y_weights = wt3.EdgeVectors(y_tucker_vectors, y_tt_vectors)
    >>> y = wt3.WeightedTuckerTensorTrain(y0, y_weights)
    >>> x_minus_y = wt3.wt3_sub(x, y)
    >>> x_dense = x.contract_edge_weights_into_cores().to_dense()
    >>> y_dense = y.contract_edge_weights_into_cores().to_dense()
    >>> x_minus_y_dense = x_minus_y.contract_edge_weights_into_cores().to_dense()
    >>> print(np.linalg.norm(x_dense - y_dense - x_minus_y_dense))
    1.8944706560546762e-13
    """
    return wt3_add(A, -B, squash=squash, use_jax=use_jax)


def wt3_inner_product(
        A: WeightedTuckerTensorTrain,
        B: WeightedTuckerTensorTrain,
        use_orthogonalization: bool = True,
        use_jax: bool = False,
):
    """Computes the Hilbert-Schmidt inner product between two Tucker tensor trains.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.weighted_tucker_tensor_train as wt3
    >>> randn = np.random.randn
    >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
    >>> x_tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
    >>> x_tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
    >>> x_weights = wt3.EdgeVectors(x_tucker_vectors, x_tt_vectors)
    >>> x = wt3.WeightedTuckerTensorTrain(x0, x_weights)
    >>> y0 = t3.t3_corewise_randn((6,7,8), (2,4,2), (1,3,1,2), stack_shape=(4,))
    >>> y_tucker_vectors = tuple([randn(4, 2), randn(4, 4), randn(4, 2)])
    >>> y_tt_vectors = tuple([randn(4, 1), randn(4, 3), randn(4, 1), randn(4, 2)])
    >>> y_weights = wt3.EdgeVectors(y_tucker_vectors, y_tt_vectors)
    >>> y = wt3.WeightedTuckerTensorTrain(y0, y_weights)
    >>> x_dot_y = wt3.wt3_inner_product(x,y)
    >>> print(x_dot_y)
    [   8.20032492   -8.97264769 -344.13076394   -3.72479156]
    >>> x_dense = x.contract_edge_weights_into_cores().to_dense()
    >>> y_dense = y.contract_edge_weights_into_cores().to_dense()
    >>> print(np.array([np.sum(x_dense[ii] * y_dense[ii]) for ii in range(4)]))
    [   8.20032492   -8.97264769 -344.13076394   -3.72479156]
    """
    return t3.t3_inner_product(
        A.contract_edge_weights_into_cores(use_jax=use_jax),
        B.contract_edge_weights_into_cores(use_jax=use_jax),
        use_orthogonalization=use_orthogonalization, use_jax=use_jax,
    )

###########################################################################
########    Scalar valued M.L.F. applies, entries, and probing    #########
###########################################################################

def wt3_apply(
        x: WeightedTuckerTensorTrain, # shape=(N0,...,N(d-1))
        vecs: typ.Sequence[NDArray], # len=d, elm_shape=V+(Ni,)
        use_jax: bool = False,
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.weighted_tucker_tensor_train as wt3
    >>> randn = np.random.randn
    >>> x0 = t3.t3_corewise_randn((6,7,8), (5,6,7), (2,3,3,1), stack_shape=(4,))
    >>> tucker_vectors = tuple([randn(4, 5), randn(4, 6), randn(4, 7)])
    >>> tt_vectors = tuple([randn(4, 2), randn(4, 3), randn(4, 3), randn(4, 1)])
    >>> weights = wt3.EdgeVectors(tucker_vectors, tt_vectors)
    >>> x = wt3.WeightedTuckerTensorTrain(x0, weights)
    >>> vecs = [np.random.randn(2,3, 6), np.random.randn(2,3, 7), np.random.randn(2,3, 8)]
    >>> result = wt3.wt3_apply(x, vecs)
    >>> result2 = np.einsum('uijk,xyi,xyj,xyk->uxy', x.to_dense(), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    6.89832231894826e-13
    '''
    return t3.t3_apply(
        x.contract_edge_weights_into_cores(use_jax=use_jax), vecs,
        use_jax=use_jax,
    )



def wt3_get_entries(
        x: WeightedTuckerTensorTrain, # shape=(N0,...,N(d-1))
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
    >>> result = t3.t3_get_entries(x, index)
    >>> result2 = x.to_dense()[9, 4, 7]
    >>> print(np.abs(result - result2))
    1.3322676295501878e-15

    Compute multiple entries (vectorized):

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> index = [[9,8], [4,10], [7,13]] # get entries (9,4,7) and (8,10,13)
    >>> entries = t3.t3_get_entries(x, index)
    >>> x_dense = x.to_dense(x)
    >>> entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
    >>> print(np.linalg.norm(entries - entries2))
    1.7763568394002505e-15

    Vectorize over entries AND T3s

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (2,3,2,2), stack_shape=(2,3))
    >>> index = [[9,0], [4,0], [7,0]] # get entries (9,4,7) and (0,0,0)
    >>> entries = t3.t3_get_entries(x, index)
    >>> x_dense = x.to_dense(x)
    >>> entries2 = np.moveaxis(np.array([x_dense[:,:, 9,4,7], x_dense[:,:, 0,0,0]]), 0,2)
    >>> print(np.linalg.norm(entries - entries2))
    9.22170384909466e-15

    Example using jax jit compiling:

	>>> import numpy as np
    >>> import jax
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> get_entry_123 = lambda x: t3.t3_get_entries(x, (1,2,3), use_jax=True)
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
    >>> get_entry_123 = lambda x: t3.t3_get_entries(x, (1,2,3), use_jax=True)
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

    return entries.t3_get_entries(x.data, index, use_jax=use_jax)


def probe_wt3(
        ww: typ.Sequence[NDArray], # len=d, elm_shape=W+(Ni,)
        x: WeightedTuckerTensorTrain, # x.stack_shape=X
        use_jax: bool=False,
) -> typ.Sequence[NDArray]: # zz, len=d, elm_shape=X+W+(Ni,)
    """Probe a TuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.core.probing as probing
    >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2))
    >>> ww = (np.random.randn(10), np.random.randn(11), np.random.randn(12))
    >>> zz = t3.probe_t3(ww, x)
    >>> x_dense = x.to_dense()
    >>> zz2 = probing.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.0259410400851746e-12, 1.0909087370186656e-12, 3.620283224238675e-13]

    Vectorize over probes:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.core.probing as probing
    >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2))
    >>> ww = (np.random.randn(2,3, 10), np.random.randn(2,3, 11), np.random.randn(2,3, 12))
    >>> zz = t3.probe_t3(ww, x)
    >>> x_dense = x.to_dense()
    >>> zz2 = probing.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [2.9290244450205316e-12, 2.0347746956505754e-12, 1.7784156096697445e-12]

    Vectorize over probes and T3s:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.core.probing as probing
    >>> x = t3.t3_corewise_randn((10,11,12),(5,6,4),(2,3,4,2), stack_shape=(4,5))
    >>> ww = (np.random.randn(2,3, 10), np.random.randn(2,3, 11), np.random.randn(2,3, 12))
    >>> zz = t3.probe_t3(ww, x)
    >>> x_dense = x.to_dense()
    >>> zz2 = probing.probe_dense(ww, x_dense)
    >>> print([np.linalg.norm(z - z2) for z, z2 in zip(zz, zz2)])
    [1.4471391818397927e-11, 1.0485601346346092e-11, 1.437623640611662e-11]
    """
    return probing.probe_t3(ww, x.data, use_jax=use_jax)