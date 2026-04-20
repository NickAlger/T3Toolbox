# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.core.tucker_tensor_train.uniform.uniform_t3_operations as uniform_ops
from t3toolbox.common import *

__all__ = [
    'UniformTuckerTensorTrain',
    #
    'check_ut3',
    'get_uniform_structure',
    'get_original_structure',
    'pack_tensors',
    'unpack',
    'make_uniform_masks',
    'apply_masks',
    'uniform_squash_tails',
    'uniform_randn',
    'uniform_zeros',
    #
    't3_to_ut3',
    'ut3_to_t3',
    #
    'ut3_to_dense',
    'are_ut3_ranks_minimal',
    'ut3_get_entries',
    'ut3_apply',
    # Linear algebra core:
    'ut3_add',
    'ut3_scale',
    'ut3_neg',
    'ut3_sub',
]


@dataclass(frozen=True)
class UniformTuckerTensorTrain:
    """Uniform Tucker tensor train.

    Uniform Tucker tensor trains are created by padding a Tucker tensor train
    so that the ranks are uniform, then stacking the TT cores and Tucker cores into
    "supercores", which have one more dimension.

    Original core shapes are tracked with boolean mask arrays associated with the edges.
    """
    tucker_supercore:   NDArray  #             shape=(d,)   + stack_shape + (n,N)
    tt_supercore:       NDArray  #             shape=(d+1,) + stack_shape + (r,n,r)
    shape_mask:         NDArray  # dtype=bool, shape=(d,)   + stack_shape + (N,)
    tucker_edge_mask:   NDArray  # dtype=bool, shape=(d,)   + stack_shape + (n,)
    tt_edge_mask:       NDArray  # dtype=bool, shape=(d+1,) + stack_shape + (r,)

    @ft.cached_property
    def data(self) -> typ.Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        return (
            self.tucker_supercore, self.tt_supercore,
            self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
        )

    @ft.cached_property
    def d(self) -> int:
        """Number of indices of the tensor.
        """
        return self.tucker_supercore.shape[0]

    @ft.cached_property
    def n(self) -> int:
        """Padded Tucker rank. n >= max(n0,...,n(d-1)), where ni are the original (unpadded) Tucker ranks.
        """
        return self.tucker_supercore.shape[-2]

    @ft.cached_property
    def N(self) -> int:
        """Padded index dimension. N >= max(N0,...,N(d-1)), where Ni are the original (unpadded) shapes.
        """
        return self.tucker_supercore.shape[-1]

    @ft.cached_property
    def r(self) -> int:
        """Padded TT rank. r >= max(r0,...,rd), where ri are the original (unpadded) TT ranks.
        """
        return self.tt_supercore.shape[-1]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        """If this contains many stacked uniform Tucker tensor trains, this is the stacking shape.
        """
        return self.tucker_supercore.shape[1:-2]

    @ft.cached_property
    def structure(self) -> typ.Tuple[int, int, int, int, typ.Tuple[int,...]]:
        """d, N, n, r, stack_shape"""
        return self.d, self.N, self.n, self.r, self.stack_shape

    @ft.cached_property
    def shape(self) -> NDArray: # dtype=int, shape=(d,)+stack_shape
        """Get the original shapes, not including portions of the tensors that are unmasked.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d,)+stack_shape+(r,n,r))
        >>> shape_mask = np.ones((d,)+stack_shape+(N,), dtype=bool)
        >>> tucker_edge_mask = np.ones((d,)+stack_shape+(n,), dtype=bool)
        >>> tt_edge_mask = np.ones((d+1,)+stack_shape+(r,), dtype=bool)
        >>> shape_mask[0, 1, 0] = False # first index,  second T3, first component
        >>> shape_mask[0, 1, 1] = False # first index,  second T3, second component
        >>> shape_mask[0, 1, 2] = False # first index,  second T3, third component.  N0=6-3=3
        >>> shape_mask[1, 1, 0] = False # second index, second T3, first component
        >>> shape_mask[1, 1, 1] = False # second index, second T3, second component. N1=6-2=4
        >>> shape_mask[2, 1, 0] = False # third index,  second T3, first component.  N2=6-1=5
        >>> x = ut3.UniformTuckerTensorTrain(tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask)
        >>> print(x.shape)
        [[6 3]
         [6 4]
         [6 5]]
        """
        return self.shape_mask.sum(axis=-1)

    @ft.cached_property
    def tucker_ranks(self) -> NDArray: # dtype=int, shape=(d,)+stack_shape
        """Get the original tucker ranks, not including components of the edges that are unmasked.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d,)+stack_shape+(r,n,r))
        >>> shape_mask = np.ones((d,)+stack_shape+(N,), dtype=bool)
        >>> tucker_edge_mask = np.ones((d,)+stack_shape+(n,), dtype=bool)
        >>> tt_edge_mask = np.ones((d+1,)+stack_shape+(r,), dtype=bool)
        >>> tucker_edge_mask[0, 1, 0] = False # first edge,  second T3, first component
        >>> tucker_edge_mask[0, 1, 1] = False # first edge,  second T3, second component
        >>> tucker_edge_mask[0, 1, 2] = False # first edge,  second T3, third component.  n0=5-3=2
        >>> tucker_edge_mask[1, 1, 0] = False # second edge, second T3, first component
        >>> tucker_edge_mask[1, 1, 1] = False # second edge, second T3, second component. n1=5-2=3
        >>> tucker_edge_mask[2, 1, 0] = False # third edge,  second T3, first component.  n2=5-1=4
        >>> x = ut3.UniformTuckerTensorTrain(tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask)
        >>> print(x.tucker_ranks)
        [[5 2]
         [5 3]
         [5 4]]
        """
        return self.tucker_edge_mask.sum(axis=-1)

    @ft.cached_property
    def tt_ranks(self) -> NDArray: # dtype=int, shape=(d+1,)+stack_shape
        """Get the original tucker ranks, not including components of the edges that are unmasked.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d,)+stack_shape+(r,n,r))
        >>> shape_mask = np.ones((d,)+stack_shape+(N,), dtype=bool)
        >>> tucker_edge_mask = np.ones((d,)+stack_shape+(n,), dtype=bool)
        >>> tt_edge_mask = np.ones((d+1,)+stack_shape+(r,), dtype=bool)
        >>> tt_edge_mask[0, 1, 0] = False # first edge,  second T3, first component
        >>> tt_edge_mask[0, 1, 1] = False # first edge,  second T3, second component
        >>> tt_edge_mask[0, 1, 2] = False # first edge,  second T3, third component.  r0=4-3=1
        >>> tt_edge_mask[1, 1, 0] = False # second edge, second T3, first component
        >>> tt_edge_mask[1, 1, 1] = False # second edge, second T3, second component. r1=4-2=2
        >>> tt_edge_mask[2, 1, 0] = False # third edge,  second T3, first component.  r2=4-1=3
        >>> x = ut3.UniformTuckerTensorTrain(tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask)
        >>> print(x.tt_ranks)
        [[4 1]
         [4 2]
         [4 3]
         [4 4]]
        """
        return self.tt_edge_mask.sum(axis=-1)

    def validate(self):
        assert(is_boolean_ndarray(self.shape_mask))
        assert(is_boolean_ndarray(self.tucker_edge_mask))
        assert(is_boolean_ndarray(self.tt_edge_mask))

        assert(self.tucker_supercore.shape == (self.d,)+self.stack_shape+(self.n, self.N))
        assert(self.tt_supercore.shape == (self.d,)+self.stack_shape+(self.r, self.n, self.r))
        assert(self.shape_mask.shape == (self.d,)+self.stack_shape+(self.N,))
        assert(self.tucker_edge_mask.shape == (self.d,)+self.stack_shape+(self.n,))
        assert(self.tt_edge_mask.shape == (self.d+1,)+self.stack_shape+(self.r,))

    def __post_init__(self):
        self.validate()

    def to_dense(self, use_jax: bool = False) -> NDArray:
        """Convert uniform Tucker tensor train to dense array.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14, 15, 16), (4, 6, 5), (3, 3, 2, 4), stack_shape=(2,3))
        >>> uniform_x = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
        >>> x_dense = x.to_dense()
        >>> x_dense2 = uniform_x.to_dense()
        >>> print(np.linalg.norm(x_dense - x_dense2))
        3.2298106396012192e-12
        """
        xnp, _, _ = get_backend(True, use_jax)

        all_t3s = ut3_to_t3(self, use_jax=use_jax)
        def _func(x):
            if isinstance(x, t3.TuckerTensorTrain):
                return x.to_dense(use_jax=use_jax)
            return xnp.array([_func(xi) for xi in x])

        return _func(all_t3s)

    def reverse(self) -> 'UniformTuckerTensorTrain':
        """Reversed a UniformTuckerTensorTrain.
        """
        return UniformTuckerTensorTrain(
            self.tucker_supercore[::-1],
            uniform_ops.reverse_utt(self.tt_supercore),
            self.shape_mask[::-1],
            self.tucker_edge_mask[::-1],
            self.tt_edge_mask[::-1],
        )

    def squash_tails(self) -> 'UniformTuckerTensorTrain':
        """Make the first index of the first TT supercore
        and the last index of the last TT-supercore equal to 1 by summing.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> tucker_supercore = np.random.randn(4, 2,3, 6,7)
        >>> tt_supercore = np.random.randn(4, 2,3, 5,6,5)
        >>> x = UniformTuckerTensorTrain(tucker_supercore, tt_supercore)
        >>> squashed_x = x.squash_tails()
        >>> print(np.linalg.norm(x.to_dense() - squashed_x.to_dense()))

        >>> new_tt_supercore = uniform_operations.uniform_squash_tt_tails(tt_supercore)
        >>> print(np.linalg.norm(np.sum(tt_supercore[0], axis=-3) - new_tt_supercore[0, :,:, 0,:,:]))
        0.0
        >>> print(np.linalg.norm(new_tt_supercore[0, :,:, 1:,:,:]))
        0.0
        >>> print(np.linalg.norm(np.sum(tt_supercore[-1], axis=-1) - new_tt_supercore[-1, :,:, :,:,0]))
        0.0
        >>> print(np.linalg.norm(new_tt_supercore[-1, :,:, :,:,1:]))
        0.0
        """
        new_tt_supercore = uniform_ops.uniform_squash_tt_tails(self.tt_supercore)
        return UniformTuckerTensorTrain(self.tucker_supercore, new_tt_supercore)

    def apply_masks_to_cores(self, use_jax: bool = False) -> typ.Tuple[
        NDArray, # masked_tucker_supercore
        NDArray, # masked_tt_supercore
    ]:
        """Applies masking to supercores, replacing unmasked regions with zeros.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> import t3toolbox.t3svd as t3svd
        >>> import t3toolbox.corewise as cw
        >>> x = t3.t3_corewise_randn(((10,11,12), (5,6,4), (1,3,5,1)))
        >>> uniform_x, masks = ut3.t3_to_ut3(x)
        >>> uniform_x_svd, ss1, _ = t3svd.uniform_t3_svd(uniform_x, masks)
        >>> dense_x = t3.t3_to_dense(x)
        >>> print(np.linalg.norm(ut3.ut3_to_dense(uniform_x_svd, masks) - dense_x))
        3.0208288525321468e-12
        >>> x_svd, ss2, _ = t3svd.t3_svd(x)
        >>> print(np.linalg.norm(t3.t3_to_dense(x_svd) - dense_x))
        2.9361853188555994e-12
        >>> x_svd_structure = t3.get_structure(x_svd)
        >>> uniform_x_svd_structure = ut3.get_uniform_structure(uniform_x_svd)
        >>> masks2 = ut3.make_uniform_masks(x_svd_structure, uniform_x_svd_structure)
        >>> print(np.linalg.norm(ut3.ut3_to_dense(uniform_x_svd, masks2) - dense_x))
        3.0208288525321468e-12
        >>> print(cw.corewise_relerr(ut3.apply_masks(uniform_x_svd, masks2), uniform_x_svd))
        0.0024164186526434567
        >>> print(cw.corewise_relerr(ut3.apply_masks(uniform_x_svd, masks), uniform_x_svd))
        0.0
        """
        xnp,_,_ = get_backend(True, use_jax)

        masked_tucker_supercore = xnp.einsum(
            'd...nN,d...n,d...N->d...nN',
            self.tucker_supercore, self.tucker_edge_mask, self.shape_mask,
        )
        masked_tt_supercore = xnp.einsum(
            'd...lnr,d...l,d...n,d...r->d...lnr',
            self.tt_supercore, self.tt_edge_mask[:-1], self.tucker_edge_mask, self.tt_edge_mask[1:],
        )
        return masked_tucker_supercore, masked_tt_supercore



def t3_to_ut3(
        x: t3.TuckerTensorTrain,
        use_jax: bool = False,
) -> UniformTuckerTensorTrain:
    """Convert TuckerTensorTrain to UniformTuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14, 15, 16), (4, 6, 5), (3, 3, 2, 4), stack_shape=(2,3))
    >>> uniform_x = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
    >>> x2 = ut3.ut3_to_t3(uniform_x, stack_t3s=True)  # Convert ut3 -> t3
    >>> dense_x = x.to_dense()
    >>> dense_x2 = x2.to_dense()
    >>> print(np.linalg.norm(dense_x - dense_x2))
    2.695489335865025e-12
    """
    return UniformTuckerTensorTrain(*uniform_ops.t3_to_ut3(x.data, use_jax=use_jax))


def ut3_to_t3(
        x_uniform: UniformTuckerTensorTrain,
        stack_t3s: bool = False,
        use_jax: bool = False,
) -> t3.TuckerTensorTrain:
    """
    Convert UniformTuckerTensorTrain to TuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (1,3,2,1), stack_shape=(2,))
    >>> uniform_x = ut3.t3_to_ut3(x) # Convert t3 -> ut3
    >>> print(uniform_x.structure)
    (3, 16, 6, 3, (2,))
    >>> print(uniform_x.shape)
    [[14 14]
     [15 15]
     [16 16]]
    >>> print(uniform_x.tucker_ranks)
    [[4 4]
     [6 6]
     [5 5]]
    >>> print(uniform_x.tt_ranks)
    [[1 1]
     [3 3]
     [2 2]
     [1 1]]
    >>> all_x2 = ut3.ut3_to_t3(uniform_x) # Convert ut3 -> t3 without stacking
    >>> for x2i in all_x2: print(x2i.structure)
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1), ())
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1), ())
    >>> stacked_x2 = ut3.ut3_to_t3(uniform_x, stack_t3s=True) # with stacking
    >>> print(stacked_x2.structure)
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1), (2,))
    >>> for B, B2 in zip(stacked_x2.tucker_cores, x.tucker_cores): print(np.linalg.norm(B - B2))
    0.0
    0.0
    0.0
    >>> for G, G2 in zip(stacked_x2.tt_cores, x.tt_cores): print(np.linalg.norm(G - G2))
    0.0
    0.0
    0.0
    """
    result = uniform_ops.ut3_to_t3(
        x_uniform.data, stack_t3s=stack_t3s, use_jax=use_jax,
    )
    if stack_t3s:
        return t3.TuckerTensorTrain(*result)
    else:
        def _func(x):
            if is_ndarray(x[0][0]):
                return t3.TuckerTensorTrain(*x)
            return tuple([_func(xi) for xi in x])

        return _func(result)



