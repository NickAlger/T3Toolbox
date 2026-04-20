# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ
import functools as ft
from dataclasses import dataclass

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.core.tucker_tensor_train.t3_entries as entries
import t3toolbox.core.tucker_tensor_train.t3_apply as apply
import t3toolbox.core.probing as probing
import t3toolbox.core.tucker_tensor_train.uniform.uniform_t3_operations as uniform_ops
import t3toolbox.core.tucker_tensor_train.uniform.uniform_tensor_linalg as utla
import t3toolbox.core.tucker_tensor_train.uniform.uniform_orthogonalization as uniform_orthogonalization
import t3toolbox.core.tucker_tensor_train.orthogonalization as orth
import t3toolbox.core.tucker_tensor_train.uniform.uniform_t3svd as ut3svd
from t3toolbox.common import *

jax = None
if has_jax:
    import jax

__all__ = [
    'UniformTuckerTensorTrain',
    #
    'pack_vectors',
    'unpack_vectors',
    #
    't3_to_ut3',
    'ut3_to_t3',
    #
    'ut3_get_entries',
    'ut3_apply',
    # Linear algebra core:
    'ut3_add',
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
    shape_mask:         NDArray  # dtype=bool, shape=(d,N). No stacking the shape
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
    def uniform_structure(self) -> typ.Tuple[int, int, int, int, typ.Tuple[int,...]]:
        """d, N, n, r, stack_shape"""
        return self.d, self.N, self.n, self.r, self.stack_shape

    @ft.cached_property
    def shape(self) -> typ.Tuple[int,...]: # dtype=int, len=d
        """Get the original shapes, not including portions of the tensors that are unmasked.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d,)+stack_shape+(r,n,r))
        >>> shape_mask = np.ones((d,N), dtype=bool)
        >>> tucker_edge_mask = np.ones((d,)+stack_shape+(n,), dtype=bool)
        >>> tt_edge_mask = np.ones((d+1,)+stack_shape+(r,), dtype=bool)
        >>> shape_mask[0, 0] = False # first index, first component
        >>> shape_mask[0, 1] = False # first index, second component
        >>> shape_mask[0, 2] = False # first index, third component.  N0=6-3=3
        >>> shape_mask[1, 0] = False # second index, first component
        >>> shape_mask[1, 1] = False # second index, second component. N1=6-2=4
        >>> shape_mask[2, 0] = False # third index, first component.  N2=6-1=5
        >>> x = ut3.UniformTuckerTensorTrain(tucker_supercore, tt_supercore, shape_mask, tucker_edge_mask, tt_edge_mask)
        >>> print(x.shape)
        (3, 4, 5)
        """
        shape_ndarray = self.shape_mask.sum(axis=-1)
        return tuple([int(x) for x in shape_ndarray])

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
        >>> shape_mask = np.ones((d,N), dtype=bool)
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
        >>> shape_mask = np.ones((d,N), dtype=bool)
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

    @ft.cached_property
    def structure(self) -> typ.Tuple[
        typ.Tuple[int,...], # shape
        NDArray, # tucker_ranks
        NDArray, # tt_ranks
        NDArray, # stack_shape
    ]:
        '''Structure of the original tensor.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (5,6,7), (2,3,4,3), stack_shape=(2,))
        >>> ux = ut3.t3_to_ut3(x)
        >>> shape, tucker_ranks, tt_ranks, stack_shape = ux.structure
        >>> print(shape)
        (14, 15, 16)
        >>> print(tucker_ranks)
        [[5 5]
         [6 6]
         [7 7]]
        >>> print(tt_ranks)
        [[1 1]
         [3 3]
         [4 4]
         [1 1]]
        >>> print(stack_shape)
        (2,)
        '''
        return self.shape, self.tucker_ranks, self.tt_ranks, self.stack_shape

    def validate(self):
        assert(is_boolean_ndarray(self.shape_mask))
        assert(is_boolean_ndarray(self.tucker_edge_mask))
        assert(is_boolean_ndarray(self.tt_edge_mask))

        assert(self.tucker_supercore.shape == (self.d,)+self.stack_shape+(self.n, self.N))
        assert(self.tt_supercore.shape == (self.d,)+self.stack_shape+(self.r, self.n, self.r))
        assert(self.shape_mask.shape == (self.d, self.N,))
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

    def squash_tails(self, use_jax: bool = False) -> 'UniformTuckerTensorTrain':
        """Make the first index of the first TT supercore
        and the last index of the last TT-supercore equal to 1 by summing.

        Examples
        --------

        EXAMPLE WORK IN PROGRESS
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> tucker_supercore = np.random.randn(4, 2,3, 6,7)
        >>> tt_supercore = np.random.randn(4, 2,3, 5,6,5)
        >>> x = ut3.UniformTuckerTensorTrain(tucker_supercore, tt_supercore)
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
        new_tt_supercore = uniform_ops.uniform_squash_tt_tails(self.tt_supercore, use_jax=use_jax)
        return UniformTuckerTensorTrain(
            self.tucker_supercore, new_tt_supercore,
            self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
        )

    def apply_masks_to_cores(
            self, use_jax: bool = False,
    ) -> typ.Tuple[
        NDArray, # masked_tucker_supercore
        NDArray, # masked_tt_supercore
    ]:
        """Applies masking to supercores, replacing unmasked regions with zeros.

        Examples
        --------

        EXAMPLE WORK IN PROGRESS
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
        return uniform_ops.apply_masks_to_cores(self.data)

        # xnp,_,_ = get_backend(True, use_jax)
        #
        # masked_tucker_supercore = xnp.einsum(
        #     'd...nN,d...n,dN->d...nN',
        #     self.tucker_supercore, self.tucker_edge_mask, self.shape_mask,
        # )
        # masked_tt_supercore = xnp.einsum(
        #     'd...lnr,d...l,d...n,d...r->d...lnr',
        #     self.tt_supercore, self.tt_edge_mask[:-1], self.tucker_edge_mask, self.tt_edge_mask[1:],
        # )
        # return masked_tucker_supercore, masked_tt_supercore

    def __mul__(
            self,
            s,  # scalar
            use_jax: bool = False,
    ) -> 'UniformTuckerTensorTrain':  # z = s*x
        """Scale a uniform Tucker tensor train, s,x -> s*x.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2))
        >>> ux = ut3.t3_to_ut3(x)
        >>> s = 3.5
        >>> usx = ux * s
        >>> print(np.linalg.norm(s*x.to_dense() - usx.to_dense()))
        1.6880423424147856e-12
        """
        return UniformTuckerTensorTrain(
            self.tucker_supercore,
            utla.scale_last_slice(self.tt_supercore, s, use_jax=use_jax),
            self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
        )

    def __neg__(
            self,
            use_jax: bool = False,
    ) -> 'UniformTuckerTensorTrain':  # z = s*x
        """Flip a uniform Tucker tensor train, x -> -x.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2))
        >>> ux = ut3.t3_to_ut3(x)
        >>> neg_ux = -ux
        >>> print(np.linalg.norm(x.to_dense() + neg_ux.to_dense()))
        6.440955358355001e-13
        """
        return self * (-1.0)

    def __add__(
            self,
            other: 'UniformTuckerTensorTrain',
            squash: bool = True,
            use_jax: bool = False,
    ) -> 'UniformTuckerTensorTrain':  # z = x + y
        """Add two UniformTuckerTensorTrains, x,y -> x+y.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2))
        >>> ux = ut3.t3_to_ut3(x)
        >>> y = t3.t3_corewise_randn((14,15,16), (6,7,8), (3,5,6,1))
        >>> uy = ut3.t3_to_ut3(y)
        >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - (ux + uy).to_dense()))
        2.7361685557814917e-12
        """
        return ut3_add(self, other, squash=squash, use_jax=use_jax)

    def __sub__(
            self,
            other: 'UniformTuckerTensorTrain',
            squash: bool = True,
            use_jax: bool = False,
    ) -> 'UniformTuckerTensorTrain':  # z = x + y
        """Subtract two UniformTuckerTensorTrains, x,y -> x-y.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2))
        >>> ux = ut3.t3_to_ut3(x)
        >>> y = t3.t3_corewise_randn((14,15,16), (6,7,8), (3,5,6,1))
        >>> uy = ut3.t3_to_ut3(y)
        >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - (ux - uy).to_dense()))
        2.7487527725050217e-12
        """
        return ut3_add(self, other, squash=squash, use_jax=use_jax)

    def up_orthogonalize_tucker_cores(
            self,
            use_jax: bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Orthogonalize Tucker cores upwards, pushing remainders onto TT cores above.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.up_orthogonalize_tucker_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        5.322185194708616e-12
        >>> ind = 1
        >>> B = ux_orth.data[0][ind]
        >>> print(np.linalg.norm(B @ B.T - np.eye(B.shape[0])))
        1.6933204261400423e-15

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.up_orthogonalize_tucker_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        5.306364476742805e-12
        >>> ind = 1
        >>> B = ux_orth.data[0][ind]
        >>> BtB = np.einsum('...abio,...abjo->...abij',B,B)
        >>> print(np.linalg.norm(BtB - np.eye(BtB.shape[-1])))
        4.2779520202910704e-15
        """
        new_tucker_cores, new_tt_cores = uniform_orthogonalization.up_orthogonalize_uniform_tucker_cores(
            *self.apply_masks_to_cores(use_jax=use_jax), use_jax=use_jax,
        )
        return UniformTuckerTensorTrain(
            new_tucker_cores, new_tt_cores,
            self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
        )

    def down_orthogonalize_tt_cores(
            self,
            use_jax: bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Outer orthogonalize TT cores, pushing remainders downward onto tucker cores below.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.down_orthogonalize_tt_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        4.767839174513546e-12
        >>> ind = 1
        >>> G = ux_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('...iaj,...ibj->...ab',G,G)-np.eye(G.shape[-2])))
        3.907103432830381e-15
        """
        new_tucker_cores, new_tt_cores = uniform_orthogonalization.down_orthogonalize_uniform_tt_cores(
            *self.apply_masks_to_cores(use_jax=use_jax), use_jax=use_jax,
        )
        return UniformTuckerTensorTrain(
            new_tucker_cores, new_tt_cores,
            self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
        )

    def left_orthogonalize_tt_cores(
            self,
            return_variation_cores: bool = False,
            use_jax: bool = False,
    ):
        """Left orthogonalize the TT cores, possibly returning variation cores as well.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.left_orthogonalize_tt_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        1.4070101740254461e-12
        >>> ind = 1
        >>> G = ux_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('iaj,iak->jk',G,G)-np.eye(G.shape[2])))
        1.707889450699257e-16

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.left_orthogonalize_tt_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        3.0778175131798327e-12
        >>> ind = 1
        >>> G = ux_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('...iaj,...iak->...jk',G,G)-np.eye(G.shape[2]))) # broadcast I
        1.1988396145496563e-15
        """
        result = orth.left_orthogonalize_tt_cores(
            self.apply_masks_to_cores(use_jax=use_jax)[1],
            return_variation_cores=return_variation_cores, use_jax=use_jax,
        )
        if return_variation_cores:
            return (
                UniformTuckerTensorTrain(
                    self.tucker_supercore, result[0],
                    self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
                ),
                result[1],
            )
        else:
            return UniformTuckerTensorTrain(
                self.tucker_supercore, result,
                self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
            )

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
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.right_orthogonalize_tt_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        7.049913893369159e-13
        >>> ind = 1
        >>> G = ux_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik',G,G)-np.eye(G.shape[-3])))
        5.60978567249119e-16

        Stacked:

        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
        >>> ux = ut3.t3_to_ut3(x)
        >>> ux_orth = ux.right_orthogonalize_tt_cores()
        >>> print(np.linalg.norm(ux.to_dense() - ux_orth.to_dense()))
        3.0648554023984285e-12
        >>> ind = 1
        >>> G = ux_orth.data[1][ind]
        >>> print(np.linalg.norm(np.einsum('...iaj,...kaj->...ik',G,G)-np.eye(G.shape[-3]))) # broadcast I
        2.4167107000621777e-15
        """
        result = orth.right_orthogonalize_tt_cores(
            self.apply_masks_to_cores(use_jax=use_jax)[1],
            return_variation_cores=return_variation_cores, use_jax=use_jax,
        )
        if return_variation_cores:
            return (
                UniformTuckerTensorTrain(
                    self.tucker_supercore, result[0],
                    self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
                ),
                result[1],
            )
        else:
            return UniformTuckerTensorTrain(
                self.tucker_supercore, result,
                self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask,
            )

    def norm(
            self,
            use_orthogonalization: bool = True,
            use_jax: bool = False,
    ):
        """Compute the Hilbert-Schmidt norm of this uniform Tucker tensor train.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2), stack_shape=(2,3))
        >>> ux = ut3.t3_to_ut3(x)
        >>> norm_ux = ux.norm()
        >>> norm_ux2 = np.einsum('...xyz->...', x.to_dense()**2)
        >>> print(np.linalg.norm(norm_ux - norm_ux2) / np.linalg.norm(norm_ux))
        1.4526456430189309e-15
        """
        return utla.ut3_norm(
            self.data, use_orthogonalization=use_orthogonalization, use_jax=use_jax,
        )



if has_jax:
    jax.tree_util.register_pytree_node(
        UniformTuckerTensorTrain,
        lambda x: (x.data, None),
        lambda aux_data, children: UniformTuckerTensorTrain(*children),
    )
    # jax.tree_util.register_pytree_node(
    #     UniformTuckerTensorTrain,
    #     lambda x: (x.data[:2], x.data[2:]), # treat masks statically
    #     lambda aux_data, children: UniformTuckerTensorTrain(*(children+aux_data)),
    # )


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
    >>> print(uniform_x.uniform_structure)
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
    >>> for x2i in all_x2: print(x2i.uniform_structure)
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1), ())
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1), ())
    >>> stacked_x2 = ut3.ut3_to_t3(uniform_x, stack_t3s=True) # with stacking
    >>> print(stacked_x2.uniform_structure)
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


def ut3_get_entries(
        x: UniformTuckerTensorTrain,
        index: NDArray, # dtype=int. shape=(d,) or shape=(num_entries,d)
        use_jax: bool = False,
) -> NDArray:
    """Compute entry (entries) of a uniform Tucker tensor train.

    If index is outside the tensor, the result is undefined.
    In this case, the function may either return a meaningless number,
    or raise an error.

    Examples
    --------
	>>> import numpy as np
	>>> import t3toolbox.tucker_tensor_train as t3
	>>> import t3toolbox.uniform_tucker_tensor_train as ut3
	>>> x = t3.t3_corewise_randn((14,15,16), (4,5,3), (1,4,2,1)) # T3
	>>> index = (3,1,2)
	>>> x_312 = t3.t3_get_entries(x, index)
	>>> print(x_312) # (3,1,2) entry from T3:
	58.91320690249439
	>>> uniform_x = ut3.t3_to_ut3(x) # Convert to Uniform T3
	>>> x_312_uniform = ut3.ut3_get_entries(uniform_x, index) # (3,1,2) entry from uniform T3:
	>>> print(x_312_uniform)
	58.91320690249439

    Multiple entries:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,3), (1,4,2,1))
    >>> index = ((3,9), (1,8), (2,7))
    >>> x_312_987 = t3.t3_get_entries(x, index)
    >>> print(x_312_987)
    [-13.31445318 -16.95641076]
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> x_312_987_uniform = ut3.ut3_get_entries(uniform_x, index)
    >>> print(x_312_987_uniform)
    [-13.31445318 -16.95641076]

    Multiple entries, multiple T3s:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,3), (1,4,2,1), stack_shape=(3,))
    >>> index = ((3,9), (1,8), (2,7))
    >>> x_312_987 = t3.t3_get_entries(x, index)
    >>> print(x_312_987)
    [[ 13.37754112 -14.2301319 ]
     [ 10.34271727   9.07781055]
     [ -3.47189513 -21.14557063]]
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> x_312_987_uniform = ut3.ut3_get_entries(uniform_x, index)
    >>> print(x_312_987_uniform)
    [[ 13.37754112 -14.2301319 ]
     [ 10.34271727   9.07781055]
     [ -3.47189513 -21.14557063]]

    # Gradient of entry getting function (Not supported)
    #
    # >>> import numpy as np
    # >>> import jax
    # >>> import jax.numpy as jnp
	# >>> import t3toolbox.tucker_tensor_train as t3
	# >>> import t3toolbox.uniform_tucker_tensor_train as ut3
	# >>> import t3toolbox.corewise as cw
	# >>> jax.config.update("jax_enable_x64", True)
	# >>> index = (3,1,2)
	# >>> get_312 = lambda z: t3.t3_get_entries(z, index, use_jax=True)
	# >>> x = t3.t3_corewise_randn((14,15,16), (4,5,3), (1,4,2,1)) # T3
	# >>> g = jax.grad(get_312)(x)
    # >>> get_312_uniform = lambda z: ut3.ut3_get_entries(z, index, use_jax=True)
    # >>> uniform_x = ut3.t3_to_ut3(x)
    # >>> uniform_g = jax.grad(get_312_uniform)(uniform_x)
    # >>> uniform_g2 = ut3.t3_to_ut3(g)
    # >>> cw.corewise_norm(cw.corewise_sub(uniform_g.data[:2], uniform_g2.data[:2]))
    # 1.418902271738168e-15
    # >>> for m, m2 in zip(uniform_g.data[2:], uniform_g2.data[2:]): print(np.all(m == m2))
    # True
    # True
    # True
    """
    masked_x = x.apply_masks_to_cores(use_jax=use_jax) # re-mask every time so that mask affects derivatives. It is relatively cheap.
    return entries.t3_get_entries(masked_x, index, use_jax=use_jax)


def ut3_apply(
        x: UniformTuckerTensorTrain,
        input_vectors: NDArray, # shape=(d,N) or shape=(...,d,N)
        use_jax: bool = False,
) -> NDArray: # shape=(d,N) or (...,d,N)
    """Apply a uniform Tucker tensor train to vectors. WORK IN PROGRESS

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> uvecs = ut3.pack_vectors(vecs)
    >>> result2 = ut3.ut3_apply(uniform_x, uvecs)
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over UT3s

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> uvecs = ut3.pack_vectors(vecs)
    >>> result2 = ut3.ut3_apply(uniform_x, uvecs)
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    masked_x = x.apply_masks_to_cores() # re-mask every time so that mask affects derivatives. It is relatively cheap.
    return apply.t3_apply(masked_x, input_vectors, use_jax=use_jax)


def probe_ut3(
        input_vectors: NDArray,  # shape=(d,N) or shape=(...,d,N)
        x: UniformTuckerTensorTrain,
        use_jax: bool = False,
) -> NDArray: # shape=(d,N) or (...,d,N)
    """Apply a uniform Tucker tensor train to vectors. WORK IN PROGRESS

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> zz = t3.probe_t3(vecs, x)
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> uvecs = ut3.pack_vectors(vecs)
    >>> uzz = ut3.probe_ut3(uvecs, uniform_x)
    >>> zz2 = ut3.unpack_vectors(uzz, uniform_x.shape)
    >>> for z, z2 in zip(zz, zz2): print(np.linalg.norm(z - z2))
    5.654425920339536e-13
    6.019471570221263e-13
    9.452355114682054e-13

    Vectorize over UT3s

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,5,6), (1,3,2,1), stack_shape=(2,3))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> zz = t3.probe_t3(vecs, x)
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> uvecs = ut3.pack_vectors(vecs)
    >>> uzz = ut3.probe_ut3(uvecs, uniform_x)
    >>> zz2 = ut3.unpack_vectors(uzz, uniform_x.shape)
    >>> for z, z2 in zip(zz, zz2): print(np.linalg.norm(z - z2))
    2.5704672147788592e-12
    1.724614542977838e-12
    2.394748346461898e-12
    """
    masked_x = x.apply_masks_to_cores() # re-mask every time so that mask affects derivatives. It is relatively cheap.
    return probing.probe_t3(input_vectors, masked_x, use_jax=use_jax)


pack_vectors = uniform_ops.pack_vectors
unpack_vectors = uniform_ops.unpack_vectors


def ut3_add(
        x: UniformTuckerTensorTrain,
        y: UniformTuckerTensorTrain,
        squash: bool = True,
        use_jax: bool = False,
) -> UniformTuckerTensorTrain: # z = x + y
    """Add two UniformTuckerTensorTrains, x,y -> x+y.

    Parameters
    ----------
    x_cores: UniformTuckerTensorTrainCores
        First summand cores
    x_masks: UniformTuckerTensorTrainMasks
        First summand masks
    y_cores: UniformTuckerTensorTrainCores
        Second summand cores
    y_masks: UniformTuckerTensorTrainMasks
        Second summand masks
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    UniformTuckerTensorTrainCores
        Cores for sum, x+y
    UniformTuckerTensorTrainMasks
        Cores for sum x+y

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2), stack_shape=(2,3))
    >>> ux = ut3.t3_to_ut3(x)
    >>> y = t3.t3_corewise_randn((14,15,16), (6,7,8), (3,5,6,1), stack_shape=(2,3))
    >>> uy = ut3.t3_to_ut3(y)
    >>> ux_plus_uy = ut3.ut3_add(ux, uy) # add x+y
    >>> print(np.linalg.norm(x.to_dense() + y.to_dense() - ux_plus_uy.to_dense()))
    3.250578545971108e-12
    """
    if x.d != y.d:
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent d.\n' +
            str(x.d) + ' = x.d != y.d = ' + str(y.d)
        )

    if (x.shape != y.shape).any():
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent shapes.\n' +
            str(x.shape) + ' = x.shape != y.shape = ' + str(y.shape)
        )

    if x.N != y.N:
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent N.\n' +
            str(x.N) + ' = x.N != y.N = ' + str(y.N)
        )

    if x.stack_shape != y.stack_shape:
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent stack_shapes.\n' +
            str(x.stack_shape) + ' = x.stack_shape != y.stack_shape = ' + str(y.stack_shape)
        )

    x_plus_y = UniformTuckerTensorTrain(*utla.ut3_add(x.data, y.data, use_jax=use_jax))
    if squash:
        x_plus_y = x_plus_y.squash_tails(use_jax=use_jax)

    return x_plus_y


def ut3_sub(
        x: UniformTuckerTensorTrain,
        y: UniformTuckerTensorTrain,
        squash: bool = True,
        use_jax: bool = False,
) -> UniformTuckerTensorTrain: # z = x + y
    """Subtract two UniformTuckerTensorTrains, x,y -> x-y.

    Parameters
    ----------
    x_cores: UniformTuckerTensorTrainCores
        First summand cores
    x_masks: UniformTuckerTensorTrainMasks
        First summand masks
    y_cores: UniformTuckerTensorTrainCores
        Second summand cores
    y_masks: UniformTuckerTensorTrainMasks
        Second summand masks
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    UniformTuckerTensorTrainCores
        Cores for sum, x+y
    UniformTuckerTensorTrainMasks
        Cores for sum x+y

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2), stack_shape=(2,3))
    >>> ux = ut3.t3_to_ut3(x)
    >>> y = t3.t3_corewise_randn((14,15,16), (6,7,8), (3,5,6,1), stack_shape=(2,3))
    >>> uy = ut3.t3_to_ut3(y)
    >>> ux_minus_uy = ut3.ut3_sub(ux, uy)
    >>> print(np.linalg.norm(x.to_dense() - y.to_dense() - ux_minus_uy.to_dense()))
    1.7975763647128273e-12
    """
    return ut3_add(x, -y, squash=squash, use_jax=use_jax)


def ut3_inner_product(
        x: UniformTuckerTensorTrain,
        y: UniformTuckerTensorTrain,
        use_orthogonalization: bool = True,
        use_jax: bool = False,
):
    """Compute the Hilbert-Schmidt inner product of two uniform Tucker tensor trains.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn((14,15,16), (4,6,5), (2,3,2,2), stack_shape=(2,3))
    >>> ux = ut3.t3_to_ut3(x)
    >>> y = t3.t3_corewise_randn((14,15,16), (6,7,8), (3,5,6,1), stack_shape=(2,3))
    >>> uy = ut3.t3_to_ut3(y)
    >>> ux_dot_uy = ut3.ut3_inner_product(ux, uy)
    >>> ux_dot_uy2 = np.einsum('...xyz,...xyz->...', x.to_dense(), y.to_dense())
    >>> print(np.linalg.norm(ux_dot_uy - ux_dot_uy2) / np.linalg.norm(ux_dot_uy))
    7.667494312151743e-15
    """
    return utla.ut3_inner_product(
        x.data, y.data, use_orthogonalization=use_orthogonalization, use_jax=use_jax,
    )

def uniform_t3_svd(
        cores: typ.Tuple[
            NDArray, # tucker_supercore
            NDArray, # tt_supercore
        ],
        rank_truncation_masks: typ.Tuple[
            NDArray, # shape_mask
            NDArray, # tucker_edge_mask
            NDArray, # tt_edge_mask
        ], # Can be used to truncate rank. Do not have to be the original masks
        squash_tails_first: bool = True,
        use_jax: bool = False,
) -> typ.Tuple[
    UniformTuckerTensorTrain, # new_x
    NDArray, # basis_singular_values, shape=(d, n)
    NDArray, # tt_singular_values, shape=(d+1, r)
]:
    """Compute T3-SVD of uniform Tucker tensor train.

    Masks are used for rank truncation. If the provided mask does not have minimal ranks,
    this function will create minimal rank masks from it and use those.

    Examples
    --------
    >>> import numpy as np
    >>> from t3toolbox.core.tucker_tensor_train.uniform.uniform_t3_operations import make_uniform_masks
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> shape, tucker_ranks, tt_ranks = (11,12,13), (6,7,5), (1,3,6,2)
    >>> min_tucker_ranks, min_tt_ranks = t3.compute_minimal_t3_ranks(shape, tucker_ranks, tt_ranks)
    >>> x = t3.t3_corewise_randn(shape, tucker_ranks, tt_ranks)
    >>> ux = ut3.t3_to_ut3(x)
    >>> min_masks = make_uniform_masks(shape, min_tucker_ranks, min_tt_ranks, ux.stack_shape, ux.N, ux.n, ux.r)
    >>> ux2, ss_tucker_from_ut3, ss_tt_from_ut3 = ut3.uniform_t3_svd((ux.tucker_supercore, ux.tt_supercore), min_masks) # Uniform T3-SVD
    >>> print(np.linalg.norm(ux2.to_dense() - x.to_dense()))
    1.2664289217892565e-11
    >>> print(ux2.structure)
    ((11, 12, 13), array([3, 7, 5]), array([1, 3, 5, 1]), ())
    >>> _, ss_tucker, ss_tt = t3.t3_svd(x) # Non-uniform T3-SVD
    >>> print(ss_tt[1])
    [2271.96541132 2004.56681783  471.59876959]
    >>> print(ss_tt_from_ut3[1])
    [2271.96541132 2004.56681783  471.59876959    0.            0.            0.        ]
    >>> print(ss_tucker[0])
    [2271.96541132 2004.56681783  471.59876959]
    >>> print(ss_tucker_from_ut3[0])
    [2271.96541132 2004.56681783  471.59876959    0.            0.            0.            0.        ]

    Using stacking:

    >>> import numpy as np
    >>> from t3toolbox.core.tucker_tensor_train.uniform.uniform_t3_operations import make_uniform_masks
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> shape, tucker_ranks, tt_ranks, stack_shape = (11,12,13), (6,7,5), (1,3,6,2), (2,)
    >>> min_tucker_ranks, min_tt_ranks = t3.compute_minimal_t3_ranks(shape, tucker_ranks, tt_ranks)
    >>> x = t3.t3_corewise_randn(shape, tucker_ranks, tt_ranks, stack_shape=stack_shape)
    >>> ux = ut3.t3_to_ut3(x)
    >>> min_masks = make_uniform_masks(shape, min_tucker_ranks, min_tt_ranks, ux.stack_shape, ux.N, ux.n, ux.r)
    >>> ux2, ss_tucker_from_ut3, ss_tt_from_ut3 = ut3.uniform_t3_svd((ux.tucker_supercore, ux.tt_supercore), min_masks) # Uniform T3-SVD
    >>> print(np.linalg.norm(ux2.to_dense() - x.to_dense()))
    2.193805472670695e-11
    >>> print(ux2.tucker_ranks)
    [[3 3]
     [7 7]
     [5 5]]
    >>> print(ux2.tt_ranks)
    [[1 1]
     [3 3]
     [5 5]
     [1 1]]
    >>> xx = x.unstack()
    >>> _, ss_tucker_a, ss_tt_a = t3.t3_svd(xx[0]) # Non-uniform T3-SVD of first T3 in stack
    >>> _, ss_tucker_b, ss_tt_b = t3.t3_svd(xx[1]) # Second T3 in stack
    >>> print(str(ss_tt_a[1]) + '\n' + str(ss_tt_b[1]))
    [3522.08053706  986.93239042  360.30264481]
    [4185.51756339 2423.23109837 1564.5114264 ]
    >>> print(ss_tt_from_ut3[1])
    [[3522.08053706  986.93239042  360.30264481    0.            0.            0.        ]
     [4185.51756339 2423.23109837 1564.5114264     0.            0.            0.        ]]
    >>> print(str(ss_tucker_a[0]) + '\n' + str(ss_tucker_b[0]))
    [3522.08053706  986.93239042  360.30264481]
    [4185.51756339 2423.23109837 1564.5114264 ]
    >>> print(ss_tucker_from_ut3[0])
    [[3522.08053706  986.93239042  360.30264481    0.            0.            0.            0.        ]
     [4185.51756339 2423.23109837 1564.5114264     0.            0.            0.            0.        ]]
    """
    new_cores, tucker_singular_values, tt_singular_values  = ut3svd.uniform_t3_svd(cores, rank_truncation_masks, use_jax=use_jax)
    return UniformTuckerTensorTrain(*(new_cores + rank_truncation_masks)), tucker_singular_values, tt_singular_values

