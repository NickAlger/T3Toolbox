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
    tucker_supercore:   NDArray  #             shape=(d,)   + stack_shape + (n,N)
    tt_supercore:       NDArray  #             shape=(d+1,) + stack_shape + (r,n,r)
    shape_mask:         NDArray  # dtype=bool, shape=(d,)   + stack_shape + (N,)
    tucker_edge_mask:   NDArray  # dtype=bool, shape=(d,)   + stack_shape + (n,)
    tt_edge_mask:       NDArray  # dtype=bool, shape=(d+1,) + stack_shape + (r,)

    @ft.cached_property
    def d(self) -> int:
        return self.tucker_supercore.shape[0]

    @ft.cached_property
    def n(self) -> int:
        return self.tucker_supercore.shape[-2]

    @ft.cached_property
    def N(self) -> int:
        return self.tucker_supercore.shape[-1]

    @ft.cached_property
    def r(self) -> int:
        return self.tt_supercore.shape[-1]

    @ft.cached_property
    def stack_shape(self) -> typ.Tuple[int,...]:
        return self.tucker_supercore.shape[1:-2]

    @ft.cached_property
    def shape(self) -> NDArray: # dtype=int, shape=(d,)+stack_shape
        """Get the original shapes, not including portions of the tensors that are ignored by masking.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d+1,)+stack_shape+(r,n,r))
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
        """Get the original tucker ranks, not including components of the edges that are ignored by masking.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d+1,)+stack_shape+(r,n,r))
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
        """Get the original tucker ranks, not including components of the edges that are ignored by masking.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> d, N, n, r = 3, 6, 5, 4
        >>> stack_shape = (2,)
        >>> tucker_supercore = np.ones((d,)+stack_shape+(n,N))
        >>> tt_supercore = np.ones((d+1,)+stack_shape+(r,n,r))
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
        assert(self.tt_supercore.shape == (self.d+1,)+self.stack_shape+(self.r, self.n, self.r))
        assert(self.shape_mask.shape == (self.d,)+self.stack_shape+(self.N,))
        assert(self.tucker_edge_mask.shape == (self.d,)+self.stack_shape+(self.n,))
        assert(self.tt_edge_mask.shape == (self.d+1,)+self.stack_shape+(self.r,))

    def __post_init__(self):
        self.validate()

