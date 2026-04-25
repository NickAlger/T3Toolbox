# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'tree_depth',
    'get_first_leaf',
    'trees_have_same_structure',
    'apply_func_to_leaf_subtrees',
    'stack',
    'unstack',
    'sum_stack',
]

def tree_depth(t):
    if not isinstance(t, typ.Sequence):
        return 0
    return tree_depth(t[0])+1

def get_first_leaf(xx):
    if not isinstance(xx, typ.Sequence):
        return xx
    return get_first_leaf(xx[0])


def trees_have_same_structure(
        tree1, # array-like structure of nested tuples
        tree2, # target structure
):
    """Checks if two trees (nested sequences) have the same structure.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T = ((1, (2,3)),4,(5,6,7))
    >>> LS = ((None, (None, None)), None, (None, None, None))
    >>> stacking.trees_have_same_structure(T, LS)
    True

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T = ((1, (2,3)),4,(5,6,7))
    >>> LS = ((None, (None)), None, (None, None, None))
    >>> stacking.trees_have_same_structure(T, LS)
    False

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T = ((1, (2,3)),4,(5,6,7), ())
    >>> LS = ((None, (None)), None, (None, None, None))
    >>> stacking.trees_have_same_structure(T, LS)
    False

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T1 = ((1, (2,3)),4,(5,6,7))
    >>> T2 = ((8, (9,10)),11,(12,13,14))
    >>> T = (T1, T2)
    >>> LS = ((None, (None, None)), None, (None, None, None))
    >>> stacking.trees_have_same_structure(T, LS)
    False

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T = ()
    >>> LS = ()
    >>> stacking.trees_have_same_structure(T, LS)
    True

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T = (1,2,3)
    >>> LS = (4,5,6)
    >>> stacking.trees_have_same_structure(T, LS)
    True

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T = (1,2,3)
    >>> LS = (4,5,6,7)
    >>> stacking.trees_have_same_structure(T, LS)
    False
    """
    if not isinstance(tree1, typ.Sequence): # t1 is a leaf -> t2 must be a leaf
        return not isinstance(tree2, typ.Sequence)
    elif not isinstance(tree2, typ.Sequence): # t2 is a leaf -> t1 must be a leaf
        return False
    else: # recurse subtrees
        if len(tree1) != len(tree2):
            return False
        return all([trees_have_same_structure(sub1, sub2) for sub1, sub2 in zip(tree1, tree2)])


def apply_func_to_leaf_subtrees(
        tree,
        func: typ.Callable, # function to be applied to all leafs
        leaf_structure, # tree structure of a leaf
):
    """Apply a function to all "leafs" in a tree.
    A "leaf" is, itself, a subtree with the structure given in leaft_structure.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> func = lambda x: (x[0] - x[1][0], x[0] + x[1][1]) # (a,(b,c)) -> (a-b, a+c)
    >>> LS = (None, (None, None))
    >>> T1 = (1, (2, 3))
    >>> T2 = (4, (5, 6))
    >>> T3 = (7, (8, 9))
    >>> T = ((T1, T2), ((T3,),))
    >>> print(stacking.apply_func_to_leaf_subtrees(T, func, LS))
    (((-1, 4), (-1, 10)), (((-1, 16),),))
    """
    if trees_have_same_structure(tree, leaf_structure):
        return func(tree)
    else:
        return tuple([apply_func_to_leaf_subtrees(x, func, leaf_structure) for x in tree])


def stack(
        T,
        axes,
        use_jax: bool = False,
):
    """Stack array-like nested tree structure.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> a00, a01, a10, a11 = randn(3), randn(3), randn(3), randn(3)
    >>> b00, b01, b10, b11 = randn(4,5), randn(4,5), randn(4,5), randn(4,5)
    >>> c00, c01, c10, c11 = randn(), randn(), randn(), randn()
    >>> T00 = (a00, (b00, c00))
    >>> T01 = (a01, (b01, c01))
    >>> T10 = (a10, (b10, c10))
    >>> T11 = (a11, (b11, c11))
    >>> T = ((T00, T01), (T10, T11))
    >>> (a, (b, c)) = stacking.stack(T, axes=(0,1))
    >>> np.linalg.norm(a - np.array([[a00, a01], [a10, a11]]))
    0.0
    >>> np.linalg.norm(b - np.array([[b00, b01], [b10, b11]]))
    0.0
    >>> np.linalg.norm(c - np.array([[c00, c01], [c10, c11]]))
    0.0

    Stacking along different axes

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> a00, a01, a10, a11 = randn(3,2,1), randn(3,2,1), randn(3,2,1), randn(3,2,1)
    >>> b00, b01, b10, b11 = randn(4,5,6,9), randn(4,5,6,9), randn(4,5,6,9), randn(4,5,6,9)
    >>> c00, c01, c10, c11 = randn(7,8), randn(7,8), randn(7,8), randn(7,8)
    >>> T00 = (a00, (b00, c00))
    >>> T01 = (a01, (b01, c01))
    >>> T10 = (a10, (b10, c10))
    >>> T11 = (a11, (b11, c11))
    >>> T = ((T00, T01), (T10, T11))
    >>> (a, (b, c)) = stacking.stack(T, axes=(1,2))
    >>> np.linalg.norm(a - np.moveaxis(np.array([[a00, a01], [a10, a11]]), 2, 0))
    0.0
    >>> np.linalg.norm(b - np.moveaxis(np.array([[b00, b01], [b10, b11]]), 2, 0))
    0.0
    >>> np.linalg.norm(c - np.moveaxis(np.array([[c00, c01], [c10, c11]]), 2, 0))
    0.0

    Stacking when there is only one, non-nested, object

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> a, b, c = randn(3), randn(4,5), randn()
    >>> T = (a, (b, c))
    >>> (a2, (b2, c2)) = stacking.stack(T, ())
    >>> print(np.linalg.norm(a - a2))
    0.0
    >>> print(np.linalg.norm(b - b2))
    0.0
    >>> print(np.linalg.norm(c - c2))
    0.0

    Stack non-nested single array

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> T = randn(3)
    >>> LS = None
    >>> T2 = stacking.stack(T, ())
    >>> print(np.linalg.norm(T - T2))
    0.0

    Stack nothing

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> T = ()
    >>> LS = ()
    >>> print(stacking.stack(T, ()))
    ()
    """
    xnp, _, _ = get_backend(False, use_jax)

    # 1. Drill down to find the 'template' of the original structure
    def get_template(obj, depth):
        if depth == 0: return obj
        return get_template(obj[0], depth - 1)

    num_stacking_levels = len(axes)
    template = get_template(T, num_stacking_levels)

    # 2. Recursive function to rebuild the structure
    def reconstruct(template_node, path_to_leaf):
        if not isinstance(template_node, typ.Sequence):
            # Collect the sliced pieces from across the 'unstacked' tree
            def collect(current_tree, current_depth):
                if current_depth == num_stacking_levels:
                    # Navigate the original tree structure to find this specific leaf
                    node = current_tree
                    for step in path_to_leaf:
                        node = node[step]
                    return node
                return [collect(branch, current_depth + 1) for branch in current_tree]

            # xnp.array() nests the lists, putting stacking axes at the front
            stacked = xnp.array(collect(T, 0))

            # Move axes from the front (0, 1, ...) to their target positions
            stacked = xnp.moveaxis(stacked, source=xnp.arange(len(axes)), destination=axes)

            # Ensure the final array is in contiguous memory order
            return xnp.ascontiguousarray(stacked)

        if isinstance(template_node, typ.Sequence):
            return tuple(reconstruct(template_node[i], path_to_leaf + [i])
                         for i in range(len(template_node)))
        return template_node

    return reconstruct(template, [])



def unstack(
        S,
        axes,
):
    """Unstack nested sequence of arrays along specificed array axes.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> A = np.random.randn(4, 2,3, 5,6)
    >>> B = np.random.randn(7, 2,3, 8)
    >>> C = np.random.randn(9, 2,3)
    >>> S = ((A, B), C)
    >>> T = stacking.unstack(S, axes=(1,2))
    >>> ii, jj = 1, 2
    >>> ((Aij, Bij), Cij) = T[ii][jj]
    >>> print(np.linalg.norm(Aij - A[:,ii,jj,:,:]))
    0.0
    >>> print(np.linalg.norm(Bij - B[:,ii,jj,:]))
    0.0
    >>> print(np.linalg.norm(Cij - C[:,ii,jj]))
    0.0

    Stack then unstack:

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> import t3toolbox.corewise as cw
    >>> randn = np.random.randn
    >>> a00, a01, a10, a11 = randn(3,2), randn(3,2), randn(3,2), randn(3,2)
    >>> b00, b01, b10, b11 = randn(4,5), randn(4,5), randn(4,5), randn(4,5)
    >>> c00, c01, c10, c11 = randn(7), randn(7), randn(7), randn(7)
    >>> T00 = (a00, (b00, c00))
    >>> T01 = (a01, (b01, c01))
    >>> T10 = (a10, (b10, c10))
    >>> T11 = (a11, (b11, c11))
    >>> T = ((T00, T01), (T10, T11))
    >>> S = stacking.stack(T, axes=(0,2))
    >>> T2 = stacking.unstack(S, axes=(0,2))
    >>> print(cw.corewise_norm(cw.corewise_sub(T, T2)))
    0.0

    Unstack then stack:

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> A = np.random.randn(4, 2,3, 5,6)
    >>> B = np.random.randn(7, 2,3, 8)
    >>> C = np.random.randn(9, 2,3)
    >>> S = ((A, B), C)
    >>> T = stacking.unstack(S, axes=(1,2))
    >>> ii, jj = 1, 2
    >>> S2 = stacking.stack(T, axes=(1,2))
    >>> ((A2, B2), C2) = S2
    >>> print(np.linalg.norm(A - A2))
    0.0
    >>> print(np.linalg.norm(B - B2))
    0.0
    >>> print(np.linalg.norm(C - C2))
    0.0

    When there are no axes to unstack:

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> A = np.random.randn(4, 5, 6)
    >>> B = np.random.randn(7, 8)
    >>> C = np.random.randn(9)
    >>> S = ((A, B), C)
    >>> T = stacking.unstack(S, axes=())
    >>> ((A2, B2), C2) = T
    >>> print(np.linalg.norm(A2 - A))
    0.0
    >>> print(np.linalg.norm(B2 - B))
    0.0
    >>> print(np.linalg.norm(C2 - C))
    0.0

    When the tree is a single object:

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> A = np.random.randn(4, 2,3, 5,6)
    >>> T = stacking.unstack(A, axes=(1,2))
    >>> ii, jj = 1, 2
    >>> Aij = T[ii][jj]
    >>> print(np.linalg.norm(Aij - A[:, ii, jj, :]))
    0.0

    When the tree is a single object and there are no objects to unstack

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> A = np.random.randn(4, 5,6)
    >>> T = stacking.unstack(A, axes=())
    >>> print(np.linalg.norm(A - T))
    0.0
    """
    def get_stack_info(x):
        if is_ndarray(x):
            return x.shape, x.ndim
        for item in x:
            res = get_stack_info(item)
            if res: return res

    full_shape, ndim = get_stack_info(S)

    # Normalize axes: convert negative integers to positive equivalents
    norm_axes = [ax if ax >= 0 else ax + ndim for ax in axes]
    stack_shape = [full_shape[ax] for ax in norm_axes]

    print('norm_axes=', norm_axes)
    print('stack_shape=', stack_shape)

    def slice_leaves(obj, current_indices):
        if is_ndarray(obj):
            idx = [slice(None)] * obj.ndim
            for ax, val in zip(norm_axes, current_indices):
                idx[ax] = val
            return obj[tuple(idx)]
        if isinstance(obj, (list, tuple)):
            return tuple(slice_leaves(item, current_indices) for item in obj)
        return obj

    def build_tree(dim_idx, current_indices):
        if dim_idx == len(axes):
            return slice_leaves(S, current_indices)
        return tuple(build_tree(dim_idx + 1, current_indices + [ii]) for ii in range(stack_shape[dim_idx]))

    return build_tree(0, [])



def sum_stack(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
        use_jax: bool=False,
) -> typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]:  # (summed_tucker_cores, summed_tt_cores)
    """If this object contains multiple stacked T3s, this sums them.
    """
    xnp, _, _ = get_backend(False, use_jax=use_jax)
    tucker_cores, tt_cores = x
    vsv = tucker_cores[0].shape[:-2]
    N_vsv = np.prod(vsv, dtype=int)

    summed_tucker_cores = []
    for B in tucker_cores:
        B_sum = xnp.sum(B.reshape((N_vsv,) + B.shape[-2:]), axis=0)
        summed_tucker_cores.append(B_sum)

    summed_tt_cores = []
    for G in tt_cores:
        G_sum = xnp.sum(G.reshape((N_vsv,) + G.shape[-3:]), axis=0)
        summed_tt_cores.append(G_sum)

    return tuple(summed_tucker_cores), tuple(summed_tt_cores)

