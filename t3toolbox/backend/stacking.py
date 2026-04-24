# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

from t3toolbox.backend.common import *

__all__ = [
    'trees_have_same_structure',
    'apply_func_to_leaf_subtrees',
    'stack_arrays_at_lowest_level_of_trees',
    # 'first_elm_k_levels_deep',
    'stack',
    'unstack',
    'sum_stack',
]


# def first_elm_k_levels_deep(xx, k):
#     """Returns first element k levels deep in nested sequence.
#
#     first_elm_k_levels_deep(xx, 0) = xx
#     first_elm_k_levels_deep(xx, 1) = xx[0]
#     first_elm_k_levels_deep(xx, 2) = xx[0][0]
#     ...
#     """
#     if k < 0:
#         raise ValueError(str(k) + ' = k < 0.')
#     elif k == 0:
#         return xx
#     else:
#         return first_elm_k_levels_deep(xx[0], k-1)
#
#
# def stack_leaf_sequence(LL, use_jax: bool = False,):
#     xnp, _, _ = get_backend(False, use_jax)
#
#     if is_ndarray(LL[0]):
#         return xnp.stack(LL)
#     else:
#         pass


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


def stack_arrays_at_lowest_level_of_trees(
        sequence_of_trees,
        axis: int = 0,
        use_jax: bool = False,
):
    """Given a sequence of trees which have the same structure, with NDArrays at the leafs,
    construct the tree which has the same structure, but has leafs given by stacking the
    leafs of the input trees.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> T1 = (np.array([1,2,3]),   (np.array([[4,5],   [6,7]]),   np.array(8)))
    >>> T2 = (np.array([9,10,11]), (np.array([[12,13], [14,15]]), np.array(16)))
    >>> (a, (b, c)) = stacking.stack_arrays_at_lowest_level_of_trees((T1, T2))
    >>> print(a)
    [[ 1  2  3]
     [ 9 10 11]]
    >>> print(b[0,:,:])
    [[4 5]
     [6 7]]
    >>> print(b[1,:,:])
    [[12 13]
     [14 15]]
    >>> print(c)
    [ 8 16]
    """
    xnp, _, _ = get_backend(False, use_jax)

    if not isinstance(sequence_of_trees[0], typ.Sequence):
        return xnp.stack(sequence_of_trees, axis=axis)
    else:
        N = len(sequence_of_trees[0])
        sequence_of_subtrees = [[x[ii] for x in sequence_of_trees] for ii in range(N)]
        return tuple([
            stack_arrays_at_lowest_level_of_trees(x, axis=axis, use_jax=use_jax)
            for x in sequence_of_subtrees
        ])



def stack(
        xx, # array-like structure of nested tuples containing arrays
        leaf_structure: typ.Tuple[typ.Union[typ.Tuple, None],...], # tree structure of a leaf subtree
        stacking_axis: int = 0,
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
    >>> LS = (None, (None, None))
    >>> (a, (b, c)) = stacking.stack(T, LS)
    >>> np.linalg.norm(a - np.array([[a00, a01], [a10, a11]]))
    0.0
    >>> np.linalg.norm(b - np.array([[b00, b01], [b10, b11]]))
    0.0
    >>> np.linalg.norm(c - np.array([[c00, c01], [c10, c11]]))
    0.0

    Stacking when there is only one, non-nested, object

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> a, b, c = randn(3), randn(4,5), randn()
    >>> T = (a, (b, c))
    >>> LS = (None, (None, None))
    >>> (a2, (b2, c2)) = stacking.stack(T, LS)
    >>> print(np.linalg.norm(a - a2))
    0.0
    >>> print(np.linalg.norm(b - b2))
    0.0
    >>> print(np.linalg.norm(c - c2))
    0.0

    Stack nothing

    >>> import numpy as np
    >>> import t3toolbox.backend.stacking as stacking
    >>> randn = np.random.randn
    >>> T = ()
    >>> LS = ()
    >>> print(stacking.stack(T, LS))
    ()
    """
    xnp,_,_ = get_backend(False, use_jax)

    if trees_have_same_structure(xx, leaf_structure):
        return xx
    else:
        stacked_subtrees = tuple([stack(x, leaf_structure, stacking_axis=stacking_axis, use_jax=use_jax) for x in xx])
        return stack_arrays_at_lowest_level_of_trees(
            stacked_subtrees, axis=stacking_axis, use_jax=use_jax,
        )


def unstack(
        x: typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]],  # (tucker_cores, tt_cores)
): # returns an array-like structure of nested tuples containing Tucker tensor trains
    """Given multiple stacked T3s, this unstacks them
    into an array-like structure of nested tuples with the same "shape" as the stacking shape.
    """
    tucker_cores, tt_cores = x
    stack_shape = tucker_cores[0].shape[:-2]

    if not stack_shape:
        return x

    n = tucker_cores[0].shape[0]
    unstacked_x = []
    for ii in range(n):
        BB = tuple([B[ii] for B in tucker_cores])
        GG = tuple([G[ii] for G in tt_cores])
        xi = (BB, GG)
        unstacked_x.append(t3_unstack(xi))

    return tuple(unstacked_x)


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

