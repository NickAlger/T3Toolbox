# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import numpy as np
import typing as typ

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.common as common
import t3toolbox.base_variation_format as bvf
from t3toolbox.common import *

__all__ = [
    'UniformTuckerTensorTrainCores',
    'UniformEdgeWeights',
    'UniformT3Base',
    'UniformT3Variation',
    'UniformBVEdgeWeights',
    'UniformBVStructure',
    #
    'get_uniform_base_structure',
    'get_uniform_base_hole_shapes',
    'check_ut3',
    'get_uniform_structure',
    'get_original_structure',
    'pack_tensors',
    'unpack',
    'make_uniform_masks',
    'apply_masks',
    'apply_masks_to_base',
    'apply_masks_to_variation',
    'uniform_squash_tails',
    'uniform_randn',
    'uniform_zeros',
    #
    't3_to_ut3',
    'ut3_to_t3',
    #
    'bv_to_ubv',
    # 'ubv_to_bv',
    #
    'ut3_to_dense',
    'are_ut3_ranks_minimal',
    'ut3_entry',
    # Linear algebra operations:
    'ut3_add',
    'ut3_scale',
    'ut3_neg',
    'ut3_sub',
]


###################################################
########    Uniform Tucker Tensor Train    ########
###################################################

UniformTuckerTensorTrainCores = typ.Tuple[
    NDArray, # tucker_supercore, shape=(d, n, N)
    NDArray, # tt_supercore, shape=(d, r, n, r)
]
"""
Tuple containing supercores of a Uniform Tucker tensor train

Uniform Tucker tensor trains are created by padding a Tucker tensor train 
so that the ranks are uniform, then stacking the TT cores and Tucker cores into
"supercores", which have one more dimension.

Padding may be tracked with boolean mask arrays associated with the edges.

See Also
--------
UniformTuckerTensorTrainMasks

**Components**
    - tucker_supercore:      NDArray. shape=(d, n, N).       Stacked padded tucker cores
    - tt_supercore:         NDArray. shape=(d, r, n, r).    Stacked padded TT-cores
    
Here:
    - d = num_cores
    - N = padded_shape
    - n = padded_tucker_rank
    - r = padded_tt_rank

Examples
--------
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> import t3toolbox.uniform_tucker_tensor_train as ut3
>>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
>>> cores, masks = ut3.t3_to_ut3(x)
>>> print(ut3.get_padded_structure(cores))
(3, 16, 6, 3)
>>> print(ut3.original_structure(masks))
((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
"""


UniformStructure = typ.Tuple[
    int, # d, num_cores
    int, # N, padded_index_size
    int, # n, padded_tucker_rank
    int, # r, padded_tt_rank
]
"""Tuple containing structure of a uniform Tucker tensor train.

**Components**
    - d: int, num_cores
    - N: int, padded_index_size
    - n: int, padded_tucker_rank
    - r: int, padded_tt_rank
"""


UniformEdgeWeights = typ.Tuple[
    NDArray,  # shape_masks, shape=(d,N)
    NDArray,  # tucker_masks, shape=(d, n)
    NDArray,  # tt_masks, shape=(d+1, r)
]
"""
Tuple containing edge masks for a Uniform Tucker tensor train. Often used for masking.

See Also
--------
UniformTuckerTensorTrainCores

**Components**
    - shape_masks:  NDArray. shape=(d,N).   Weights for edges between Tucker cores and exterior of tensor
    - tucker_masks: NDArray. shape=(d,n).   Weights for edges between Tucker cores and adjacent TT-cores
    - tt_masks:     NDArray. shape=(d+1,r). Weights for edges between adjacent TT-cores

Here:
    - d = num_cores
    - N = padded_shape
    - n = padded_tucker_rank
    - r = padded_tt_rank

Examples
--------
>>> import numpy as np
>>> import t3toolbox.tucker_tensor_train as t3
>>> import t3toolbox.uniform_tucker_tensor_train as ut3
>>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
>>> cores, masks = ut3.t3_to_ut3(x)
>>> print(ut3.get_padded_structure(cores))
(3, 16, 6, 3)
>>> print(ut3.original_structure(masks))
((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
"""

UniformT3Base = typ.Tuple[
    NDArray,  # up_tucker_supercore. shape=(d, n, N),      up orthogonal elements
    NDArray,  # left_tt_supercore.   shape=(d, rL, n, rR), left orthogonal elements
    NDArray,  # right_tt_supercore.  shape=(d, rL, n, rR), right orthogonal elements
    NDArray,  # outer_tt_supercores. shape=(d, rL, n, rR), outer orthogonal elements
]
"""
Base supercores for base-variation representation of Uniform Tucker tensor train.

The components of T3Base are the "base cores":
    - up_tucker_cores: NDArray, shape=(d, nU, N),    up orthogonal elements 
    - left_tt_cores:   NDArray, shape=(d, r, nU, r), left orthogonal elements
    - right_tt_cores:  NDArray, shape=(d, r, nU, r), right orthogonal elements
    - outer_tt_cores:  NDArray, shape=(d, r, nO, r), outer orthogonal elements

See Also
--------
t3toolbox.tucker_tensor_train.T3Base
UniformT3Variation
"""

UniformT3Variation = typ.Tuple[
    NDArray,  # var_tucker_supercore.
    NDArray,  # var_tt_supercore.
]
"""
Variation supercores for base-variation representation of Uniform Tucker tensor train.

*Components*
    - var_tucker_supercore: NDArray, shape=(d,nO,N)
    - var_tt_supercore:     NDArray, shape=(d,r,nU,r)

The variation components should fit in the "holes" of a UniformT3Variation.

See Also
--------
t3toolbox.tucker_tensor_train.T3Variation
UniformT3Base
"""


UniformBVEdgeWeights = typ.Tuple[
    NDArray,  # shape_weights, shape=(d,Ni)
    NDArray,  # up_tucker_weights, shape=(d,nUi,)
    NDArray,  # outer_tucker_weights, shape=(d,nOi)
    NDArray,  # left_tt_weights, len=d, shape=(d+1,rLi,)
    NDArray,  # right_tt_weights, len=d, shape=(d+1,rRi)
]
"""Edge weights for base-variation format.

*Components*
    - shape_weights:        NDArray, shape=(d,Ni)
    - up_tucker_weights:    NDArray, shape=(d,nUi)
    - outer_tucker_weights: NDArray, shape=(d,nOi)
    - left_tt_weights:      NDArray, shape=(d,rLi)
    - right_tt_weights:     NDArray, shape=(d,rRi)

See Also
--------
t3tools.tucker_tensor_train.EdgeWeights
t3tools.base_variation_format.BVEdgeWeights
UniformT3Variation
UniformT3Base
"""


UniformBVStructure = typ.Tuple[
    int, # d
    int, # N
    int, # nU
    int, # NO
    int, # rL
    int, # rR
]
"""Shape and rank structure of a uniform base-variation T3 representation.

*Components*
    - d: int, number of tensor indices
    - N: size of external indices
    - nU: up tucker rank
    - nO: outer tucker rank
    - rL: left TT rank
    - rR: right TT rank

See Also
--------
UniformT3Base
UniformT3Variation
"""


def get_uniform_base_structure(
        base: UniformT3Base,
) -> UniformBVStructure:
    """Get the structore of a uniform base.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.uniform as ut3
    >>> d, N, nU, nO, rL, rR = 4, 10, 5, 4, 3, 2
    >>> up_tucker_supercore = np.random.randn(d, nU, N)
    >>> left_tt_supercore = np.random.randn(d, rL, nU, rL)
    >>> right_tt_supercore = np.random.randn(d, rR, nU, rR)
    >>> outer_tt_supercore = np.random.randn(d, rL, nO, rR)
    >>> base = (up_tucker_supercore, left_tt_supercore, right_tt_supercore, outer_tt_supercore)
    >>> print(ut3.get_uniform_base_structure(base))
    (4, 10, 5, 4, 3, 2)
    """
    up_tucker_supercore, left_tt_supercore, right_tt_supercore, outer_tt_supercore = base
    d, nU, N = up_tucker_supercore.shape
    _, rL, _, _ = left_tt_supercore.shape
    _, rR, _, _ = right_tt_supercore.shape
    _, _, nO, _ = outer_tt_supercore.shape
    return d, N, nU, nO, rL, rR


def get_uniform_base_hole_shapes(
        base: UniformT3Base,
) -> typ.Tuple[
    typ.Tuple[int, int, int], # (d, nO, N)
    typ.Tuple[int, int, int, int], # (d, rL, nU, rR)
]:
    """Get the hole shapes for a uniform base.

    Examples:
    ---------
    >>> import numpy as np
    >>> import t3toolbox.uniform as ut3
    >>> d, N, nU, nO, rL, rR = 6, 5, 4, 3, 2, 1
    >>> up_tucker_supercore = np.random.randn(d, nU, N)
    >>> left_tt_supercore = np.random.randn(d, rL, nU, rL)
    >>> right_tt_supercore = np.random.randn(d, rR, nU, rR)
    >>> outer_tt_supercore = np.random.randn(d, rL, nO, rR)
    >>> base = (up_tucker_supercore, left_tt_supercore, right_tt_supercore, outer_tt_supercore)
    >>> print(ut3.get_uniform_base_hole_shapes(base))
    ((6, 3, 5), (6, 2, 4, 1))
    """
    d, N, nU, nO, rL, rR = get_uniform_base_structure(base)
    return ((d, nO, N), (d, rL, nU, rR))


def get_uniform_variation_shapes(
        variation: UniformT3Variation,
) -> typ.Tuple[
    typ.Tuple[int, int, int], # (d, nO, N)
    typ.Tuple[int, int, int, int], # (d, rL, nU, rR)
]:
    """Get the shapes of the cores in a uniform variation.
    """
    var_tucker_supercore, var_tt_supercore = variation
    return var_tucker_supercore.shape, var_tt_supercore.shape

#

def check_ut3(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformEdgeWeights,
) -> None:
    """Check internal shape consistency of UniformTuckerTensorTrain.

    Parameters
    ----------
    cores: UniformTuckerTensorTrainCores
        Cores of the uniform Tucker tensor train
    masks: UniformTuckerTensorTrainEdgeMasks
        Edge masks for the uniform Tucker tensor train

    Raises:
    -------
    RuntimeError:
        Raised if internal shapes are inconsistent.
    """
    tucker_supercore, tt_supercore = cores
    shape_masks, tucker_masks, tt_masks = masks

    d, n, N = tucker_supercore.shape
    _, r, _, _ = tt_supercore.shape

    shapes_string =     'tucker_supercore.shape = ' + str(tucker_supercore.shape) + '\n'
    shapes_string +=    'tt_supercore.shape = '    + str(tt_supercore.shape) + '\n'
    shapes_string +=    'shape_masks.shape = '     + str(shape_masks.shape) + '\n'
    shapes_string +=    'tucker_masks.shape = '    + str(tucker_masks.shape) + '\n'
    shapes_string +=    'tt_masks.shape = '        + str(tt_masks.shape)

    if tucker_supercore.shape != (d,n,N):
        raise RuntimeError(
            'tucker_supercore has incorrect shape.\n' + shapes_string
        )

    if tt_supercore.shape != (d,r,n,r):
        raise RuntimeError(
            'tt_supercore has incorrect shape.\n' + shapes_string
        )

    if shape_masks.shape != (d,N):
        raise RuntimeError(
            'shape_masks has incorrect shape.\n' + shapes_string
        )

    if tucker_masks.shape != (d,n):
        raise RuntimeError(
            'tucker_masks has incorrect shape.\n' + shapes_string
        )

    if tt_masks.shape != (d+1,r):
        raise RuntimeError(
            'tt_masks has incorrect shape.\n' + shapes_string
        )


def get_uniform_structure(
        cores: UniformTuckerTensorTrainCores,
) -> UniformStructure:
    """Get padded structure of uniform Tucker tensor train.

    Parameters
    ----------
    cores: UniformTuckerTensorTrainCores
        Cores of the uniform Tucker tensor train

    Returns
    -------
    num_cores: int
        Number of cores
    N: int
        Size of each index
    n: int
        padded Tucker rank
    r: int
        padded TT-rank

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.get_uniform_structure(cores))
    (3, 16, 6, 3)
    """
    tucker_supercore, tt_supercore = cores

    d, n, N = tucker_supercore.shape
    r = tt_supercore.shape[1]
    return d, N, n, r


def get_original_structure(
        masks: UniformEdgeWeights,
) -> t3.T3Structure:
    """Get original (unpadded) structure of a uniform Tucker tensor train.

    Parameters
    ----------
    masks: UniformTuckerTensorTrainEdgeMasks
        Edge masks for the uniform Tucker tensor train

    Returns
    -------
    t3.T3Structure
        Structure of the unpadded Tucker tensor train

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.get_uniform_structure(cores))
    (3, 16, 6, 3)
    >>> print(ut3.get_original_structure(masks))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    """
    shape_masks, tucker_masks, tt_masks = masks

    original_shape = tuple([
        sum([int(e) for e in list(m)])
        for m in shape_masks
    ])

    original_tucker_ranks = tuple([
        sum([int(e) for e in list(m)])
        for m in tucker_masks
    ])

    original_tt_ranks = tuple([
        sum([int(e) for e in list(m)])
        for m in tt_masks
    ])

    return original_shape, original_tucker_ranks, original_tt_ranks


def apply_masks(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformEdgeWeights,
        use_jax: bool = False,
) -> UniformTuckerTensorTrainCores: # cores with masks applied
    """Apply masks to uniform Tucker tensor train cores.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
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
    xnp, xmap, xscan = get_backend(True, use_jax)

    shape_mask, tucker_mask, tt_mask = masks
    BB, GG = cores

    BB_mask = xnp.einsum('da,do->dao', tucker_mask, shape_mask)
    BB = BB * BB_mask

    GG_mask = xnp.einsum('di,da,dj->diaj', tt_mask[:-1], tucker_mask, tt_mask[1:])
    GG = GG * GG_mask

    masked_cores = (BB, GG)
    return masked_cores


def apply_masks_to_base(
        base: UniformT3Base,
        masks: UniformBVEdgeWeights,
        use_jax: bool = False,
) -> UniformT3Base:  # cores with masks applied
    """Apply masks to uniform Tucker tensor train base.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    shape_mask, up_mask, outer_mask, left_mask, right_mask = masks
    UU, LL, RR, OO = base

    UU_mask = xnp.einsum('da,do->dao', up_mask, shape_mask)
    UU = UU * UU_mask

    d, N, nU, nO, rL, rR = get_uniform_base_structure(base)

    left_mask_extended = xnp.concatenate([left_mask, xnp.ones((1,rL), dtype=bool)], axis=0) # len=d+1
    LL_mask = xnp.einsum(
        'di,da,dj->diaj', left_mask_extended[:-1], up_mask, left_mask_extended[1:],
    )
    LL = LL * LL_mask

    right_mask_extended = xnp.concatenate([xnp.ones((1,rR), dtype=bool), right_mask], axis=0) # len=d+1
    RR_mask = xnp.einsum(
        'di,da,dj->diaj', right_mask_extended[:-1], up_mask, right_mask_extended[1:],
    )
    RR = RR * RR_mask

    OO_mask = xnp.einsum(
        'di,da,dj->diaj', left_mask, outer_mask, right_mask,
    )
    OO = OO * OO_mask

    masked_base = (UU, LL, RR, OO)
    return masked_base


def apply_masks_to_variation(
        variation: UniformT3Variation,
        masks: UniformBVEdgeWeights,
        use_jax: bool = False,
) -> UniformT3Variation:  # cores with masks applied
    """Apply masks to uniform Tucker tensor train variation.
    """
    xnp, xmap, xscan = get_backend(True, use_jax)

    shape_mask, up_mask, outer_mask, left_mask, right_mask = masks
    VV, HH = variation

    VV_mask = xnp.einsum('da,do->dao', outer_mask, shape_mask)
    VV = VV * VV_mask

    HH_mask = xnp.einsum(
        'di,da,dj->diaj', left_mask, up_mask, right_mask,
    )
    HH = HH * HH_mask

    masked_variation = (VV, HH)
    return masked_variation


def uniform_squash_tails(
        x: UniformTuckerTensorTrainCores,
        use_jax: bool = False,
) -> UniformTuckerTensorTrainCores:
    """Squash tails of uniform Tucker tensor train.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> import t3toolbox.t3svd as t3svd
    >>> import t3toolbox.corewise as cw
    >>> x = t3.t3_corewise_randn(((10,11,12), (5,6,4), (1,3,5,1)))
    >>> uniform_x, masks = ut3.t3_to_ut3(x)
    >>> uniform_x2 = ut3.uniform_squash_tails(uniform_x)
    >>> dense_ux = ut3.ut3_to_dense(uniform_x, masks)
    >>> dense_ux2 = ut3.ut3_to_dense(uniform_x2, masks)
    >>> print(np.linalg.norm(dense_ux - dense_ux2))
    0.0
    """
    xnp, xmap, xscan = get_backend(True, use_jax)
    tucker_supercore, tt_supercore = x

    _, r, n, _ = tt_supercore.shape

    G0 = tt_supercore[0]
    new_G0 = xnp.concatenate([
        xnp.sum(G0, axis=0).reshape((1,n,r)),
        xnp.zeros((r-1,n,r))],
        axis=0,
    )

    GG_mid = tt_supercore[1:-1]

    Gf = tt_supercore[-1]
    new_Gf = xnp.concatenate([
        xnp.sum(Gf, axis=2).reshape((r,n,1)),
        xnp.zeros((r,n,r-1))],
        axis=2,
    )

    new_tt_supercore = xnp.concatenate([
        new_G0.reshape((1,r,n,r)),
        GG_mid,
        new_Gf.reshape((1,r,n,r))],
        axis=0,
    )
    return tucker_supercore, new_tt_supercore


def pack_tensors(
        unpacked_tensors = typ.Sequence[NDArray], # len=d, ith_elm.shape=(m1i, ..., mki)
        use_jax: bool = False,
) -> NDArray: # packed_tensors, shape=(d,)+(m1,...,mk), where mj=max(mj1, ..., mjd)
    """Use zero-padding to pack several tensors with ragged shapes into one tensor with an extra dimension.
    """
    xnp, _, _ = get_backend(False, use_jax)

    #

    if not unpacked_tensors:
        return xnp.array(())

    k = len(unpacked_tensors[0].shape)
    packed_shape = [max([x.shape[ii] for x in unpacked_tensors]) for ii in range(k)]

    delta_shapes = [[packed_shape[ii] - x.shape[ii] for ii in range(k)] for x in unpacked_tensors]

    padded_tensors_list = []
    for x, delta in zip(unpacked_tensors, delta_shapes):
        pads = [(0, d) for d in delta]
        padded_x = xnp.pad(
            x, pads,
        )
        padded_tensors_list.append(padded_x)

    packed_tensors = xnp.stack(padded_tensors_list)
    return packed_tensors


def unpack(
        packed_edge_tensors: NDArray, # shape=(...,c,m) or (c,m). E.g., (num_vecs,d,N) or (d,N)
        submask: NDArray, # shape=(c,m). Typical use case: component of UniformTuckerTensorTrainMasks
        xnp = np,
) -> typ.Tuple[
    NDArray, # shape=(...,mi) or (mi,). E.g., (num_vecs,Ni) or (Ni,)
    ...
]: # len=c, e.g., len=d
    """Get ragged (variable length) edge vectors from uniform edge vectors.

    Example
    -------
    >>> import numpy as np
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> E = np.array([[1,2,3,4],[5,6,7,8]])
    >>> submask = [[True, False, True, True],[False, True, False, False]]
    >>> print(ut3.unpack(E, submask))
    (array([1, 3, 4]), array([6]))

    Get a tensor from each "edge":

    >>> import numpy as np
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> E = np.random.randn(6,5,4,3,2)
    >>> submask = [[False, False],[False, True], [True, True]]
    >>> ee = ut3.unpack(E, submask)
    >>> print([e.shape for e in ee])
    [(6, 5, 4, 0), (6, 5, 4, 1), (6, 5, 4, 2)]

    Practical use case: remove zero singular values from uniform T3-SVD

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> import t3toolbox.t3svd as t3svd
    >>> s0 = ((11,12,13), (6,7,5), (1,3,6,2))
    >>> s = (s0[0],) + t3.compute_minimal_ranks(s0)
    >>> x = t3.t3_corewise_randn(s)
    >>> _, _, ss_tt = t3svd.t3_svd(x)
    >>> print(ss_tt[1])
    [2627.79225375  441.12769204  328.73617961]
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> _, _, ss_tt_from_ut3 = ut3.ut3_svd(cores, masks)
    >>> print(ss_tt_from_ut3[1])
    [2627.79225375  441.12769204  328.73617961    0.            0.        ]
    >>> print(ut3.unpack(ss_tt_from_ut3, masks[2])[1])
    [2627.79225375  441.12769204  328.73617961]
    """
    c = packed_edge_tensors.shape[-2]
    unpacked_edge_vectors = []
    for ii in range(c):
        TTi = xnp.take(packed_edge_tensors, ii, axis=-2)
        Ti = TTi[..., submask[ii]]
        unpacked_edge_vectors.append(Ti)
    return tuple(unpacked_edge_vectors)


def make_uniform_masks(
        structure: t3.T3Structure,
        uniform_structure: UniformStructure,
        use_jax: bool = False,
) -> UniformEdgeWeights:
    xnp, xmap, xscan = get_backend(False, use_jax)

    shape, tucker_ranks, tt_ranks = structure
    d, N, n, r = uniform_structure

    shape_masks = xnp.stack([
        xnp.concatenate([xnp.ones(Ni, dtype=bool), xnp.zeros(N-Ni, dtype=bool)])
        for Ni in shape
    ])

    tucker_masks = xnp.stack([
        xnp.concatenate([xnp.ones(ni, dtype=bool), xnp.zeros(n - ni, dtype=bool)])
        for ni in tucker_ranks
    ])

    tt_masks = xnp.stack([
        xnp.concatenate([xnp.ones(ri, dtype=bool), xnp.zeros(r - ri, dtype=bool)])
        for ri in tt_ranks
    ])

    return shape_masks, tucker_masks, tt_masks


def uniform_randn(
        structure: UniformStructure,
        masks: UniformEdgeWeights = None,
        use_jax: bool = False,
) -> UniformTuckerTensorTrainCores:
    """Makes a uniform Tucker tensor train with random cores.
    """
    xnp, _, _ = get_backend(True, use_jax)

    d, N, n, r = structure
    tucker_supercore = randn(d,n,N, use_jax=use_jax)
    tt_supercore = randn(d,r,n,r, use_jax=use_jax)

    x = (tucker_supercore, tt_supercore)
    x = apply_masks(x, masks) if masks is not None else x
    return x


def uniform_zeros(
        structure: UniformStructure,
        use_jax: bool = False,
) -> UniformTuckerTensorTrainCores:
    """Makes a uniform Tucker tensor train with cores filled with zeros.
    """
    xnp, _, _ = get_backend(True, use_jax)

    d, N, n, r = structure
    tucker_supercore = xnp.zeros((d,n,N))
    tt_supercore = xnp.zeros((d,r,n,r))
    x = (tucker_supercore, tt_supercore)
    return x


def t3_to_ut3(
        x: t3.TuckerTensorTrain,
        squash_tails: bool = True,
        xnp = np,
) -> typ.Tuple[
    UniformTuckerTensorTrainCores,
    UniformEdgeWeights,
]:
    """Convert TuckerTensorTrain to UniformTuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> x = t3.t3_corewise_randn(((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)))
    >>> cores, masks = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
    >>> x2 = ut3.ut3_to_t3(cores, masks)  # Convert ut3 -> t3
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_x2 = t3.t3_to_dense(x2)
    >>> print(np.linalg.norm(dense_x - dense_x2))
    0.0
    """
    if squash_tails:
        x = t3.squash_tails(x)

    shape, tucker_ranks, tt_ranks = t3.get_structure(x)

    d = len(shape)
    N = max(shape)
    n = max(tucker_ranks)
    r = max(tt_ranks)

    padded_shape = (N,)*d
    padded_tucker_ranks = (n,)*d
    padded_tt_ranks = (r,)*(d+1)

    padded_tucker_cores, padded_tt_cores = t3.change_structure(
        x, (padded_shape, padded_tucker_ranks, padded_tt_ranks),
    )

    tucker_supercore = xnp.stack(padded_tucker_cores)
    tt_supercore = xnp.stack(padded_tt_cores)

    shape_masks, tucker_masks, tt_masks = make_uniform_masks(
        (shape, tucker_ranks, tt_ranks), (d, N, n, r)
    )

    return (tucker_supercore, tt_supercore), (shape_masks, tucker_masks, tt_masks)


def ut3_to_t3(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformEdgeWeights,
        xnp = np,
) -> t3.TuckerTensorTrain:
    '''Convert UniformTuckerTensorTrain to TuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x) # Convert t3 -> ut3
    >>> print(ut3.get_uniform_structure(cores))
    (3, 16, 6, 3)
    >>> print(ut3.get_original_structure(masks))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    >>> x2 = ut3.ut3_to_t3(cores, masks) # Convert ut3 -> t3
    >>> print(t3.get_structure(x2))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    >>> print([np.linalg.norm(B - B2) for B, B2  in zip(x[0], x2[0])])
    [0.0, 0.0, 0.0]
    >>> print([np.linalg.norm(G - G2) for G, G2  in zip(x[1], x2[1])])
    [0.0, 0.0, 0.0]
    '''
    tucker_supercore, tt_supercore = cores
    shape_masks, tucker_masks, tt_masks = masks

    shape_inds  = [xnp.argwhere(em).reshape(-1) for em in list(shape_masks)]
    tucker_inds = [xnp.argwhere(em).reshape(-1) for em in list(tucker_masks)]
    tt_inds     = [xnp.argwhere(em).reshape(-1) for em in list(tt_masks)]

    tucker_cores = tuple([
        B[ii,:][:,jj]
        for ii, jj, B
        in zip(tucker_inds, shape_inds, list(tucker_supercore))
    ])

    tt_cores = tuple([
        G[ii, :, :][:,aa,:][:, :, jj]
        for ii, aa, jj, G
        in zip(tt_inds[:-1], tucker_inds, tt_inds[1:], list(tt_supercore))
    ])

    return tucker_cores, tt_cores


def bv_to_ubv(
        variation: bvf.T3Variation,
        base: bvf.T3Base,
        use_jax: bool = False,
) -> typ.Tuple[
    UniformT3Variation,
    UniformT3Base,
    UniformBVEdgeWeights, # masks
]:
    xnp, _, _ = get_backend(False, use_jax)

    var_tucker_cores, var_tt_cores = variation
    up_tucker_cores, left_tt_cores, right_tt_cores, outer_tt_cores = base

    NN = [U.shape[1] for U in up_tucker_cores]
    nnU = [U.shape[0] for U in up_tucker_cores]
    rrL = [L.shape[0] for L in left_tt_cores] + [left_tt_cores[-1].shape[2]]
    rrR = [R.shape[0] for R in right_tt_cores] + [right_tt_cores[-1].shape[2]]
    nnO = [O.shape[1] for O in outer_tt_cores]

    d = len(var_tucker_cores)
    N = max(NN)
    nU = max(nnU)
    nO = max(nnO)
    r = max(rrL + rrR)

    padded_shape = (N,) * d
    padded_up_tucker_ranks = (nU,) * d
    padded_outer_tucker_ranks = (nO,) * d
    padded_tt_ranks = (r,) * (d + 1)

    #

    var_tucker_supercore = xnp.stack(
        t3.change_tucker_core_shapes(var_tucker_cores, padded_shape, padded_outer_tucker_ranks)
    )
    var_tt_supercore = xnp.stack(
        t3.change_tt_core_shapes(var_tt_cores, padded_up_tucker_ranks, padded_tt_ranks)
    )

    #

    up_tucker_supercore = xnp.stack(
        t3.change_tucker_core_shapes(up_tucker_cores, padded_shape, padded_up_tucker_ranks)
    )
    left_tt_supercore = xnp.stack(
        t3.change_tt_core_shapes(left_tt_cores, padded_up_tucker_ranks, padded_tt_ranks)
    )
    right_tt_supercore = xnp.stack(
        t3.change_tt_core_shapes(right_tt_cores, padded_up_tucker_ranks, padded_tt_ranks)
    )
    outer_tt_supercore = xnp.stack(
        t3.change_tt_core_shapes(outer_tt_cores, padded_outer_tucker_ranks, padded_tt_ranks)
    )

    uniform_variation = (var_tucker_supercore, var_tt_supercore)
    uniform_base = (up_tucker_supercore, left_tt_supercore, right_tt_supercore, outer_tt_supercore)

    #

    shape_masks         = pack_tensors([xnp.ones(Ni, dtype=bool) for Ni in NN])
    up_tucker_masks     = pack_tensors([xnp.ones(nUi, dtype=bool) for nUi in nnU])
    outer_tucker_masks  = pack_tensors([xnp.ones(nOi, dtype=bool) for nOi in nnO])
    left_tt_masks       = pack_tensors([xnp.ones(rLi, dtype=bool) for rLi in rrL[:-1]])
    right_tt_masks      = pack_tensors([xnp.ones(rRi, dtype=bool) for rRi in rrR[1:]])

    masks = (shape_masks, up_tucker_masks, outer_tucker_masks, left_tt_masks, right_tt_masks)

    return uniform_variation, uniform_base, masks


#

def ut3_to_dense(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformEdgeWeights,
        xnp = np,
) -> NDArray:
    """Construct dense tensor represented by uniform Tucker tensor train

    Parameters
    ----------
    cores: UniformTuckerTensorTrainCores,
        Cores for the uniform Tucker tensor train
    masks: UniformTuckerTensorTrainMasks,
        Masks for the uniform Tucker tensor train

    Returns
    -------
    NDArray
        The dense tensor represented by uniform_x

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (2,3,2,2)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> x_dense = t3.t3_to_dense(x)
    >>> x_dense2 = ut3.ut3_to_dense(cores, masks)
    >>> print(np.linalg.norm(x_dense - x_dense2))
    0.0
    """
    check_ut3(cores, masks)
    return t3.t3_to_dense(ut3_to_t3(cores, masks), xnp=xnp)


def are_ut3_ranks_minimal(
        masks: UniformEdgeWeights,
) -> bool:
    """Checks if the ranks of a uniform Tucker train are minimal.

    Example
    -------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,4,9,7,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.are_ut3_ranks_minimal(masks))
    True

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,99,9,7,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.are_ut3_ranks_minimal(masks))
    False
    """
    s = get_original_structure(masks)
    _, tucker_ranks, tt_ranks = s
    minimal_tucker_ranks, minimal_tt_ranks = t3.compute_minimal_ranks(s)
    return (tucker_ranks == minimal_tucker_ranks) and (tt_ranks == minimal_tt_ranks)


def ut3_entry(
        cores: UniformTuckerTensorTrainCores,
        index: NDArray, # dtype=int. shape=(d,) or shape=(num_entries,d)
        xnp = np,
        scan = common.numpy_scan,
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
	>>> x = t3.t3_corewise_randn(((14,15,16), (4,5,3), (1,4,2,1))) # T3
	>>> index = (3,1,2)
	>>> x_312 = t3.t3_entry(x, index)
	>>> print(x_312) # (3,1,2) entry from T3:
	-1.4931654579929192
	>>> cores, masks = ut3.t3_to_ut3(x) # Convert to Uniform T3
	>>> print(ut3.get_original_structure(masks)) # original (shape, tucker_ranks, tt_ranks):
	((14, 15, 16), (4, 5, 3), (1, 4, 2, 1))
	>>> print(ut3.get_uniform_structure(cores)) # uniform shape and ranks, (d,N,n,r):
	(3, 16, 5, 4)
	>>> x_312_uniform = ut3.ut3_entry(cores, index) # (3,1,2) entry from uniform T3:
	>>> print(x_312_uniform)
	-1.4931654579929197

    Multiple entries:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,3), (1,4,2,1)))
    >>> index = ((3,10), (1,9), (2,8))
    >>> x_312 = t3.t3_entry(x, index)
    >>> print(x_312)
    -6.127319174475167
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> x_312_uniform = ut3.ut3_entry(cores, index)
    >>> print(x_312_uniform)
    -6.127319174475165

    """
    tucker_supercore, tt_supercore = cores

    d, N, n, r = get_uniform_structure(cores)

    index = xnp.array(index)

    vectorized = True
    if len(index.shape) == 1:
        vectorized = False
        index = index.reshape((-1,1))

    num_entries = index.shape[1]
    assert(index.shape == (d, num_entries))

    def _func(mu_na, x):
        ind, B_xi, G_axb = x
        v_xn = xnp.take_along_axis(B_xi, ind.reshape((1,-1)), 1)
        g_anb = xnp.einsum('axb,xn->anb', G_axb, v_xn)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        return mu_nb, 0

    init = xnp.ones((num_entries, r))
    xs = (index, tucker_supercore, tt_supercore)
    final_mu = scan(_func, init, xs, length=None)[0]
    result = xnp.einsum('na->n', final_mu)

    if not vectorized:
        result = result[0]

    return result


def ut3_apply(
        cores: UniformTuckerTensorTrainCores,
        input_vectors: NDArray, # shape=(d,N) or shape=(...,d,N)
        xnp = np,
        scan = common.numpy_scan,
) -> NDArray: # shape=(d,N) or (...,d,N)
    """Apply a uniform Tucker tensor train to vectors. WORK IN PROGRESS

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> uvecs =
    >>> print(np.linalg.norm(result - result2))

	>>> import numpy as np
	>>> import t3toolbox.uniform_tucker_tensor_train as ut3
	>>> d = 3
	>>> N = 11
	>>> n = 7
	>>> r = 4
	>>> tucker_supercore = np.random.randn(d,n,N)
	>>> tt_supercore = np.random.randn(d,r,n,r)
	>>> cores = (tucker_supercore, basis_supercore)
	>>> ww = np.random.randn(d, N)
	>>> result = ut3.ut3_apply(cores, ww)
	>>> result2 = xnp.einsum('di,dxi,da', ww, tucker_supercore, tt_supercore)
	>>> x_312 = t3.t3_entry(x, index)
	>>> print(x_312) # (3,1,2) entry from T3:
	-1.4931654579929192
	>>> cores, masks = ut3.t3_to_ut3(x) # Convert to Uniform T3
	>>> print(ut3.get_original_structure(masks)) # original (shape, tucker_ranks, tt_ranks):
	((14, 15, 16), (4, 5, 3), (1, 4, 2, 1))
	>>> print(ut3.get_uniform_structure(cores)) # uniform shape and ranks, (d,N,n,r):
	(3, 16, 5, 4)
	>>> x_312_uniform = ut3.ut3_entry(cores, index) # (3,1,2) entry from uniform T3:
	>>> print(x_312_uniform)
	-1.4931654579929197

    Multiple entries:

    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,3), (1,4,2,1)))
    >>> index = ((3,10), (1,9), (2,8))
    >>> x_312 = t3.t3_entry(x, index)
    >>> print(x_312)
    -6.127319174475167
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> x_312_uniform = ut3.ut3_entry(cores, index)
    >>> print(x_312_uniform)
    -6.127319174475165

    """
    tucker_supercore, tt_supercore = cores

    d, N, n, r = get_uniform_structure(cores)

    def _func(mu_Na, x):
        w_Ni, B_xi, G_axb = x
        xi_Nx = xnp.einsum('...i,xi->...x', w_Ni, B_xi)
        tmp_Nxb = xnp.einsum('...a,axb->...xb', mu_Na, G_axb)
        mu_Nb = xnp.einsum('...xb,...Nx->...b', tmp_Nxb, xi_Nx)
        return mu_Nb, 0

    init = xnp.ones((input_vectors.shape[:-2], r))
    xs = (input_vectors.moveaxis(-2,0), tucker_supercore, tt_supercore)
    final_mu = scan(_func, init, xs, length=None)[0]
    result = xnp.einsum('...a->...', final_mu)

    return result

###################################################
#########    Linear algebra operations    #########
###################################################

def ut3_add(
        x_cores: UniformTuckerTensorTrainCores,
        x_masks: UniformEdgeWeights,
        y_cores: UniformTuckerTensorTrainCores,
        y_masks: UniformEdgeWeights,
        xnp = np,
) -> typ.Tuple[
    UniformTuckerTensorTrainCores, # x+y cores
    UniformEdgeWeights, # x+y masks
]: # z = x + y
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
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (2,3,2,2)))
    >>> x_cores, x_masks = ut3.t3_to_ut3(x)
    >>> y = t3.t3_corewise_randn(((14,15,16), (6,7,8), (3,5,6,1)))
    >>> y_cores, y_masks = ut3.t3_to_ut3(y)
    >>> x_plus_y_cores, x_plus_y_masks = ut3.ut3_add(x_cores, x_masks, y_cores, y_masks) # add x+y
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_y = t3.t3_to_dense(y)
    >>> dense_x_plus_y = ut3.ut3_to_dense(x_plus_y_cores, x_plus_y_masks)
    >>> print(np.linalg.norm(dense_x + dense_y - dense_x_plus_y))
    0.0
    """
    check_ut3(x_cores, x_masks)
    check_ut3(y_cores, y_masks)

    d_x, N_x, n_x, r_x = get_uniform_structure(x_cores)
    d_y, N_y, n_y, r_y = get_uniform_structure(y_cores)
    if N_x != N_y or d_x != d_y:
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent shapes.\n'
            'Must have d_x=d_y and N_x=N_y:\n'
            + '(d_x, N_x, n_x, r_x) = ' + str(get_uniform_structure(x_cores)) + '\n'
            + '(d_y, N_y, n_y, r_y) = ' + str(get_uniform_structure(y_cores))
        )

    x_tucker_supercore, x_tt_supercore = x_cores
    x_shape_masks, x_tucker_masks, x_tt_masks = x_masks

    y_tucker_supercore, y_tt_supercore = y_cores
    y_shape_masks, y_tucker_masks, y_tt_masks = y_masks

    z_shape_masks  = x_shape_masks + y_shape_masks # Addition is nonsensical if these are not the same.
    z_tucker_masks = xnp.concatenate([x_tucker_masks,   y_tucker_masks],   axis=1)
    z_tt_masks     = xnp.concatenate([x_tt_masks,       y_tt_masks],       axis=1)

    z_tucker_supercore = xnp.concatenate([x_tucker_supercore, y_tucker_supercore], axis=1)

    r0, n0 = r_x, n_x
    r1, n1 = r_y, n_y
    d = d_x
    G000 = x_tt_supercore
    G001 = xnp.zeros((d, r0, n0, r1))
    G010 = xnp.zeros((d, r0, n1, r0))
    G011 = xnp.zeros((d, r0, n1, r1))
    G100 = xnp.zeros((d, r1, n0, r0))
    G101 = xnp.zeros((d, r1, n0, r1))
    G110 = xnp.zeros((d, r1, n1, r0))
    G111 = y_tt_supercore
    z_tt_supercore = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])

    return (z_tucker_supercore, z_tt_supercore), (z_shape_masks, z_tucker_masks, z_tt_masks)


def ut3_scale(
        x_cores: UniformTuckerTensorTrainCores,
        s, # scalar
        xnp = np,
) -> UniformTuckerTensorTrainCores: # cores for z = s*x
    """Scale a uniform Tucker tensor train, s,x -> s*x.

    Parameters
    ----------
    x_cores: UniformTuckerTensorTrainCores
        Original uniform Tucker tensor train cores
    s: scalar
        Scaling factor
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    UniformTuckerTensorTrainCores
        Cores for scaled uniform Tucker tensor train, s*x

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (2,3,2,2)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> s = 3.5
    >>> sx_cores = ut3.ut3_scale(cores, s) # scale x
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_sx = ut3.ut3_to_dense(sx_cores, masks)
    >>> print(np.linalg.norm(s*dense_x - dense_sx))
    1.4502362601421634e-12
    """
    x_tucker_supercore, x_tt_supercore = x_cores

    first_x_tucker_supercore = x_tucker_supercore[:1,:,:]
    rest_x_tucker_supercore = x_tucker_supercore[1:, :, :]
    sx_tucker_supercore = xnp.concatenate([s*first_x_tucker_supercore, rest_x_tucker_supercore], axis=0)

    return sx_tucker_supercore, x_tt_supercore


def ut3_neg(
        x_cores: UniformTuckerTensorTrainCores,
        xnp = np,
) -> UniformTuckerTensorTrainCores: # cores for z = -x
    """Flip a uniform Tucker tensor train, x -> -x.

    Parameters
    ----------
    x_cores: UniformTuckerTensorTrainCores
        Original uniform Tucker tensor train cores

    Returns
    -------
    UniformTuckerTensorTrainCores
        Cores for flipped uniform Tucker tensor train, -x

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (2,3,2,2)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> neg_cores = ut3.ut3_neg(cores) # flip x
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_neg_x = ut3.ut3_to_dense(neg_cores, masks)
    >>> print(np.linalg.norm(-dense_x - dense_neg_x))
    0.0
    """
    return ut3_scale(x_cores, -1.0, xnp=xnp)


def ut3_sub(
        x_cores: UniformTuckerTensorTrainCores,
        x_masks: UniformEdgeWeights,
        y_cores: UniformTuckerTensorTrainCores,
        y_masks: UniformEdgeWeights,
        xnp = np
) -> typ.Tuple[
    UniformTuckerTensorTrainCores, # x-y cores
    UniformEdgeWeights, # x-y masks
]: # z = x - y
    """Subtract two UniformTuckerTensorTrains, x,y -> x-y.

    Parameters
    ----------
    x_cores: UniformTuckerTensorTrainCores
        First term cores
    x_masks: UniformTuckerTensorTrainMasks
        First term masks
    y_cores: UniformTuckerTensorTrainCores
        Second term cores
    y_masks: UniformTuckerTensorTrainMasks
        Second term masks
    xnp:
        Linear algebra backend. Default: np (numpy)

    Returns
    -------
    UniformTuckerTensorTrainCores
        Cores for difference, x-y
    UniformTuckerTensorTrainMasks
        Cores for difference x-y

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (2,3,2,2)))
    >>> x_cores, x_masks = ut3.t3_to_ut3(x)
    >>> y = t3.t3_corewise_randn(((14,15,16), (6,7,8), (3,5,6,1)))
    >>> y_cores, y_masks = ut3.t3_to_ut3(y)
    >>> x_minus_y_cores, x_minus_y_masks = ut3.ut3_sub(x_cores, x_masks, y_cores, y_masks) # add x+y
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_y = t3.t3_to_dense(y)
    >>> dense_x_minus_y = ut3.ut3_to_dense(x_minus_y_cores, x_minus_y_masks)
    >>> print(np.linalg.norm(dense_x - dense_y - dense_x_minus_y))
    0.0
    """
    return ut3_add(x_cores, x_masks, ut3_neg(y_cores, xnp=xnp), y_masks, xnp=xnp)




#


#

# # # #
#
# __all__ = [
#     'ut3_tangent_vector_to_ut3',
#     'ut3_attached_tangent_vector_to_ut3',
#     'left_orthogonalize_utt',
#     'right_orthogonalize_utt',
#     'orthogonalize_ut3_tucker_cores',
#     'ut3_svd_masked',
#     'ut3_retract',
#     'construct_ut3_base_representations',
#     'make_ut3_masks',
#     'ut3_to_t3',
#     't3_to_ut3',
#     'ut3_to_dense',
#     'ut3_project_dense_tensor_onto_tangent_space',
#     'apply_masks',
# ]
#
# #### WORK IN PROGRESS DO NOT USE
#

#
#
# def make_ut3_masks(
#         padded_N:           int, # padded_shape=(N,N,...,N)
#         padded_tucker_rank: int,
#         padded_tt_rank:     int,
#         unpadded_shape:         typ.Sequence[int], # len=d
#         unpadded_tucker_ranks:  typ.Sequence[int], # len=d
#         unpadded_tt_ranks:      typ.Sequence[int], # len=d+1
# ) -> typ.Tuple[
#     jnp.ndarray,  # shape_masks, shape=(d, N)
#     jnp.ndarray,  # tucker_mask, shape=(d, n)
#     jnp.ndarray,  # tt_mask, shape=(d+1, r)
# ]:  # use to specify ranks
#     shape_masks = jnp.stack([
#         jnp.concatenate([jnp.ones(N), jnp.zeros(padded_N - N)])
#         for N in unpadded_shape
#     ])
#     tucker_masks = jnp.stack([
#         jnp.concatenate([jnp.ones(n), jnp.zeros(padded_tucker_rank - n)])
#         for n in unpadded_tucker_ranks
#     ])
#     tt_masks = jnp.stack([
#         jnp.concatenate([jnp.ones(r), jnp.zeros(padded_tt_rank - r)])
#         for r in unpadded_tt_ranks
#     ])
#     masks = (shape_masks, tucker_masks, tt_masks)
#     return masks
#
#

#
