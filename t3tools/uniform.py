# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

import t3tools.tucker_tensor_train as t3
# from t3tools.common import jnp, NDArray, numpy_scan, jax_scan
import t3tools.common as common

# xnp = np
# scan = common.numpy_scan
NDArray = np.ndarray

__all__ = [
    'UniformTuckerTensorTrainCores',
    'UniformTuckerTensorTrainMasks',
    'get_padded_structure',
    'get_original_structure',
    'unpack_edge_tensors',
    'apply_masks',
    't3_to_ut3',
    'ut3_to_t3',
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
    NDArray, # basis_supercore, shape=(d, n, N)
    NDArray, # tt_supercore, shape=(d, r, n, r)
]
"""
Tuple containing supercores of a Uniform Tucker tensor train

Uniform Tucker tensor trains are created by padding a Tucker tensor train 
so that the ranks are uniform, then stacking the TT-cores and basis cores into
"supercores", which have one more dimension.

Padding may be tracked with boolean mask arrays associated with the edges.

See Also
--------
UniformTuckerTensorTrainMasks

**Components**
    - basis_supercore:      NDArray. shape=(d, n, N).       Stacked padded basis cores
    - tt_supercore:         NDArray. shape=(d, r, n, r).    Stacked padded TT-cores
    
Here:
    - d = num_cores
    - N = padded_shape
    - n = padded_tucker_rank
    - r = padded_tt_rank

Examples
--------
>>> import numpy as np
>>> import t3tools.tucker_tensor_train as t3
>>> import t3tools.uniform_tucker_tensor_train as ut3
>>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
>>> cores, masks = ut3.t3_to_ut3(x)
>>> print(ut3.get_padded_structure(cores))
(3, 16, 6, 3)
>>> print(ut3.original_structure(masks))
((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
"""


UniformTuckerTensorTrainMasks = typ.Tuple[
    NDArray,  # shape_masks, shape=(d,N)
    NDArray,  # tucker_masks, shape=(d, n)
    NDArray,  # tt_masks, shape=(d+1, r)
]
"""
Tuple containing edge masks for a Uniform Tucker tensor train

See Also
--------
UniformTuckerTensorTrainCores

**Components**
    - shape_masks:  NDArray. shape=(d,N).   Stacked masks for edges between basis cores and exterior of tensor
    - tucker_masks: NDArray. shape=(d,n).   Stacked masks for edges between basis cores and adjacent TT-cores
    - tt_masks:     NDArray. shape=(d+1,r). Stacked masks for edges between adjacent TT-cores

Here:
    - d = num_cores
    - N = padded_shape
    - n = padded_tucker_rank
    - r = padded_tt_rank

Examples
--------
>>> import numpy as np
>>> import t3tools.tucker_tensor_train as t3
>>> import t3tools.uniform_tucker_tensor_train as ut3
>>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
>>> cores, masks = ut3.t3_to_ut3(x)
>>> print(ut3.get_padded_structure(cores))
(3, 16, 6, 3)
>>> print(ut3.original_structure(masks))
((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
"""


def check_ut3(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformTuckerTensorTrainMasks,
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
    basis_supercore, tt_supercore = cores
    shape_masks, tucker_masks, tt_masks = masks

    d, n, N = basis_supercore.shape
    _, r, _, _ = tt_supercore.shape

    shapes_string =     'basis_supercore.shape = ' + str(basis_supercore.shape) + '\n'
    shapes_string +=    'tt_supercore.shape = '    + str(tt_supercore.shape) + '\n'
    shapes_string +=    'shape_masks.shape = '     + str(shape_masks.shape) + '\n'
    shapes_string +=    'tucker_masks.shape = '    + str(tucker_masks.shape) + '\n'
    shapes_string +=    'tt_masks.shape = '        + str(tt_masks.shape)

    if basis_supercore.shape != (d,n,N):
        raise RuntimeError(
            'basis_supercore has incorrect shape.\n' + shapes_string
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


def get_padded_structure(
        cores: UniformTuckerTensorTrainCores,
) -> typ.Tuple[
    int, # d, num_cores
    int, # N, padded_index_size
    int, # n, padded_tucker_rank
    int, # r, padded_tt_rank
]:
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.get_padded_structure(cores))
    (3, 16, 6, 3)
    """
    basis_supercore, tt_supercore = cores

    d, n, N = basis_supercore.shape
    r = tt_supercore.shape[1]
    return d, N, n, r


def get_original_structure(
        masks: UniformTuckerTensorTrainMasks,
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.get_padded_structure(cores))
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
        masks: UniformTuckerTensorTrainMasks,
        xnp = np,
) -> UniformTuckerTensorTrainCores: # cores with masks applied
    """Apply masks to uniform Tucker tensor train cores to zero out superflous entries.
    """
    shape_mask, tucker_mask, tt_mask = masks
    BB, GG = cores
    BB = xnp.einsum('dao,do->dao', BB, shape_mask)
    BB = xnp.einsum('dao,da->dao', BB, tucker_mask)
    GG = xnp.einsum('diaj,di->diaj', GG, tt_mask[:-1])
    GG = xnp.einsum('diaj,da->diaj', GG, tucker_mask)
    GG = xnp.einsum('diaj,dj->diaj', GG, tt_mask[1:])
    masked_cores = (BB, GG)
    return masked_cores


def unpack_edge_tensors(
        packed_edge_tensors: NDArray, # shape=(...,c,m) or (c,m). E.g., (num_vecs,d,N) or (d,N)
        submask: NDArray, # shape=(c,m). Typical use case: component of UniformTuckerTensorTrainMasks
        xnp = np,
) -> typ.Tuple[
    NDArray, # shape=(...,mi) or (mi,). E.g., (num_vecs,Ni) or (Ni,)
]: # len=c, e.g., len=d
    """Get ragged (variable length) edge vectors from uniform edge vectors.

    Example
    -------
    >>> import numpy as np
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> E = np.array([[1,2,3,4],[5,6,7,8]])
    >>> submask = [[True, False, True, True],[False, True, False, False]]
    >>> print(ut3.unpack_edge_tensors(E, submask))
    (array([1, 3, 4]), array([6]))

    Get a tensor from each "edge":

    >>> import numpy as np
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> E = np.random.randn(6,5,4,3,2)
    >>> submask = [[False, False],[False, True], [True, True]]
    >>> ee = ut3.unpack_edge_tensors(E, submask)
    >>> print([e.shape for e in ee])
    [(6, 5, 4, 0), (6, 5, 4, 1), (6, 5, 4, 2)]

    Practical use case: remove zero singular values from uniform T3-SVD

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> import t3tools.t3svd as t3svd
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
    >>> print(ut3.unpack_edge_tensors(ss_tt_from_ut3, masks[2])[1])
    [2627.79225375  441.12769204  328.73617961]
    """
    c = packed_edge_tensors.shape[-2]
    unpacked_edge_vectors = []
    for ii in range(c):
        TTi = xnp.take(packed_edge_tensors, ii, axis=-2)
        Ti = TTi[..., submask[ii]]
        unpacked_edge_vectors.append(Ti)
    return tuple(unpacked_edge_vectors)


def t3_to_ut3(
        x: t3.TuckerTensorTrain,
        xnp = np,
) -> typ.Tuple[
    UniformTuckerTensorTrainCores,
    UniformTuckerTensorTrainMasks,
]:
    """Convert TuckerTensorTrain to UniformTuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14, 15, 16), (4, 6, 5), (3, 3, 2, 4)))
    >>> cores, masks = ut3.t3_to_ut3(x)  # Convert t3 -> ut3
    >>> x2 = ut3.ut3_to_t3(cores, masks)  # Convert ut3 -> t3
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_x2 = t3.t3_to_dense(x2)
    >>> print(np.linalg.norm(dense_x - dense_x2))
    0.0
    """
    shape, tucker_ranks, tt_ranks = t3.get_structure(x)

    d = len(shape)
    N = max(shape)
    n = max(tucker_ranks)
    r = max(tt_ranks)

    padded_shape = (N,)*d
    padded_tucker_ranks = (n,)*d
    padded_tt_ranks = (r,)*(d+1)

    padded_basis_cores, padded_tt_cores = t3.pad_t3(
        x, (padded_shape, padded_tucker_ranks, padded_tt_ranks),
    )

    basis_supercore = xnp.stack(padded_basis_cores)
    tt_supercore = xnp.stack(padded_tt_cores)

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

    return (basis_supercore, tt_supercore), (shape_masks, tucker_masks, tt_masks)


def ut3_to_t3(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformTuckerTensorTrainMasks,
        xnp = np,
) -> t3.TuckerTensorTrain:
    '''Convert UniformTuckerTensorTrain to TuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x) # Convert t3 -> ut3
    >>> print(ut3.get_padded_structure(cores))
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
    basis_supercore, tt_supercore = cores
    shape_masks, tucker_masks, tt_masks = masks

    shape_inds  = [xnp.argwhere(em).reshape(-1) for em in list(shape_masks)]
    tucker_inds = [xnp.argwhere(em).reshape(-1) for em in list(tucker_masks)]
    tt_inds     = [xnp.argwhere(em).reshape(-1) for em in list(tt_masks)]

    basis_cores = tuple([
        B[ii,:][:,jj]
        for ii, jj, B
        in zip(tucker_inds, shape_inds, list(basis_supercore))
    ])

    tt_cores = tuple([
        G[ii, :, :][:,aa,:][:, :, jj]
        for ii, aa, jj, G
        in zip(tt_inds[:-1], tucker_inds, tt_inds[1:], list(tt_supercore))
    ])

    return basis_cores, tt_cores


def ut3_to_dense(
        cores: UniformTuckerTensorTrainCores,
        masks: UniformTuckerTensorTrainMasks,
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
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
        masks: UniformTuckerTensorTrainMasks,
) -> bool:
    """Checks if the ranks of a uniform Tucker train are minimal.

    Example
    -------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,4,9,7,1)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> print(ut3.are_ut3_ranks_minimal(masks))
    True

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
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
	>>> import t3tools.tucker_tensor_train as t3
	>>> import t3tools.uniform_tucker_tensor_train as ut3
	>>> x = t3.t3_corewise_randn(((14,15,16), (4,5,3), (1,4,2,1))) # T3
	>>> index = (3,1,2)
	>>> x_312 = t3.t3_entry(x, index)
	>>> print(x_312) # (3,1,2) entry from T3:
	-1.4931654579929192
	>>> cores, masks = ut3.t3_to_ut3(x) # Convert to Uniform T3
	>>> print(ut3.get_original_structure(masks)) # original (shape, tucker_ranks, tt_ranks):
	((14, 15, 16), (4, 5, 3), (1, 4, 2, 1))
	>>> print(ut3.get_padded_structure(cores)) # uniform shape and ranks, (d,N,n,r):
	(3, 16, 5, 4)
	>>> x_312_uniform = ut3.ut3_entry(cores, index) # (3,1,2) entry from uniform T3:
	>>> print(x_312_uniform)
	-1.4931654579929197

    Multiple entries:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
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
    basis_supercore, tt_supercore = cores

    d, N, n, r = get_padded_structure(cores)

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
    xs = (index, basis_supercore, tt_supercore)
    final_mu = scan(_func, init, xs, length=None)[0]
    result = xnp.einsum('na->n', final_mu)

    if not vectorized:
        result = result[0]

    return result

###################################################
#########    Linear algebra operations    #########
###################################################

def ut3_add(
        x_cores: UniformTuckerTensorTrainCores,
        x_masks: UniformTuckerTensorTrainMasks,
        y_cores: UniformTuckerTensorTrainCores,
        y_masks: UniformTuckerTensorTrainMasks,
        xnp = np,
) -> typ.Tuple[
    UniformTuckerTensorTrainCores, # x+y cores
    UniformTuckerTensorTrainMasks, # x+y masks
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
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

    d_x, N_x, n_x, r_x = get_padded_structure(x_cores)
    d_y, N_y, n_y, r_y = get_padded_structure(y_cores)
    if N_x != N_y or d_x != d_y:
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent shapes.\n'
            'Must have d_x=d_y and N_x=N_y:\n'
            + '(d_x, N_x, n_x, r_x) = ' + str(get_padded_structure(x_cores)) + '\n'
            + '(d_y, N_y, n_y, r_y) = ' + str(get_padded_structure(y_cores))
        )

    x_basis_supercore, x_tt_supercore = x_cores
    x_shape_masks, x_tucker_masks, x_tt_masks = x_masks

    y_basis_supercore, y_tt_supercore = y_cores
    y_shape_masks, y_tucker_masks, y_tt_masks = y_masks

    z_shape_masks  = x_shape_masks + y_shape_masks # Addition is nonsensical if these are not the same.
    z_tucker_masks = xnp.concatenate([x_tucker_masks,   y_tucker_masks],   axis=1)
    z_tt_masks     = xnp.concatenate([x_tt_masks,       y_tt_masks],       axis=1)

    z_basis_supercore = xnp.concatenate([x_basis_supercore, y_basis_supercore], axis=1)

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

    return (z_basis_supercore, z_tt_supercore), (z_shape_masks, z_tucker_masks, z_tt_masks)


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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (2,3,2,2)))
    >>> cores, masks = ut3.t3_to_ut3(x)
    >>> s = 3.5
    >>> sx_cores = ut3.ut3_scale(cores, s) # scale x
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_sx = ut3.ut3_to_dense(sx_cores, masks)
    >>> print(np.linalg.norm(s*dense_x - dense_sx))
    1.4502362601421634e-12
    """
    x_basis_supercore, x_tt_supercore = x_cores

    first_x_basis_supercore = x_basis_supercore[:1,:,:]
    rest_x_basis_supercore = x_basis_supercore[1:, :, :]
    sx_basis_supercore = xnp.concatenate([s*first_x_basis_supercore, rest_x_basis_supercore], axis=0)

    return sx_basis_supercore, x_tt_supercore


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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
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
        x_masks: UniformTuckerTensorTrainMasks,
        y_cores: UniformTuckerTensorTrainCores,
        y_masks: UniformTuckerTensorTrainMasks,
        xnp = np
) -> typ.Tuple[
    UniformTuckerTensorTrainCores, # x-y cores
    UniformTuckerTensorTrainMasks, # x-y masks
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
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
#     'orthogonalize_ut3_basis_cores',
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
#     jnp.ndarray,  # basis_mask, shape=(d, n)
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
