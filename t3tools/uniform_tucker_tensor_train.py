# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
import t3tools.tucker_tensor_train as t3

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]

__all__ = [
    'UniformTuckerTensorTrainCores',
    'UniformTuckerTensorTrainMasks',
    'padded_structure',
    'original_structure',
    't3_to_ut3',
    'ut3_to_t3',
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
>>> print(ut3.padded_structure(cores))
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
>>> print(ut3.padded_structure(cores))
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



def padded_structure(
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
    >>> print(ut3.padded_structure(cores))
    (3, 16, 6, 3)
    """
    basis_supercore, tt_supercore = cores

    d, n, N = basis_supercore.shape
    r = tt_supercore.shape[1]
    return d, N, n, r


def original_structure(
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
    >>> print(ut3.padded_structure(cores))
    (3, 16, 6, 3)
    >>> print(ut3.original_structure(masks))
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


def t3_to_ut3(
        x: t3.TuckerTensorTrain,
        use_jax: bool = False,
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
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x) # Convert t3 -> ut3
    >>> print(ut3.padded_structure(cores))
    (3, 16, 6, 3)
    >>> print(ut3.original_structure(masks))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    >>> x2 = ut3.ut3_to_t3(cores, masks) # Convert ut3 -> t3
    >>> print(t3.structure(x2))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    >>> print([np.linalg.norm(B - B2) for B, B2  in zip(x[0], x2[0])])
    [0.0, 0.0, 0.0]
    >>> print([np.linalg.norm(G - G2) for G, G2  in zip(x[1], x2[1])])
    [0.0, 0.0, 0.0]
    """
    xnp = jnp if use_jax else np

    shape, tucker_ranks, tt_ranks = t3.structure(x)

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
        use_jax: bool = False,
) -> t3.TuckerTensorTrain:
    '''Convert UniformTuckerTensorTrain to TuckerTensorTrain.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.uniform_tucker_tensor_train as ut3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> cores, masks = ut3.t3_to_ut3(x) # Convert t3 -> ut3
    >>> print(ut3.padded_structure(cores))
    (3, 16, 6, 3)
    >>> print(ut3.original_structure(masks))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    >>> x2 = ut3.ut3_to_t3(cores, masks) # Convert ut3 -> t3
    >>> print(t3.structure(x2))
    ((14, 15, 16), (4, 6, 5), (1, 3, 2, 1))
    >>> print([np.linalg.norm(B - B2) for B, B2  in zip(x[0], x2[0])])
    [0.0, 0.0, 0.0]
    >>> print([np.linalg.norm(G - G2) for G, G2  in zip(x[1], x2[1])])
    [0.0, 0.0, 0.0]
    '''
    xnp = jnp if use_jax else np

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
        use_jax: bool = False,
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
    return t3.t3_to_dense(ut3_to_t3(cores, masks, use_jax=use_jax), use_jax=use_jax)


###################################################
#########    Linear algebra operations    #########
###################################################

def ut3_add(
        x_cores: UniformTuckerTensorTrainCores,
        x_masks: UniformTuckerTensorTrainMasks,
        y_cores: UniformTuckerTensorTrainCores,
        y_masks: UniformTuckerTensorTrainMasks,
        use_jax: bool = False,
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
    >>> dense_x_plus_y = ut3.ut3_to_dense(x_cores, x_masks)
    >>> print(np.linalg.norm(dense_x + dense_y - dense_x_plus_y))
    0.0
    """
    xnp = jnp if use_jax else np

    check_ut3(x_cores, x_masks)
    check_ut3(y_cores, y_masks)

    d_x, N_x, n_x, r_x = padded_structure(x_cores)
    d_y, N_y, n_y, r_y = padded_structure(y_cores)
    if N_x != N_y or d_x != d_y:
        raise RuntimeError(
            'Attempted to add UniformTuckerTensorTrains x+y with inconsistent shapes.\n'
            'Must have d_x=d_y and N_x=N_y:\n'
            + '(d_x, N_x, n_x, r_x) = ' + str(padded_structure(x_cores)) + '\n'
            + '(d_y, N_y, n_y, r_y) = ' + str(padded_structure(y_cores))
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
        use_jax: bool = False,
) -> UniformTuckerTensorTrainCores: # cores for z = s*x
    """Scale a uniform Tucker tensor train, s,x -> s*x.

    Parameters
    ----------
    x_cores: UniformTuckerTensorTrainCores
        Original uniform Tucker tensor train cores
    s: scalar
        Scaling factor

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
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> s = 3.5
    >>> uniform_sx = ut3.ut3_scale(uniform_x, s) # scale x
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_sx = ut3.ut3_to_dense(uniform_sx)
    >>> print(np.linalg.norm(s*dense_x - dense_sx))
    1.4502362601421634e-12
    """
    xnp = jnp if use_jax else np
    x_basis_supercore, x_tt_supercore = x_cores

    first_x_basis_supercore = x_basis_supercore[:1,:,:]
    rest_x_basis_supercore = x_basis_supercore[1:, :, :]
    sx_basis_supercore = xnp.concatenate([s*first_x_basis_supercore, rest_x_basis_supercore], axis=0)

    return sx_basis_supercore, x_tt_supercore


def ut3_neg(
        x_cores: UniformTuckerTensorTrainCores,
        use_jax: bool = False,
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
    >>> uniform_x = ut3.t3_to_ut3(x)
    >>> uniform_neg_x = ut3.ut3_neg(uniform_x) # flip x
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_neg_x = ut3.ut3_to_dense(uniform_neg_x)
    >>> print(np.linalg.norm(-dense_x - dense_neg_x))
    1.4502362601421634e-12
    """
    return ut3_scale(x_cores, -1.0, use_jax=use_jax)


def ut3_sub(
        x_cores: UniformTuckerTensorTrainCores,
        x_masks: UniformTuckerTensorTrainMasks,
        y_cores: UniformTuckerTensorTrainCores,
        y_masks: UniformTuckerTensorTrainMasks,
        use_jax: bool = False,
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
    >>> x_plus_y_cores, x_plus_y_masks = ut3.ut3_sub(x_cores, x_masks, y_cores, y_masks) # add x+y
    >>> dense_x = t3.t3_to_dense(x)
    >>> dense_y = t3.t3_to_dense(y)
    >>> dense_x_plus_y = ut3.ut3_to_dense(x_cores, x_masks)
    >>> print(np.linalg.norm(dense_x - dense_y - dense_x_plus_y))
    0.0
    """
    return ut3_add(x_cores, x_masks, *ut3_neg(y_cores,use_jax=use_jax), y_masks, use_jax=use_jax)


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
# def ut3_tangent_vector_to_ut3(
#         variations: typ.Tuple[
#             jnp.ndarray, # basis_variations, shape=(d, n, N)
#             jnp.ndarray, # tt_variations, shape=(d, r, n, r)
#         ],
#         orthogonal_basis_cores: jnp.ndarray,  # shape=(d, n, N)
#         left_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
#         right_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
#         up_orthogonal_tt_cores: jnp.ndarray,  # shape=(d, r, n, r)
# ) -> typ.Tuple[
#     jnp.ndarray, # tangent_basis_cores, shape=(d, 2n, N)
#     jnp.ndarray, # tangent_tt_cores, shape=(d, 2r, 2n, 2r)
# ]:
#     '''Rank 2r Tucker tensor train representation of tangent vector:
#             u(x,y,z,w) = ([dU1(B x) P1(B x)]) ([Q2(B y)        0]) ([Q3(B z)        0]) ([Q4(B w) ])
#                          (                  ) ([dU2(B y) P2(B y)]) ([dU3(B z) P3(B z)]) ([dU4(B w)])
#                          (         +        ) (         +        ) (        +         ) (    +     )
#                          ([R1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
#                          (                  ) ([R2(dB y)       0]) ([R3(dB z)       0]) ([R4(dB w)])
#     '''
#     basis_variations, tt_variations = variations
#
#     d, n, N = basis_variations.shape
#     r = tt_variations.shape[1]
#
#     tangent_basis_cores = jnp.concatenate([orthogonal_basis_cores, basis_variations], axis=1)
#
#     dU = tt_variations
#     R = up_orthogonal_tt_cores
#     P = left_orthogonal_tt_cores
#     Q = right_orthogonal_tt_cores
#     Z = jnp.zeros((d, r, n, r))
#
#     first_tangent_tt_core = jnp.concatenate([
#         jnp.concatenate([
#             jnp.concatenate([dU[:1], P[:1]], axis=3),
#             jnp.concatenate([Z[:1], Z[:1]], axis=3),
#         ], axis=1),
#         jnp.concatenate([
#             jnp.concatenate([R[:1], Z[:1]], axis=3),
#             jnp.concatenate([Z[:1], Z[:1]], axis=3),
#         ], axis=1)
#     ], axis=2)
#
#     mid_tangent_tt_cores = jnp.concatenate([
#         jnp.concatenate([
#             jnp.concatenate([Q[1:-1], Z[1:-1]], axis=3),
#             jnp.concatenate([dU[1:-1], P[1:-1]], axis=3),
#         ], axis=1),
#         jnp.concatenate([
#             jnp.concatenate([Z[1:-1], Z[1:-1]], axis=3),
#             jnp.concatenate([R[1:-1], Z[1:-1]], axis=3),
#         ], axis=1)
#     ], axis=2)
#
#     last_tangent_tt_core = jnp.concatenate([
#         jnp.concatenate([
#             jnp.concatenate([Q[-1:], Z[-1:]], axis=3),
#             jnp.concatenate([dU[-1:], Z[-1:]], axis=3),
#         ], axis=1),
#         jnp.concatenate([
#             jnp.concatenate([Z[-1:], Z[-1:]], axis=3),
#             jnp.concatenate([R[-1:], Z[-1:]], axis=3),
#         ], axis=1)
#     ], axis=2)
#
#     tangent_tt_cores = jnp.concatenate(
#         [first_tangent_tt_core, mid_tangent_tt_cores, last_tangent_tt_core],
#         axis=0
#     )
#
#     return tangent_basis_cores, tangent_tt_cores
#
#
# def ut3_attached_tangent_vector_to_ut3(
#         variations: typ.Tuple[
#             jnp.ndarray, # basis_variations, shape=(d, n, N)
#             jnp.ndarray, # tt_variations, shape=(d, r, n, r)
#         ],
#         orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
#         left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
#         right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
#         up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
# ) -> typ.Tuple[
#     jnp.ndarray, # tangent_basis_cores, shape=(d, 2n, N)
#     jnp.ndarray, # tangent_tt_cores, shape=(2, 2r, 2n, 2r)
# ]:
#     '''Rank 2r Tucker tensor train representation of *attached* tangent vector:
#             u(x,y,z,w) = ([dU1(B x) P1(B x)]) ([Q2(B y)        0]) ([Q3(B z)        0]) ([Q4(B w) ])
#                          (                  ) ([dU2(B y) P2(B y)]) ([dU3(B z) P3(B z)]) ([P4(B w) + dU4(B w)])
#                          (         +        ) (         +        ) (        +         ) (    +     )
#                          ([R1(dB x)       0]) ([0              0]) ([0              0]) ([0       ])
#                          (                  ) ([R2(dB y)       0]) ([R3(dB z)       0]) ([R4(dB w)])
#     '''
#     basis_variations, tt_variations = variations
#     # (orthogonal_basis_cores,
#     #  left_orthogonal_tt_cores,
#     #  right_orthogonal_tt_cores,
#     #  up_orthogonal_tt_cores,
#     #  _, _,
#     #  ) = base_representations
#
#     d, n, N = basis_variations.shape
#     r = tt_variations.shape[1]
#
#     tangent_basis_cores = jnp.concatenate([orthogonal_basis_cores, basis_variations], axis=1)
#
#     dU = tt_variations
#     R = up_orthogonal_tt_cores
#     P = left_orthogonal_tt_cores
#     Q = right_orthogonal_tt_cores
#     Z = jnp.zeros((d, r, n, r))
#
#     first_tangent_tt_core = jnp.concatenate([
#         jnp.concatenate([
#             jnp.concatenate([dU[:1], P[:1]], axis=3),
#             jnp.concatenate([Z[:1], Z[:1]], axis=3),
#         ], axis=1),
#         jnp.concatenate([
#             jnp.concatenate([R[:1], Z[:1]], axis=3),
#             jnp.concatenate([Z[:1], Z[:1]], axis=3),
#         ], axis=1)
#     ], axis=2)
#
#     mid_tangent_tt_cores = jnp.concatenate([
#         jnp.concatenate([
#             jnp.concatenate([Q[1:-1], Z[1:-1]], axis=3),
#             jnp.concatenate([dU[1:-1], P[1:-1]], axis=3),
#         ], axis=1),
#         jnp.concatenate([
#             jnp.concatenate([Z[1:-1], Z[1:-1]], axis=3),
#             jnp.concatenate([R[1:-1], Z[1:-1]], axis=3),
#         ], axis=1)
#     ], axis=2)
#
#     last_tangent_tt_core = jnp.concatenate([
#         jnp.concatenate([
#             jnp.concatenate([Q[-1:], Z[-1:]], axis=3),
#             jnp.concatenate([P[-1:] + dU[-1:], Z[-1:]], axis=3),
#         ], axis=1),
#         jnp.concatenate([
#             jnp.concatenate([Z[-1:], Z[-1:]], axis=3),
#             jnp.concatenate([R[-1:], Z[-1:]], axis=3),
#         ], axis=1)
#     ], axis=2)
#
#     tangent_tt_cores = jnp.concatenate(
#         [first_tangent_tt_core, mid_tangent_tt_cores, last_tangent_tt_core],
#         axis=0
#     )
#
#     return tangent_basis_cores, tangent_tt_cores
#
#
# def left_orthogonalize_utt(
#         tt_cores:       jnp.ndarray,  # shape=(d, r, n, r)
# ) -> jnp.ndarray: # new_tt_cores
#     d, r, n, _ = tt_cores.shape
#
#     def _orth_one_core(
#             prev_R: jnp.ndarray, # shape=(r, r)
#             G: jnp.ndarray, # shape=(r, n, r)
#     ):
#         G = jnp.einsum('ij,jak->iak', prev_R, G)
#         # Q, R = jnp.linalg.qr(G.reshape((r*n, r)), mode='reduced')
#         Q, ss, Vt = jnp.linalg.svd(G.reshape((r*n, r)), full_matrices=False)
#         R = jnp.einsum('i,ij->ij', ss, Vt)
#
#         G = Q.reshape((r, n, r))
#         return R, G
#
#     R0 = jnp.eye(r)
#     R, first_new_tt_cores = jax.lax.scan(_orth_one_core, R0, tt_cores[:-1])
#     last_new_tt_core = jnp.einsum('ij,djak->diak', R, tt_cores[-1:])
#     new_tt_cores = jnp.concatenate([first_new_tt_cores, last_new_tt_core], axis=0)
#     return new_tt_cores
#
#
# def right_orthogonalize_utt(
#         tt_cores:       jnp.ndarray,  # shape=(d, r, n, r)
# ) -> jnp.ndarray: # new_tt_cores
#     return left_orthogonalize_utt(tt_cores[::-1].swapaxes(1,3))[::-1].swapaxes(3,1)
#
#
# def orthogonalize_ut3_basis_cores(
#         basis_cores,  # shape=(d, n, N)
#         tt_cores,  # shape=(d, r, n, r)
# ) -> typ.Tuple[
#     jnp.ndarray, # new_basis_cores
#     jnp.ndarray, # new_tt_cores
# ]:
#     d, n, N = basis_cores.shape
#     r = tt_cores.shape[1]
#
#     # QQ, RR = jnp.linalg.qr(basis_cores.swapaxes(1,2), mode='reduced')
#     QQ, sss, VVt = jnp.linalg.svd(basis_cores.swapaxes(1, 2), full_matrices=False) # use SVD because QR sometimes yields nans
#     RR = jnp.einsum('da,dab->dab', sss, VVt)
#
#     new_basis_cores = QQ.swapaxes(2,1)
#     new_tt_cores = jnp.einsum('dab,dibj->diaj', RR, tt_cores)
#
#     n2 = QQ.shape[-1]
#     new_basis_cores = jnp.concatenate([new_basis_cores, jnp.zeros((d, n-n2, N))], axis=1)
#     new_tt_cores = jnp.concatenate([new_tt_cores, jnp.zeros((d, r, n-n2, r))], axis=2)
#
#     return new_basis_cores, new_tt_cores
#
#
# @jax.jit
# def ut3_svd_masked(
#         uniform_t3: typ.Tuple[
#             jnp.ndarray, # basis_cores, shape=(d, n, N)
#             jnp.ndarray, # tt_cores, shape=(d, r, N, r)
#         ],
#         masks: typ.Tuple[
#             jnp.ndarray,  # basis_cores_mask, shape=(d, n)
#             jnp.ndarray,  # tt_cores_mask, shape=(d+1, r)
#         ], # use to control rank truncation
# ) -> typ.Tuple[
#     typ.Tuple[
#         jnp.ndarray, # new_basis_cores, shape=(d, n, N)
#         jnp.ndarray, # new_tt_cores, shape=(d, r, n, r)
#     ],
#     jnp.ndarray, # basis_singular_values, shape=(d, n)
#     jnp.ndarray, # tt_singular_values, shape=(d+1, r)
# ]:
#     basis_cores, tt_cores = uniform_t3
#
#     shape_mask, basis_masks, tt_masks = masks
#
#     d, n, N = basis_cores.shape
#     r = tt_cores.shape[1]
#
#     basis_cores, tt_cores = orthogonalize_ut3_basis_cores(basis_cores, tt_cores)
#     tt_cores = right_orthogonalize_utt(tt_cores)
#
#     _, ss_tt00, _ = jnp.linalg.svd(tt_cores[0].reshape((r, n*r)), full_matrices=False)
#     ss_tt0 = jnp.concatenate([ss_tt00, jnp.zeros(r-len(ss_tt00))], axis=0)
#     if len(ss_tt0) != len(tt_masks[0]):
#         print('shape_mask.shape=', shape_mask.shape)
#         print('basis_masks.shape=', basis_masks.shape)
#         print('tt_masks.shape=', tt_masks.shape)
#         print('basis_cores.shape=', basis_cores.shape)
#         print('tt_cores.shape=', tt_cores.shape)
#         print('d=', d)
#         print('n=', n)
#         print('N=', N)
#         print('r=', r)
#         print('ss_tt0.shape=', ss_tt0.shape)
#         print('tt_masks[0].shape=', tt_masks[0].shape)
#         print('tt_cores[0].shape=', tt_cores[0].shape)
#         print('ss_tt00.shape=', ss_tt00.shape)
#
#
#
#     ss_tt0 = ss_tt0 * tt_masks[0]
#
#     def _step(
#             carry: jnp.ndarray,
#             x,
#     ):
#         Y = carry # shape=(r, r)
#         B, G, basis_mask, tt_mask = x
#
#         G = jnp.einsum('ij,jak->iak', Y, G) # shape=(r, n, r)
#         # Note: B.shape=(n, N)
#
#         M = G.swapaxes(1,2).reshape((r*r, n))
#         U, ss_basis, Vt = jnp.linalg.svd(M, full_matrices=False)
#         n2 = len(ss_basis)
#         U           = jnp.concatenate([U,           jnp.zeros((r*r, n-n2))],    axis=1)
#         ss_basis    = jnp.concatenate([ss_basis,    jnp.zeros((n-n2, ))],           axis=0)
#         Vt          = jnp.concatenate([Vt,          jnp.zeros((n-n2, n))],      axis=0)
#
#         U           = U         * basis_mask.reshape((1,-1))
#         ss_basis    = ss_basis  * basis_mask
#         Vt          = Vt        * basis_mask.reshape((-1,1))
#
#         new_B = jnp.einsum('ij,jk->ik', Vt, B)
#
#         M = jnp.einsum('ij,j->ij', U, ss_basis).reshape((r, r, n)).swapaxes(1,2).reshape((r*n, r))
#         U, ss_tt, Vt = jnp.linalg.svd(M, full_matrices=False)
#
#         U       = U     * tt_mask.reshape((1,-1))
#         ss_tt   = ss_tt * tt_mask
#         Vt      = Vt    * tt_mask.reshape((-1,1))
#
#         new_G = U.reshape((r, n, r))
#
#         Y_next = jnp.einsum('i,ij->ij', ss_tt, Vt)  # shape=(r, r)
#
#         return Y_next, (new_B, new_G, ss_basis, ss_tt)
#
#     Y0 = jnp.eye(r)
#     Yf, (new_basis_cores, new_tt_cores, basis_singular_values, tt_singular_values0) = jax.lax.scan(
#         _step,
#         Y0,
#         (basis_cores, tt_cores, basis_masks, tt_masks[1:]),
#     )
#
#     # G_last = jnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)[:, :, :, :r]
#     G_last = jnp.einsum('diaj,jk->diak', new_tt_cores[-1:], Yf)
#     new_tt_cores = jnp.concatenate([new_tt_cores[:-1], G_last], axis=0)
#
#     tt_singular_values = jnp.concatenate([ss_tt0.reshape((1, r)), tt_singular_values0], axis=0)
#     return (new_basis_cores, new_tt_cores), basis_singular_values, tt_singular_values
#
#
# @jax.jit
# def ut3_retract(
#         variations: typ.Tuple[
#             jnp.ndarray, # basis_variations, shape=(d, n, N)
#             jnp.ndarray, # tt_variations, shape=(d, r, n, r)
#         ],
#         orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
#         left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
#         right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
#         up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
#         doubled_rank_masks: typ.Tuple[
#             jnp.ndarray,  # shape_mask, shape=(d, N)
#             jnp.ndarray,  # tucker_cores_mask, shape=(d, 2*n)
#             jnp.ndarray,  # tt_cores_mask, shape=(d+1, 2*r)
#         ],  # use to specify ranks
# ) -> typ.Tuple[
#     jnp.ndarray, # retracted_basis_cores, shape=(d, n, N)
#     jnp.ndarray, # retracted_tt_cores, shape=(d, r, n, r)
# ]:
#     basis_variations, tt_variations = variations
#
#     d, n, N = basis_variations.shape
#     r = tt_variations.shape[1]
#
#     X = ut3_attached_tangent_vector_to_ut3(
#         variations,
#         orthogonal_basis_cores,
#         left_orthogonal_tt_cores,
#         right_orthogonal_tt_cores,
#         up_orthogonal_tt_cores,
#     )
#
#     (basis_cores0, tt_cores0), _, _ = ut3_svd_masked(
#         X,
#         doubled_rank_masks,
#         # basis_ranks, tt_ranks,
#     )
#
#     basis_cores = basis_cores0[:, :n, :]
#     tt_cores = tt_cores0[:, :r, :n, :r]
#     return basis_cores, tt_cores
#
#
# @jax.jit
# def construct_ut3_base_representations(
#         base_point: typ.Tuple[
#             jnp.ndarray, # basis_cores, shape=(d, n, N)
#             jnp.ndarray, # tt_cores, shape=(d, r, N, r)
#         ],
#         masks: typ.Tuple[
#             jnp.ndarray,  # basis_cores_mask, shape=(d, n)
#             jnp.ndarray,  # tt_cores_mask, shape=(d+1, r)
#         ],
# ) -> typ.Tuple[
#     jnp.ndarray, # orthogonal_basis_cores,      shape=(d, n, N)
#     jnp.ndarray, # left_orthogonal_tt_cores,    shape=(d, r, n, r)
#     jnp.ndarray, # right_orthogonal_tt_cores,   shape=(d, r, n, r)
#     jnp.ndarray, # up_orthogonal_tt_cores,      shape=(d, r, n, r)
#     jnp.ndarray, # nonorthogonal_basis_cores,   shape=(d, n, N)
#     jnp.ndarray, # nonorthogonal_tt_cores,      shape=(d, r, n, r)
# ]:
#     basis_cores, tt_cores = base_point
#     d, n, N = basis_cores.shape
#     r = tt_cores.shape[1]
#
#     shape_mask, tucker_mask, tt_mask = masks
#     basis_cores_mask = jnp.einsum('da,do->dao', tucker_mask, shape_mask)
#     tt_cores_mask = jnp.einsum('di,da,dj->diaj', tt_mask[:-1], tucker_mask, tt_mask[1:])
#
#     QQ, sss, VVt = jnp.linalg.svd(basis_cores.swapaxes(1, 2), full_matrices=False) # use SVD because QR sometimes yields nans
#     RR = jnp.einsum('da,dab->dab', sss, VVt)
#
#     orthogonal_basis_cores = QQ.swapaxes(2,1)
#
#     tt_cores = jnp.einsum('dab,dibj->diaj', RR, tt_cores)
#
#     n2 = QQ.shape[-1]
#     orthogonal_basis_cores = jnp.concatenate([orthogonal_basis_cores, jnp.zeros((d, n-n2, N))], axis=1)
#     tt_cores = jnp.concatenate([tt_cores, jnp.zeros((d, r, n-n2, r))], axis=2)
#
#     orthogonal_basis_cores = orthogonal_basis_cores * basis_cores_mask
#
#     right_orthogonal_tt_cores = right_orthogonalize_utt(tt_cores) * tt_cores_mask
#
#     def _process_one_core(
#             prev_R: jnp.ndarray,  # shape=(r, r)
#             x: jnp.ndarray,
#     ):
#         G, B, G_mask, B_mask  = x
#
#         G_tilde = jnp.einsum('ij,jak->iak', prev_R, G)
#         G_tilde = G_tilde * G_mask
#
#         Q, ss, Vt = jnp.linalg.svd(G_tilde.swapaxes(1, 2).reshape((r * r, n)), full_matrices=False)
#         R = jnp.einsum('a,ab->ab', ss, Vt)
#         B_tilde = jnp.einsum('ij,jo->io', R, B)
#         B_tilde = B_tilde * B_mask
#
#         n2 = Q.shape[1]
#         Q = jnp.concatenate([Q, jnp.zeros((r*r, n-n2))], axis=1)
#         G_up = Q.reshape(r,r,n).swapaxes(1,2)
#         G_up = G_up * G_mask
#
#         Q, ss, Vt = jnp.linalg.svd(G_tilde.reshape((r*n, r)), full_matrices=False)
#         R = jnp.einsum('a,ab->ab', ss, Vt)
#
#         G_left = Q.reshape((r, n, r))
#         G_left = G_left * G_mask
#
#         return R, (G_up, G_left, B_tilde, G_tilde)
#
#     R0 = jnp.eye(r)
#     R, (up_orthogonal_tt_cores, left_orthogonal_tt_cores0, nonorthogonal_basis_cores, nonorthogonal_tt_cores) = jax.lax.scan(
#         _process_one_core, R0,
#         (right_orthogonal_tt_cores, orthogonal_basis_cores, tt_cores_mask, basis_cores_mask),
#     )
#
#     first_left_orthogonal_tt_cores = left_orthogonal_tt_cores0[:-1]
#     last_left_orthogonal_tt_core0 = left_orthogonal_tt_cores0[-1:] * tt_cores_mask[-1:]
#     last_left_orthogonal_tt_core = jnp.einsum('diaj,jk->diak', last_left_orthogonal_tt_core0, R)
#     left_orthogonal_tt_cores = jnp.concatenate(
#         [first_left_orthogonal_tt_cores, last_left_orthogonal_tt_core],
#         axis=0,
#     )
#
#     return (
#         orthogonal_basis_cores,
#         left_orthogonal_tt_cores,
#         right_orthogonal_tt_cores,
#         up_orthogonal_tt_cores,
#         nonorthogonal_basis_cores,
#         nonorthogonal_tt_cores,
#     )
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
# @jax.jit
# def apply_masks(
#         X: typ.Tuple[
#             jnp.ndarray, # basis_cores, shape=(d, n, N)
#             jnp.ndarray, # tt_cores, shape=(d, r, n, r)
#         ],
#         masks: typ.Tuple[
#             jnp.ndarray,  # shape_masks, shape=(d, N)
#             jnp.ndarray,  # basis_mask, shape=(d, n)
#             jnp.ndarray,  # tt_mask, shape=(d+1, r)
#         ],
# ) -> typ.Tuple[
#     jnp.ndarray, # masked_basis_cores, shape=(d, n, N)
#     jnp.ndarray, # masked_tt_cores, shape=(d, r, n, r)
# ]:
#     shape_mask, tucker_mask, tt_mask = masks
#     BB, GG = X
#     BB = jnp.einsum('dao,do->dao', BB, shape_mask)
#     BB = jnp.einsum('dao,da->dao', BB, tucker_mask)
#     GG = jnp.einsum('diaj,di->diaj', GG, tt_mask[:-1])
#     GG = jnp.einsum('diaj,da->diaj', GG, tucker_mask)
#     GG = jnp.einsum('diaj,dj->diaj', GG, tt_mask[1:])
#     return BB, GG
#
#
#
#
#
# def ut3_to_dense(
#         uniform_t3: typ.Tuple[
#             jnp.ndarray, # uniform_basis_cores, shape=(d, n, N)
#             jnp.ndarray, # uniform_tt_cores, shape=(d, r, n, r)
#         ],
#         original_shape,
# ):
#     uniform_basis_cores, uniform_tt_cores = uniform_t3
#     d, n, N = uniform_basis_cores.shape
#     r = uniform_tt_cores.shape[1]
#
#     tt_ranks = tuple([1] + [r]*(d-1) + [1])
#     tucker_ranks = tuple([n]*(d))
#
#     t3 = ut3_to_t3(uniform_t3, original_shape, tucker_ranks, tt_ranks)
#     return t3_to_dense(t3)
#
#
#
# def ut3_project_dense_tensor_onto_tangent_space(
#         T:                          jnp.ndarray, # shape=(N, N, ..., N)
#         orthogonal_basis_cores:     jnp.ndarray, # shape=(d, n, N)
#         left_orthogonal_tt_cores:   jnp.ndarray, # shape=(d, r, n, r)
#         right_orthogonal_tt_cores:  jnp.ndarray, # shape=(d, r, n, r)
#         up_orthogonal_tt_cores:     jnp.ndarray, # shape=(d, r, n, r)
# ):
#     '''Very expensive, probably only useful for debugging other functions'''
#     d, n, N = orthogonal_basis_cores.shape
#     r = left_orthogonal_tt_cores.shape[1]
#
#     XX = []
#     for ii in range(d):
#         X = T.reshape((1,) + T.shape + (1,))
#         X = jnp.pad(
#             X,
#             [(0, r-1)] + [(0, 0)]*d + [(0, r-1)],
#         )
#         for jj in range(ii):
#             B = orthogonal_basis_cores[jj]
#             P = left_orthogonal_tt_cores[jj]
#             X = jnp.einsum('ac,ec...->ea...', B, X)
#             X = jnp.einsum('axb,ax...->b...', P, X)
#
#         for jj in range(d-1, ii, -1):
#             B = orthogonal_basis_cores[jj]
#             Q = right_orthogonal_tt_cores[jj]
#             X = jnp.einsum('bc,...ce->...be', B, X)
#             X = jnp.einsum('...xb,axb->...a', X, Q)
#
#         XX.append(X)
#
#     BB_tilde = []
#     GG_tilde = []
#     for ii in range(d):
#         X = XX[ii]
#         B = orthogonal_basis_cores[ii]
#         S = up_orthogonal_tt_cores[ii]
#         G_tilde = jnp.einsum('aob,ko->akb', X, B)
#         B_tilde = jnp.einsum('aob,akb->ko', X, S)
#         BB_tilde.append(B_tilde)
#         GG_tilde.append(G_tilde)
#
#     ungauged_basis_variations = jnp.stack(BB_tilde)
#     ungauged_tt_variations = jnp.stack(GG_tilde)
#     ungauged_variations = (ungauged_basis_variations, ungauged_tt_variations)
#
#     variations = ut3_orthogonal_gauge_projection_using_map(
#         ungauged_variations,
#         orthogonal_basis_cores,
#         left_orthogonal_tt_cores,
#     )
#     return variations
#
