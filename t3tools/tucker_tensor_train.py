# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ

try:
    import jax.numpy as jnp
except:
    print('jax import failed in tucker_tensor_train. Defaulting to numpy.')
    jnp = np

NDArray = typ.Union[np.ndarray, jnp.ndarray]


###########################################
########    Tucker Tensor Train    ########
###########################################

__all__ = [
    'TuckerTensorTrain',
    'T3Structure',
    't3_structure',
    't3_apply',
    't3_entry',
    't3_to_dense',
    't3_reverse',
    't3_check',
    't3_zeros',
    't3_minimal_ranks',
    't3_pad_ranks',
    't3_save',
    't3_load',
]

#####################################################
####################    Types    ####################
#####################################################

TuckerTensorTrain = typ.Tuple[
    typ.Sequence[NDArray], # basis_cores, len=d, elm_shape=(ni, Ni)
    typ.Sequence[NDArray], # tt_cores, len=d, elm_shape=(ri, ni, r(i+1))
]
"""
Tuple containing Tucker Tensor Train basis cores and TT-cores.

Tensor network diagram::

    1 -- G0 -- G1 -- G2 -- G3 -- 1
         |     |     |     |
         B0    B1    B2    B3
         |     |     |     |
    
Components:
    - **basis_cores** : *Sequence[NDArray]*
        Basis matrices (B0, ..., Bd) with shape (ni, Ni) for i=1,...,d. len(basis_cores)=d.
    - **tt_cores** : *Sequence[NDArray]*
        Tensor train cores (G0, ..., Gd) with shape (ri, ni, r(i+1)) for i=1,...,d. len(tt_cores)=d. r0=rd=1.

Structure:
    - shape: (N1, ..., Nd)
    - tucker ranks: (n1, ..., nd)
    - tt ranks: (1, r1, ..., r(d-1), 1)

Examples
--------
>>> from numpy.random import randn
>>> from t3tools.tucker_tensor_train import *
>>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
>>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
>>> x = (basis_cores, tt_cores)
>>> print(isinstance(x, TuckerTensorTrain))
    True
>>> t3_check_correctness(x) # does nothing if everything is ok
>>> shape, tucker_ranks, tt_ranks = t3_structure(x)
>>> print(shape)
    (14, 15, 16)
>>> print(tucker_ranks)
    (4, 5, 6)
>>> print(tt_ranks)
    (1, 3, 2, 1)
"""


T3Structure = typ.Tuple[
    typ.Sequence[int], # shape, len=d
    typ.Sequence[int], # tucker_ranks, len=d
    typ.Sequence[int], # tt_ranks, len=d+1
]
"""
Tuple containing the structure of a Tucker Tensor Train.

Tensor network diagram::

    1 -[r0]- G0 -[r1]- G1 -[r2]- G2 -[r3]- G3 -[r4]- 1
             |         |         |         |
             [n0]      [n1]      [n2]      [n3]
             |         |         |         |
             B0        B1        B2        B3
             |         |         |         |
             [N0]      [N1]      [N2]      [N3]
             |         |     |             |
         

Components:
    - **shape** : *Sequence[NDArray]*
        Shape of the represented tensor, (N1, ..., Nd). len=d
    - **tucker_ranks** : *Sequence[NDArray]*
        Tucker ranks, (n1, ..., nd). len=d
    - **tt_ranks** : *Sequence[NDArray]*
        TT-ranks, (1, r1, ..., r(d-1), 1). len=d+1. tt_ranks[0]=tt_ranks[-1]=1

Examples
--------
>>> from numpy.random import randn
>>> from t3tools.tucker_tensor_train import *
>>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
>>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
>>> x = (basis_cores, tt_cores)
>>> shape, tucker_ranks, tt_ranks = t3_structure(x)
>>> print(shape)
    (14, 15, 16)
>>> print(tucker_ranks)
    (4, 5, 6)
>>> print(tt_ranks)
    (1, 3, 2, 1)
"""


#####################################################################
########    Structural properties and consistency checks    #########
#####################################################################

def t3_structure(
        x: TuckerTensorTrain,
) -> T3Structure:
    """Get the structure of a Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train with shape (N1, ..., Nd), Tucker ranks (n1, ..., nd), and TT-ranks (1, r1, ..., r(d-1), 1)).

    Returns
    -------
    T3Structure
        ((N1, ..., Nd), (n1, ..., nd), (1, r1, ... r(d-1), 1)), the structure of the Tucker tensor train.

    See Also
    --------
    TuckerTensorTrain
    T3Structure
    t3_shape
    t3_tucker_ranks
    t3_tt_ranks

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> shape, tucker_ranks, tt_ranks = t3_structure(x)
    >>> print(shape)
        (14, 15, 16)
    >>> print(tucker_ranks)
        (4, 5, 6)
    >>> print(tt_ranks)
        (1, 3, 2, 1)
    """
    basis_cores, tt_cores = x
    shape = tuple([B.shape[1] for B in basis_cores])
    tucker_ranks = tuple([B.shape[0] for B in basis_cores])
    tt_ranks = tuple([int(tt_cores[0].shape[0])] + [int(G.shape[2]) for G in tt_cores])
    return shape, tucker_ranks, tt_ranks


def t3_check(
        x: TuckerTensorTrain,
) -> None:
    '''Check correctness / consistency of Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain

    Raises
    ------
    RuntimeError
        Error raised if the cores of the Tucker tensor train have inconsistent shapes.

    See Also
    --------
    TuckerTensorTrain
    T3Structure
    t3_shape
    t3_tucker_ranks
    t3_tt_ranks
    t3_structure

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x) # Nothing happens because T3 is consistent

    >>> basis_cores = (randn(4,14), randn(5,15))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        2 = len(basis_cores) != len(tt_cores) = 3

    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        tt_cores[0] is not a 3-tensor. shape=(4, 3)

    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(9,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        First TT rank is not one. tt_ranks = (9, 3, 2, 1)

    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,9))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        Last TT rank is not one. tt_ranks = (1, 3, 2, 9)

    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,9), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        (1, 3, 2, 1) = left_tt_ranks != right_tt_ranks = (1, 9, 2, 1)

    >>> basis_cores = (randn(4,14), randn(5,15,3), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        basis_cores[1] is not a matrix. shape=(5, 15, 3)

    >>> basis_cores = (randn(4,14), randn(5,15), randn(9,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> t3_check(x)
        RuntimeError: Inconsistent TuckerTensorTrain.
        9 = basis_cores[2].shape[0] != tt_cores[2].shape[1] = 6
    '''
    basis_cores, tt_cores = x
    if len(basis_cores) != len(tt_cores):
        raise RuntimeError(
            'Inconsistent TuckerTensorTrain.\n'
            + str(len(basis_cores)) + ' = len(basis_cores) != len(tt_cores) = ' + str(len(tt_cores))
        )

    for ii, G in enumerate(tt_cores):
        if len(G.shape) != 3:
            raise RuntimeError(
                'Inconsistent TuckerTensorTrain.\n'
                + 'tt_cores[' + str(ii) + '] is not a 3-tensor. shape=' + str(G.shape)
            )

    right_tt_ranks = tuple([int(tt_cores[0].shape[0])] + [int(G.shape[2]) for G in tt_cores])
    left_tt_ranks = tuple([int(G.shape[0]) for G in tt_cores] + [int(tt_cores[-1].shape[2])])
    if left_tt_ranks != right_tt_ranks:
        raise RuntimeError(
            'Inconsistent TuckerTensorTrain.\n'
            + str(left_tt_ranks) + ' = left_tt_ranks != right_tt_ranks = ' + str(right_tt_ranks)
        )

    if right_tt_ranks[0] != 1:
        raise RuntimeError(
            'Inconsistent TuckerTensorTrain.\n'
            + str('First TT rank is not one. tt_ranks = ' + str(right_tt_ranks))
        )

    if right_tt_ranks[-1] != 1:
        raise RuntimeError(
            'Inconsistent TuckerTensorTrain.\n'
            + str('Last TT rank is not one. tt_ranks = ' + str(right_tt_ranks))
        )

    for ii, B in enumerate(basis_cores):
        if len(B.shape) != 2:
            raise RuntimeError(
                'Inconsistent TuckerTensorTrain.\n'
                + 'basis_cores['+str(ii)+'] is not a matrix. shape='+str(B.shape)
            )

    for ii, (B, G) in enumerate(zip(basis_cores, tt_cores)):
        if B.shape[0] != G.shape[1]:
            raise RuntimeError(
                'Inconsistent TuckerTensorTrain.\n'
                + str(B.shape[0]) + ' = basis_cores[' + str(ii) + '].shape[0]'
                + ' != '
                + 'tt_cores[' + str(ii) + '].shape[1] = ' + str(G.shape[1])
            )


###########################################################
################    Basic T3 functions    #################
###########################################################

def t3_to_dense(
        x: TuckerTensorTrain,
        use_jax: bool = False,
) -> NDArray:
    """Contract Tucker tensor train to dense tensor.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train with shape (N1, ..., Nd).
    use_jax: bool
        Use jax if True, numpy if False. Default: False

    Returns
    -------
    NDArray
        Dense tensor represented by x, which has shape (N1, ..., Nd)

    See Also
    --------
    TuckerTensorTrain
    t3_shape

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> x_dense = t3_to_dense(x)
    >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', *(basis_cores + tt_cores))
    >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
        7.48952547844518e-16
    """
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    big_tt_cores = [xnp.einsum('iaj,ab->ibj', G, U) for G, U in zip(tt_cores, basis_cores)]

    G = big_tt_cores[0]
    rL, n, rR = G.shape
    T = G.reshape((n, rR))
    for G in big_tt_cores[1:-1]:
        T = xnp.tensordot(T, G, axes=1)
    G = big_tt_cores[-1]
    rL, n, rR = G.shape
    T = xnp.tensordot(T, G.reshape((rL, n)), axes=1)

    return T


def t3_reverse(
        x: TuckerTensorTrain,
) -> NDArray:
    """Reverse Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train with shape=(N1, ..., Nd), tucker_ranks=(n1,...,nd), tt_ranks=(1,r1,...,r(d-1),1)

    Returns
    -------
    reversed_x : TuckerTensorTrain
        Tucker tensor train with index order reversed. shape=(Nd, ..., N1), tucker_ranks=(nd,...,n1), tt_ranks=(1,r(d-1),...,r1,1)

    See Also
    --------
    TuckerTensorTrain
    t3_structure

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> print(t3_structure(x))
        ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
    >>> reversed_x = t3_reverse(x)
    >>> print(t3_structure(reversed_x))
        ((16, 15, 14), (6, 5, 4), (1, 2, 3, 1))
    >>> x_dense = t3_to_dense(x)
    >>> reversed_x_dense = t3_to_dense(reversed_x)
    >>> x_dense2 = reversed_x_dense.transpose([2,1,0])
    >>> print(np.linalg.norm(x_dense - x_dense2))
        1.859018050214056e-13
    """
    basis_cores, tt_cores = x

    reversed_basis_cores = tuple([B.copy() for B in basis_cores[::-1]])
    reversed_tt_cores = tuple([G.swapaxes(0,2).copy() for G in tt_cores[::-1]])
    reversed_x = (reversed_basis_cores, reversed_tt_cores)
    return reversed_x


def t3_zeros(
        structure:  T3Structure,
        use_jax:    bool = False,
) -> TuckerTensorTrain:
    """Construct Tucker tensor train of zeros.

    Parameters
    ----------
    structure:  T3Structure
        Tucker tensor train structure, (shape, tucker_ranks, tt_ranks)=((N1,...,Nd), (n1,...,nd), (1,r1,...,r(d-1),1))).
    use_jax: bool
        Use jax if True, numpy if False. Default: False

    Returns
    -------
    NDArray
        Dense tensor represented by x, which has shape (N1, ..., Nd)

    See Also
    --------
    TuckerTensorTrain
    T3Structure

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> shape = (14, 15, 16)
    >>> tucker_ranks = (4, 5, 6)
    >>> tt_ranks = (1, 3, 2, 1)
    >>> structure = (shape, tucker_ranks, tt_ranks)
    >>> z = t3_zeros(structure)
    >>> print(np.linalg.norm(t3_to_dense(z)))
        0.0
    """
    xnp = jnp if use_jax else np

    shape, tucker_ranks, tt_ranks = structure

    tt_cores = tuple([xnp.zeros((tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])) for ii in range(len(tucker_ranks))])
    basis_cores = tuple([xnp.zeros((n, N)) for n, N  in zip(tucker_ranks, shape)])
    z = (basis_cores, tt_cores)
    return z


def t3_save(
        file,
        x: TuckerTensorTrain,
) -> None:
    """Save Tucker tensor train to file with numpy.savez()

    Parameters
    ----------
    file:  str or file
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    x: TuckerTensorTrain
        The Tucker tensor train to save

    Raises
    ------
    RuntimeError
        Error raised if the Tucker tensor train is inconsistent, or fails to save.

    See Also
    --------
    TuckerTensorTrain
    t3_load
    t3_check

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> fname = 't3_file'
    >>> t3_save(fname, x)
    >>> x2 = t3_load(fname, use_jax=False)
    >>> basis_cores2, tt_cores2 = x2
    >>> print([np.linalg.norm(B - B2) for B, B2 in zip(basis_cores, basis_cores2)])
        [0.0, 0.0, 0.0]
    >>> print([np.linalg.norm(G - G2) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
    """
    t3_check(x)
    basis_cores, tt_cores = x
    cores_dict = {'basis_cores_'+str(ii): basis_cores[ii] for ii in range(len(basis_cores))}
    cores_dict.update({'tt_cores_'+str(ii): tt_cores[ii] for ii in range(len(tt_cores))})

    try:
        np.savez(file, **cores_dict)
    except RuntimeError:
        print('Failed to save TuckerTensorTrain to file')


def t3_load(
        file,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    """Save Tucker tensor train to file with numpy.savez()

    Parameters
    ----------
    file:  str or file
        Either the filename (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the filename if it is not
        already there.
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train loaded from the file

    Raises
    ------
    RuntimeError
        Error raised if the Tucker tensor train fails to load, or is inconsistent.

    See Also
    --------
    TuckerTensorTrain
    t3_save
    t3_check

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> fname = 't3_file'
    >>> t3_save(fname, x)
    >>> x2 = t3_load(fname)
    >>> basis_cores2, tt_cores2 = x2
    >>> print([np.linalg.norm(B - B2) for B, B2 in zip(basis_cores, basis_cores2)])
        [0.0, 0.0, 0.0]
    >>> print([np.linalg.norm(G - G2) for G, G2 in zip(tt_cores, tt_cores2)])
        [0.0, 0.0, 0.0]
    """
    if isinstance(file, str):
        if not file.endswith('.npz'):
            file = file + '.npz'

    try:
        d = np.load(file)
    except RuntimeError:
        print('Failed to load TuckerTensorTrain from file')

    assert (len(d.files) % 2 == 0)
    num_cores = len(d.files) // 2
    basis_cores = [d['basis_cores_' + str(ii)] for ii in range(num_cores)]
    tt_cores = [d['tt_cores_' + str(ii)] for ii in range(num_cores)]

    if use_jax:
        basis_cores = [jnp.array(B) for B in basis_cores]
        tt_cores = [jnp.array(G) for G in tt_cores]

    x = (tuple(basis_cores), tuple(tt_cores))
    t3_check(x)
    return x


###############################################################################
########    Scalar valued multilinear function applies and entries    #########
###############################################################################

def t3_apply(
        x: TuckerTensorTrain, # shape=(N1,...,Nd)
        vecs: typ.Sequence[NDArray], # len=d, elm_shape=(Ni,) or (num_applies, Ni)
        use_jax: bool = False,
) -> NDArray:
    '''Contract TuckerTensorTrain with vectors in all indices.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train. shape=(N1,...,Nd)
    vecs: typ.Sequence[NDArray]
        Vectors to contract with indices of x. len=d, elm_shape=(Ni,) or (num_applies, Ni) if vectorized
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    NDArray or scalar
        Result of contracting x with the vectors in all indices.
        scalar if vecs elements are vectors, NDArray with shape (num_applies,) if vecs elements are matrices (i.e., vectorized applies)

    Raises
    ------
    RuntimeError
        Error raised if the provided vectors in vecs are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_entry

    Notes
    -----
    Algorithm contracts vectors with cores of the TuckerTensorTrains in a zippering fashion from left to right.

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> vecs = [randn(14), randn(15), randn(16)]
    >>> result = t3_apply(x, vecs)
    >>> result2 = np.einsum('ijk,i,j,k', t3_to_dense(x), vecs[0], vecs[1], vecs[2])
    >>> print(np.abs(result - result2))

    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> vecs = [randn(3,14), randn(3,15), randn(3,16)]
    >>> result = t3_apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', t3_to_dense(x), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
        1.3334750051052994e-12
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    if len(vecs)  != len(shape):
        raise RuntimeError(
            'Attempted to apply TuckerTensorTrain to wrong number of vectors.'
            + str(str(len(shape)) + ' = num_indices != len(vecs) = ' + str(len(vecs)))
        )

    vecs_dims = [len(v.shape) for v in vecs]
    if vecs_dims != [vecs_dims[0]]*len(vecs_dims):
        raise RuntimeError(
            'Inconsistent array dimensions for vecs.'
            + '[len(v.shape) for v in vecs]=' + str([len(v.shape) for v in vecs])
        )

    if vecs_dims[0] == 1:
        vecs = [v.reshape((1,-1)) for v in vecs]

    num_applies = vecs[0].shape[0]
    if [v.shape[0] for v in vecs] != [num_applies] * len(vecs):
        raise RuntimeError(
            'Inconsistent numbers of applies per index.'
            + '[v.shape[0] for v in vecs]=' + str([v.shape[0] for v in vecs])
        )

    vector_sizes = tuple([v.shape[1] for v in vecs])
    if vector_sizes != shape:
        raise RuntimeError(
            'Input vector sizes to not match tensor shape.'
            + str(vector_sizes) + ' = vector_sizes != x_shape = ' + str(shape)
        )

    mu_na = xnp.ones((num_applies, 1))
    for V_ni, B_xi, G_axb in zip(vecs, basis_cores, tt_cores):
        v_nx = xnp.einsum('ni,xi->nx', V_ni, B_xi)
        g_anb = xnp.einsum('axb,nx->anb', G_axb, v_nx)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        mu_na = mu_nb

    result = mu_na.reshape((num_applies,))
    if num_applies == 1:
        result = result[0]

    return result


def t3_entry(
        x: TuckerTensorTrain, # shape=(N1,...,Nd)
        index: typ.Union[typ.Sequence[int], typ.Sequence[typ.Sequence[int]]], # len=d. one entry: typ.Sequence[int]. many entries: typ.Sequence[typ.Sequence[int]], elm_size=num_entries
        use_jax: bool = False,
) -> NDArray:
    '''Compute an entry (or multiple entries) of a TuckerTensorTrain.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train. shape=(N1,...,Nd)
    index: typ.Union[typ.Sequence[int], typ.Sequence[typ.Sequence[int]]]
        Index of the desired entry (typ.Sequence[int]), or indices of desired entries (typ.Sequence[typ.Sequence[int]])
        len(index)=d. If many entries: elm_size=num_entries
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    scalar or NDArray
        Desired entry or entries.
        scalar if one entry, NDArray with shape (num_entries,) if many entries (i.e., vectorized entry computation)

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_apply

    Notes
    -----
    Algorithm contracts core slices of the TuckerTensorTrains in a zippering fashion from left to right.

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> index = [9, 4, 7]
    >>> result = t3_entry(x, index)
    >>> result2 = t3_to_dense(x)[9, 4, 7]
    >>> print(np.abs(result - result2))
        0.0

    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> index = [[9,8], [4,10], [7,13]] # get 2 entries at once
    >>> entries = t3_entry(x, index)
    >>> x_dense = t3_to_dense(x)
    >>> entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
    >>> print(np.linalg.norm(entries - entries2))
        1.7763568394002505e-15
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    if len(index)  != len(shape):
        raise RuntimeError(
            'Wrong number of indices for TuckerTensorTrain.'
            + str(str(len(shape)) + ' = num tensor indices != num provided indices = ' + str(len(index)))
        )

    if isinstance(index[0], int):
        index = [[ind] for ind in index]
    else:
        index = [list(ind) for ind in index]

    num_entries = len(index[0])
    if [len(ind) for ind in index] != [num_entries] * len(shape):
        raise RuntimeError(
            'Inconsistent numbers of index entries across different dimensions. The following should be all the same:'
            + '[len(ind) for ind in index]=' + str([len(ind) for ind in index])
        )

    mu_na = xnp.ones((num_entries, 1))
    for ind, B_xi, G_axb in zip(index, basis_cores, tt_cores):
        v_xn = B_xi[:, ind]
        g_anb = xnp.einsum('axb,xn->anb', G_axb, v_xn)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        mu_na = mu_nb

    result = mu_na.reshape((num_entries,))
    if num_entries == 1:
        result = result[0]

    return result


########################################################################
########################    Rank adjustment    #########################
########################################################################

def t3_minimal_ranks(
        structure: T3Structure,
) -> typ.Tuple[
    typ.Tuple[int,...], # new_tucker_ranks
    typ.Tuple[int,...], # new_tt_ranks
]:
    '''Find minimal ranks for a TuckerTensorTrain with a given structure. (remove useless rank)
    '''
    shape, tucker_ranks, tt_ranks = structure
    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = int(np.minimum(new_tucker_ranks[ii], shape[ii]))

    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = int(np.minimum(rL, n*rR))

    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = int(np.minimum(n, rL*rR))
        rR =int(np.minimum(rR, rL*n))
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    return tuple(new_tucker_ranks), tuple(new_tt_ranks)


def t3_pad_ranks(
        x:                  TuckerTensorTrain,
        new_tucker_ranks:   typ.Sequence[int],
        new_tt_ranks:       typ.Sequence[int],
) -> TuckerTensorTrain:
    '''Increase TuckerTensorTrain ranks via zero padding.
    '''
    shape, old_tucker_ranks, old_tt_ranks = t3_structure(x)
    num_cores = len(shape)
    assert(len(old_tucker_ranks) == len(new_tucker_ranks))
    assert(len(old_tt_ranks) == len(new_tt_ranks))

    delta_tucker_ranks  = [r_new - r_old for r_new, r_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    basis_cores, tt_cores = x

    new_basis_cores = []
    for ii in range(num_cores):
        new_basis_cores.append(jnp.pad(
            basis_cores[ii],
            ((0,delta_tucker_ranks[ii]), (0,0)),
        ))

    new_tt_cores = []
    for ii in range(num_cores):
        new_tt_cores.append(jnp.pad(
            tt_cores[ii],
            (
                (0,delta_tt_ranks[ii]),
                (0,delta_tucker_ranks[ii]),
                (0,delta_tt_ranks[ii+1]),
            ),
        ))

    return tuple(new_basis_cores), tuple(new_tt_cores)







