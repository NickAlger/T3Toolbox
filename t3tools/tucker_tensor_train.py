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


###########################################
########    Tucker Tensor Train    ########
###########################################

__all__ = [
    'TuckerTensorTrain',
    'T3Structure',
    't3_shape',
    't3_tucker_ranks',
    't3_tt_ranks',
    't3_structure',
    't3_add',
    't3_scale',
    't3_dot_t3',
    't3_apply',
    't3_entry',
    't3_svd',
    't3_to_dense',
    't3_reverse',
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
    'orthogonalize_ith_basis_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'inner_svd_ith_tt_core',
    'outer_svd_ith_tt_core',
    't3_check',
    't3_zeros',
    't3_remove_useless_rank',
    't3_pad_rank',
    'tucker_svd_dense',
    't3_svd_dense',
    't3_save',
    't3_load',
    #'t3_reverse',
]



#####################################################
####################    Types    ####################
#####################################################

NDArray = typ.Union[np.ndarray, jnp.ndarray]


#: Tuple containing Tucker Tensor Train basis cores and TT-cores.
#:
#: Components:
#:  - **basis_cores** : *Sequence[NDArray]*
#:    Basis matrices with shape (ni, Ni) for i=1,...,d. len(basis_cores)=d.
#:  - **tt_cores** : *Sequence[NDArray]*
#:    Tensor train cores with shape (ri, ni, r(i+1)) for i=1,...,d. len(tt_cores)=d. r0=rd=1.
#:
#: Structure:
#:  - shape: (N1, ..., Nd)
#:  - tucker ranks: (n1, ..., nd)
#:  - tt ranks: (1, r1, ..., r(d-1), 1)
#:
#: Examples
#: --------
#:  >>> from numpy.random import randn
#:  >>> from t3tools.tucker_tensor_train import *
#:  >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
#:  >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
#:  >>> x = (basis_cores, tt_cores)
#:  >>> print(isinstance(x, TuckerTensorTrain))
#:      True
#:  >>> t3_check_correctness(x) # does nothing if everything is ok
#:  >>> shape, tucker_ranks, tt_ranks = t3_structure(x)
#:  >>> print(shape)
#:      (14, 15, 16)
#:  >>> print(tucker_ranks)
#:      (4, 5, 6)
#:  >>> print(tt_ranks)
#:      (1, 3, 2, 1)
TuckerTensorTrain = typ.Tuple[
    typ.Sequence[NDArray], # basis_cores, len=d, elm_shape=(ni, Ni)
    typ.Sequence[NDArray], # tt_cores, len=d, elm_shape=(ri, ni, r(i+1))
]


#: Tuple containing the structure of a Tucker Tensor Train.
#:
#: Components:
#:  - **shape** : *Sequence[NDArray]*
#:    Shape of the represented tensor, (N1, ..., Nd). len=d
#:  - **tucker_ranks** : *Sequence[NDArray]*
#:    Tucker ranks, (n1, ..., nd). len=d
#:  - **tt_ranks** : *Sequence[NDArray]*
#:     TT-ranks, (1, r1, ..., r(d-1), 1). len=d+1. tt_ranks[0]=tt_ranks[-1]=1
#:
#: Examples
#: --------
#:  >>> from numpy.random import randn
#:  >>> from t3tools.tucker_tensor_train import *
#:  >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
#:  >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
#:  >>> x = (basis_cores, tt_cores)
#:  >>> shape, tucker_ranks, tt_ranks = t3_structure(x)
#:  >>> print(shape)
#:      (14, 15, 16)
#:  >>> print(tucker_ranks)
#:      (4, 5, 6)
#:  >>> print(tt_ranks)
#:      (1, 3, 2, 1)
T3Structure = typ.Tuple[
    typ.Sequence[int], # shape, len=d
    typ.Sequence[int], # tucker_ranks, len=d
    typ.Sequence[int], # tt_ranks, len=d+1
]


#####################################################################
########    Structural properties and consistency checks    #########
#####################################################################

def t3_shape(
        x: TuckerTensorTrain,
) -> typ.Tuple[int,...]: # shape of x, (N1,N2,...,Nd), len=d
    """Get the shape of the tensor represented by a Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train representation of a tensor of shape (N1, N2, ..., Nd).

    Returns
    -------
    typ.Tuple[int,...]
        (N1, N2, ..., Nd), the shape of tensor represented by x.

    See Also
    --------
    TuckerTensorTrain
    T3Structure
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
    >>> print(t3_shape(x))
        (14, 15, 16)
    >>> print(t3_to_dense(x).shape == t3_shape(x))
        True
    """
    basis_cores, tt_cores = x
    return tuple([B.shape[1] for B in basis_cores])


def t3_tucker_ranks(
        x: TuckerTensorTrain,
) -> typ.Tuple[int,...]: # tucker_ranks=(n1,n2,...,nd)
    '''Get the Tucker ranks of a Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train with Tucker ranks (n1, n2, ..., nd)
        
    Returns
    -------
    typ.Tuple[int,...]
        (n1, n2, ..., nd), the Tucker ranks.

    See Also
    --------
    TuckerTensorTrain
    T3Structure
    t3_shape
    t3_tt_ranks
    t3_structure

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> print(t3_tucker_ranks(x))
        (4, 5, 6)
    '''
    basis_cores, tt_cores = x
    return tuple([B.shape[0] for B in basis_cores])


def t3_tt_ranks(
        x: TuckerTensorTrain,
) -> typ.Tuple[int,...]: # tt_ranks=(r1,r2,...,n(d+1))
    '''Get the TT-ranks of a Tucker tensor train.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train with TT-ranks (1, r1, r2, ..., r(d-1), 1)

    Returns
    -------
    typ.Tuple[int,...]
        (1, r1, r2, ..., r(d-1), 1), the TT-ranks.

    See Also
    --------
    TuckerTensorTrain
    T3Structure
    t3_shape
    t3_tucker_ranks
    t3_structure

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> print(t3_tt_ranks(x))
        (1,3,2,1)
    '''
    basis_cores, tt_cores = x
    return tuple([int(tt_cores[0].shape[0])] + [int(G.shape[2]) for G in tt_cores])


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
    return t3_shape(x), t3_tucker_ranks(x), t3_tt_ranks(x)


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


###########################################################################
########    Linear algebra operations, inner product, and norm    #########
###########################################################################

def t3_add(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    """Add TuckerTensorTrains x, y with the same shape, yielding a TuckerTensorTrain z=x+y with summed ranks.

    Parameters
    ----------
    x: TuckerTensorTrain
        First summand. structure=((N1,...,Nd), (n1,...,nd), (1, r1,...,r(d-1),1))
    y: TuckerTensorTrain
        Second summand. structure=((N1,...,Nd), (m1,...,md), (1, q1,...,q(d-1),1))
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    z: TuckerTensorTrain
        Sum of Tucker tensor trains, z=x+y. structure=((N1,...,Nd), (n1+m1,...,nd+md), (1, r1+q1,...,r(d-1)+q(d-1),1))

    Raises
    ------
    RuntimeError
        Error raised if the Tucker tensor trains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_scale

    Notes
    -----
    The basis cores for z are vertically stacked versions of the basis cores for x and y

    The first TT-core for z is a block 1x2x2 tensor, with:
        - the first TT-core for x in the (0,0,0) block,
        - the first TT-core for y in the (0,1,1) block,
        - zeros elsewhere

    Intermediate TT-cores for z are block 2x2x2 tensors, with:
        - the TT-cores for x in the (0,0,0) block,
        - the TT-cores for y in the (1,1,1) block,
        - zeros elsewhere.

    The last TT-cores for z is a block 2x2x1 tensor, with:
        - the last TT-core for x in the (0,0,0) block,
        - the last TT-core for y in the (1,1,0) block,
        - zeros elsewhere

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> basis_cores_y = (randn(5,14), randn(6,15), randn(7,16))
    >>> tt_cores_y = (randn(1,5,4), randn(4,6,3), randn(3,7,1))
    >>> y = (basis_cores_y, tt_cores_y)
    >>> z = t3_add(x, y)
    >>> print(t3_structure(x))
        ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
    >>> print(t3_structure(y))
        ((14, 15, 16), (5, 6, 7), (1, 4, 3, 1))
    >>> print(t3_structure(z))
        ((14, 15, 16), (9, 11, 13), (1, 7, 5, 1))
    >>> print(np.linalg.norm(t3_to_dense(x) + t3_to_dense(y) - t3_to_dense(z)))
        6.524094086845177e-13
    """
    x_shape = t3_shape(x)
    y_shape = t3_shape(y)
    if x_shape != y_shape:
        raise RuntimeError(
            'Attempted to add TuckerTensorTrains x+y with inconsistent shapes.'
            + str(x_shape) + ' = x_shape != y_shape = ' + str(y_shape)
        )

    xnp = jnp if use_jax else np

    basis_cores_x, tt_cores_x = x
    basis_cores_y, tt_cores_y = y
    basis_cores_z = [xnp.concatenate([Bx, By], axis=0) for Bx, By in zip(basis_cores_x, basis_cores_y)]

    tt_cores_z = []

    Gx = tt_cores_x[0]
    Gy = tt_cores_y[0]
    G000 = Gx
    G001 = xnp.zeros((1, Gx.shape[1], Gy.shape[2]))
    G010 = xnp.zeros((1, Gy.shape[1], Gx.shape[2]))
    G011 = Gy
    Gz = xnp.block([[G000, G001], [G010, G011]])
    tt_cores_z.append(Gz)

    for Gx, Gy in zip(tt_cores_x[1:-1], tt_cores_y[1:-1]):
        G000 = Gx
        G001 = xnp.zeros((Gx.shape[0], Gx.shape[1], Gy.shape[2]))
        G010 = xnp.zeros((Gx.shape[0], Gy.shape[1], Gx.shape[2]))
        G011 = xnp.zeros((Gx.shape[0], Gy.shape[1], Gy.shape[2]))
        G100 = xnp.zeros((Gy.shape[0], Gx.shape[1], Gx.shape[2]))
        G101 = xnp.zeros((Gy.shape[0], Gx.shape[1], Gy.shape[2]))
        G110 = xnp.zeros((Gy.shape[0], Gy.shape[1], Gx.shape[2]))
        G111 = Gy
        Gz = xnp.block([[[G000, G001], [G010, G011]], [[G100, G101], [G110, G111]]])
        tt_cores_z.append(Gz)

    Gx = tt_cores_x[-1]
    Gy = tt_cores_y[-1]
    G000 = Gx.transpose([2,0,1])
    G001 = xnp.zeros((1, Gx.shape[0], Gy.shape[1]))
    G010 = xnp.zeros((1, Gy.shape[0], Gx.shape[1]))
    G011 = Gy.transpose([2,0,1])
    Gz = xnp.block([
        [G000, G001],
        [G010, G011],
    ]).transpose([1,2,0])
    tt_cores_z.append(Gz)

    z = (tuple(basis_cores_z), tuple(tt_cores_z))
    return z


def t3_scale(
        x: TuckerTensorTrain,
        s, # scalar
) -> TuckerTensorTrain:
    """Scale TuckerTensorTrain x by a scaling factor s.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train
    s: scalar
        scaling factor

    Returns
    -------
    z: TuckerTensorTrain
        scaled TuckerTensorTrain z=s*x, with the same structure as x.

    See Also
    --------
    TuckerTensorTrain
    t3_add

    Notes
    -----
    Scales the last basis core of x, leaving all other cores unchanged

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores, tt_cores)
    >>> s = randn()
    >>> z = t3_scale(x, s)
    >>> print(np.linalg.norm(s*t3_to_dense(x) - t3_to_dense(z)))
        1.6268482531988893e-13
    """
    basis_cores, tt_cores = x

    scaled_basis_cores = [B.copy() for B in basis_cores]
    scaled_basis_cores[-1] = s*scaled_basis_cores[-1]

    copied_tt_cores = [G.copy() for G in tt_cores]

    z = (tuple(scaled_basis_cores), tuple(copied_tt_cores))
    return z


def t3_dot_t3(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt (dot) product of two TuckerTensorTrains x, y with the same shape, (x, y)_HS.

    Parameters
    ----------
    x: TuckerTensorTrain
        First Tucker tensor train. shape=(N1,...,Nd)
    y: TuckerTensorTrain
        Second Tucker tensor train. shape=(N1,...,Nd)
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    scalar
        Hilbert-Schmidt (dot) product of Tucker tensor trains, (x, y)_HS.

    Raises
    ------
    RuntimeError
        Error raised if the Tucker tensor trains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_add
    t3_scale

    Notes
    -----
    Algorithm contracts the TuckerTensorTrains in a zippering fashion from left to right.

    Examples
    --------
    >>> from numpy.random import randn
    >>> from t3tools.tucker_tensor_train import *
    >>> basis_cores_x = (randn(4,14), randn(5,15), randn(6,16))
    >>> tt_cores_x = (randn(1,4,3), randn(3,5,2), randn(2,6,1))
    >>> x = (basis_cores_x, tt_cores_x)
    >>> basis_cores_y = (randn(5,14), randn(6,15), randn(7,16))
    >>> tt_cores_y = (randn(1,5,4), randn(4,6,3), randn(3,7,1))
    >>> y = (basis_cores_y, tt_cores_y)
    >>> x_dot_y = t3_dot_t3(x, y)
    >>> x_dot_y2 = np.sum(t3_to_dense(x) * t3_to_dense(y))
    >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
        8.731149137020111e-11
    """
    xnp = jnp if use_jax else np

    x_shape = t3_shape(x)
    y_shape = t3_shape(y)
    if x_shape != y_shape:
        raise RuntimeError(
            'Attempted to dot TuckerTensorTrains (x,y)_HS with inconsistent shapes.'
            + str(x_shape) + ' = x_shape != y_shape = ' + str(y_shape)
        )

    basis_cores_x, tt_cores_x = x
    basis_cores_y, tt_cores_y = y

    M_sp = xnp.ones((1,1))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(basis_cores_x, tt_cores_x, basis_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('ai,bi->ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('sat,ab->sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('sp,sbt->pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('pbt,pbq->tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    return M_sp[0,0]


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


#########################################################################
########    Orthogonalization, T3-SVD, and related functions    #########
#########################################################################

def left_svd_3tensor(
        G0_i_a_j: jnp.ndarray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
) -> typ.Tuple[
    jnp.ndarray, # U_i_a_x, shape=(ni, na, nx)
    jnp.ndarray, # ss_x,    shape=(nx,)
    jnp.ndarray, # Vt_x_j,  shape=(nx, nj)
]:
    rtol1 = 0.0 if rtol is None else rtol
    atol1 = 0.0 if atol is None else atol

    ni, na, nj = G0_i_a_j.shape
    G0_ia_j = G0_i_a_j.reshape((ni*na, nj))

    U_ia_y, ss_y, Vt_y_j = jnp.linalg.svd(G0_ia_j, full_matrices=False)

    tol = jnp.maximum(ss_y[0] * rtol1, atol1)

    if forced_rank is not None:
        nx = forced_rank
    elif rtol is None and atol is None:
        nx = len(ss_y)
    else:
        nx = jnp.sum(ss_y >= tol)
        if max_rank is not None:
            nx = jnp.minimum(nx, max_rank)

    U_ia_x = U_ia_y[:, :nx]
    ss_x = ss_y[:nx]
    Vt_x_j = Vt_y_j[:nx, :]

    nx = len(ss_x)
    U_i_a_x = U_ia_x.reshape((ni, na, nx))
    return U_i_a_x, ss_x, Vt_x_j


def right_svd_3tensor(
        G0_i_a_j: jnp.ndarray, # shape=(ni, na, nj)
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    jnp.ndarray, # U_i_x,       shape=(ni, nx)
    jnp.ndarray, # ss_x,        shape=(nx,)
    jnp.ndarray, # Vt_x_a_j,    shape=(nx, na, nj)
]:
    G0_j_a_i = G0_i_a_j.swapaxes(0, 2)
    Vt_j_a_x, ss_x, U_x_i = left_svd_3tensor(G0_j_a_i, max_rank, rtol, atol, forced_rank)
    Vt_x_a_j = Vt_j_a_x.swapaxes(0, 2)
    U_i_x = U_x_i.swapaxes(0,1)
    return U_i_x, ss_x, Vt_x_a_j


def outer_svd_3tensor(
        G0_i_a_j: jnp.ndarray, # shape=(ni, na, nj)
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    jnp.ndarray, # U_i_x_j, shape=(ni, nx, nj),
    jnp.ndarray, # ss_x,    shape=(nx,)
    jnp.ndarray, # Vt_x_a,  shape=(nx, na)
]:
    G0_i_j_a = G0_i_a_j.swapaxes(1, 2)
    U_i_j_x, ss_x, Vt_x_a = left_svd_3tensor(G0_i_j_a, max_rank, rtol, atol, forced_rank)
    U_i_x_j = U_i_j_x.swapaxes(1, 2)
    return U_i_x_j, ss_x, Vt_x_a


def orthogonalize_ith_basis_core(
        ii: int, # which base core to orthogonalize
        x: TuckerTensorTrain,
) -> TuckerTensorTrain:
    basis_cores, tt_cores = x
    G_a_i_b = tt_cores[ii]
    U_i_o = basis_cores[ii]
    U_o_i = U_i_o.T
    U2_o_x, R_x_i = jnp.linalg.qr(U_o_i, mode='reduced')
    G2_a_x_b = jnp.einsum('aib,xi->axb', G_a_i_b, R_x_i)
    U2_x_o = U2_o_x.T

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G2_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = U2_x_o

    return tuple(new_basis_cores), tuple(new_tt_cores)


def left_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(r(i+1),)
]:
    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii+1]

    A_a_i_x, ss_x, Vt_x_b = left_svd_3tensor(A0_a_i_b, max_rank, rtol, atol, forced_rank)
    B_x_j_c = jnp.tensordot(ss_x.reshape((-1,1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii+1] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(ri,)
]:
    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii-1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c, max_rank, rtol, atol, forced_rank)
    A_a_i_x = jnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1,-1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii-1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def up_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(ri,)
]:
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, max_rank, rtol, atol, forced_rank)

    G_a_x_b = jnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = jnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def down_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        forced_rank: int = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    jnp.ndarray, # singular values, shape=(ri,)
]:
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, max_rank, rtol, atol, forced_rank)

    Q_x_o = (ss_x.reshape((-1,1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def t3_svd(
        x: TuckerTensorTrain,
        max_tt_ranks:       typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        max_basis_ranks:    typ.Sequence[int] = None, # len=k,
        rtol: float = None,
        atol: float = None,
        forced_tt_ranks:    typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        forced_basis_ranks: typ.Sequence[int] = None, # len=k
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[jnp.ndarray,...], # basis singular values, len=k
    typ.Tuple[jnp.ndarray,...], # tt singular values, len=k+1
]:
    basis_cores, tt_cores = x

    num_cores = len(tt_cores)

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x = orthogonalize_ith_basis_core(ii, x)

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x, _ = right_svd_ith_tt_core(ii, x)

    G0 = x[1][0]
    _, ss_first, _ = right_svd_3tensor(G0)
    # _, ss_first, _ = right_svd_3tensor(tt_cores[0])

    # Sweep left to right computing SVDS
    all_ss_basis = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        max_r_basis = max_basis_ranks[ii] if max_basis_ranks is not None else None
        forced_r_basis = forced_basis_ranks[ii] if forced_basis_ranks is not None else None
        # SVD inbetween tt core and basis core
        x, ss_basis = inner_svd_ith_tt_core(
            ii, x, max_r_basis, rtol, atol, forced_rank=forced_r_basis,
        )
        all_ss_basis.append(ss_basis)

        if ii < num_cores-1:
            max_r_tt = max_tt_ranks[ii] if max_tt_ranks is not None else None
            forced_r_tt = forced_tt_ranks[ii] if forced_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = left_svd_ith_tt_core(
                ii, x, max_r_tt, rtol, atol, forced_rank=forced_r_tt,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = left_svd_3tensor(Gf)
            # _, ss_tt, _ = left_svd_3tensor(tt_cores[-1])
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


#

def tucker_svd_dense(
        T: jnp.ndarray, # shape=(N1, N2, .., Nd)
        max_ranks: typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[jnp.ndarray, ...], # Tucker bases, ith_elm_shape=(ni, Ni)
        jnp.ndarray, # Tucker core, shape=(n1,n2,...,nd)
    ],
    typ.Tuple[jnp.ndarray], # singular values of unfoldings
]:
    rtol = 0.0 if rtol is None else rtol
    atol = 0.0 if atol is None else atol

    bases = []
    singular_values_of_unfoldings = []
    C = T
    for ii in range(len(T.shape)):
        C_swap = C.swapaxes(ii,0)
        old_shape_swap = C_swap.shape

        C_swap_mat = C_swap.reshape((old_shape_swap[0], -1))
        U, ss, Vt = jnp.linalg.svd(C_swap_mat, full_matrices=False)

        if max_ranks is not None:
            ss = ss[:max_ranks[ii]]

        tol = jnp.maximum(ss[0] * rtol, atol)

        rM_new = jnp.sum(ss >= tol)
        U = U[:, :rM_new]
        ss = ss[:rM_new]
        Vt = Vt[:rM_new, :]

        singular_values_of_unfoldings.append(ss)
        bases.append(U.T)
        C_swap = (ss.reshape((-1,1)) * Vt).reshape((rM_new,) + old_shape_swap[1:])
        C = C_swap.swapaxes(0, ii)

    return (tuple(bases), C), tuple(singular_values_of_unfoldings)


def t3_svd_dense(
        T: jnp.ndarray, # shape=(N1, N2, .., Nd)
        max_mid_tt_ranks:       typ.Sequence[int] = None, # len=k-1, i.e., Correct: (r1, ..., r_{k-1}), Incorrect: (1,r1, ..., r_{k-1},1)
        max_basis_ranks:        typ.Sequence[int] = None, # len=k,
        rtol: float = None,
        atol: float = None,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[jnp.ndarray,...], # basis singular values, len=k
    typ.Tuple[jnp.ndarray,...], # tt singular values, len=k-1
]:
    (basis_cores, tucker_core), ss_tucker = tucker_svd_dense(T, max_basis_ranks, rtol=rtol, atol=atol)
    tt_cores, ss_tt = tt_svd_dense(tucker_core, max_mid_tt_ranks, rtol=rtol, atol=atol)
    return (basis_cores, tt_cores), ss_tucker, ss_tt


########################################################################
########################    Rank adjustment    #########################
########################################################################

def t3_remove_useless_rank(
        shape:          typ.Sequence[int], # len=d
        tucker_ranks:   typ.Sequence[int], # len=d
        tt_ranks:       typ.Sequence[int], # len=d+1
) -> typ.Tuple[
    typ.Tuple[int,...], # new_tucker_ranks
    typ.Tuple[int,...], # new_tt_ranks
]:
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


def t3_pad_rank(
        cores:              TuckerTensorTrain,
        new_tucker_ranks:   typ.Sequence[int],
        new_tt_ranks:       typ.Sequence[int],
) -> TuckerTensorTrain:
    t3_check_correctness(cores)
    shape = t3_get_shape(cores)
    num_cores = len(shape)
    old_tt_ranks        = t3_get_tt_ranks(cores)
    old_tucker_ranks    = t3_get_tucker_ranks(cores)
    assert(len(old_tucker_ranks) == len(new_tucker_ranks))
    assert(len(old_tt_ranks) == len(new_tt_ranks))

    delta_tucker_ranks  = [r_new - r_old for r_new, r_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    tucker_cores, tt_cores = cores

    new_tucker_cores = []
    for ii in range(num_cores):
        new_tucker_cores.append(jnp.pad(
            tucker_cores[ii],
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

    return tuple(new_tucker_cores), tuple(new_tt_cores)







