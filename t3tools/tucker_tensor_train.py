# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy as np
import typing as typ
import t3tools.dense as dense

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
    # Tucker tensor train
    'TuckerTensorTrain',
    'T3Structure',
    't3_structure',
    't3_apply',
    't3_entry',
    't3_to_dense',
    't3_reverse',
    'check_t3',
    't3_zeros',
    't3_corewise_randn',
    'compute_minimal_ranks',
    'pad_ranks',
    't3_save',
    't3_load',
    # Orthogonalization
    'up_svd_ith_basis_core',
    'left_svd_ith_tt_core',
    'right_svd_ith_tt_core',
    'up_svd_ith_tt_core',
    'down_svd_ith_tt_core',
    'orthogonalize_relative_to_ith_basis_core',
    'orthogonalize_relative_to_ith_tt_core',
    # Linear algebra
    't3_add',
    't3_scale',
    't3_neg',
    't3_sub',
    't3_dot_t3',
    't3_norm',
    # T3-SVD
    't3_svd',
    't3_svd_dense',
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
>>> import numpy as np
>>> import t3tools.tucker_tensor_train as t3
>>> basis_cores = [np.ones((4,14)),np.ones((5,15)),np.ones((6,16))]
>>> tt_cores = [np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))]
>>> x = (basis_cores, tt_cores) # TuckerTensorTrain, all cores filled with ones
>>> t3.check_t3(x) # does nothing because t3 core shapes are consistent
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
>>> import numpy as np
>>> import t3tools.tucker_tensor_train as t3
>>> basis_cores = [np.ones((4,14)),np.ones((5,15)),np.ones((6,16))]
>>> tt_cores = [np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))]
>>> x = (basis_cores, tt_cores) # TuckerTensorTrain, all cores filled with ones
>>> shape, tucker_ranks, tt_ranks = t3.t3_structure(x)
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
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
    >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
    >>> x = (basis_cores, tt_cores)
    >>> shape, tucker_ranks, tt_ranks = t3.t3_structure(x)
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


def check_t3(
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

    (Good) Consistent Tucker tensor train:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = [np.ones((4,14)),np.ones((5,15)),np.ones((6,16))]
    >>> tt_cores = [np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))]
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x) # Nothing happens because T3 is consistent

    (Bad) Mismatch between number of basis cores and number of TT-cores:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15))) # one too few basis cores
    >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
    RuntimeError: Inconsistent TuckerTensorTrain.
    2 = len(basis_cores) != len(tt_cores) = 3

    (Bad) One of the TT-cores is not a 3-tensor:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
    >>> tt_cores = (np.ones((4,3)), np.ones((3,5,2)), np.ones((2,6,1))) # first TT-core is not a 3-tensor
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
    RuntimeError: Inconsistent TuckerTensorTrain.
    tt_cores[0] is not a 3-tensor. shape=(4, 3)

    (Bad) First TT-rank is not 1:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16))) # First TT-rank is not 1
    >>> tt_cores = (np.ones((9,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
    RuntimeError: Inconsistent TuckerTensorTrain.
    First TT rank is not one. tt_ranks = (9, 3, 2, 1)

    (Bad) Last TT-rank is not 1:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
    >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,9))) # Last TT-rank is not 1
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
    RuntimeError: Inconsistent TuckerTensorTrain.
    Last TT rank is not one. tt_ranks = (1, 3, 2, 9)

    (Bad) TT-core shapes inconsistent with each other:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((6,16)))
    >>> tt_cores = (np.ones((1,4,9)), np.ones((3,5,2)), np.ones((2,6,1))) # Inconsistent TT-core shapes
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
    RuntimeError: Inconsistent TuckerTensorTrain.
    (1, 3, 2, 1) = left_tt_ranks != right_tt_ranks = (1, 9, 2, 1)

    (Bad) Basis core is not a matrix:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15,3)), np.ones((6,16))) # Basis core 2 is not a matrix
    >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1)))
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
    RuntimeError: Inconsistent TuckerTensorTrain.
    basis_cores[1] is not a matrix. shape=(5, 15, 3)

    (Bad) Inconsist shapes for basis core and adjacent TT-core

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> basis_cores = (np.ones((4,14)), np.ones((5,15)), np.ones((9,16)))
    >>> tt_cores = (np.ones((1,4,3)), np.ones((3,5,2)), np.ones((2,6,1))) # Last basis and TT-cores inconsistent
    >>> x = (basis_cores, tt_cores)
    >>> t3.check_t3(x)
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
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16),(4,5,6),(1,3,2,1))) # make TuckerTensorTrain
    >>> x_dense = t3.t3_to_dense(x) # Convert TuckerTensorTrain to dense tensor
    >>> ((B0,B1,B2), (G0,G1,G2)) = x
    >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
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
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1))) # Make TuckerTensorTrain
    >>> print(t3.t3_structure(x))
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
    >>> reversed_x = t3.t3_reverse(x)
    >>> print(t3.t3_structure(reversed_x))
    ((16, 15, 14), (6, 5, 4), (1, 2, 3, 1))
    >>> x_dense = t3.t3_to_dense(x)
    >>> reversed_x_dense = t3.t3_to_dense(reversed_x)
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
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> shape = (14, 15, 16)
    >>> tucker_ranks = (4, 5, 6)
    >>> tt_ranks = (1, 3, 2, 1)
    >>> structure = (shape, tucker_ranks, tt_ranks)
    >>> z = t3.t3_zeros(structure)
    >>> print(np.linalg.norm(t3.t3_to_dense(z)))
    0.0
    """
    xnp = jnp if use_jax else np

    shape, tucker_ranks, tt_ranks = structure

    tt_cores = tuple([xnp.zeros((tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1])) for ii in range(len(tucker_ranks))])
    basis_cores = tuple([xnp.zeros((n, N)) for n, N  in zip(tucker_ranks, shape)])
    z = (basis_cores, tt_cores)
    return z


def t3_corewise_randn(
        structure:  T3Structure,
        use_jax:    bool = False,
) -> TuckerTensorTrain:
    """Construct Tucker tensor train with random cores (i.i.d. N(0,1) entries).

    Parameters
    ----------
    structure:  T3Structure
        Tucker tensor train structure
        (shape, tucker_ranks, tt_ranks)=((N1,...,Nd), (n1,...,nd), (1,r1,...,r(d-1),1))).
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
    >>> from t3tools import *
    >>> import t3tools.tucker_tensor_train as t3
    >>> shape = (14, 15, 16)
    >>> tucker_ranks = (4, 5, 6)
    >>> tt_ranks = (1, 3, 2, 1)
    >>> structure = (shape, tucker_ranks, tt_ranks)
    >>> x = t3.t3_corewise_randn(structure) # TuckerTensorTrain with random cores
    """
    shape, tucker_ranks, tt_ranks = structure

    if use_jax:
        _randn = lambda x: jnp.array(np.random.randn(x))
    else:
        _randn = np.random.randn

    tt_cores = tuple([_randn(tt_ranks[ii], tucker_ranks[ii], tt_ranks[ii+1]) for ii in range(len(tucker_ranks))])
    basis_cores = tuple([_randn(n, N) for n, N  in zip(tucker_ranks, shape)])
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
    check_t3

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> fname = 't3_file'
    >>> t3.t3_save(fname, x) # Save to file 't3_file.npz'
    >>> x2 = t3.t3_load(fname) # Load from file
    >>> basis_cores, tt_cores = x
    >>> basis_cores2, tt_cores2 = x2
    >>> print([np.linalg.norm(B - B2) for B, B2 in zip(basis_cores, basis_cores2)])
    [0.0, 0.0, 0.0]
    >>> print([np.linalg.norm(G - G2) for G, G2 in zip(tt_cores, tt_cores2)])
    [0.0, 0.0, 0.0]
    """
    check_t3(x)
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
    check_t3

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> fname = 't3_file'
    >>> t3.t3_save(fname, x) # Save to file 't3_file.npz'
    >>> x2 = t3.t3_load(fname) # Load from file
    >>> basis_cores, tt_cores = x
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
    check_t3(x)
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

    Apply to one set of vectors:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> vecs = [np.random.randn(14), np.random.randn(15), np.random.randn(16)]
    >>> result = t3.t3_apply(x, vecs) # <-- contract x with vecs in all indices
    >>> result2 = np.einsum('ijk,i,j,k', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
    >>> print(np.abs(result - result2))
    5.229594535194337e-12

    Apply to multiple sets of vectors (vectorized):

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    3.1271953680324864e-12

    Example using jax automatic differentiation:

	>>> import numpy as np
    >>> import jax
    >>> import t3tools.tucker_tensor_train as t3
    >>> jax.config.update("jax_enable_x64", True)
    >>> A = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 Tucker tensor train
    >>> apply_A_sym = lambda u: t3.t3_apply(A, (u,u,u), use_jax=True) # symmetric apply function
    >>> u0 = np.random.randn(10)
    >>> Auuu0 = apply_A_sym(u0)
    >>> g0 = jax.grad(apply_A_sym)(u0) # gradient using automatic differentiation
    >>> du = np.random.randn(10)
    >>> dAuuu = np.dot(g0, du) # derivative in direction du
    >>> print(dAuuu)
    766.5390335764645
    >>> s = 1e-7
    >>> u1 = u0 + s*du
    >>> Auuu1 = apply_A_sym(u1)
    >>> dAuuu_diff = (Auuu1 - Auuu0) / s # finite difference approximation
    >>> print(dAuuu_diff)
    766.5390504030256
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

    vectorized = True
    if vecs_dims[0] == 1:
        vectorized = False
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
    if not vectorized:
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

    Compute one entry:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> index = [9, 4, 7] # get entry (9,4,7)
    >>> result = t3.t3_entry(x, index)
    >>> result2 = t3.t3_to_dense(x)[9, 4, 7]
    >>> print(np.abs(result - result2))
    1.3322676295501878e-15

    Compute multiple entries (vectorized):

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> index = [[9,8], [4,10], [7,13]] # get entries (9,4,7) and (8,10,13)
    >>> entries = t3.t3_entry(x, index)
    >>> x_dense = t3.t3_to_dense(x)
    >>> entries2 = np.array([x_dense[9, 4, 7], x_dense[8, 10, 13]])
    >>> print(np.linalg.norm(entries - entries2))
    1.7763568394002505e-15

    Example using jax jit compiling:

	>>> import numpy as np
    >>> import jax
    >>> import t3tools.tucker_tensor_train as t3
    >>> get_entry_123 = lambda x: t3.t3_entry(x, (1,2,3), use_jax=True)
    >>> A = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 Tucker tensor train
    >>> a123 = get_entry_123(A)
    >>> print(a123)
    11.756762
    >>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
    >>> a123_jit = get_entry_123_jit(A)
    >>> print(a123_jit)
    11.756762
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    if len(index)  != len(shape):
        raise RuntimeError(
            'Wrong number of indices for TuckerTensorTrain.'
            + str(str(len(shape)) + ' = num tensor indices != num provided indices = ' + str(len(index)))
        )

    vectorized = True
    if isinstance(index[0], int):
        vectorized = False
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
    if not vectorized:
        result = result[0]

    return result


########################################################################
########################    Rank adjustment    #########################
########################################################################

def compute_minimal_ranks(
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


def pad_ranks(
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


##########################################
########    Orthogonalization    #########
##########################################

def up_svd_ith_basis_core(
        ii: int, # which base core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # ss_x. singular values
]:
    '''Compute SVD of ith basis core and contract non-orthogonal factor up into the TT-core above.

    Parameters
    ----------
    ii: int
        index of basis core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith basis core orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
        new_basis_cores[ii] @ new_basis_cores[ii].T = identity matrix
    ss_x: NDArray
        Singular values of prior ith basis core. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = t3.up_svd_ith_basis_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    5.772851635866132e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> rank = len(ss)
    >>> B = basis_cores2[ind]
    >>> print(np.linalg.norm(B @ B.T - np.eye(rank))) # basis core is orthogonal
    8.456498415401757e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    G_a_i_b = tt_cores[ii]
    U_i_o = basis_cores[ii]
    U_o_i = U_i_o.T

    U2_o_x, ss_x, Vt_x_i = dense.truncated_svd(U_o_i, min_rank, max_rank, rtol, atol, use_jax)
    R_x_i = xnp.einsum('x,xi->xi', ss_x, Vt_x_i)
    # U2_o_x, R_x_i = xnp.linalg.qr(U_o_i, mode='reduced')

    G2_a_x_b = xnp.einsum('aib,xi->axb', G_a_i_b, R_x_i)
    U2_x_o = U2_o_x.T

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G2_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = U2_x_o

    new_x = (tuple(new_basis_cores), tuple(new_tt_cores))

    return new_x, ss_x


def left_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(r(i+1),)
]:
    '''Compute SVD of ith TT-core left unfolding and contract non-orthogonal factor into the TT-core to the right.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
        new_tt_cores[ii].shape = (ri, ni, new_r(i+1))
        new_tt_cores[ii+1].shape = (new_r(i+1), n(i+1), r(i+2))
        einsum('iaj,iak->jk', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core left unfolding. shape=(new_r(i+1),).

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    up_svd_ith_basis_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = t3.left_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
        5.186463661974644e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(np.einsum('iaj,iak->jk', G, G) - np.eye(G.shape[2]))) # TT-core is left-orthogonal
        4.453244025338311e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii]
    B0_b_j_c = tt_cores[ii+1]

    A_a_i_x, ss_x, Vt_x_b = dense.left_svd_3tensor(A0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)
    B_x_j_c = xnp.tensordot(ss_x.reshape((-1,1)) * Vt_x_b, B0_b_j_c, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = A_a_i_x
    new_tt_cores[ii+1] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def right_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ri,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor into the TT-core to the left.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core orthogonal.
        new_tt_cores[ii].shape = (new_ri, ni, r(i+1))
        new_tt_cores[ii-1].shape = (r(i-1), n(i-1), new_ri)
        einsum('iaj,kaj->ik', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core right unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = t3.right_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
        5.304678679078675e-13
    >>> basis_cores2, tt_cores2 = x2
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(np.einsum('iaj,kaj->ik', G, G) - np.eye(G.shape[0]))) # TT-core is right orthogonal
        4.207841813173725e-16
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    A0_a_i_b = tt_cores[ii-1]
    B0_b_j_c = tt_cores[ii]

    U_b_x, ss_x, B_x_j_c = dense.right_svd_3tensor(B0_b_j_c, min_rank, max_rank, rtol, atol, use_jax)
    A_a_i_x = xnp.tensordot(A0_a_i_b, U_b_x * ss_x.reshape((1,-1)), axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii-1] = A_a_i_x
    new_tt_cores[ii] = B_x_j_c

    return (tuple(basis_cores), tuple(new_tt_cores)), ss_x


def up_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core outer unfolding and keep non-orthogonal factor with this core.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ri,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> x2, ss = t3.up_svd_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    1.002901486286745e-12
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    U_a_x_b, ss_x, Vt_x_i = dense.outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

    G_a_x_b = xnp.einsum('axb,x->axb', U_a_x_b, ss_x)
    Q_x_o = xnp.tensordot(Vt_x_i, Q0_i_o, axes=1)

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def down_svd_ith_tt_core(
        ii: int, # which tt core to orthogonalize
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    NDArray, # singular values, shape=(new_ni,)
]:
    '''Compute SVD of ith TT-core right unfolding and contract non-orthogonal factor down into the basis core below.

    Parameters
    ----------
    ii: int
        index of TT-core to SVD
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but with ith TT-core outer orthogonal.
        new_tt_cores[ii].shape = (ri, new_ni, r(i+1))
        new_basis_cores[ii].shape = (new_ni, Ni)
        einsum('iaj,ibj->ab', new_tt_cores[ii], new_tt_cores[ii]) = identity matrix
    ss_x: NDArray
        Singular values of prior ith TT-core outer unfolding. shape=(new_ni,).

    See Also
    --------
    truncated_svd
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> ind = 1
    >>> x2, ss = t3.down_svd_ith_tt_core(ind, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    4.367311712704942e-12
    >>> basis_cores2, tt_cores2 = x2
    >>> G = tt_cores2[ind]
    >>> print(np.linalg.norm(np.einsum('iaj,ibj->ab', G, G) - np.eye(G.shape[1]))) # TT-core is outer orthogonal
    1.0643458053135608e-15
    '''
    basis_cores, tt_cores = x

    G0_a_i_b = tt_cores[ii]
    Q0_i_o = basis_cores[ii]

    G_a_x_b, ss_x, Vt_x_i = dense.outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

    Q_x_o = (ss_x.reshape((-1,1)) * Vt_x_i) @ Q0_i_o

    new_tt_cores = list(tt_cores)
    new_tt_cores[ii] = G_a_x_b

    new_basis_cores = list(basis_cores)
    new_basis_cores[ii] = Q_x_o

    return (tuple(new_basis_cores), tuple(new_tt_cores)), ss_x


def orthogonalize_relative_to_ith_basis_core(
        ii: int,
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith basis core.

    Orthogonal is done relative to the ith basis core:
        - ith basis core is not orthogonalized
        - All other basis cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - TT-core directly above is outer orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of basis core that is not orthogonalized
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith basis core.

    See Also
    --------
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> x2 = t3.orthogonalize_relative_to_ith_basis_core(1, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> X = np.einsum('xi,axb,byc,czd,zk->iyk', B0, G0, G1, G2, B2) # Contraction of everything except B1
    >>> print(np.linalg.norm(np.einsum('iyk,iwk->yw', X, X) - np.eye(B1.shape[0]))) # Complement of B1 is orthogonal
    1.7116160385376214e-15
    '''
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    new_x = down_svd_ith_tt_core(ii, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
    return new_x


def orthogonalize_relative_to_ith_tt_core(
        ii: int,
        x: TuckerTensorTrain,
        min_rank: int = None,
        max_rank: int = None,
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Orthogonalize all cores in the TuckerTensorTrain except for the ith TT-core.

    Orthogonal is done relative to the ith TT-core:
        - All basis cores are orthogonalized.
        - TT-cores to the left are left orthogonalized.
        - ith TT-core is not orthogonalized.
        - TT-cores to the right are right orthogonalized.

    Parameters
    ----------
    ii: int
        index of TT-core that is not orthogonalized
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_rank: int
        Minimum rank for truncation.
    min_rank: int
        Maximum rank for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    See Also
    --------
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core

    Returns
    -------
    new_x: NDArray
        New TuckerTensorTrain representing the same tensor, but orthogonalized relative to the ith TT-core.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> x2 = t3.orthogonalize_relative_to_ith_tt_core(1, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    8.800032152216517e-13
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> XL = np.einsum('axb,xi -> aib', G0, B0) # Everything to the left of G1
    >>> print(np.linalg.norm(np.einsum('aib,aic->bc', XL, XL) - np.eye(G1.shape[0]))) # Left subtree is left orthogonal
    9.820411604510197e-16
    >>> print(np.linalg.norm(np.einsum('xi,yi->xy', B1, B1) - np.eye(G1.shape[1]))) # Core below G1 is up orthogonal
    2.1875310121178e-15
    >>> XR = np.einsum('axb,xi->aib', G2, B2) # Everything to the right of G1
    >>> print(np.linalg.norm(np.einsum('aib,cib->ac', XR, XR) - np.eye(G1.shape[2]))) # Right subtree is right orthogonal
    1.180550381921849e-15
    '''
    shape, tucker_ranks, tt_ranks = t3_structure(x)

    new_x = x
    for jj in range(ii):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = left_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    for jj in range(len(shape)-1, ii, -1):
        new_x = down_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = up_svd_ith_basis_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
        new_x = right_svd_ith_tt_core(jj, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]

    new_x = up_svd_ith_basis_core(ii, new_x, min_rank, max_rank, rtol, atol, use_jax)[0]
    return new_x


###############################
########    T3-SVD    #########
###############################

def t3_svd(
        x: TuckerTensorTrain,
        min_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        min_tucker_ranks:   typ.Sequence[int] = None,  # len=d
        max_tt_ranks:       typ.Sequence[int] = None, # len=d+1
        max_tucker_ranks:   typ.Sequence[int] = None, # len=d
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,

) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # basis singular values, len=d
    typ.Tuple[NDArray,...], # tt singular values, len=d+1
]:
    '''Compute (truncated) T3-SVD of TuckerTensorTrain.

    Parameters
    ----------
    x: TuckerTensorTrain
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (1,r1,...r(d-1),1))
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation.
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation.
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation.
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation.
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    NDArray
        New TuckerTensorTrain representing the same tensor (or a truncated version), but with modified cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between basis cores and TT-cores
    typ.Tuple[NDArray,...]
        Singular values associated with edges between adjacent TT-cores

    See Also
    --------
    left_svd_3tensor
    right_svd_3tensor
    outer_svd_3tensor
    up_svd_ith_basis_core
    left_svd_ith_tt_core
    right_svd_ith_tt_core
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    truncated_svd

    Examples
    --------

    T3-SVD with no truncation:
    (ranks may decrease to minimal values, but no approximation error)

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((5,6,3), (4,4,3), (1,3,2,1)))
    >>> x2, ss_basis, ss_tt = t3.t3_svd(x) # Compute T3-SVD
    >>> x_dense = t3.t3_to_dense(x)
    >>> x2_dense = t3.t3_to_dense(x2)
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    7.556835759880194e-13
    >>> ss_tt1 = np.linalg.svd(x_dense.reshape((5, 6*3)))[1] # Singular values of unfolding 1
    >>> print(ss_tt1); print(ss_tt[1])
    [1.75326490e+02 3.41363029e+01 9.31164204e+00 1.33610061e-14 4.11601708e-15]
    [175.32648969  34.13630287   9.31164204]
    >>> ss_basis2 = np.linalg.svd(x_dense.transpose([2,0,1]).reshape((3,5*6)))[1] # Singular values of matricization 2
    >>> print(ss_basis2); print(ss_basis[2])
    [1.71350937e+02 5.12857505e+01 1.36927051e-14]
    [171.35093708  51.28575045]

    T3-SVD with truncation based on relative tolerance:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> B0 = np.random.randn(35,40) @ np.diag(1.0 / np.arange(1, 41)**2) # preconditioned indices
    >>> B1 = np.random.randn(45,50) @ np.diag(1.0 / np.arange(1, 51)**2)
    >>> B2 = np.random.randn(55,60) @ np.diag(1.0 / np.arange(1, 61)**2)
    >>> G0 = np.random.randn(1,35,30)
    >>> G1 = np.random.randn(30,45,40)
    >>> G2 = np.random.randn(40,55,1)
    >>> basis_cores_x = (B0, B1, B2)
    >>> tt_cores_x = (G0, G1, G2)
    >>> x = (basis_cores_x, tt_cores_x) # Tensor has spectral decay due to preconditioning
    >>> x2, ss_basis, ss_tt = t3.t3_svd(x, rtol=1e-2) # Truncate singular values to reduce rank
    >>> print(t3.t3_structure(x))
    ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1))
    >>> print(t3.t3_structure(x2))
    ((40, 50, 60), (6, 6, 5), (1, 6, 5, 1))
    >>> x_dense = t3.t3_to_dense(x)
    >>> x2_dense = t3.t3_to_dense(x2)
    >>> print(np.linalg.norm(x_dense - x2_dense)/np.linalg.norm(x_dense)) # Should be near rtol=1e-2
    0.013078458673911168

    T3-SVD with truncation based on absolute tolerance:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (10,11,12), (1,8,9,1)))
    >>> x2, ss_basis, ss_tt = t3.t3_svd(x, max_tucker_ranks=(3,3,3), max_tt_ranks=(1,2,2,1)) # Truncate based on ranks
    >>> print(t3.t3_structure(x))
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1))
    >>> print(t3.t3_structure(x2))
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1))
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    num_cores = len(tt_cores)

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x, _ = up_svd_ith_basis_core(ii, x, use_jax=use_jax)

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x, _ = right_svd_ith_tt_core(ii, x, use_jax=use_jax)

    G0 = x[1][0]
    _, ss_first, _ = dense.right_svd_3tensor(G0, use_jax=use_jax)

    # Sweep left to right computing SVDS
    all_ss_basis = []
    all_ss_tt = [ss_first]
    for ii in range(num_cores):
        min_rank = min_tucker_ranks[ii] if min_tucker_ranks is not None else None
        max_rank = max_tucker_ranks[ii] if max_tucker_ranks is not None else None
        # SVD inbetween tt core and basis core
        x, ss_basis = up_svd_ith_tt_core(
            ii, x, min_rank, max_rank, rtol, atol, use_jax,
        )
        all_ss_basis.append(ss_basis)

        if ii < num_cores-1:
            min_rank = min_tt_ranks[ii+1] if min_tt_ranks is not None else None
            max_rank = max_tt_ranks[ii+1] if max_tt_ranks is not None else None
            # SVD inbetween ith tt core and (i+1)th tt core
            x, ss_tt = left_svd_ith_tt_core(
                ii, x, min_rank, max_rank, rtol, atol, use_jax,
            )
        else:
            Gf = x[1][-1]
            _, ss_tt, _ = dense.left_svd_3tensor(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


###########################################################
##################    Linear algebra    ###################
###########################################################

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
        - Error raised if either of the TuckerTensorTrains are internally inconsistent
        - Error raised if the TuckerTensorTrains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_scale
    t3_sub
    t3_neg

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
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> z = t3.t3_add(x, y)
    >>> print(t3.t3_structure(z))
    ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1))
    >>> print(np.linalg.norm(t3.t3_to_dense(x) + t3.t3_to_dense(y) - t3.t3_to_dense(z)))
    6.524094086845177e-13
    """
    check_t3(x)
    check_t3(y)

    x_shape = t3_structure(x)[0]
    y_shape = t3_structure(y)[0]
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

    Raises
    ------
    RuntimeError
        - Error raised if the TuckerTensorTrains are internally inconsistent

    See Also
    --------
    TuckerTensorTrain
    t3_add
    t3_neg
    t3_sub

    Notes
    -----
    Scales the last basis core of x, leaving all other cores unchanged

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> s = 3.2
    >>> z = t3.t3_scale(x, s)
    >>> print(np.linalg.norm(s*t3.t3_to_dense(x) - t3.t3_to_dense(z)))
    1.6268482531988893e-13
    """
    check_t3(x)

    basis_cores, tt_cores = x

    scaled_basis_cores = [B.copy() for B in basis_cores]
    scaled_basis_cores[-1] = s*scaled_basis_cores[-1]

    copied_tt_cores = [G.copy() for G in tt_cores]

    z = (tuple(scaled_basis_cores), tuple(copied_tt_cores))
    return z


def t3_neg(
        x: TuckerTensorTrain,
) -> TuckerTensorTrain:
    """Scale TuckerTensorTrain x by a scaling factor -1.

    Parameters
    ----------
    x: TuckerTensorTrain
        Tucker tensor train

    Returns
    -------
    TuckerTensorTrain
        scaled TuckerTensorTrain -x, with the same structure as x.

        Raises
    ------
    RuntimeError
        - Error raised if the TuckerTensorTrains is internally inconsistent

    See Also
    --------
    TuckerTensorTrain
    t3_add
    t3_scale
    t3_sub

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> neg_x = t3.t3_neg(x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) + t3.t3_to_dense(neg_x)))
    0.0
    """
    return t3_scale(x, -1.0)


def t3_sub(
        x: TuckerTensorTrain,
        y: TuckerTensorTrain,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    """Subtract TuckerTensorTrains x, y with the same shape, yielding a TuckerTensorTrain z=x-y with summed ranks.

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
        Difference of Tucker tensor trains, z=x-y. structure=((N1,...,Nd), (n1+m1,...,nd+md), (1, r1+q1,...,r(d-1)+q(d-1),1))

    Raises
    ------
    RuntimeError
        - Error raised if either of the TuckerTensorTrains are internally inconsistent
        - Error raised if the TuckerTensorTrains have different shapes.

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_add
    t3_scale
    t3_neg

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> z = t3.t3_sub(x, y)
    >>> print(t3.t3_structure(z))
    ((14, 15, 16), (7, 12, 8), (1, 8, 8, 1))
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(y) - t3.t3_to_dense(z)))
    3.5875705233607603e-13
    """
    return t3_add(x, t3_neg(y), use_jax=use_jax)


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
        - Error raised if either of the TuckerTensorTrains are internally inconsistent
        - Error raised if the TuckerTensorTrains have different shapes.

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
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> x_dot_y = t3.t3_dot_t3(x, y)
    >>> x_dot_y2 = np.sum(t3.t3_to_dense(x) * t3.t3_to_dense(y))
    >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
    8.731149137020111e-11
    """
    check_t3(x)
    check_t3(y)

    xnp = jnp if use_jax else np

    x_shape = t3_structure(x)[0]
    y_shape = t3_structure(y)[0]
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


def t3_norm(
        x: TuckerTensorTrain,
        use_orthogonalization: bool = True,
        use_jax: bool = False,
):
    """Compute Hilbert-Schmidt (dot) product of two TuckerTensorTrains x, y with the same shape, (x, y)_HS.

    Parameters
    ----------
    x: TuckerTensorTrain
        First Tucker tensor train. shape=(N1,...,Nd)
    use_orthogonalization: bool
        If True, use orthogonalization-based algorithm (more stable).
        If False, use zippering algorithm (faster, easier to differentiate).
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    scalar
        Hilbert-Schmidt (Frobenius) norm of Tucker tensor trains, ||x||_HS

    Raises
    ------
    RuntimeError
        - Error raised if the TuckerTensorTrain is internally inconsistent

    See Also
    --------
    TuckerTensorTrain
    t3_shape
    t3_dot_t3

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> norm_x = t3.t3_norm(x)
    >>> print(np.abs(norm_x - np.linalg.norm(t3.t3_to_dense(x))))
    1.3642420526593924e-12
    """
    check_t3(x)
    xnp = jnp if use_jax else np

    if use_orthogonalization:
        shape, _, _ = t3_structure(x)
        last_ind = len(shape)-1
        x2 = orthogonalize_relative_to_ith_basis_core(last_ind, x, use_jax=use_jax)
        basis_cores, tt_cores = x2
        return xnp.linalg.norm(basis_cores[-1])
    else:
        return xnp.sqrt(t3_dot_t3(x, x, use_jax=use_jax))


def t3_svd_dense(
        T: NDArray, # shape=(N1, N2, .., Nd)
        min_tucker_ranks:  typ.Sequence[int] = None, # len=d
        max_tucker_ranks:  typ.Sequence[int] = None,  # len=d
        min_tt_ranks:  typ.Sequence[int] = None, # len=d+1
        max_tt_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    TuckerTensorTrain, # new_x
    typ.Tuple[NDArray,...], # basis singular values, len=k
    typ.Tuple[NDArray,...], # tt singular values, len=k-1
]:
    '''Compute TuckerTensorTrain and edge singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_tucker_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation. len=d. e.g., (3,3,3)
    max_tucker_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation. len=d. e.g., (5,5,5)
    min_tt_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation. len=d+1. e.g., (1,3,3,3,1)
    max_tt_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation. len=d+1. e.g., (1,5,5,5,1)
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    TuckerTensorTrain
        Tucker tensor train approxiamtion of T
    typ.Tuple[NDArray,...]
        Singular values of matricizations. len=d. elm_shape=(ni,)
    typ.Tuple[NDArray,...]
        Singular values of unfoldings. len=d+1. elm_shape=(ri,)

    See Also
    --------
    truncated_svd
    tucker_svd_dense
    tt_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> x, ss_tucker, ss_tt = t3.t3_svd_dense(T, rtol=1e-3) # Truncate T3-SVD to reduce rank
    >>> print(t3.t3_structure(x))
    ((40, 50, 60), (12, 11, 12), (1, 12, 12, 1))
    >>> T2 = t3.t3_to_dense(x)
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.0025147026955504846
    '''
    (basis_cores, tucker_core), ss_tucker = dense.tucker_svd_dense(T, min_tucker_ranks, max_tucker_ranks, rtol, atol, use_jax)
    tt_cores, ss_tt = dense.tt_svd_dense(tucker_core, min_tt_ranks, max_tt_ranks, rtol, atol, use_jax)
    return (basis_cores, tt_cores), ss_tucker, ss_tt

