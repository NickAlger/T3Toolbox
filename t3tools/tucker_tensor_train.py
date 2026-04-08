# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# https://github.com/NickAlger/TuckerTensorTrainTools
import numpy
import numpy as np
import typing as typ

import t3tools
import t3tools.util as util
from t3tools.util import NDArray, truncated_svd

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
    'structure',
    't3_apply',
    't3_entry',
    't3_to_dense',
    't3_reverse',
    'check_t3',
    't3_zeros',
    't3_corewise_randn',
    'compute_minimal_ranks',
    'are_t3_ranks_minimal',
    'pad_t3',
    't3_save',
    't3_load',
    # Orthogonalization
    'left_svd_3tensor',
    'right_svd_3tensor',
    'outer_svd_3tensor',
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
    'tucker_svd_dense',
    'tt_svd_dense',
    't3_svd_dense',
    't3_svd',
]

u = typ.Tuple[int]
"""
Test

Examples
--------
>>> import numpy as np
>>> import t3tools.tucker_tensor_train as t3
>>> randn = np.random.randn
>>> x_basis_cores = [randn(4, 14), randn(5, 15), randn(6, 16)]
>>> x_tt_cores = [randn(1, 4, 3), randn(3, 5, 2), randn(2, 6, 1)]
>>> x = (x_basis_cores, x_tt_cores)  # TuckerTensorTrain, x
>>> y_basis_cores = [randn(7, 14), randn(6, 15), randn(8, 16)]
>>> y_tt_cores = [randn(1, 7, 4), randn(4, 6, 5), randn(5, 6, 1)]
>>> y = (basis_cores, tt_cores)  # TuckerTensorTrain, y
>>> x_plus_y = t3.t3_add(x, y)
>>> print(t3_structure(x_plus_y)
"""

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
    - tt ranks: (r0, r1, ..., r(d-1), rd)
    
Note: typically r0=rd=1, and "1" in the diagram is the number 1. 
However, it is allowed for these ranks to not be 1, in which case
the "1"s in the diagram are vectors of ones.

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

        r0        r1        r2        r2        r4
    1 ------ G0 ------ G1 ------ G2 ------ G3 ------ 1
             |         |         |         |
             |n0       |n1       |n2       |n3
             |         |         |         |
             B0        B1        B2        B3
             |         |         |         |
             |N0       |N1       |N2       |N3
             |         |         |         |
         

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
>>> shape, tucker_ranks, tt_ranks = t3.structure(x)
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

def structure(
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
    >>> shape, tucker_ranks, tt_ranks = t3.structure(x)
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
    structure

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

    (Bad) Inconsistent shapes for basis core and adjacent TT-core

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
        contract_ones: bool = True,
        use_jax: bool = False,
) -> NDArray:
    """Contract Tucker tensor train to dense tensor.

    Parameters
    ----------
    x : TuckerTensorTrain
        Tucker tensor train with shape (N1, ..., Nd).
    contract_ones: bool
        If true (default), contract with leading and training 1's, yielding shape=(N1,...,Nd).
        If false, do not contract with leading and trailing 1's, yielding shape=(r0,N1,...,Nd,rd).
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

    Case where the first and last TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16),(4,5,6),(2,3,2,4))) # make TuckerTensorTrain
    >>> x_dense = t3.t3_to_dense(x) # Convert TuckerTensorTrain to dense tensor
    >>> ((B0,B1,B2), (G0,G1,G2)) = x
    >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->ijk', B0, B1, B2, G0, G1, G2)
    >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
    1.4635914598284152e-15

    Example where leading and trailing ones are not contracted

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16),(4,5,6),(2,3,4,2))) # make TuckerTensorTrain
    >>> x_dense = t3.t3_to_dense(x, contract_ones=False) # Convert TuckerTensorTrain to dense tensor
    >>> print(x_dense.shape)
    (2, 14, 15, 16, 2)
    >>> ((B0,B1,B2), (G0,G1,G2)) = x
    >>> x_dense2 = np.einsum('xi,yj,zk,axb,byc,czd->aijkd', B0, B1, B2, G0, G1, G2)
    >>> print(np.linalg.norm(x_dense - x_dense2) / np.linalg.norm(x_dense))
    1.1217675019342066e-15
    """
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    big_tt_cores = [xnp.einsum('iaj,ab->ibj', G, U) for G, U in zip(tt_cores, basis_cores)]

    # T = xnp.ones(big_tt_cores[0].shape[0])

    T = big_tt_cores[0]
    for G in big_tt_cores[1:]:
        T = xnp.tensordot(T, G, axes=1)

    if contract_ones:
        mu_L = xnp.ones(big_tt_cores[0].shape[0])
        mu_R = xnp.ones(big_tt_cores[-1].shape[2])

        T = xnp.tensordot(mu_L, T, axes=1)
        T = xnp.tensordot(T, mu_R, axes=1)

    # T = xnp.tensordot(T, xnp.ones(big_tt_cores[-1].shape[-1]), axes=1)

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
    structure

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1))) # Make TuckerTensorTrain
    >>> print(t3.structure(x))
    ((14, 15, 16), (4, 5, 6), (1, 3, 2, 1))
    >>> reversed_x = t3.t3_reverse(x)
    >>> print(t3.structure(reversed_x))
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

    First and last TT-ranks are not ones:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,4)))
    >>> vecs = [np.random.randn(3,14), np.random.randn(3,15), np.random.randn(3,16)]
    >>> result = t3.t3_apply(x, vecs)
    >>> result2 = np.einsum('ijk,ni,nj,nk->n', t3.t3_to_dense(x), vecs[0], vecs[1], vecs[2])
    >>> print(np.linalg.norm(result - result2))
    6.481396196459234e-12

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
    >>> print(dAuuu_diff) #ths same as dAuuu
    766.5390504030256
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    shape, tucker_ranks, tt_ranks = structure(x)

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

    mu_na = xnp.ones((num_applies, tt_cores[0].shape[0]))
    for V_ni, B_xi, G_axb in zip(vecs, basis_cores, tt_cores):
        v_nx = xnp.einsum('ni,xi->nx', V_ni, B_xi)
        g_anb = xnp.einsum('axb,nx->anb', G_axb, v_nx)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        mu_na = mu_nb
    result = xnp.einsum('na->n', mu_na)

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
    -1.3764521
    >>> get_entry_123_jit = jax.jit(get_entry_123) # jit compile
    >>> a123_jit = get_entry_123_jit(A)
    >>> print(a123_jit)
    -1.3764523

    Example using jax automatic differentiation

    >>> import numpy as np
    >>> import jax
    >>> import t3tools.tucker_tensor_train as t3
    >>> import t3tools.util as util
    >>> jax.config.update("jax_enable_x64", True) # enable double precision for finite difference
    >>> get_entry_123 = lambda x: t3.t3_entry(x, (1,2,3), use_jax=True)
    >>> A0 = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 Tucker tensor train
    >>> f0 = get_entry_123(A0)
    >>> G0 = jax.grad(get_entry_123)(A0) # gradient using automatic differentiation
    >>> dA = t3.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1)))
    >>> df = util.corewise_dot(dA, G0) # sensitivity in direction dA
    >>> print(df)
    -7.418801772515241
    >>> s = 1e-7
    >>> A1 = util.corewise_add(A0, util.corewise_scale(dA, s)) # A1 = A0 + s*dA
    >>> f1 = get_entry_123(A1)
    >>> df_diff = (f1 - f0) / s # finite difference
    >>> print(df_diff)
    -7.418812309825662
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x
    shape, tucker_ranks, tt_ranks = structure(x)

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

    mu_na = xnp.ones((num_entries, tt_cores[0].shape[0]))
    for ind, B_xi, G_axb in zip(index, basis_cores, tt_cores):
        v_xn = B_xi[:, ind]
        g_anb = xnp.einsum('axb,xn->anb', G_axb, v_xn)
        mu_nb = xnp.einsum('na,anb->nb', mu_na, g_anb)
        mu_na = mu_nb
    result = xnp.einsum('na->n', mu_na)

    if not vectorized:
        result = result[0]

    return result


########################################################################
########################    Rank adjustment    #########################
########################################################################

def compute_minimal_ranks(
        s: T3Structure,
) -> typ.Tuple[
    typ.Tuple[int,...], # new_tucker_ranks
    typ.Tuple[int,...], # new_tt_ranks
]:
    '''Find minimal ranks for a TuckerTensorTrain with a given structure. (such that there is no useless rank)

    Examples
    --------
    >>> import t3tools.tucker_tensor_train as t3
    >>> print(t3.compute_minimal_ranks(((10,11,12,13), (14,15,16,17), (98,99,100,101,102))))
    ((10, 11, 12, 13), (1, 10, 100, 13, 1))
    '''
    shape, tucker_ranks, tt_ranks = s
    d = len(shape)
    assert(len(tucker_ranks) == d)
    assert(len(tt_ranks) == d+1)

    new_tucker_ranks   = list(tucker_ranks)
    new_tt_ranks       = list(tt_ranks)

    for ii in range(d):
        new_tucker_ranks[ii] = int(np.minimum(new_tucker_ranks[ii], shape[ii]))

    new_tt_ranks[-1] = 1
    for ii in range(d-1, 0, -1):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        new_tt_ranks[ii] = int(np.minimum(rL, n*rR))

    new_tt_ranks[0] = 1
    for ii in range(d):
        n   = new_tucker_ranks[ii]
        rL  = new_tt_ranks[ii]
        rR  = new_tt_ranks[ii+1]

        n = int(np.minimum(n, rL*rR))
        rR =int(np.minimum(rR, rL*n))
        new_tucker_ranks[ii] = n
        new_tt_ranks[ii+1] = rR

    return tuple(new_tucker_ranks), tuple(new_tt_ranks)


def are_t3_ranks_minimal(
        x: TuckerTensorTrain,
) -> bool:
    """Checks if the ranks of a Tucker train are minimal.

    Example
    -------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,4,9,7,1)))
    >>> print(t3.are_t3_ranks_minimal(x))
    True

    Using T3-SVD to make equivalent T3 with minimal ranks:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((13,14,15,16), (4,5,6,7), (1,99,9,7,1)))
    >>> print(t3.are_t3_ranks_minimal(x))
    False
    >>> x2 = t3.t3_svd(x)[0]
    >>> print(t3.are_t3_ranks_minimal(x2))
    True
    """
    s = structure(x)
    _, tucker_ranks, tt_ranks = s
    minimal_tucker_ranks, minimal_tt_ranks = compute_minimal_ranks(s)
    return (tucker_ranks == minimal_tucker_ranks) and (tt_ranks == minimal_tt_ranks)


def pad_t3(
        x:                  TuckerTensorTrain,
        new_structure:      T3Structure,
        use_jax: bool = False,
) -> TuckerTensorTrain:
    '''Increase TuckerTensorTrain ranks via zero padding.

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (1,3,2,1)))
    >>> new_structure = ((17,18,17), (8,8,8), (1,5,6,1))
    >>> padded_x = t3.pad_t3(x, new_structure)
    >>> print(t3.structure(padded_x))
    ((17, 18, 17), (8, 8, 8), (1, 5, 6, 1))

    Example where first and last ranks are nonzero:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,6,5), (3,3,2,4)))
    >>> new_structure = ((17,18,17), (8,8,8), (5,5,6,7))
    >>> padded_x = t3.pad_t3(x, new_structure)
    >>> print(t3.structure(padded_x))
    ((17, 18, 17), (8, 8, 8), (5, 5, 6, 7))
    '''
    xnp = jnp if use_jax else np

    new_shape, new_tucker_ranks, new_tt_ranks = new_structure

    old_shape, old_tucker_ranks, old_tt_ranks = structure(x)
    num_cores = len(old_shape)
    assert(len(old_shape) == len(new_shape))
    assert(len(old_tucker_ranks) == len(new_tucker_ranks))
    assert(len(old_tt_ranks) == len(new_tt_ranks))

    delta_shape         = [N_new - N_old for N_new, N_old in zip(new_shape, old_shape)]
    delta_tucker_ranks  = [n_new - n_old for n_new, n_old in zip(new_tucker_ranks, old_tucker_ranks)]
    delta_tt_ranks      = [r_new - r_old for r_new, r_old in zip(new_tt_ranks, old_tt_ranks)]

    basis_cores, tt_cores = x

    new_basis_cores = []
    for ii in range(num_cores):
        new_basis_cores.append(xnp.pad(
            basis_cores[ii],
            (
                (0,delta_tucker_ranks[ii]),
                (0,delta_shape[ii]),
            ),
        ))

    new_tt_cores = []
    for ii in range(num_cores):
        new_tt_cores.append(xnp.pad(
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

def left_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U_i_a_x, shape=(ni, na, nx)
    NDArray, # ss_x,    shape=(nx,)
    NDArray, # Vt_x_j,  shape=(nx, nj)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor left unfolding.

    First two indices of the tensor are grouped for the SVD.

    G0_i_a_j = einsum('iax,x,xj->ixj', U_i_a_x, ss_x, Vt_x_j).
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    G0_i_a_j: NDArray
        3-tensor. shape=(ni, na, nj)
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
    U_i_a_x: NDArray
        Left singular vectors, reshaped into 3-tensor. shape=(ni, na, nx).
        einsum('iax,iay->xy', U_i_a_x, U_i_a_x) = identity matrix
    ss_x: NDArray
        Singular values. Non-negative. shape=(nx,).
    Vt_x_j: NDArray
        Right singular vectors. shape=(nx, nj)
        einsum('xj,yj->xy', Vt_x_j, Vt_x_j) = identity matrix

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    truncated_svd
    right_svd_3tensor
    outer_svd_3tensor
    left_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_a_x, ss_x, Vt_x_j = t3.left_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('iax,x,xj->iaj', U_i_a_x, ss_x, Vt_x_j)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2)) # SVD exact to numerical precision
    1.8290510387826402e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('iax,iay->xy', U_i_a_x, U_i_a_x) - np.eye(rank))) # U is left-orthogonal
    1.6194412284045956e-15
    >>> print(np.linalg.norm(np.einsum('xj,yj->xy', Vt_x_j, Vt_x_j) - np.eye(rank))) # V is orthogonal
    1.4738004835812172e-15
    '''
    ni, na, nj = G0_i_a_j.shape
    G0_ia_j = G0_i_a_j.reshape((ni*na, nj))

    U_ia_x, ss_x, Vt_x_j = truncated_svd(G0_ia_j, min_rank, max_rank, rtol, atol, use_jax)

    nx = len(ss_x)
    U_i_a_x = U_ia_x.reshape((ni, na, nx))
    return U_i_a_x, ss_x, Vt_x_j


def right_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U_i_x,       shape=(ni, nx)
    NDArray, # ss_x,        shape=(nx,)
    NDArray, # Vt_x_a_j,    shape=(nx, na, nj)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor right unfolding.

    Last two indices of the tensor are grouped for the SVD.

    G0_i_a_j = einsum('iax,x,xj->ixj', U_i_x, ss_x, Vt_x_a_j).
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    G0_i_a_j: NDArray
        3-tensor. shape=(ni, na, nj)
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
    U_i_x: NDArray
        Left singular vectors. shape=(ni, nx).
        einsum('ix,iy->xy', U_i_x, U_i_x) = identity matrix
    ss_x: NDArray
        Singular values. Non-negative. shape=(nx,).
    Vt_x_a_j: NDArray
        Right singular vectors, reshaped into 3-tensor. shape=(nx, na, nj)
        einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) = identity matrix

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    outer_svd_3tensor
    right_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x, ss_x, Vt_x_a_j = t3.right_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('ix,x,xaj->iaj', U_i_x, ss_x, Vt_x_a_j)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2)) # SVD exact to numerical precision
    1.2503321403334437e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('ix,iy->xy', U_i_x, U_i_x) - np.eye(rank))) # U is orthogonal
    1.6591938592301729e-15
    >>> print(np.linalg.norm(np.einsum('xaj,yaj->xy', Vt_x_a_j, Vt_x_a_j) - np.eye(rank))) # Vt is right-orthogonal
    1.9466202162000267e-15
    '''
    G0_j_a_i = G0_i_a_j.swapaxes(0, 2)
    Vt_j_a_x, ss_x, U_x_i = left_svd_3tensor(G0_j_a_i, min_rank, max_rank, rtol, atol, use_jax)
    Vt_x_a_j = Vt_j_a_x.swapaxes(0, 2)
    U_i_x = U_x_i.swapaxes(0,1)
    return U_i_x, ss_x, Vt_x_a_j


def outer_svd_3tensor(
        G0_i_a_j: NDArray, # shape=(ni, na, nj)
        min_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        max_rank: int = None, # 1 <= min_rank <= max_rank <= minimum(ni*na, nj)
        rtol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        atol: float = None, # removes singular values satisfying sigma < maximum(atol, rtol*sigma1)
        use_jax: bool = False,
) -> typ.Tuple[
    NDArray, # U_i_x_j, shape=(ni, nx, nj),
    NDArray, # ss_x,    shape=(nx,)
    NDArray, # Vt_x_a,  shape=(nx, na)
]:
    '''Compute (truncated) singular value decomposition of 3-tensor outer unfolding.

    First and last indices of the tensor are grouped to form rows for the SVD.
    Middle index forms columns.

    G0_i_a_j = einsum('iax,x,xj->ixj', U_i_x_j, ss_x, Vt_x_a).
    Equality may be approximate if truncation is used.

    Parameters
    ----------
    G0_i_a_j: NDArray
        3-tensor. shape=(ni, na, nj)
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
    U_i_x_j: NDArray
        Left singular vectors. shape=(ni, nx, nj).
        einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) = identity matrix
    ss_x: NDArray
        Singular values. Non-negative. shape=(nx,).
    Vt_x_a: NDArray
        Right singular vectors. shape=(nx, na)
        einsum('xa,ya->xy', Vt_x_a, Vt_x_a) = identity matrix

    Raises
    ------
    RuntimeError
        Error raised if the provided indices in index are inconsistent with each other or the Tucker tensor train x.

    See Also
    --------
    truncated_svd
    left_svd_3tensor
    right_svd_3tensor
    up_svd_ith_tt_core
    down_svd_ith_tt_core
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> G_i_a_j = np.random.randn(5,7,6)
    >>> U_i_x_j, ss_x, Vt_x_a = t3.outer_svd_3tensor(G_i_a_j)
    >>> G_i_a_j2 = np.einsum('ixj,x,xa->iaj', U_i_x_j, ss_x, Vt_x_a)
    >>> print(np.linalg.norm(G_i_a_j - G_i_a_j2)) # SVD exact to numerical precision
    1.4102138928233928e-14
    >>> rank = len(ss_x)
    >>> print(np.linalg.norm(np.einsum('ixj,iyj->xy', U_i_x_j, U_i_x_j) - np.eye(rank))) # U is outer-orthogonal
    3.3426764835898436e-15
    >>> print(np.linalg.norm(np.einsum('xa,ya->xy', Vt_x_a, Vt_x_a) - np.eye(rank))) # Vt is orthogonal
    1.8969691003092744e-15
    '''
    G0_i_j_a = G0_i_a_j.swapaxes(1, 2)
    U_i_j_x, ss_x, Vt_x_a = left_svd_3tensor(G0_i_j_a, min_rank, max_rank, rtol, atol, use_jax)
    U_i_x_j = U_i_j_x.swapaxes(1, 2)
    return U_i_x_j, ss_x, Vt_x_a


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
        The Tucker tensor train. structure=((N1,...,Nd), (n1,...,nd), (r0,r1,...r(d-1),rd))
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

    U2_o_x, ss_x, Vt_x_i = util.truncated_svd(U_o_i, min_rank, max_rank, rtol, atol, use_jax)
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

    A_a_i_x, ss_x, Vt_x_b = left_svd_3tensor(A0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)
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

    U_b_x, ss_x, B_x_j_c = right_svd_3tensor(B0_b_j_c, min_rank, max_rank, rtol, atol, use_jax)
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

    U_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

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

    G_a_x_b, ss_x, Vt_x_i = outer_svd_3tensor(G0_a_i_b, min_rank, max_rank, rtol, atol, use_jax)

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

    Example where first and last TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> x2 = t3.orthogonalize_relative_to_ith_basis_core(0, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    5.152424496985265e-12
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> X = np.einsum('yj,zk,axb,byc,czd->axjkd', B1, B2, G0, G1, G2) # Contraction of everything except B0
    >>> print(np.linalg.norm(np.einsum('axjkd,ayjkd->xy', X, X) - np.eye(B0.shape[0]))) # Complement of B1 is orthogonal
    2.3594586449868743e-15
    '''
    shape, tucker_ranks, tt_ranks = structure(x)

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

    Example where first and last TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> x2 = t3.orthogonalize_relative_to_ith_tt_core(0, x)
    >>> print(np.linalg.norm(t3.t3_to_dense(x) - t3.t3_to_dense(x2))) # Tensor unchanged
    5.4708999671349535e-12
    >>> ((B0, B1, B2), (G0, G1, G2)) = x2
    >>> XR = np.einsum('yi,zj,byc,czd->bijd', B1, B2, G1, G2) # Everything to the right of G0
    >>> print(np.linalg.norm(np.einsum('bijd,cijd->bc', XR, XR) - np.eye(G0.shape[2]))) # Right subtree is right orthogonal
    8.816596607002667e-16
    '''
    shape, tucker_ranks, tt_ranks = structure(x)

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
        First summand. structure=((N1,...,Nd), (n1,...,nd), (r0, r1,...,rd))
    y: TuckerTensorTrain
        Second summand. structure=((N1,...,Nd), (m1,...,md), (q0, q1,...,qd))
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    z: TuckerTensorTrain
        Sum of Tucker tensor trains, z=x+y. structure=((N1,...,Nd), (n1+m1,...,nd+md), (r0+q0, r1+q1,...,rd+qd))

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

    TT-cores for z are block 2x2x2 tensors, with:
        - the TT-cores for x in the (0,0,0) block,
        - the TT-cores for y in the (1,1,1) block,
        - zeros elsewhere.


    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (1,3,2,1)))
    >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (1,5,6,1)))
    >>> z = t3.t3_add(x, y)
    >>> print(t3.structure(z))
    ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2))
    >>> print(np.linalg.norm(t3.t3_to_dense(x) + t3.t3_to_dense(y) - t3.t3_to_dense(z)))
    6.524094086845177e-13
    """
    check_t3(x)
    check_t3(y)

    x_shape = structure(x)[0]
    y_shape = structure(y)[0]
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

    for Gx, Gy in zip(tt_cores_x, tt_cores_y):
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
        First summand. structure=((N1,...,Nd), (n1,...,nd), (r0, r1,...,rd))
    y: TuckerTensorTrain
        Second summand. structure=((N1,...,Nd), (m1,...,md), (q0, q1,...,qd))
    use_jax: bool
        If True, returned TuckerTensorTrain cores are jnp.ndaray. Otherwise, np.ndarray. Default: False

    Returns
    -------
    z: TuckerTensorTrain
        Difference of Tucker tensor trains, z=x-y. structure=((N1,...,Nd), (n1+m1,...,nd+md), (r0+q0, r1+q1, rd+qd))

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
    >>> print(t3.structure(z))
    ((14, 15, 16), (7, 12, 8), (2, 8, 8, 2))
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

    Example where leading and trailing TT-ranks are not 1:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> y = t3.t3_corewise_randn(((14,15,16), (3,7,2), (3,5,6,3)))
    >>> x_dot_y = t3.t3_dot_t3(x, y)
    >>> x_dot_y2 = np.sum(t3.t3_to_dense(x) * t3.t3_to_dense(y))
    >>> print(np.linalg.norm(x_dot_y - x_dot_y2))
    1.3096723705530167e-10
    """
    check_t3(x)
    check_t3(y)

    xnp = jnp if use_jax else np

    x_shape = structure(x)[0]
    y_shape = structure(y)[0]
    if x_shape != y_shape:
        raise RuntimeError(
            'Attempted to dot TuckerTensorTrains (x,y)_HS with inconsistent shapes.'
            + str(x_shape) + ' = x_shape != y_shape = ' + str(y_shape)
        )

    basis_cores_x, tt_cores_x = x
    basis_cores_y, tt_cores_y = y

    r0_x = tt_cores_x[0].shape[0]
    r0_y = tt_cores_y[0].shape[0]

    M_sp = xnp.ones((r0_x, r0_y))
    for Bx_ai, Gx_sat, By_bi, Gy_pbq in zip(basis_cores_x, tt_cores_x, basis_cores_y, tt_cores_y):
        tmp_ab = xnp.einsum('ai,bi->ab', Bx_ai, By_bi)
        tmp_sbt = xnp.einsum('sat,ab->sbt', Gx_sat, tmp_ab)
        tmp_pbt = xnp.einsum('sp,sbt->pbt', M_sp, tmp_sbt)
        tmp_tq = xnp.einsum('pbt,pbq->tq', tmp_pbt, Gy_pbq)
        M_sp = tmp_tq

    rd_x = tt_cores_x[-1].shape[2]
    rd_y = tt_cores_y[-1].shape[2]

    result = xnp.einsum('tq,t,q', M_sp, np.ones(rd_x), np.ones(rd_y))
    return result


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
    >>> x = t3.t3_corewise_randn(((14,15,16), (4,5,6), (2,3,2,2)))
    >>> norm_x = t3.t3_norm(x)
    >>> print(np.abs(norm_x - np.linalg.norm(t3.t3_to_dense(x))))
    1.3642420526593924e-12
    """
    check_t3(x)
    xnp = jnp if use_jax else np

    return xnp.sqrt(t3_dot_t3(x, x, use_jax=use_jax))


###############################################
#############    Dense T3-SVD    ##############
###############################################

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
    >>> print(t3.structure(x))
    ((40, 50, 60), (35, 45, 55), (1, 30, 40, 1))
    >>> print(t3.structure(x2))
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
    >>> print(t3.structure(x))
        ((14, 15, 16), (10, 11, 12), (1, 8, 9, 1))
    >>> print(t3.structure(x2))
        ((14, 15, 16), (3, 3, 2), (1, 2, 2, 1))

    Example where first and last ranks are not ones:

    >>> import numpy as np
    >>> import t3tools.tucker_tensor_train as t3
    >>> x = t3.t3_corewise_randn(((5,6,3), (4,4,3), (2,3,2,2)))
    >>> x2, ss_basis, ss_tt = t3.t3_svd(x) # Compute T3-SVD
    >>> x_dense = t3.t3_to_dense(x, contract_ones=False)
    >>> x2_dense = t3.t3_to_dense(x2, contract_ones=False)
    >>> print(np.linalg.norm(x_dense - x2_dense)) # Tensor unchanged
    5.486408687260824e-13
    >>> ss_tt0 = np.linalg.svd(x_dense.reshape((2,5*6*3*2)))[1] # Singular values of leading unfolding
    >>> print(ss_tt0); print(ss_tt[0])
    [303.0474449   88.85034392]
    [303.0474449   88.85034392]
    >>> ss_tt3 = np.linalg.svd(x_dense.reshape((2*5*6*3,2)))[1] # Singular values of trailing unfolding
    >>> print(ss_tt3); print(ss_tt[3])
    [299.45433768 100.29574828]
    [299.45433768 100.29574828]
    '''
    xnp = jnp if use_jax else np

    basis_cores, tt_cores = x

    num_cores = len(tt_cores)

    # make leading and trailing TT-ranks equal to 1
    G0 = x[1][0]
    G0 = xnp.tensordot(xnp.ones((1, G0.shape[0])), G0, axes=1)

    Gf = x[1][-1]
    Gf = xnp.tensordot(Gf, xnp.ones((Gf.shape[2], 1)), axes=1)

    x = (tuple(x[0]), (G0,) + tuple(x[1][1:-1]) + (Gf,))

    # Orthogonalize basis matrices
    for ii in range(num_cores):
        x, _ = up_svd_ith_basis_core(ii, x, use_jax=use_jax)

    # Right orthogonalize
    for ii in range(num_cores-1, 0, -1): # num_cores-1, num_cores-2, ..., 1
        x, _ = right_svd_ith_tt_core(ii, x, use_jax=use_jax)

    G0 = x[1][0]
    _, ss_first, _ = right_svd_3tensor(G0, use_jax=use_jax)

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
            _, ss_tt, _ = left_svd_3tensor(Gf, use_jax=use_jax)
        all_ss_tt.append(ss_tt)

    return x, tuple(all_ss_basis), tuple(all_ss_tt)


def tucker_svd_dense(
        T: NDArray, # shape=(N1, N2, .., Nd)
        min_ranks:  typ.Sequence[int] = None, # len=d
        max_ranks:  typ.Sequence[int] = None,  # len=d
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[
        typ.Tuple[NDArray,...], # Tucker bases, ith_elm_shape=(ni, Ni)
        NDArray, # Tucker core, shape=(n1,n2,...,nd)
    ],
    typ.Tuple[NDArray,...], # singular values of matricizations
]:
    '''Compute Tucker decomposition and matricization singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_ranks: typ.Sequence[int]
        Minimum Tucker ranks for truncation. len=d
    max_ranks: typ.Sequence[int]
        Maximum Tucker ranks for truncation. len=d
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    typ.Tuple[typ.Tuple[NDArray,...],NDArray]
        Tucker decomposition (tucker_bases, tucker_core). tucker_bases[ii].shape=(ni,Ni). tucker_core.shape=(n1,...,nd)
    typ.Tuple[NDArray,...]
        Singular values of matricizations

    See Also
    --------
    truncated_svd
    tt_svd_dense
    t3_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> (bases, core), ss = dense.tucker_svd_dense(T, rtol=1e-3) # Truncate Tucker SVD to reduce rank
    >>> print(core.shape)
    (9, 9, 9)
    >>> T2 = np.einsum('abc, ai,bj,ck->ijk', core, bases[0], bases[1], bases[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.002418671417862558
    '''
    xnp = jnp if use_jax else np

    bases = []
    singular_values_of_matricizations = []
    C = T
    for ii in range(len(T.shape)):
        C_swap = C.swapaxes(ii,0)
        old_shape_swap = C_swap.shape

        min_rank = None if min_ranks is None else min_ranks[ii]
        max_rank = None if max_ranks is None else max_ranks[ii]

        C_swap_mat = C_swap.reshape((old_shape_swap[0], -1))
        U, ss, Vt = util.truncated_svd(C_swap_mat, min_rank, max_rank, rtol, atol, use_jax)
        rM_new = len(ss)

        singular_values_of_matricizations.append(ss)
        bases.append(U.T)
        C_swap = (ss.reshape((-1,1)) * Vt).reshape((rM_new,) + old_shape_swap[1:])
        C = C_swap.swapaxes(0, ii)

    return (tuple(bases), C), tuple(singular_values_of_matricizations)


def tt_svd_dense(
        T: NDArray,
        min_ranks:  typ.Sequence[int] = None, # len=d+1
        max_ranks:  typ.Sequence[int] = None,  # len=d+1
        rtol: float = None,
        atol: float = None,
        use_jax: bool = False,
) -> typ.Tuple[
    typ.Tuple[NDArray,...], # tt_cores
    typ.Tuple[NDArray,...], # singular values of unfoldings
]:
    '''Compute tensor train (TT) decomposition and unfolding singular values for dense tensor.

    Parameters
    ----------
    T: NDArray
        The dense tensor. shape=(N1, ..., Nd)
    min_ranks: typ.Sequence[int]
        Minimum TT-ranks for truncation. len=d+1. e.g., (1,3,3,3,1)
    max_ranks: typ.Sequence[int]
        Maximum TT-ranks for truncation. len=d+1. e.g., (1,5,5,5,1)
    rtol: float
        Relative tolerance for truncation.
    atol: float
        Absolute tolerance for truncation.
    use_jax: bool
        If True, use jax operations. Otherwise, numpy. Default: False

    Returns
    -------
    typ.Tuple[NDArray,...]
        TT cores. len=d. elm_shape=(ri, ni, r(i+1))
    typ.Tuple[NDArray,...]
        Singular values of unfoldings. len=d+1. elm_shape=(ri,)

    See Also
    --------
    truncated_svd
    tucker_svd_dense
    t3_svd_dense
    t3_svd

    Examples
    --------
    >>> import numpy as np
    >>> import t3tools.dense as dense
    >>> T0 = np.random.randn(40, 50, 60)
    >>> c0 = 1.0 / np.arange(1, 41)**2
    >>> c1 = 1.0 / np.arange(1, 51)**2
    >>> c2 = 1.0 / np.arange(1, 61)**2
    >>> T = np.einsum('ijk,i,j,k->ijk', T0, c0, c1, c2) # Preconditioned random tensor
    >>> cores, ss = dense.tt_svd_dense(T, rtol=1e-3) # Truncate TT-SVD to reduce rank
    >>> print([G.shape for G in cores])
    [(1, 40, 13), (13, 50, 13), (13, 60, 1)]
    >>> T2 = np.einsum('aib,bjc,ckd->ijk', cores[0], cores[1], cores[2])
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.0023999063535883633
    '''
    nn = T.shape

    X = T.reshape((1,) + T.shape)
    singular_values_of_unfoldings = []
    cores = []
    for ii in range(len(nn)-1):
        rL = X.shape[0]

        min_rank = None if min_ranks is None else min_ranks[ii+1]
        max_rank = None if max_ranks is None else max_ranks[ii+1]

        U, ss, Vt = util.truncated_svd(X.reshape((rL * nn[ii], -1)), min_rank, max_rank, rtol, atol, use_jax)
        rR = len(ss)

        singular_values_of_unfoldings.append(ss)
        cores.append(U.reshape((rL, nn[ii], rR)))
        X = ss.reshape((-1,1)) * Vt
    cores.append(X.reshape(X.shape + (1,)))

    norm_T_vec = np.array([np.linalg.norm(T)])
    singular_values_of_unfoldings = [norm_T_vec,] + singular_values_of_unfoldings + [norm_T_vec,]

    return tuple(cores), tuple(singular_values_of_unfoldings)



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
    TuckerTensorTrain, # Approximation of T by Tucker tensor train
    typ.Tuple[NDArray,...], # basis singular values, len=d
    typ.Tuple[NDArray,...], # tt singular values, len=d+1
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
    >>> print(t3.structure(x))
    ((40, 50, 60), (12, 11, 12), (1, 12, 12, 1))
    >>> T2 = t3.t3_to_dense(x)
    >>> print(np.linalg.norm(T - T2) / np.linalg.norm(T)) # should be slightly more than rtol=1e-3
    0.0025147026955504846
    '''
    (basis_cores, tucker_core), ss_tucker = tucker_svd_dense(T, min_tucker_ranks, max_tucker_ranks, rtol, atol, use_jax)
    tt_cores, ss_tt = tt_svd_dense(tucker_core, min_tt_ranks, max_tt_ranks, rtol, atol, use_jax)
    return (basis_cores, tt_cores), ss_tucker, ss_tt

