# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ
import numpy as np

from t3toolbox.common import *

__all__ = [
    'Na_Maib_Ni_to_NMb',
    'MNa_Maib_No_Mio_to_MNb',
    'MNa_Maib_MiN_to_MNb',
    'Mio_No_to_MNi',
    'dMio_dNo_to_dMNi',
    'MNa_Maib_MNb_to_MNi',
    'dMNa_dMaib_dMNb_to_dMNi',
    'MNi_Mio_to_MNo',
    'dMNi_dMio_to_dMNo',
]

NDArray = typ.TypeVar('NDArray') # Generic stand-in for np.ndarray, jnp.ndarray, or other array backend


#############################################
########    Special contractions    #########
#############################################

def Na_Maib_Ni_to_NMb(
        Na: NDArray,
        Maib: NDArray,
        Ni: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum a,aib,i->b, with vectorization over a and i, or aib, or both.

    N and M are the vectorization indices, which may be groups of indices.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_Ni_to_NMb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> xyz_i = np.random.randn(2,3,4, 11)
    >>> NMb = Na_Maib_Ni_to_NMb(xyz_a, uv_aib, xyz_i)
    >>> NMb_true = np.einsum('xyza,uvaib,xyzi->xyzuvb', xyz_a, uv_aib, xyz_i)
    >>> print(NMb.shape == NMb_true.shape)
    True
    >>> print(np.linalg.norm(NMb - NMb_true))
    3.5869432063566724e-13

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_Ni_to_NMb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> aib = np.random.randn(10,11,12)
    >>> xyz_i = np.random.randn(2,3,4, 11)
    >>> Nb = Na_Maib_Ni_to_NMb(xyz_a, aib, xyz_i)
    >>> Nb_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
    >>> print(Nb.shape == Nb_true.shape)
    True
    >>> print(np.linalg.norm(Nb - Nb_true))
    7.459556385862986e-14

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_Ni_to_NMb
    >>> a = np.random.randn(10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> i = np.random.randn(11)
    >>> Mb = Na_Maib_Ni_to_NMb(a, uv_aib, i)
    >>> Mb_true = np.einsum('a,uvaib,i->uvb', a, uv_aib, i)
    >>> print(Mb.shape == Mb_true.shape)
    True
    >>> print(np.linalg.norm(Mb - Mb_true))
    1.254699383909023e-14

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_Ni_to_NMb
    >>> a = np.random.randn(10)
    >>> aib = np.random.randn(10,11,12)
    >>> i = np.random.randn(11)
    >>> b = Na_Maib_Ni_to_NMb(a, aib, i)
    >>> b_true = np.einsum('a,aib,i->b', a, aib, i)
    >>> print(b.shape == b_true.shape)
    True
    >>> print(np.linalg.norm(b - b_true))
    6.108244889215317e-15
    """
    xnp, _, _ = get_backend(True, use_jax)

    N_shape = Na.shape[:-1]
    M_shape = Maib.shape[:-3]

    a_shape = Na.shape[-1:]
    aib_shape = Maib.shape[-3:]
    b_shape = Maib.shape[-1:]
    i_shape = Ni.shape[-1:]

    Na      = Na.reshape((-1,)      + a_shape)
    Maib    = Maib.reshape((-1,)    + aib_shape)
    Ni      = Ni.reshape((-1,)      + i_shape)

    path = [
        'einsum_path',
        (0,1), # Na,Maib,Ni -> Ni, NMib
        (0,1), # Ni, NaMib -> NMb
    ] # contract(Ni, contract(Na, Maib))

    if use_jax:
        NMb = xnp.einsum('Na,Maib,Ni->NMb', Na, Maib, Ni)
    else:
        NMb = xnp.einsum('Na,Maib,Ni->NMb', Na, Maib, Ni, optimize=path)

    NMb = NMb.reshape(N_shape + M_shape + b_shape)
    return NMb


def MNa_Maib_No_Mio_to_MNb(
        MNa: NDArray,
        Maib: NDArray,
        No: NDArray,
        Mio: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum a,aib,o,io->b, with vectorization over a and i, or aib and io, or both.

    N and M are the vectorization indices, which may be groups of indices.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_No_Mio_to_MNb
    >>> uv_xyz_a = np.random.randn(5,6, 2,3,4, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> xyz_o = np.random.randn(2,3,4, 13)
    >>> uv_io = np.random.randn(5,6, 11,13)
    >>> MNb = MNa_Maib_No_Mio_to_MNb(uv_xyz_a, uv_aib, xyz_o, uv_io)
    >>> MNb_true = np.einsum('uvxyza,uvaib,xyzo,uvio->uvxyzb', uv_xyz_a, uv_aib, xyz_o, uv_io)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    1.4784891826885966e-12

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_No_Mio_to_MNb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> aib = np.random.randn(10,11,12)
    >>> xyz_o = np.random.randn(2,3,4, 13)
    >>> io = np.random.randn(11,13)
    >>> MNb = MNa_Maib_No_Mio_to_MNb(xyz_a, aib, xyz_o, io)
    >>> MNb_true = np.einsum('xyza,aib,xyzo,io->xyzb', xyz_a, aib, xyz_o, io)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    4.083260418474411e-13

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_No_Mio_to_MNb
    >>> uv_a = np.random.randn(5,6, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> o = np.random.randn(13)
    >>> uv_io = np.random.randn(5,6, 11,13)
    >>> MNb = MNa_Maib_No_Mio_to_MNb(uv_a, uv_aib, o, uv_io)
    >>> MNb_true = np.einsum('uva,uvaib,o,uvio->uvb', uv_a, uv_aib, o, uv_io)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    2.859552860272838e-13

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_No_Mio_to_MNb
    >>> a = np.random.randn(10)
    >>> aib = np.random.randn(10,11,12)
    >>> o = np.random.randn(13)
    >>> io = np.random.randn(11,13)
    >>> MNb = MNa_Maib_No_Mio_to_MNb(a, aib, o, io)
    >>> MNb_true = np.einsum('a,aib,o,io->b', a, aib, o, io)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    3.638551654418504e-14
    """
    xnp, _, _ = get_backend(True, use_jax)

    N_shape = No.shape[:-1]
    M_shape = Maib.shape[:-3]
    a_shape = (Maib.shape[-3],)

    aib_shape = Maib.shape[-3:]
    io_shape = Mio.shape[-2:]
    b_shape = Maib.shape[-1:]
    o_shape = No.shape[-1:]

    size_M = np.prod(M_shape, dtype=int)
    size_N = np.prod(N_shape, dtype=int)

    MNa     = MNa.reshape((size_M,) + (size_N,)      + a_shape)
    Maib    = Maib.reshape((size_M,) + aib_shape)
    No      = No.reshape((size_N,) + o_shape)
    Mio     = Mio.reshape((size_M,) + io_shape)

    path = [
        'einsum_path',
        (0, 1),
        (0, 1),
        (0, 1)
    ]

    if use_jax:
        MNb = xnp.einsum('MNa,Maib,No,Mio->MNb', MNa, Maib, No, Mio) # let the compiler figure out the best path
    else:
        MNb = xnp.einsum('MNa,Maib,No,Mio->MNb', MNa, Maib, No, Mio, optimize=path)

    MNb = MNb.reshape(M_shape + N_shape + b_shape)
    return MNb


def MNa_Maib_MiN_to_MNb(
        MNa: NDArray,
        Maib: NDArray,
        MiN: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum a,aib,i->b, with vectorization over a and i, or aib and i, or both.

    N and M are the vectorization indices, which may be groups of indices.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MiN_to_MNb
    >>> uv_xyz_a = np.random.randn(5,6,  2,3,4, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> uv_i_xyz = np.random.randn(5,6, 11, 2,3,4)
    >>> MNb = MNa_Maib_MiN_to_MNb(uv_xyz_a, uv_aib, uv_i_xyz)
    >>> MNb_true = np.einsum('uvxyza,uvaib,uvixyz->uvxyzb', uv_xyz_a, uv_aib, uv_i_xyz)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    4.0539884184333385e-13

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MiN_to_MNb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> aib = np.random.randn(10,11,12)
    >>> i_xyz = np.random.randn(11, 2,3,4)
    >>> MNb = MNa_Maib_MiN_to_MNb(xyz_a, aib, i_xyz)
    >>> MNb_true = np.einsum('xyza,aib,ixyz->xyzb', xyz_a, aib, i_xyz)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    7.867752051442911e-14

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MiN_to_MNb
    >>> uv_a = np.random.randn(5,6, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> uv_i = np.random.randn(5,6, 11)
    >>> MNb = MNa_Maib_MiN_to_MNb(uv_a, uv_aib, uv_i)
    >>> MNb_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    6.647117027933763e-14

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MiN_to_MNb
    >>> a = np.random.randn(10)
    >>> aib = np.random.randn(10,11,12)
    >>> i = np.random.randn(11)
    >>> MNb = MNa_Maib_MiN_to_MNb(a, aib, i)
    >>> MNb_true = np.einsum('a,aib,i->b', a, aib, i)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    8.510422543011842e-15
    """
    xnp, _, _ = get_backend(True, use_jax)

    M_shape = Maib.shape[:-3]
    i_shape = (Maib.shape[-2],)
    N_shape = MNa.shape[len(M_shape):-1]

    a_shape = MNa.shape[-1:]
    aib_shape = Maib.shape[-3:]
    b_shape = Maib.shape[-1:]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    MNa     = MNa.reshape((size_M,) + (size_N,) + a_shape)
    Maib    = Maib.reshape((size_M,) + aib_shape)
    MiN     = MiN.reshape((size_M,) + i_shape + (size_N,))

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        MNb = xnp.einsum('MNa,Maib,MiN->MNb', MNa, Maib, MiN)
    else:
        MNb = xnp.einsum('MNa,Maib,MiN->MNb', MNa, Maib, MiN, optimize=path)

    MNb = MNb.reshape(M_shape + N_shape + b_shape)
    return MNb


def MNa_Maib_MNi_to_MNb(
        MNa: NDArray,
        Maib: NDArray,
        MNi: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum a,aib,i->b, with vectorization over a and i, or aib and i, or both.

    N and M are the vectorization indices, which may be groups of indices.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MNi_to_MNb
    >>> uv_xyz_a = np.random.randn(5,6,  2,3,4, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> uv_xyz_i = np.random.randn(5,6, 2,3,4, 11)
    >>> MNb = MNa_Maib_MNi_to_MNb(uv_xyz_a, uv_aib, uv_xyz_i)
    >>> MNb_true = np.einsum('uvxyza,uvaib,uvxyzi->uvxyzb', uv_xyz_a, uv_aib, uv_xyz_i)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    3.7987073746093894e-13

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MNi_to_MNb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> aib = np.random.randn(10,11,12)
    >>> xyz_i = np.random.randn(2,3,4, 11)
    >>> MNb = MNa_Maib_MNi_to_MNb(xyz_a, aib, xyz_i)
    >>> MNb_true = np.einsum('xyza,aib,xyzi->xyzb', xyz_a, aib, xyz_i)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    6.195833601977675e-14

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MiN_to_MNb
    >>> uv_a = np.random.randn(5,6, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> uv_i = np.random.randn(5,6, 11)
    >>> MNb = MNa_Maib_MiN_to_MNb(uv_a, uv_aib, uv_i)
    >>> MNb_true = np.einsum('uva,uvaib,uvi->uvb', uv_a, uv_aib, uv_i)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    7.699394299441565e-14

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MiN_to_MNb
    >>> a = np.random.randn(10)
    >>> aib = np.random.randn(10,11,12)
    >>> i = np.random.randn(11)
    >>> MNb = MNa_Maib_MiN_to_MNb(a, aib, i)
    >>> MNb_true = np.einsum('a,aib,i->b', a, aib, i)
    >>> print(MNb.shape == MNb_true.shape)
    True
    >>> print(np.linalg.norm(MNb - MNb_true))
    9.090279413851974e-15
    """
    xnp, _, _ = get_backend(True, use_jax)

    M_shape = Maib.shape[:-3]
    i_shape = (Maib.shape[-2],)
    N_shape = MNa.shape[len(M_shape):-1]

    a_shape = MNa.shape[-1:]
    aib_shape = Maib.shape[-3:]
    b_shape = Maib.shape[-1:]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    MNa     = MNa.reshape((size_M,) + (size_N,) + a_shape)
    Maib    = Maib.reshape((size_M,) + aib_shape)
    MNi     = MNi.reshape((size_M,) + (size_N,) + i_shape)

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        MNb = xnp.einsum('MNa,Maib,MNi->MNb', MNa, Maib, MNi)
    else:
        MNb = xnp.einsum('MNa,Maib,MNi->MNb', MNa, Maib, MNi, optimize=path)

    MNb = MNb.reshape(M_shape + N_shape + b_shape)
    return MNb


def Mio_No_to_MNi(
        Mio: NDArray,
        No: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum io,o->i, with vectorization over io, o, or both

    N and M are the vectorization indices, which may be groups of indices.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Mio_No_to_MNi
    >>> Mio = np.random.randn(5,6, 10,13)
    >>> No = np.random.randn(2,3,4, 13)
    >>> result = Mio_No_to_MNi(Mio, No)
    >>> result2 = np.einsum('uvio,xyzo->uvxyzi', Mio, No)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Mio_No_to_MNi
    >>> Mio = np.random.randn(10,13)
    >>> No = np.random.randn(2,3,4, 13)
    >>> result = Mio_No_to_MNi(Mio, No)
    >>> result2 = np.einsum('io,xyzo->xyzi', Mio, No)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Mio_No_to_MNi
    >>> Mio = np.random.randn(5,6, 10,13)
    >>> No = np.random.randn(13)
    >>> result = Mio_No_to_MNi(Mio, No)
    >>> result2 = np.einsum('uvio,o->uvi', Mio, No)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Mio_No_to_MNi
    >>> Mio = np.random.randn(10,13)
    >>> No = np.random.randn(13)
    >>> result = Mio_No_to_MNi(Mio, No)
    >>> result2 = np.einsum('io,o->i', Mio, No)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)

    M_shape = Mio.shape[:-2]
    i_shape = (Mio.shape[-2],)
    o_shape = (Mio.shape[-1],)
    N_shape = No.shape[:-1]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    Mio = Mio.reshape((size_M,) + i_shape + o_shape)
    No  = No.reshape((size_N,) + o_shape)

    MNi = xnp.einsum('Mio,No->MNi', Mio, No)

    MNi = MNi.reshape(M_shape + N_shape + i_shape)
    return MNi


def dMio_dNo_to_dMNi(
        dMio: NDArray,
        dNo: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes contraction dMio,dNo->dMNi.

    N and M may be individual indices, groups of indices, or nonexistent.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMio_dNo_to_dMNi
    >>> dMio = np.random.randn(8, 5,6, 10,13)
    >>> dNo = np.random.randn(8, 2,3,4, 13)
    >>> result = dMio_dNo_to_dMNi(dMio, dNo)
    >>> result2 = np.einsum('duvio,dxyzo->duvxyzi', dMio, dNo)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMio_dNo_to_dMNi
    >>> dMio = np.random.randn(8, 10,13)
    >>> dNo = np.random.randn(8, 2,3,4, 13)
    >>> result = dMio_dNo_to_dMNi(dMio, dNo)
    >>> result2 = np.einsum('dio,dxyzo->dxyzi', dMio, dNo)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMio_dNo_to_dMNi
    >>> dMio = np.random.randn(8, 5,6, 10,13)
    >>> dNo = np.random.randn(8, 13)
    >>> result = dMio_dNo_to_dMNi(dMio, dNo)
    >>> result2 = np.einsum('duvio,do->duvi', dMio, dNo)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMio_dNo_to_dMNi
    >>> dMio = np.random.randn(8, 10,13)
    >>> dNo = np.random.randn(8, 13)
    >>> result = dMio_dNo_to_dMNi(dMio, dNo)
    >>> result2 = np.einsum('dio,do->di', dMio, dNo)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dMio.shape[0],)
    M_shape = dMio.shape[1:-2]
    i_shape = (dMio.shape[-2],)
    o_shape = (dMio.shape[-1],)
    N_shape = dNo.shape[1:-1]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    dMio = dMio.reshape(d_shape + (size_M,) + i_shape + o_shape)
    dNo  = dNo.reshape(d_shape + (size_N,) + o_shape)

    dMNi = xnp.einsum('dMio,dNo->dMNi', dMio, dNo)

    dMNi = dMNi.reshape(d_shape + M_shape + N_shape + i_shape)
    return dMNi


def MNa_Maib_MNb_to_MNi(
        MNa: NDArray,
        Maib: NDArray,
        MNb: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes contraction MNa,Maib,MNb->MNi.

    N and M may be individual indices, groups of indices, or nonexistent.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MNb_to_MNi
    >>> MNa = np.random.randn(2,3, 4,5,6, 10)
    >>> Maib = np.random.randn(2,3, 10,11,12)
    >>> MNb = np.random.randn(2,3, 4,5,6, 12)
    >>> result = MNa_Maib_MNb_to_MNi(MNa, Maib, MNb)
    >>> result2 = np.einsum('uvxyza,uvaib,uvxyzb->uvxyzi', MNa, Maib, MNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MNb_to_MNi
    >>> MNa = np.random.randn(4,5,6, 10)
    >>> Maib = np.random.randn(10,11,12)
    >>> MNb = np.random.randn(4,5,6, 12)
    >>> result = MNa_Maib_MNb_to_MNi(MNa, Maib, MNb)
    >>> result2 = np.einsum('xyza,aib,xyzb->xyzi', MNa, Maib, MNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MNb_to_MNi
    >>> MNa = np.random.randn(2,3, 10)
    >>> Maib = np.random.randn(2,3, 10,11,12)
    >>> MNb = np.random.randn(2,3, 12)
    >>> result = MNa_Maib_MNb_to_MNi(MNa, Maib, MNb)
    >>> result2 = np.einsum('uva,uvaib,uvb->uvi', MNa, Maib, MNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNa_Maib_MNb_to_MNi
    >>> MNa = np.random.randn(10)
    >>> Maib = np.random.randn(10,11,12)
    >>> MNb = np.random.randn(12)
    >>> result = MNa_Maib_MNb_to_MNi(MNa, Maib, MNb)
    >>> result2 = np.einsum('a,aib,b->i', MNa, Maib, MNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)

    M_shape = Maib.shape[:-3]
    a_shape = (Maib.shape[-3],)
    i_shape = (Maib.shape[-2],)
    b_shape = (Maib.shape[-1],)
    N_shape = MNa.shape[len(M_shape):-1]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    MNa     = MNa.reshape((size_M,) + (size_N,) + a_shape)
    Maib    = Maib.reshape((size_M,) + a_shape + i_shape + b_shape)
    MNb     = MNb.reshape((size_M,) + (size_N,) + b_shape)

    MNi = xnp.einsum('MNa,Maib,MNb->MNi', MNa, Maib, MNb)

    MNi = MNi.reshape(M_shape + N_shape + i_shape)
    return MNi


def dMNa_dMaib_dMNb_to_dMNi(
        dMNa: NDArray,
        dMaib: NDArray,
        dMNb: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes contraction MNa,dMaib,MNb->dMNi.

    N and M may be individual indices, groups of indices, or nonexistent.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNa_dMaib_dMNb_to_dMNi
    >>> dMNa = np.random.randn(8, 2,3, 4,5,6, 10)
    >>> dMaib = np.random.randn(8, 2,3, 10,11,12)
    >>> dMNb = np.random.randn(8, 2,3, 4,5,6, 12)
    >>> result = dMNa_dMaib_dMNb_to_dMNi(dMNa, dMaib, dMNb)
    >>> result2 = np.einsum('duvxyza,duvaib,duvxyzb->duvxyzi', dMNa, dMaib, dMNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNa_dMaib_dMNb_to_dMNi
    >>> dMNa = np.random.randn(8, 4,5,6, 10)
    >>> dMaib = np.random.randn(8, 10,11,12)
    >>> dMNb = np.random.randn(8, 4,5,6, 12)
    >>> result = dMNa_dMaib_dMNb_to_dMNi(dMNa, dMaib, dMNb)
    >>> result2 = np.einsum('dxyza,daib,dxyzb->dxyzi', dMNa, dMaib, dMNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNa_dMaib_dMNb_to_dMNi
    >>> dMNa = np.random.randn(8, 2,3, 10)
    >>> dMaib = np.random.randn(8, 2,3, 10,11,12)
    >>> dMNb = np.random.randn(8, 2,3, 12)
    >>> result = dMNa_dMaib_dMNb_to_dMNi(dMNa, dMaib, dMNb)
    >>> result2 = np.einsum('duva,duvaib,duvb->duvi', dMNa, dMaib, dMNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNa_dMaib_dMNb_to_dMNi
    >>> dMNa = np.random.randn(8, 10)
    >>> dMaib = np.random.randn(8, 10,11,12)
    >>> dMNb = np.random.randn(8, 12)
    >>> result = dMNa_dMaib_dMNb_to_dMNi(dMNa, dMaib, dMNb)
    >>> result2 = np.einsum('da,daib,db->di', dMNa, dMaib, dMNb)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dMaib.shape[0],)
    M_shape = dMaib.shape[1:-3]
    a_shape = (dMaib.shape[-3],)
    i_shape = (dMaib.shape[-2],)
    b_shape = (dMaib.shape[-1],)
    N_shape = dMNa.shape[1+len(M_shape):-1]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    dMNa    = dMNa.reshape(d_shape + (size_M,) + (size_N,) + a_shape)
    dMaib   = dMaib.reshape(d_shape + (size_M,) + a_shape + i_shape + b_shape)
    dMNb    = dMNb.reshape(d_shape + (size_M,) + (size_N,) + b_shape)

    dMNi = xnp.einsum('dMNa,dMaib,dMNb->dMNi', dMNa, dMaib, dMNb)

    dMNi = dMNi.reshape(d_shape + M_shape + N_shape + i_shape)
    return dMNi


def MNi_Mio_to_MNo(
        MNi: NDArray,
        Mio: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes contraction i,io->o.

    N and M may be individual indices, groups of indices, or nonexistent.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNi_Mio_to_MNo
    >>> Ni = np.random.randn(2,3,4, 10)
    >>> Mio = np.random.randn(5,6, 10,13)
    >>> result = MNi_Mio_to_MNo(Ni, Mio)
    >>> result2 = np.einsum('xyzi,uvio->uvxyzo', Ni, Mio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNi_Mio_to_MNo
    >>> Ni = np.random.randn(2,3,4, 10)
    >>> Mio = np.random.randn(10,13)
    >>> result = MNi_Mio_to_MNo(Ni, Mio)
    >>> result2 = np.einsum('xyzi,io->xyzo', Ni, Mio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNi_Mio_to_MNo
    >>> Ni = np.random.randn(10)
    >>> Mio = np.random.randn(5,6, 10,13)
    >>> result = MNi_Mio_to_MNo(Ni, Mio)
    >>> result2 = np.einsum('i,uvio->uvo', Ni, Mio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import MNi_Mio_to_MNo
    >>> Ni = np.random.randn(10)
    >>> Mio = np.random.randn(10,13)
    >>> result = MNi_Mio_to_MNo(Ni, Mio)
    >>> result2 = np.einsum('i,io->o', Ni, Mio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)

    M_shape = Mio.shape[:-2]
    i_shape = (Mio.shape[-2],)
    o_shape = (Mio.shape[-1],)
    N_shape = MNi.shape[len(M_shape):-1]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    Mio = Mio.reshape((size_M,) + i_shape + o_shape)
    MNi = MNi.reshape((size_M,) + (size_N,) + i_shape)

    MNo = xnp.einsum('MNi,Mio->MNo', MNi, Mio)

    MNo = MNo.reshape(M_shape + N_shape + o_shape)
    return MNo


def dMNi_dMio_to_dMNo(
        dMNi: NDArray,
        dMio: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes named contraction.

    N and M may be individual indices, groups of indices, or nonexistent.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNi_dMio_to_dMNo
    >>> dMNi = np.random.randn(8, 5,6, 2,3,4, 10)
    >>> dMio = np.random.randn(8, 5,6, 10,13)
    >>> result = dMNi_dMio_to_dMNo(dMNi, dMio)
    >>> result2 = np.einsum('duvxyzi,duvio->duvxyzo', dMNi, dMio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNi_dMio_to_dMNo
    >>> dMNi = np.random.randn(8, 2,3,4, 10)
    >>> dMio = np.random.randn(8, 10,13)
    >>> result = dMNi_dMio_to_dMNo(dMNi, dMio)
    >>> result2 = np.einsum('dxyzi,dio->dxyzo', dMNi, dMio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNi_dMio_to_dMNo
    >>> dMNi = np.random.randn(8, 5,6, 10)
    >>> dMio = np.random.randn(8, 5,6, 10,13)
    >>> result = dMNi_dMio_to_dMNo(dMNi, dMio)
    >>> result2 = np.einsum('duvi,duvio->duvo', dMNi, dMio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import dMNi_dMio_to_dMNo
    >>> dMNi = np.random.randn(8, 10)
    >>> dMio = np.random.randn(8, 10,13)
    >>> result = dMNi_dMio_to_dMNo(dMNi, dMio)
    >>> result2 = np.einsum('di,dio->do', dMNi, dMio)
    >>> print(result.shape == result2.shape)
    True
    >>> print(np.linalg.norm(result - result2))
    0.0
    """
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dMio.shape[0],)
    M_shape = dMio.shape[1:-2]
    i_shape = (dMio.shape[-2],)
    o_shape = (dMio.shape[-1],)
    N_shape = dMNi.shape[1+len(M_shape):-1]

    size_N = np.prod(N_shape, dtype=int) # yes, np. We want this done statically. dtype: () -> int 1
    size_M = np.prod(M_shape, dtype=int)

    dMio = dMio.reshape(d_shape + (size_M,) + i_shape + o_shape)
    dMNi  = dMNi.reshape(d_shape + (size_M,) + (size_N,) + i_shape)

    dMNo = xnp.einsum('dMNi,dMio->dMNo', dMNi, dMio)

    dMNo = dMNo.reshape(d_shape + M_shape + N_shape + o_shape)
    return dMNo