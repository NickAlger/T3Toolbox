# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/TuckerTensorTrainTools
# Documentation: https://nickalger.github.io/TuckerTensorTrainTools/index.html
import typing as typ
import numpy as np

from t3toolbox.common import *

__all__ = [
    'Na_Maib_Ni_to_NMb',
    'Na_Maib_No_Mio_to_NMb',
    'Na_Maib_Ni_to_NMb',
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

    N denotes the vectorization index over a and i
    M denotes the vectorization index over aib

    Cases supported:
        - a,aib,i->b
        - Na,aib,Ni->Nb
        - a,Maib,i->Mb
        - Na,Maib,Ni->NMb

    N and M may be grouped indices. E.g., for N=xyz, M=uv we have
        xyza,uvaib,xyzi->xyzuvb

    Seems impossible to make this work for all cases without checking shapes manually.

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

    shape_string = (
        'Na.shape=' + str(Na.shape) + '\n' +
        'Maib.shape=' + str(Maib.shape) + '\n' +
        'Ni.shape=' + str(Ni.shape)
    )

    if len(Na.shape) < 1:
        raise RuntimeError(
            'Na must have nonempty shape.\n' + shape_string
        )

    if len(Maib.shape) < 3:
        raise RuntimeError(
            'Maib must have at least 3 indices.\n' + shape_string
        )

    if len(Ni.shape) < 1:
        raise RuntimeError(
            'Ni must have nonempty shape.\n' + shape_string
        )

    N_shape = Na.shape[:-1]
    M_shape = Maib.shape[:-3]
    N2_shape = Ni.shape[:-1]

    if N_shape != N2_shape:
        raise RuntimeError(
            'Vectorization indices N must be the same on Na and Ni.\n' + shape_string
        )

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


def Na_Maib_No_Mio_to_NMb(
        Na: NDArray,
        Maib: NDArray,
        No: NDArray,
        Mio: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum a,aib,o,io->b, with vectorization over a and i, or aib and io, or both.

    N denotes the vectorization index over a and i
    M denotes the vectorization index over aib and io

    Cases supported:
        - a,aib,o,io->b
        - Na,aib,No,io->Nb
        - a,Maib,o,Mio->Mb
        - Na,Maib,No,Mio->NMb

    N and M may be grouped indices. E.g., for N=xyz, M=uv we have
        xyza,uvaib,xyzo,uvio->xyzuvb

    Seems impossible to make this work for all cases without checking shapes manually.

    Examples
    --------

    Vectorize over both N and M:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_No_Mio_to_NMb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> uv_aib = np.random.randn(5,6, 10,11,12)
    >>> xyz_o = np.random.randn(2,3,4, 13)
    >>> uv_io = np.random.randn(5,6, 11,13)
    >>> NMb = Na_Maib_No_Mio_to_NMb(xyz_a, uv_aib, xyz_o, uv_io)
    >>> NMb_true = np.einsum('xyza,uvaib,xyzo,uvio->xyzuvb', xyz_a, uv_aib, xyz_o, uv_io)
    >>> print(NMb.shape == NMb_true.shape)
    True
    >>> print(np.linalg.norm(NMb - NMb_true))
    1.4784891826885966e-12

    Vectorize over N only

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_No_Mio_to_NMb
    >>> xyz_a = np.random.randn(2,3,4, 10)
    >>> aib = np.random.randn(10,11,12)
    >>> xyz_o = np.random.randn(2,3,4, 13)
    >>> io = np.random.randn(11,13)
    >>> Nb = Na_Maib_No_Mio_to_NMb(xyz_a, aib, xyz_o, io)
    >>> Nb_true = np.einsum('xyza,aib,xyzo,io->xyzb', xyz_a, aib, xyz_o, io)
    >>> print(Nb.shape == Nb_true.shape)
    True
    >>> print(np.linalg.norm(Nb - Nb_true))
    4.083260418474411e-13

    Vectorize over both M only:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_No_Mio_to_NMb
    >>> a = np.random.randn(10)
    >>> uva_ib = np.random.randn(5,6, 10,11,12)
    >>> o = np.random.randn(13)
    >>> uv_io = np.random.randn(5,6, 11,13)
    >>> Mb = Na_Maib_No_Mio_to_NMb(a, uva_ib, o, uv_io)
    >>> Mb_true = np.einsum('a,uvaib,o,uvio->uvb', a, uva_ib, o, uv_io)
    >>> print(Mb.shape == Mb_true.shape)
    True
    >>> print(np.linalg.norm(Mb - Mb_true))
    2.859552860272838e-13

    No vectorization:

    >>> import numpy as np
    >>> from t3toolbox.utils.contractions import Na_Maib_No_Mio_to_NMb
    >>> a = np.random.randn(10)
    >>> aib = np.random.randn(10,11,12)
    >>> o = np.random.randn(13)
    >>> io = np.random.randn(11,13)
    >>> b = Na_Maib_No_Mio_to_NMb(a, aib, o, io)
    >>> b_true = np.einsum('a,aib,o,io->b', a, aib, o, io)
    >>> print(b.shape == b_true.shape)
    True
    >>> print(np.linalg.norm(b - b_true))
    3.638551654418504e-14
    """
    xnp, _, _ = get_backend(True, use_jax)

    shape_string = (
        'Na.shape='     + str(Na.shape)     + '\n' +
        'Maib.shape='   + str(Maib.shape)   + '\n' +
        'No.shape='     + str(No.shape)     + '\n' +
        'Mio.shape='    + str(Mio.shape)

    )

    if len(Na.shape) < 1:
        raise RuntimeError(
            'Na must have nonempty shape.\n' + shape_string
        )

    if len(Maib.shape) < 3:
        raise RuntimeError(
            'Maib must have at least 3 indices.\n' + shape_string
        )

    if len(Mio.shape) < 2:
        raise RuntimeError(
            'Mio must have at least 2 indices.\n' + shape_string
        )

    if len(No.shape) < 1:
        raise RuntimeError(
            'No must have nonempty shape.\n' + shape_string
        )

    N_shape = Na.shape[:-1]
    M_shape = Maib.shape[:-3]
    M2_shape = Mio.shape[:-2]
    N2_shape = No.shape[:-1]

    if N_shape != N2_shape:
        raise RuntimeError(
            'Vectorization indices N must be the same on Na and No.\n' + shape_string
        )

    if M_shape != M2_shape:
        raise RuntimeError(
            'Vectorization indices M must be the same on Maib and Mio.\n' + shape_string
        )

    a_shape = Na.shape[-1:]
    aib_shape = Maib.shape[-3:]
    io_shape = Mio.shape[-2:]
    b_shape = Maib.shape[-1:]
    o_shape = No.shape[-1:]

    Na      = Na.reshape((-1,)      + a_shape)
    Maib    = Maib.reshape((-1,)    + aib_shape)
    No      = No.reshape((-1,)      + o_shape)
    Mio     = Mio.reshape((-1,)      + io_shape)

    path = [
        'einsum_path',
        (0, 1),  # Na, Maib, No, Mio -> No, Mio, NMib
        (0, 1),  # No, Mio  NMib -> NMib, NMi
        (0, 1)  # NMib, NMi -> Nmb
    ]  # contract(contract(No, Mio), contract(Na, Maib))

    if use_jax:
        NMb = xnp.einsum('Na,Maib,No,Mio->NMb', Na, Maib, No, Mio) # let the compiler figure out the best path
    else:
        NMb = xnp.einsum('Na,Maib,No,Mio->NMb', Na, Maib, No, Mio, optimize=path)

    NMb = NMb.reshape(N_shape + M_shape + b_shape)
    return NMb


def MNa_Maib_MiN_to_MNb(
        MNa: NDArray,
        Maib: NDArray,
        MiN: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes vectorized einsum a,aib,i->b, with vectorization over a and i, or aib and i, or both.

    N denotes the vectorization index over a, aib, and i
    M denotes the vectorization index over aib and i

    Cases supported:
        - a,aib,ni->b
        - Na,aib,Ni->Nb
        - Ma,Maib,Mi->Mb
        - MNa,Maib,MiN->MNb

    N and M may be grouped indices. E.g., for N=xyz, M=uv we have
        xyza,uvaib,xyziuv->xyzuvb

    Seems impossible to make this work for all cases without checking shapes manually.

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

    shape_string = (
        'MNa.shape=' + str(MNa.shape) + '\n' +
        'Maib.shape=' + str(Maib.shape) + '\n' +
        'MiN.shape=' + str(MiN.shape)
    )

    if len(MNa.shape) < 1:
        raise RuntimeError(
            'MNa must have nonempty shape.\n' + shape_string
        )

    if len(Maib.shape) < 3:
        raise RuntimeError(
            'Maib must have at least 3 indices.\n' + shape_string
        )

    if len(MiN.shape) < 1:
        raise RuntimeError(
            'MiN must have nonempty shape.\n' + shape_string
        )

    M_shape = Maib.shape[:-3]
    i_shape = (Maib.shape[-2],)
    N_shape = MNa.shape[len(M_shape):-1]


    if MiN.shape != M_shape + i_shape + N_shape:
        raise RuntimeError(
            'Vectorization indices M and N must be consistent in Na, Maib and MiN.\n' + shape_string
        )

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