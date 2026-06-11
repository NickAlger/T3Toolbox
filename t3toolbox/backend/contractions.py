# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import math
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

__all__ = [
    'Fa_Gaib_Fi_to_FGb',
    'GFa_Gaib_Fo_Gio_to_GFb',
    'GFa_Gaib_GiF_to_GFb',
    'GFa_Gaib_GFi_to_GFb',
    'Gio_Fo_to_GFi',
    'dGio_dFo_to_dGFi',
    'GFa_Gaib_GFb_to_GFi',
    'dGFa_dGaib_dGFb_to_dGFi',
    'GFi_Gio_to_GFo',
    'dGFi_dGio_to_dGFo',
    'GFo_Gio_to_GFi',
    'GFo_GFa_to_Gao',
    'Fo_GFa_to_Gao',
    'GFi_GFa_GFj_to_Giaj',
]


def Fa_Gaib_Fi_to_FGb(
        Fa: NDArray,
        Gaib: NDArray,
        Fi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((Fa, Gaib, Fi))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = Fa.shape[:-1]
    G_shape = Gaib.shape[:-3]

    a_shape = Fa.shape[-1:]
    aib_shape = Gaib.shape[-3:]
    b_shape = Gaib.shape[-1:]
    i_shape = Fi.shape[-1:]

    Fa      = Fa.reshape((-1,)      + a_shape)
    Gaib    = Gaib.reshape((-1,)    + aib_shape)
    Fi      = Fi.reshape((-1,)      + i_shape)

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        FGb = xnp.einsum('Fa,Gaib,Fi->FGb', Fa, Gaib, Fi)
    else:
        FGb = xnp.einsum('Fa,Gaib,Fi->FGb', Fa, Gaib, Fi, optimize=path)

    FGb = FGb.reshape(F_shape + G_shape + b_shape)
    return FGb


def GFa_Gaib_Fo_Gio_to_GFb(
        GFa: NDArray,
        Gaib: NDArray,
        Fo: NDArray,
        Gio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((GFa, Gaib, Fo, Gio))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = Fo.shape[:-1]
    G_shape = Gaib.shape[:-3]
    a_shape = (Gaib.shape[-3],)

    aib_shape = Gaib.shape[-3:]
    io_shape = Gio.shape[-2:]
    b_shape = Gaib.shape[-1:]
    o_shape = Fo.shape[-1:]

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    GFa     = GFa.reshape((size_G,) + (size_F,)      + a_shape)
    Gaib    = Gaib.reshape((size_G,) + aib_shape)
    Fo      = Fo.reshape((size_F,) + o_shape)
    Gio     = Gio.reshape((size_G,) + io_shape)

    path = [
        'einsum_path',
        (0, 1),
        (0, 1),
        (0, 1)
    ]

    if use_jax:
        GFb = xnp.einsum('GFa,Gaib,Fo,Gio->GFb', GFa, Gaib, Fo, Gio) # let the compiler figure out the best path
    else:
        GFb = xnp.einsum('GFa,Gaib,Fo,Gio->GFb', GFa, Gaib, Fo, Gio, optimize=path)

    GFb = GFb.reshape(G_shape + F_shape + b_shape)
    return GFb


def GFa_Gaib_GiF_to_GFb(
        GFa: NDArray,
        Gaib: NDArray,
        GiF: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((GFa, Gaib, GiF))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    i_shape = (Gaib.shape[-2],)
    F_shape = GFa.shape[len(G_shape):-1]

    a_shape = GFa.shape[-1:]
    aib_shape = Gaib.shape[-3:]
    b_shape = Gaib.shape[-1:]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    GFa     = GFa.reshape((size_G,) + (size_F,) + a_shape)
    Gaib    = Gaib.reshape((size_G,) + aib_shape)
    GiF     = GiF.reshape((size_G,) + i_shape + (size_F,))

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        GFb = xnp.einsum('GFa,Gaib,GiF->GFb', GFa, Gaib, GiF)
    else:
        GFb = xnp.einsum('GFa,Gaib,GiF->GFb', GFa, Gaib, GiF, optimize=path)

    GFb = GFb.reshape(G_shape + F_shape + b_shape)
    return GFb


def GFa_Gaib_GFi_to_GFb(
        GFa: NDArray,
        Gaib: NDArray,
        GFi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((GFa, Gaib, GFi))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    i_shape = (Gaib.shape[-2],)
    F_shape = GFa.shape[len(G_shape):-1]

    a_shape = GFa.shape[-1:]
    aib_shape = Gaib.shape[-3:]
    b_shape = Gaib.shape[-1:]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    GFa     = GFa.reshape((size_G,) + (size_F,) + a_shape)
    Gaib    = Gaib.reshape((size_G,) + aib_shape)
    GFi     = GFi.reshape((size_G,) + (size_F,) + i_shape)

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        GFb = xnp.einsum('GFa,Gaib,GFi->GFb', GFa, Gaib, GFi)
    else:
        GFb = xnp.einsum('GFa,Gaib,GFi->GFb', GFa, Gaib, GFi, optimize=path)

    GFb = GFb.reshape(G_shape + F_shape + b_shape)
    return GFb


def Gio_Fo_to_GFi(
        Gio: NDArray,
        Fo: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((Gio, Fo))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gio.shape[:-2]
    i_shape = (Gio.shape[-2],)
    o_shape = (Gio.shape[-1],)
    F_shape = Fo.shape[:-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    Gio = Gio.reshape((size_G,) + i_shape + o_shape)
    Fo  = Fo.reshape((size_F,) + o_shape)

    GFi = xnp.einsum('Gio,Fo->GFi', Gio, Fo)

    GFi = GFi.reshape(G_shape + F_shape + i_shape)
    return GFi


def dGio_dFo_to_dGFi(
        dGio: NDArray,
        dFo: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dGio.shape[0],)
    G_shape = dGio.shape[1:-2]
    i_shape = (dGio.shape[-2],)
    o_shape = (dGio.shape[-1],)
    F_shape = dFo.shape[1:-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    dGio = dGio.reshape(d_shape + (size_G,) + i_shape + o_shape)
    dFo  = dFo.reshape(d_shape + (size_F,) + o_shape)

    dGFi = xnp.einsum('dGio,dFo->dGFi', dGio, dFo)

    dGFi = dGFi.reshape(d_shape + G_shape + F_shape + i_shape)
    return dGFi


def GFa_Gaib_GFb_to_GFi(
        GFa: NDArray,
        Gaib: NDArray,
        GFb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((GFa, Gaib, GFb))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    a_shape = (Gaib.shape[-3],)
    i_shape = (Gaib.shape[-2],)
    b_shape = (Gaib.shape[-1],)
    F_shape = GFa.shape[len(G_shape):-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    GFa     = GFa.reshape((size_G,) + (size_F,) + a_shape)
    Gaib    = Gaib.reshape((size_G,) + a_shape + i_shape + b_shape)
    GFb     = GFb.reshape((size_G,) + (size_F,) + b_shape)

    GFi = xnp.einsum('GFa,Gaib,GFb->GFi', GFa, Gaib, GFb)

    GFi = GFi.reshape(G_shape + F_shape + i_shape)
    return GFi


def dGFa_dGaib_dGFb_to_dGFi(
        dGFa: NDArray,
        dGaib: NDArray,
        dGFb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((dGFa, dGaib, dGFb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dGaib.shape[0],)
    G_shape = dGaib.shape[1:-3]
    a_shape = (dGaib.shape[-3],)
    i_shape = (dGaib.shape[-2],)
    b_shape = (dGaib.shape[-1],)
    F_shape = dGFa.shape[1+len(G_shape):-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    dGFa    = dGFa.reshape(d_shape + (size_G,) + (size_F,) + a_shape)
    dGaib   = dGaib.reshape(d_shape + (size_G,) + a_shape + i_shape + b_shape)
    dGFb    = dGFb.reshape(d_shape + (size_G,) + (size_F,) + b_shape)

    dGFi = xnp.einsum('dGFa,dGaib,dGFb->dGFi', dGFa, dGaib, dGFb)

    dGFi = dGFi.reshape(d_shape + G_shape + F_shape + i_shape)
    return dGFi


def GFi_Gio_to_GFo(
        GFi: NDArray,
        Gio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((GFi, Gio))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gio.shape[:-2]
    i_shape = (Gio.shape[-2],)
    o_shape = (Gio.shape[-1],)
    F_shape = GFi.shape[len(G_shape):-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    Gio = Gio.reshape((size_G,) + i_shape + o_shape)
    GFi = GFi.reshape((size_G,) + (size_F,) + i_shape)

    GFo = xnp.einsum('GFi,Gio->GFo', GFi, Gio)

    GFo = GFo.reshape(G_shape + F_shape + o_shape)
    return GFo


def dGFi_dGio_to_dGFo(
        dGFi: NDArray,
        dGio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((dGFi, dGio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dGio.shape[0],)
    G_shape = dGio.shape[1:-2]
    i_shape = (dGio.shape[-2],)
    o_shape = (dGio.shape[-1],)
    F_shape = dGFi.shape[1+len(G_shape):-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    dGio = dGio.reshape(d_shape + (size_G,) + i_shape + o_shape)
    dGFi  = dGFi.reshape(d_shape + (size_G,) + (size_F,) + i_shape)

    dGFo = xnp.einsum('dGFi,dGio->dGFo', dGFi, dGio)

    dGFo = dGFo.reshape(d_shape + G_shape + F_shape + o_shape)
    return dGFo


def GFo_Gio_to_GFi(
        GFo: NDArray,
        Gio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Unlike Gio_Fo_to_GFi (which forms an outer product over the two stacks), here G is a *shared*
    batch on both operands: Gio carries the T3 stack G only, GFo carries G and the probe stack F.
    """
    use_jax = tree_contains_jax((GFo, Gio))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gio.shape[:-2]
    i_shape = (Gio.shape[-2],)
    o_shape = (Gio.shape[-1],)
    F_shape = GFo.shape[len(G_shape):-1]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    Gio = Gio.reshape((size_G,) + i_shape + o_shape)
    GFo = GFo.reshape((size_G,) + (size_F,) + o_shape)

    GFi = xnp.einsum('GFo,Gio->GFi', GFo, Gio)

    GFi = GFi.reshape(G_shape + F_shape + i_shape)
    return GFi


def GFo_GFa_to_Gao(
        GFo: NDArray,
        GFa: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack F (kept on both operands, dropped
    from the output). Capital letters indicate grouped indices, which may be empty. n_probe is the
    number of trailing (probe-stack) batch axes to sum over.
    """
    use_jax = tree_contains_jax((GFo, GFa))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = GFo.shape[:-1]
    n_G = len(prefix) - n_probe
    G_shape = prefix[:n_G]
    F_shape = prefix[n_G:]
    o_shape = (GFo.shape[-1],)
    a_shape = (GFa.shape[-1],)

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    GFo = GFo.reshape((size_G,) + (size_F,) + o_shape)
    GFa = GFa.reshape((size_G,) + (size_F,) + a_shape)

    Gao = xnp.einsum('GFo,GFa->Gao', GFo, GFa)

    Gao = Gao.reshape(G_shape + a_shape + o_shape)
    return Gao


def Fo_GFa_to_Gao(
        Fo: NDArray,
        GFa: NDArray,
) -> NDArray:
    """Computes named contraction, summing over the probe stack F (with Fo broadcast over the T3
    stack G). Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((Fo, GFa))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = Fo.shape[:-1]
    o_shape = (Fo.shape[-1],)
    a_shape = (GFa.shape[-1],)
    G_shape = GFa.shape[:len(GFa.shape) - 1 - len(F_shape)]

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    Fo = Fo.reshape((size_F,) + o_shape)
    GFa = GFa.reshape((size_G,) + (size_F,) + a_shape)

    Gao = xnp.einsum('Fo,GFa->Gao', Fo, GFa)

    Gao = Gao.reshape(G_shape + a_shape + o_shape)
    return Gao


def GFi_GFa_GFj_to_Giaj(
        GFi: NDArray,
        GFa: NDArray,
        GFj: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack F (kept on all operands, dropped from
    the output). Capital letters indicate grouped indices, which may be empty. n_probe is the number
    of trailing (probe-stack) batch axes to sum over.
    """
    use_jax = tree_contains_jax((GFi, GFa, GFj))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = GFi.shape[:-1]
    n_G = len(prefix) - n_probe
    G_shape = prefix[:n_G]
    F_shape = prefix[n_G:]
    i_shape = (GFi.shape[-1],)
    a_shape = (GFa.shape[-1],)
    j_shape = (GFj.shape[-1],)

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    GFi = GFi.reshape((size_G,) + (size_F,) + i_shape)
    GFa = GFa.reshape((size_G,) + (size_F,) + a_shape)
    GFj = GFj.reshape((size_G,) + (size_F,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        Giaj = xnp.einsum('GFi,GFa,GFj->Giaj', GFi, GFa, GFj)
    else:
        Giaj = xnp.einsum('GFi,GFa,GFj->Giaj', GFi, GFa, GFj, optimize=path)

    Giaj = Giaj.reshape(G_shape + i_shape + a_shape + j_shape)
    return Giaj

