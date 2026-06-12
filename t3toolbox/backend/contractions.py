# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import math
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

# This is the grouped-block contraction toolkit for the case of TWO independent batch blocks on
# DIFFERENT operand subsets (the core/base stack G vs the probe/tangent stack F/V), which a single
# '...' cannot express. For the full design -- why base-inner (F/V outer, G inner), the naming
# scheme, and when to use this vs a plain '...' einsum -- see docs/batching_and_stacking.md (esp. §4).

__all__ = [
    'Fa_Gaib_Fi_to_FGb',
    'GFa_Gaib_Fo_Gio_to_GFb',
    'FGa_Gaib_Fo_Gio_to_FGb',
    'GFa_Gaib_GiF_to_GFb',
    'FGa_Gaib_FGi_to_FGb',
    'Gio_Fo_to_FGi',
    'dGio_dFo_to_dFGi',
    'FGa_Gaib_FGb_to_FGi',
    'dFGa_dGaib_dFGb_to_dFGi',
    'FGi_Gio_to_FGo',
    'dFGi_dGio_to_dFGo',
    'FGo_Gio_to_FGi',
    'FGo_FGa_to_Gao',
    'Fo_FGa_to_Gao',
    'FGi_FGa_FGj_to_Giaj',
    # Three-group (F probe, V tangent, G base) contractions for probing a V-stacked tangent.
    'FVGa_Gaib_FGi_to_FVGb',
    'FGa_Gaib_FVGi_to_FVGb',
    'FVGa_Gaib_FGb_to_FVGi',
    'FGa_Gaib_FVGb_to_FVGi',
    'FGa_VGaib_FGi_to_FVGb',
    'FGa_VGaib_FGb_to_FVGi',
    'FGi_VGio_to_FVGo',
    'FVGi_Gio_to_FVGo',
    # Transpose-assemble three-group (F,V,G) outer products that build variation cores.
    'FVGo_FGa_to_FVGao',
    'FVGo_FGa_to_VGao',
    'Fo_FVGa_to_FVGao',
    'Fo_FVGa_to_VGao',
    'FGi_FGa_FVGj_to_FVGiaj',
    'FGi_FGa_FVGj_to_VGiaj',
    'FVGi_FGa_FGj_to_FVGiaj',
    'FVGi_FGa_FGj_to_VGiaj',
    'FGi_FVGa_FGj_to_FVGiaj',
    'FGi_FVGa_FGj_to_VGiaj',
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


def FGa_Gaib_Fo_Gio_to_FGb(
        FGa: NDArray,
        Gaib: NDArray,
        Fo: NDArray,
        Gio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost. FG twin of
    GFa_Gaib_Fo_Gio_to_GFb, used by the (base-inner) apply.
    """
    use_jax = tree_contains_jax((FGa, Gaib, Fo, Gio))
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

    FGa     = FGa.reshape((size_F,) + (size_G,)      + a_shape)
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
        FGb = xnp.einsum('FGa,Gaib,Fo,Gio->FGb', FGa, Gaib, Fo, Gio)
    else:
        FGb = xnp.einsum('FGa,Gaib,Fo,Gio->FGb', FGa, Gaib, Fo, Gio, optimize=path)

    FGb = FGb.reshape(F_shape + G_shape + b_shape)
    return FGb


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


def FGa_Gaib_FGi_to_FGb(
        FGa: NDArray,
        Gaib: NDArray,
        FGi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
    """
    use_jax = tree_contains_jax((FGa, Gaib, FGi))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    i_shape = (Gaib.shape[-2],)
    F_shape = FGa.shape[:-(len(G_shape) + 1)]

    a_shape = FGa.shape[-1:]
    aib_shape = Gaib.shape[-3:]
    b_shape = Gaib.shape[-1:]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    FGa     = FGa.reshape((size_F,) + (size_G,) + a_shape)
    Gaib    = Gaib.reshape((size_G,) + aib_shape)
    FGi     = FGi.reshape((size_F,) + (size_G,) + i_shape)

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        FGb = xnp.einsum('FGa,Gaib,FGi->FGb', FGa, Gaib, FGi)
    else:
        FGb = xnp.einsum('FGa,Gaib,FGi->FGb', FGa, Gaib, FGi, optimize=path)

    FGb = FGb.reshape(F_shape + G_shape + b_shape)
    return FGb


def Gio_Fo_to_FGi(
        Gio: NDArray,
        Fo: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
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

    FGi = xnp.einsum('Gio,Fo->FGi', Gio, Fo)

    FGi = FGi.reshape(F_shape + G_shape + i_shape)
    return FGi


def dGio_dFo_to_dFGi(
        dGio: NDArray,
        dFo: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
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

    dFGi = xnp.einsum('dGio,dFo->dFGi', dGio, dFo)

    dFGi = dFGi.reshape(d_shape + F_shape + G_shape + i_shape)
    return dFGi


def FGa_Gaib_FGb_to_FGi(
        FGa: NDArray,
        Gaib: NDArray,
        FGb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
    """
    use_jax = tree_contains_jax((FGa, Gaib, FGb))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    a_shape = (Gaib.shape[-3],)
    i_shape = (Gaib.shape[-2],)
    b_shape = (Gaib.shape[-1],)
    F_shape = FGa.shape[:-(len(G_shape) + 1)]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    FGa     = FGa.reshape((size_F,) + (size_G,) + a_shape)
    Gaib    = Gaib.reshape((size_G,) + a_shape + i_shape + b_shape)
    FGb     = FGb.reshape((size_F,) + (size_G,) + b_shape)

    FGi = xnp.einsum('FGa,Gaib,FGb->FGi', FGa, Gaib, FGb)

    FGi = FGi.reshape(F_shape + G_shape + i_shape)
    return FGi


def dFGa_dGaib_dFGb_to_dFGi(
        dFGa: NDArray,
        dGaib: NDArray,
        dFGb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
    """
    use_jax = tree_contains_jax((dFGa, dGaib, dFGb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dGaib.shape[0],)
    G_shape = dGaib.shape[1:-3]
    a_shape = (dGaib.shape[-3],)
    i_shape = (dGaib.shape[-2],)
    b_shape = (dGaib.shape[-1],)
    F_shape = dFGa.shape[1:-(len(G_shape) + 1)]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    dFGa    = dFGa.reshape(d_shape + (size_F,) + (size_G,) + a_shape)
    dGaib   = dGaib.reshape(d_shape + (size_G,) + a_shape + i_shape + b_shape)
    dFGb    = dFGb.reshape(d_shape + (size_F,) + (size_G,) + b_shape)

    dFGi = xnp.einsum('dFGa,dGaib,dFGb->dFGi', dFGa, dGaib, dFGb)

    dFGi = dFGi.reshape(d_shape + F_shape + G_shape + i_shape)
    return dFGi


def FGi_Gio_to_FGo(
        FGi: NDArray,
        Gio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
    """
    use_jax = tree_contains_jax((FGi, Gio))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gio.shape[:-2]
    i_shape = (Gio.shape[-2],)
    o_shape = (Gio.shape[-1],)
    F_shape = FGi.shape[:-(len(G_shape) + 1)]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    Gio = Gio.reshape((size_G,) + i_shape + o_shape)
    FGi = FGi.reshape((size_F,) + (size_G,) + i_shape)

    FGo = xnp.einsum('FGi,Gio->FGo', FGi, Gio)

    FGo = FGo.reshape(F_shape + G_shape + o_shape)
    return FGo


def dFGi_dGio_to_dFGo(
        dFGi: NDArray,
        dGio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: F (probe/extra stack) outermost, G (core stack) innermost.
    """
    use_jax = tree_contains_jax((dFGi, dGio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dGio.shape[0],)
    G_shape = dGio.shape[1:-2]
    i_shape = (dGio.shape[-2],)
    o_shape = (dGio.shape[-1],)
    F_shape = dFGi.shape[1:-(len(G_shape) + 1)]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    dGio = dGio.reshape(d_shape + (size_G,) + i_shape + o_shape)
    dFGi  = dFGi.reshape(d_shape + (size_F,) + (size_G,) + i_shape)

    dFGo = xnp.einsum('dFGi,dGio->dFGo', dFGi, dGio)

    dFGo = dFGo.reshape(d_shape + F_shape + G_shape + o_shape)
    return dFGo


def FGo_Gio_to_FGi(
        FGo: NDArray,
        Gio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Unlike Gio_Fo_to_FGi (which forms an outer product over the two stacks), here G is a *shared*
    batch on both operands: Gio carries the T3 stack G only, FGo carries the probe stack F and G.
    Base-inner convention: F outermost, G innermost.
    """
    use_jax = tree_contains_jax((FGo, Gio))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gio.shape[:-2]
    i_shape = (Gio.shape[-2],)
    o_shape = (Gio.shape[-1],)
    F_shape = FGo.shape[:-(len(G_shape) + 1)]

    size_F = math.prod(F_shape)
    size_G = math.prod(G_shape)

    Gio = Gio.reshape((size_G,) + i_shape + o_shape)
    FGo = FGo.reshape((size_F,) + (size_G,) + o_shape)

    FGi = xnp.einsum('FGo,Gio->FGi', FGo, Gio)

    FGi = FGi.reshape(F_shape + G_shape + i_shape)
    return FGi


def FGo_FGa_to_Gao(
        FGo: NDArray,
        FGa: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack F (kept on both operands, dropped
    from the output). Capital letters indicate grouped indices, which may be empty. n_probe is the
    number of leading (probe-stack) batch axes to sum over (base-inner: F outermost, G innermost).
    """
    use_jax = tree_contains_jax((FGo, FGa))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = FGo.shape[:-1]
    F_shape = prefix[:n_probe]
    G_shape = prefix[n_probe:]
    o_shape = (FGo.shape[-1],)
    a_shape = (FGa.shape[-1],)

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    FGo = FGo.reshape((size_F,) + (size_G,) + o_shape)
    FGa = FGa.reshape((size_F,) + (size_G,) + a_shape)

    Gao = xnp.einsum('FGo,FGa->Gao', FGo, FGa)

    Gao = Gao.reshape(G_shape + a_shape + o_shape)
    return Gao


def Fo_FGa_to_Gao(
        Fo: NDArray,
        FGa: NDArray,
) -> NDArray:
    """Computes named contraction, summing over the probe stack F (with Fo broadcast over the T3
    stack G). Capital letters indicate grouped indices, which may be empty (base-inner: F outer, G inner).
    """
    use_jax = tree_contains_jax((Fo, FGa))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = Fo.shape[:-1]
    o_shape = (Fo.shape[-1],)
    a_shape = (FGa.shape[-1],)
    G_shape = FGa.shape[len(F_shape):-1]

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    Fo = Fo.reshape((size_F,) + o_shape)
    FGa = FGa.reshape((size_F,) + (size_G,) + a_shape)

    Gao = xnp.einsum('Fo,FGa->Gao', Fo, FGa)

    Gao = Gao.reshape(G_shape + a_shape + o_shape)
    return Gao


def FGi_FGa_FGj_to_Giaj(
        FGi: NDArray,
        FGa: NDArray,
        FGj: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack F (kept on all operands, dropped from
    the output). Capital letters indicate grouped indices, which may be empty. n_probe is the number
    of leading (probe-stack) batch axes to sum over (base-inner: F outermost, G innermost).
    """
    use_jax = tree_contains_jax((FGi, FGa, FGj))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = FGi.shape[:-1]
    F_shape = prefix[:n_probe]
    G_shape = prefix[n_probe:]
    i_shape = (FGi.shape[-1],)
    a_shape = (FGa.shape[-1],)
    j_shape = (FGj.shape[-1],)

    size_G = math.prod(G_shape)
    size_F = math.prod(F_shape)

    FGi = FGi.reshape((size_F,) + (size_G,) + i_shape)
    FGa = FGa.reshape((size_F,) + (size_G,) + a_shape)
    FGj = FGj.reshape((size_F,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        Giaj = xnp.einsum('FGi,FGa,FGj->Giaj', FGi, FGa, FGj)
    else:
        Giaj = xnp.einsum('FGi,FGa,FGj->Giaj', FGi, FGa, FGj, optimize=path)

    Giaj = Giaj.reshape(G_shape + i_shape + a_shape + j_shape)
    return Giaj


###############################################################################
# Three-group contractions (probing a V-stacked tangent).
#
# A third independent batch block V (a stack of tangent vectors sharing one base
# point) joins the probe stack F and base stack G. Base-inner output order is
# F + V + G (F outer, V middle, G inner -- see docs/batching_and_stacking.md).
#
# The split is recovered from shapes, never passed from the frontend. A function
# self-infers when its operands include a G-only base core (pins len(G)) and an
# F+G edge variable (pins len(F)). When the only "core" operand is a variation
# core (V+G) with no G-only operand present, len(G) is underdetermined by the
# operands alone, so n_base is supplied (computed locally in the sweep _func from
# a G-only base core it already holds -- the same precedent as n_probe above).
###############################################################################


def FVGa_Gaib_FGi_to_FVGb(
        FVGa: NDArray,  # F + V + G + (a,)   -- e.g. sigma (perturbation left edge var)
        Gaib: NDArray,  # G + (a, i, b)      -- e.g. Q base core (G-only -> pins len(G))
        FGi:  NDArray,  # F + G + (i,)       -- e.g. xi-hat base edge var (F+G -> pins len(F))
) -> NDArray:           # F + V + G + (b,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction. Self-infers the split: Gaib (G-only) pins len(G),
    FGi (F+G) pins len(F), and V is the remainder of FVGa. V rides on FVGa and the output and
    broadcasts over the operands that lack it.
    """
    use_jax = tree_contains_jax((FVGa, Gaib, FGi))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    F_shape = FGi.shape[:-(len(G_shape) + 1)]
    V_shape = FVGa.shape[len(F_shape):-(len(G_shape) + 1)]

    a_shape = (Gaib.shape[-3],)
    i_shape = (Gaib.shape[-2],)
    b_shape = (Gaib.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FVGa = FVGa.reshape((size_F,) + (size_V,) + (size_G,) + a_shape)
    Gaib = Gaib.reshape((size_G,) + a_shape + i_shape + b_shape)
    FGi  = FGi.reshape((size_F,) + (size_G,) + i_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGb = xnp.einsum('FVGa,Gaib,FGi->FVGb', FVGa, Gaib, FGi)
    else:
        FVGb = xnp.einsum('FVGa,Gaib,FGi->FVGb', FVGa, Gaib, FGi, optimize=path)

    FVGb = FVGb.reshape(F_shape + V_shape + G_shape + b_shape)
    return FVGb


def FGa_Gaib_FVGi_to_FVGb(
        FGa:  NDArray,  # F + G + (a,)       -- e.g. mu-hat base edge var (F+G -> pins len(F))
        Gaib: NDArray,  # G + (a, i, b)      -- e.g. O base core (G-only -> pins len(G))
        FVGi: NDArray,  # F + V + G + (i,)   -- e.g. delta-xi (perturbation up edge var)
) -> NDArray:           # F + V + G + (b,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction. Self-infers the split: Gaib (G-only) pins len(G),
    FGa (F+G) pins len(F), and V is the remainder of FVGi.
    """
    use_jax = tree_contains_jax((FGa, Gaib, FVGi))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    F_shape = FGa.shape[:-(len(G_shape) + 1)]
    V_shape = FVGi.shape[len(F_shape):-(len(G_shape) + 1)]

    a_shape = (Gaib.shape[-3],)
    i_shape = (Gaib.shape[-2],)
    b_shape = (Gaib.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)
    Gaib = Gaib.reshape((size_G,) + a_shape + i_shape + b_shape)
    FVGi = FVGi.reshape((size_F,) + (size_V,) + (size_G,) + i_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGb = xnp.einsum('FGa,Gaib,FVGi->FVGb', FGa, Gaib, FVGi)
    else:
        FVGb = xnp.einsum('FGa,Gaib,FVGi->FVGb', FGa, Gaib, FVGi, optimize=path)

    FVGb = FVGb.reshape(F_shape + V_shape + G_shape + b_shape)
    return FVGb


def FVGa_Gaib_FGb_to_FVGi(
        FVGa: NDArray,  # F + V + G + (a,)   -- e.g. sigma (perturbation left edge var)
        Gaib: NDArray,  # G + (a, i, b)      -- e.g. Q base core (G-only -> pins len(G))
        FGb:  NDArray,  # F + G + (b,)       -- e.g. nu-hat base edge var (F+G -> pins len(F))
) -> NDArray:           # F + V + G + (i,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction. Self-infers the split: Gaib (G-only) pins len(G),
    FGb (F+G) pins len(F), and V is the remainder of FVGa.
    """
    use_jax = tree_contains_jax((FVGa, Gaib, FGb))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    F_shape = FGb.shape[:-(len(G_shape) + 1)]
    V_shape = FVGa.shape[len(F_shape):-(len(G_shape) + 1)]

    a_shape = (Gaib.shape[-3],)
    i_shape = (Gaib.shape[-2],)
    b_shape = (Gaib.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FVGa = FVGa.reshape((size_F,) + (size_V,) + (size_G,) + a_shape)
    Gaib = Gaib.reshape((size_G,) + a_shape + i_shape + b_shape)
    FGb  = FGb.reshape((size_F,) + (size_G,) + b_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGi = xnp.einsum('FVGa,Gaib,FGb->FVGi', FVGa, Gaib, FGb)
    else:
        FVGi = xnp.einsum('FVGa,Gaib,FGb->FVGi', FVGa, Gaib, FGb, optimize=path)

    FVGi = FVGi.reshape(F_shape + V_shape + G_shape + i_shape)
    return FVGi


def FGa_Gaib_FVGb_to_FVGi(
        FGa:  NDArray,  # F + G + (a,)       -- e.g. mu-hat base edge var (F+G -> pins len(F))
        Gaib: NDArray,  # G + (a, i, b)      -- e.g. P base core (G-only -> pins len(G))
        FVGb: NDArray,  # F + V + G + (b,)   -- e.g. tau (perturbation right edge var)
) -> NDArray:           # F + V + G + (i,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction. Self-infers the split: Gaib (G-only) pins len(G),
    FGa (F+G) pins len(F), and V is the remainder of FVGb.
    """
    use_jax = tree_contains_jax((FGa, Gaib, FVGb))
    xnp, _, _ = get_backend(True, use_jax)

    G_shape = Gaib.shape[:-3]
    F_shape = FGa.shape[:-(len(G_shape) + 1)]
    V_shape = FVGb.shape[len(F_shape):-(len(G_shape) + 1)]

    a_shape = (Gaib.shape[-3],)
    i_shape = (Gaib.shape[-2],)
    b_shape = (Gaib.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)
    Gaib = Gaib.reshape((size_G,) + a_shape + i_shape + b_shape)
    FVGb = FVGb.reshape((size_F,) + (size_V,) + (size_G,) + b_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGi = xnp.einsum('FGa,Gaib,FVGb->FVGi', FGa, Gaib, FVGb)
    else:
        FVGi = xnp.einsum('FGa,Gaib,FVGb->FVGi', FGa, Gaib, FVGb, optimize=path)

    FVGi = FVGi.reshape(F_shape + V_shape + G_shape + i_shape)
    return FVGi


def FGa_VGaib_FGi_to_FVGb(
        FGa:   NDArray,  # F + G + (a,)        -- e.g. mu-hat base edge var (F+G)
        VGaib: NDArray,  # V + G + (a, i, b)   -- e.g. delta-G variation tt core (V+G)
        FGi:   NDArray,  # F + G + (i,)        -- e.g. xi-hat base edge var (F+G)
        n_base: int,     # len(G). The only core operand (VGaib) is V+G, so len(G) cannot be
                         # recovered from these operands -- it is supplied (the n_probe precedent).
) -> NDArray:            # F + V + G + (b,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction whose only core operand is a variation core (V+G).
    The operands {F+G, V+G, F+G} do not pin len(G), so it is passed as ``n_base``.
    """
    use_jax = tree_contains_jax((FGa, VGaib, FGi))
    xnp, _, _ = get_backend(True, use_jax)

    VG_shape = VGaib.shape[:-3]
    G_shape = VG_shape[len(VG_shape) - n_base:]
    V_shape = VG_shape[:len(VG_shape) - n_base]
    F_shape = FGa.shape[:len(FGa.shape) - 1 - n_base]

    a_shape = (VGaib.shape[-3],)
    i_shape = (VGaib.shape[-2],)
    b_shape = (VGaib.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGa   = FGa.reshape((size_F,) + (size_G,) + a_shape)
    VGaib = VGaib.reshape((size_V,) + (size_G,) + a_shape + i_shape + b_shape)
    FGi   = FGi.reshape((size_F,) + (size_G,) + i_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGb = xnp.einsum('FGa,VGaib,FGi->FVGb', FGa, VGaib, FGi)
    else:
        FVGb = xnp.einsum('FGa,VGaib,FGi->FVGb', FGa, VGaib, FGi, optimize=path)

    FVGb = FVGb.reshape(F_shape + V_shape + G_shape + b_shape)
    return FVGb


def FGa_VGaib_FGb_to_FVGi(
        FGa:   NDArray,  # F + G + (a,)        -- e.g. mu-hat base edge var (F+G)
        VGaib: NDArray,  # V + G + (a, i, b)   -- e.g. delta-G variation tt core (V+G)
        FGb:   NDArray,  # F + G + (b,)        -- e.g. nu-hat base edge var (F+G)
        n_base: int,     # len(G) (supplied; VGaib is V+G with no G-only operand to pin it).
) -> NDArray:            # F + V + G + (i,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction whose only core operand is a variation core (V+G).
    The operands {F+G, V+G, F+G} do not pin len(G), so it is passed as ``n_base``.
    """
    use_jax = tree_contains_jax((FGa, VGaib, FGb))
    xnp, _, _ = get_backend(True, use_jax)

    VG_shape = VGaib.shape[:-3]
    G_shape = VG_shape[len(VG_shape) - n_base:]
    V_shape = VG_shape[:len(VG_shape) - n_base]
    F_shape = FGa.shape[:len(FGa.shape) - 1 - n_base]

    a_shape = (VGaib.shape[-3],)
    i_shape = (VGaib.shape[-2],)
    b_shape = (VGaib.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGa   = FGa.reshape((size_F,) + (size_G,) + a_shape)
    VGaib = VGaib.reshape((size_V,) + (size_G,) + a_shape + i_shape + b_shape)
    FGb   = FGb.reshape((size_F,) + (size_G,) + b_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGi = xnp.einsum('FGa,VGaib,FGb->FVGi', FGa, VGaib, FGb)
    else:
        FVGi = xnp.einsum('FGa,VGaib,FGb->FVGi', FGa, VGaib, FGb, optimize=path)

    FVGi = FVGi.reshape(F_shape + V_shape + G_shape + i_shape)
    return FVGi


def FGi_VGio_to_FVGo(
        FGi:   NDArray,  # F + G + (i,)        -- e.g. eta-hat base edge var (F+G)
        VGio:  NDArray,  # V + G + (i, o)      -- e.g. delta-U variation tucker core (V+G)
        n_base: int,     # len(G) (supplied; VGio is V+G with no G-only operand to pin it).
) -> NDArray:            # F + V + G + (o,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (F, V, G) base-inner contraction whose only core operand is a variation core (V+G).
    The operands {F+G, V+G} do not pin len(G), so it is passed as ``n_base``.
    """
    use_jax = tree_contains_jax((FGi, VGio))
    xnp, _, _ = get_backend(True, use_jax)

    VG_shape = VGio.shape[:-2]
    G_shape = VG_shape[len(VG_shape) - n_base:]
    V_shape = VG_shape[:len(VG_shape) - n_base]
    F_shape = FGi.shape[:len(FGi.shape) - 1 - n_base]

    i_shape = (VGio.shape[-2],)
    o_shape = (VGio.shape[-1],)

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGi  = FGi.reshape((size_F,) + (size_G,) + i_shape)
    VGio = VGio.reshape((size_V,) + (size_G,) + i_shape + o_shape)

    FVGo = xnp.einsum('FGi,VGio->FVGo', FGi, VGio)

    FVGo = FVGo.reshape(F_shape + V_shape + G_shape + o_shape)
    return FVGo


def FVGi_Gio_to_FVGo(
        FVGi: NDArray,  # F + V + G + (i,)   -- e.g. delta-eta (perturbation down edge var)
        Gio:  NDArray,  # G + (i, o)         -- e.g. U base tucker core (G-only)
) -> NDArray:           # F + V + G + (o,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group name for readability; F and V fuse into one outer block and Gio is G-only, so this
    is exactly the two-group ``FGi_Gio_to_FGo`` with the outer block being F+V. Delegates to it.
    """
    return FGi_Gio_to_FGo(FVGi, Gio)


###############################################################################
# Transpose-assemble three-group contractions (V-stacked tangent transpose).
#
# These build variation cores by OUTER-PRODUCTING edge variables (the indices a/i/j/o are free, not
# contracted), the adjoint analogue of the forward assembly. The tangent stack V rides on the
# residual-derived operand; the base edge vars stay F+G. Each comes in a keep-F (output F+V+G+...)
# and a sum-F (output V+G+..., the probe stack summed) form; the sum-F forms generalize the existing
# FGo_FGa_to_Gao / Fo_FGa_to_Gao / FGi_FGa_FGj_to_Giaj (their V=() case).
#
# len(F) (n_probe) is supplied where no operand pins it; the w-bearing ones self-infer F from the
# F-only probe vector. See docs/batching_and_stacking.md and docs/probing_section6_notes.md.
###############################################################################


def FVGo_FGa_to_FVGao(
        FVGo:   NDArray,  # F + V + G + (o,)   -- z-tilde residual (carries V)
        FGa:    NDArray,  # F + G + (a,)       -- eta-hat base edge var (F+G -> pins G given n_probe)
        n_probe: int,     # len(F); {F+V+G, F+G} do not pin it, so it is supplied
) -> NDArray:             # F + V + G + (a, o)
    """Computes named contraction (outer product over a, o). Capitals are grouped indices, may be empty.

    Transpose-assemble (z-tilde (x) eta-hat), keeping the probe stack F.
    """
    use_jax = tree_contains_jax((FVGo, FGa))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGa.shape[:n_probe]
    G_shape = FGa.shape[n_probe:-1]
    o_shape = (FVGo.shape[-1],)
    a_shape = (FGa.shape[-1],)
    V_shape = FVGo.shape[n_probe:len(FVGo.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FVGo = FVGo.reshape((size_F,) + (size_V,) + (size_G,) + o_shape)
    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)

    FVGao = xnp.einsum('FVGo,FGa->FVGao', FVGo, FGa)

    FVGao = FVGao.reshape(F_shape + V_shape + G_shape + a_shape + o_shape)
    return FVGao


def FVGo_FGa_to_VGao(
        FVGo:   NDArray,  # F + V + G + (o,)   -- z-tilde residual (carries V)
        FGa:    NDArray,  # F + G + (a,)       -- eta-hat base edge var
        n_probe: int,     # len(F), summed out
) -> NDArray:             # V + G + (a, o)
    """Computes named contraction (outer product over a, o; probe stack F summed out).

    Transpose-assemble (z-tilde (x) eta-hat), summing over the probe stack F.
    """
    use_jax = tree_contains_jax((FVGo, FGa))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGa.shape[:n_probe]
    G_shape = FGa.shape[n_probe:-1]
    o_shape = (FVGo.shape[-1],)
    a_shape = (FGa.shape[-1],)
    V_shape = FVGo.shape[n_probe:len(FVGo.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FVGo = FVGo.reshape((size_F,) + (size_V,) + (size_G,) + o_shape)
    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)

    VGao = xnp.einsum('FVGo,FGa->VGao', FVGo, FGa)

    VGao = VGao.reshape(V_shape + G_shape + a_shape + o_shape)
    return VGao


def Fo_FVGa_to_FVGao(
        Fo:   NDArray,  # F + (o,)           -- probe vector w (F-only -> self-pins len(F))
        FVGa: NDArray,  # F + V + G + (a,)   -- delta-xi-tilde (carries V)
) -> NDArray:           # F + V + G + (a, o)
    """Computes named contraction (outer product over a, o). Capitals are grouped indices, may be empty.

    Transpose-assemble (w (x) delta-xi-tilde), keeping the probe stack F. Self-infers len(F) from the
    F-only probe vector Fo; V and G never need separating here (no operand carries G without V), so
    they ride as one combined block.
    """
    use_jax = tree_contains_jax((Fo, FVGa))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = Fo.shape[:-1]
    o_shape = (Fo.shape[-1],)
    a_shape = (FVGa.shape[-1],)
    VG_shape = FVGa.shape[len(F_shape):-1]

    size_F = math.prod(F_shape)
    size_VG = math.prod(VG_shape)

    Fo   = Fo.reshape((size_F,) + o_shape)
    FVGa = FVGa.reshape((size_F,) + (size_VG,) + a_shape)

    FVGao = xnp.einsum('Fo,FXa->FXao', Fo, FVGa)

    FVGao = FVGao.reshape(F_shape + VG_shape + a_shape + o_shape)
    return FVGao


def Fo_FVGa_to_VGao(
        Fo:   NDArray,  # F + (o,)           -- probe vector w (F-only -> self-pins len(F))
        FVGa: NDArray,  # F + V + G + (a,)   -- delta-xi-tilde (carries V)
) -> NDArray:           # V + G + (a, o)
    """Computes named contraction (outer product over a, o; probe stack F summed out).

    Transpose-assemble (w (x) delta-xi-tilde), summing over the probe stack F. Self-infers len(F);
    V and G ride combined.
    """
    use_jax = tree_contains_jax((Fo, FVGa))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = Fo.shape[:-1]
    o_shape = (Fo.shape[-1],)
    a_shape = (FVGa.shape[-1],)
    VG_shape = FVGa.shape[len(F_shape):-1]

    size_F = math.prod(F_shape)
    size_VG = math.prod(VG_shape)

    Fo   = Fo.reshape((size_F,) + o_shape)
    FVGa = FVGa.reshape((size_F,) + (size_VG,) + a_shape)

    VGao = xnp.einsum('Fo,FXa->Xao', Fo, FVGa)

    VGao = VGao.reshape(VG_shape + a_shape + o_shape)
    return VGao


def FGi_FGa_FVGj_to_FVGiaj(
        FGi:    NDArray,  # F + G + (i,)       -- base edge var (F+G -> pins G given n_probe)
        FGa:    NDArray,  # F + G + (a,)       -- base edge var
        FVGj:   NDArray,  # F + V + G + (j,)   -- residual-derived edge var (carries V)
        n_probe: int,     # len(F); {F+G, F+G, F+V+G} do not pin it, so it is supplied
) -> NDArray:             # F + V + G + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with V on the third (j) edge var, keeping the probe stack F.
    """
    use_jax = tree_contains_jax((FGi, FGa, FVGj))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGi.shape[:n_probe]
    G_shape = FGi.shape[n_probe:-1]
    i_shape = (FGi.shape[-1],)
    a_shape = (FGa.shape[-1],)
    j_shape = (FVGj.shape[-1],)
    V_shape = FVGj.shape[n_probe:len(FVGj.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGi  = FGi.reshape((size_F,) + (size_G,) + i_shape)
    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)
    FVGj = FVGj.reshape((size_F,) + (size_V,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGiaj = xnp.einsum('FGi,FGa,FVGj->FVGiaj', FGi, FGa, FVGj)
    else:
        FVGiaj = xnp.einsum('FGi,FGa,FVGj->FVGiaj', FGi, FGa, FVGj, optimize=path)

    FVGiaj = FVGiaj.reshape(F_shape + V_shape + G_shape + i_shape + a_shape + j_shape)
    return FVGiaj


def FGi_FGa_FVGj_to_VGiaj(
        FGi:    NDArray,  # F + G + (i,)
        FGa:    NDArray,  # F + G + (a,)
        FVGj:   NDArray,  # F + V + G + (j,)
        n_probe: int,     # len(F), summed out
) -> NDArray:             # V + G + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j; probe stack F summed out).

    Transpose tt-assemble term with V on the third (j) edge var, summing over the probe stack F.
    """
    use_jax = tree_contains_jax((FGi, FGa, FVGj))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGi.shape[:n_probe]
    G_shape = FGi.shape[n_probe:-1]
    i_shape = (FGi.shape[-1],)
    a_shape = (FGa.shape[-1],)
    j_shape = (FVGj.shape[-1],)
    V_shape = FVGj.shape[n_probe:len(FVGj.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGi  = FGi.reshape((size_F,) + (size_G,) + i_shape)
    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)
    FVGj = FVGj.reshape((size_F,) + (size_V,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        VGiaj = xnp.einsum('FGi,FGa,FVGj->VGiaj', FGi, FGa, FVGj)
    else:
        VGiaj = xnp.einsum('FGi,FGa,FVGj->VGiaj', FGi, FGa, FVGj, optimize=path)

    VGiaj = VGiaj.reshape(V_shape + G_shape + i_shape + a_shape + j_shape)
    return VGiaj


def FVGi_FGa_FGj_to_FVGiaj(
        FVGi:   NDArray,  # F + V + G + (i,)   -- residual-derived edge var (carries V)
        FGa:    NDArray,  # F + G + (a,)       -- base edge var (F+G -> pins G given n_probe)
        FGj:    NDArray,  # F + G + (j,)       -- base edge var
        n_probe: int,     # len(F); supplied
) -> NDArray:             # F + V + G + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with V on the first (i) edge var, keeping the probe stack F.
    """
    use_jax = tree_contains_jax((FVGi, FGa, FGj))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGa.shape[:n_probe]
    G_shape = FGa.shape[n_probe:-1]
    i_shape = (FVGi.shape[-1],)
    a_shape = (FGa.shape[-1],)
    j_shape = (FGj.shape[-1],)
    V_shape = FVGi.shape[n_probe:len(FVGi.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FVGi = FVGi.reshape((size_F,) + (size_V,) + (size_G,) + i_shape)
    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)
    FGj  = FGj.reshape((size_F,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGiaj = xnp.einsum('FVGi,FGa,FGj->FVGiaj', FVGi, FGa, FGj)
    else:
        FVGiaj = xnp.einsum('FVGi,FGa,FGj->FVGiaj', FVGi, FGa, FGj, optimize=path)

    FVGiaj = FVGiaj.reshape(F_shape + V_shape + G_shape + i_shape + a_shape + j_shape)
    return FVGiaj


def FVGi_FGa_FGj_to_VGiaj(
        FVGi:   NDArray,  # F + V + G + (i,)
        FGa:    NDArray,  # F + G + (a,)
        FGj:    NDArray,  # F + G + (j,)
        n_probe: int,     # len(F), summed out
) -> NDArray:             # V + G + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j; probe stack F summed out).

    Transpose tt-assemble term with V on the first (i) edge var, summing over the probe stack F.
    """
    use_jax = tree_contains_jax((FVGi, FGa, FGj))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGa.shape[:n_probe]
    G_shape = FGa.shape[n_probe:-1]
    i_shape = (FVGi.shape[-1],)
    a_shape = (FGa.shape[-1],)
    j_shape = (FGj.shape[-1],)
    V_shape = FVGi.shape[n_probe:len(FVGi.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FVGi = FVGi.reshape((size_F,) + (size_V,) + (size_G,) + i_shape)
    FGa  = FGa.reshape((size_F,) + (size_G,) + a_shape)
    FGj  = FGj.reshape((size_F,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        VGiaj = xnp.einsum('FVGi,FGa,FGj->VGiaj', FVGi, FGa, FGj)
    else:
        VGiaj = xnp.einsum('FVGi,FGa,FGj->VGiaj', FVGi, FGa, FGj, optimize=path)

    VGiaj = VGiaj.reshape(V_shape + G_shape + i_shape + a_shape + j_shape)
    return VGiaj


def FGi_FVGa_FGj_to_FVGiaj(
        FGi:    NDArray,  # F + G + (i,)       -- base edge var (F+G -> pins G given n_probe)
        FVGa:   NDArray,  # F + V + G + (a,)   -- residual-derived edge var (carries V)
        FGj:    NDArray,  # F + G + (j,)       -- base edge var
        n_probe: int,     # len(F); supplied
) -> NDArray:             # F + V + G + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with V on the middle (a) edge var, keeping the probe stack F.
    """
    use_jax = tree_contains_jax((FGi, FVGa, FGj))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGi.shape[:n_probe]
    G_shape = FGi.shape[n_probe:-1]
    i_shape = (FGi.shape[-1],)
    a_shape = (FVGa.shape[-1],)
    j_shape = (FGj.shape[-1],)
    V_shape = FVGa.shape[n_probe:len(FVGa.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGi  = FGi.reshape((size_F,) + (size_G,) + i_shape)
    FVGa = FVGa.reshape((size_F,) + (size_V,) + (size_G,) + a_shape)
    FGj  = FGj.reshape((size_F,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        FVGiaj = xnp.einsum('FGi,FVGa,FGj->FVGiaj', FGi, FVGa, FGj)
    else:
        FVGiaj = xnp.einsum('FGi,FVGa,FGj->FVGiaj', FGi, FVGa, FGj, optimize=path)

    FVGiaj = FVGiaj.reshape(F_shape + V_shape + G_shape + i_shape + a_shape + j_shape)
    return FVGiaj


def FGi_FVGa_FGj_to_VGiaj(
        FGi:    NDArray,  # F + G + (i,)
        FVGa:   NDArray,  # F + V + G + (a,)
        FGj:    NDArray,  # F + G + (j,)
        n_probe: int,     # len(F), summed out
) -> NDArray:             # V + G + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j; probe stack F summed out).

    Transpose tt-assemble term with V on the middle (a) edge var, summing over the probe stack F.
    """
    use_jax = tree_contains_jax((FGi, FVGa, FGj))
    xnp, _, _ = get_backend(True, use_jax)

    F_shape = FGi.shape[:n_probe]
    G_shape = FGi.shape[n_probe:-1]
    i_shape = (FGi.shape[-1],)
    a_shape = (FVGa.shape[-1],)
    j_shape = (FGj.shape[-1],)
    V_shape = FVGa.shape[n_probe:len(FVGa.shape) - 1 - len(G_shape)]

    size_F = math.prod(F_shape)
    size_V = math.prod(V_shape)
    size_G = math.prod(G_shape)

    FGi  = FGi.reshape((size_F,) + (size_G,) + i_shape)
    FVGa = FVGa.reshape((size_F,) + (size_V,) + (size_G,) + a_shape)
    FGj  = FGj.reshape((size_F,) + (size_G,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        VGiaj = xnp.einsum('FGi,FVGa,FGj->VGiaj', FGi, FVGa, FGj)
    else:
        VGiaj = xnp.einsum('FGi,FVGa,FGj->VGiaj', FGi, FVGa, FGj, optimize=path)

    VGiaj = VGiaj.reshape(V_shape + G_shape + i_shape + a_shape + j_shape)
    return VGiaj

