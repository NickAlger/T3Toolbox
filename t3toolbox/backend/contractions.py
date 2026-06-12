# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import math
import typing as typ
import numpy as np

from t3toolbox.backend.common import *

# This is the grouped-block contraction toolkit for the case of TWO independent batch blocks on
# DIFFERENT operand subsets (the core/base stack C vs the probe/tangent stack W/K), which a single
# '...' cannot express. For the full design -- why base-inner (W/K outer, C inner), the naming
# scheme, and when to use this vs a plain '...' einsum -- see docs/batching_and_stacking.md (esp. §4).

__all__ = [
    'Wa_Caib_Wi_to_WCb',
    'CWa_Caib_Wo_Cio_to_CWb',
    'WCa_Caib_Wo_Cio_to_WCb',
    'CWa_Caib_CiW_to_CWb',
    'WCa_Caib_WCi_to_WCb',
    'Cio_Wo_to_WCi',
    'dCio_dWo_to_dWCi',
    'WCa_Caib_WCb_to_WCi',
    'dWCa_dCaib_dWCb_to_dWCi',
    'WCi_Cio_to_WCo',
    'dWCi_dCio_to_dWCo',
    'WCo_Cio_to_WCi',
    'WCo_WCa_to_Cao',
    'Wo_WCa_to_Cao',
    'WCi_WCa_WCj_to_Ciaj',
    # Three-group (W probe, K tangent, C base) contractions for probing a K-stacked tangent.
    'WKCa_Caib_WCi_to_WKCb',
    'WCa_Caib_WKCi_to_WKCb',
    'WKCa_Caib_WCb_to_WKCi',
    'WCa_Caib_WKCb_to_WKCi',
    'WCa_KCaib_WCi_to_WKCb',
    'WCa_KCaib_WCb_to_WKCi',
    'WCi_KCio_to_WKCo',
    'WKCi_Cio_to_WKCo',
    # Transpose-assemble three-group (W,K,C) outer products that build variation cores.
    'WKCo_WCa_to_WKCao',
    'WKCo_WCa_to_KCao',
    'Wo_WKCa_to_WKCao',
    'Wo_WKCa_to_KCao',
    'WCi_WCa_WKCj_to_WKCiaj',
    'WCi_WCa_WKCj_to_KCiaj',
    'WKCi_WCa_WCj_to_WKCiaj',
    'WKCi_WCa_WCj_to_KCiaj',
    'WCi_WKCa_WCj_to_WKCiaj',
    'WCi_WKCa_WCj_to_KCiaj',
]


def Wa_Caib_Wi_to_WCb(
        Wa: NDArray,
        Caib: NDArray,
        Wi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((Wa, Caib, Wi))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wa.shape[:-1]
    C_shape = Caib.shape[:-3]

    a_shape = Wa.shape[-1:]
    aib_shape = Caib.shape[-3:]
    b_shape = Caib.shape[-1:]
    i_shape = Wi.shape[-1:]

    Wa      = Wa.reshape((-1,)      + a_shape)
    Caib    = Caib.reshape((-1,)    + aib_shape)
    Wi      = Wi.reshape((-1,)      + i_shape)

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        WCb = xnp.einsum('Wa,Caib,Wi->WCb', Wa, Caib, Wi)
    else:
        WCb = xnp.einsum('Wa,Caib,Wi->WCb', Wa, Caib, Wi, optimize=path)

    WCb = WCb.reshape(W_shape + C_shape + b_shape)
    return WCb


def CWa_Caib_Wo_Cio_to_CWb(
        CWa: NDArray,
        Caib: NDArray,
        Wo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((CWa, Caib, Wo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)

    aib_shape = Caib.shape[-3:]
    io_shape = Cio.shape[-2:]
    b_shape = Caib.shape[-1:]
    o_shape = Wo.shape[-1:]

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    CWa     = CWa.reshape((size_C,) + (size_W,)      + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    Wo      = Wo.reshape((size_W,) + o_shape)
    Cio     = Cio.reshape((size_C,) + io_shape)

    path = [
        'einsum_path',
        (0, 1),
        (0, 1),
        (0, 1)
    ]

    if use_jax:
        CWb = xnp.einsum('CWa,Caib,Wo,Cio->CWb', CWa, Caib, Wo, Cio) # let the compiler figure out the best path
    else:
        CWb = xnp.einsum('CWa,Caib,Wo,Cio->CWb', CWa, Caib, Wo, Cio, optimize=path)

    CWb = CWb.reshape(C_shape + W_shape + b_shape)
    return CWb


def WCa_Caib_Wo_Cio_to_WCb(
        WCa: NDArray,
        Caib: NDArray,
        Wo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost. WC twin of
    CWa_Caib_Wo_Cio_to_CWb, used by the (base-inner) apply.
    """
    use_jax = tree_contains_jax((WCa, Caib, Wo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)

    aib_shape = Caib.shape[-3:]
    io_shape = Cio.shape[-2:]
    b_shape = Caib.shape[-1:]
    o_shape = Wo.shape[-1:]

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    WCa     = WCa.reshape((size_W,) + (size_C,)      + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    Wo      = Wo.reshape((size_W,) + o_shape)
    Cio     = Cio.reshape((size_C,) + io_shape)

    path = [
        'einsum_path',
        (0, 1),
        (0, 1),
        (0, 1)
    ]

    if use_jax:
        WCb = xnp.einsum('WCa,Caib,Wo,Cio->WCb', WCa, Caib, Wo, Cio)
    else:
        WCb = xnp.einsum('WCa,Caib,Wo,Cio->WCb', WCa, Caib, Wo, Cio, optimize=path)

    WCb = WCb.reshape(W_shape + C_shape + b_shape)
    return WCb


def CWa_Caib_CiW_to_CWb(
        CWa: NDArray,
        Caib: NDArray,
        CiW: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.
    """
    use_jax = tree_contains_jax((CWa, Caib, CiW))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    i_shape = (Caib.shape[-2],)
    W_shape = CWa.shape[len(C_shape):-1]

    a_shape = CWa.shape[-1:]
    aib_shape = Caib.shape[-3:]
    b_shape = Caib.shape[-1:]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    CWa     = CWa.reshape((size_C,) + (size_W,) + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    CiW     = CiW.reshape((size_C,) + i_shape + (size_W,))

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        CWb = xnp.einsum('CWa,Caib,CiW->CWb', CWa, Caib, CiW)
    else:
        CWb = xnp.einsum('CWa,Caib,CiW->CWb', CWa, Caib, CiW, optimize=path)

    CWb = CWb.reshape(C_shape + W_shape + b_shape)
    return CWb


def WCa_Caib_WCi_to_WCb(
        WCa: NDArray,
        Caib: NDArray,
        WCi: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((WCa, Caib, WCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    i_shape = (Caib.shape[-2],)
    W_shape = WCa.shape[:-(len(C_shape) + 1)]

    a_shape = WCa.shape[-1:]
    aib_shape = Caib.shape[-3:]
    b_shape = Caib.shape[-1:]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    WCa     = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib    = Caib.reshape((size_C,) + aib_shape)
    WCi     = WCi.reshape((size_W,) + (size_C,) + i_shape)

    path = [
        'einsum_path',
        (0,1),
        (0,1),
    ]

    if use_jax:
        WCb = xnp.einsum('WCa,Caib,WCi->WCb', WCa, Caib, WCi)
    else:
        WCb = xnp.einsum('WCa,Caib,WCi->WCb', WCa, Caib, WCi, optimize=path)

    WCb = WCb.reshape(W_shape + C_shape + b_shape)
    return WCb


def Cio_Wo_to_WCi(
        Cio: NDArray,
        Wo: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((Cio, Wo))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = Wo.shape[:-1]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio = Cio.reshape((size_C,) + i_shape + o_shape)
    Wo  = Wo.reshape((size_W,) + o_shape)

    WCi = xnp.einsum('Cio,Wo->WCi', Cio, Wo)

    WCi = WCi.reshape(W_shape + C_shape + i_shape)
    return WCi


def dCio_dWo_to_dWCi(
        dCio: NDArray,
        dWo: NDArray,
        use_jax: bool = False,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    W_shape = dWo.shape[1:-1]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dWo  = dWo.reshape(d_shape + (size_W,) + o_shape)

    dWCi = xnp.einsum('dCio,dWo->dWCi', dCio, dWo)

    dWCi = dWCi.reshape(d_shape + W_shape + C_shape + i_shape)
    return dWCi


def WCa_Caib_WCb_to_WCi(
        WCa: NDArray,
        Caib: NDArray,
        WCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((WCa, Caib, WCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)
    W_shape = WCa.shape[:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    WCa     = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib    = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WCb     = WCb.reshape((size_W,) + (size_C,) + b_shape)

    WCi = xnp.einsum('WCa,Caib,WCb->WCi', WCa, Caib, WCb)

    WCi = WCi.reshape(W_shape + C_shape + i_shape)
    return WCi


def dWCa_dCaib_dWCb_to_dWCi(
        dWCa: NDArray,
        dCaib: NDArray,
        dWCb: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((dWCa, dCaib, dWCb))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCaib.shape[0],)
    C_shape = dCaib.shape[1:-3]
    a_shape = (dCaib.shape[-3],)
    i_shape = (dCaib.shape[-2],)
    b_shape = (dCaib.shape[-1],)
    W_shape = dWCa.shape[1:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dWCa    = dWCa.reshape(d_shape + (size_W,) + (size_C,) + a_shape)
    dCaib   = dCaib.reshape(d_shape + (size_C,) + a_shape + i_shape + b_shape)
    dWCb    = dWCb.reshape(d_shape + (size_W,) + (size_C,) + b_shape)

    dWCi = xnp.einsum('dWCa,dCaib,dWCb->dWCi', dWCa, dCaib, dWCb)

    dWCi = dWCi.reshape(d_shape + W_shape + C_shape + i_shape)
    return dWCi


def WCi_Cio_to_WCo(
        WCi: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((WCi, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = WCi.shape[:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio = Cio.reshape((size_C,) + i_shape + o_shape)
    WCi = WCi.reshape((size_W,) + (size_C,) + i_shape)

    WCo = xnp.einsum('WCi,Cio->WCo', WCi, Cio)

    WCo = WCo.reshape(W_shape + C_shape + o_shape)
    return WCo


def dWCi_dCio_to_dWCo(
        dWCi: NDArray,
        dCio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Base-inner convention: W (probe/extra stack) outermost, C (core stack) innermost.
    """
    use_jax = tree_contains_jax((dWCi, dCio))
    xnp, _, _ = get_backend(True, use_jax)

    d_shape = (dCio.shape[0],)
    C_shape = dCio.shape[1:-2]
    i_shape = (dCio.shape[-2],)
    o_shape = (dCio.shape[-1],)
    W_shape = dWCi.shape[1:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    dCio = dCio.reshape(d_shape + (size_C,) + i_shape + o_shape)
    dWCi  = dWCi.reshape(d_shape + (size_W,) + (size_C,) + i_shape)

    dWCo = xnp.einsum('dWCi,dCio->dWCo', dWCi, dCio)

    dWCo = dWCo.reshape(d_shape + W_shape + C_shape + o_shape)
    return dWCo


def WCo_Cio_to_WCi(
        WCo: NDArray,
        Cio: NDArray,
) -> NDArray:
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Unlike Cio_Wo_to_WCi (which forms an outer product over the two stacks), here C is a *shared*
    batch on both operands: Cio carries the T3 stack C only, WCo carries the probe stack W and C.
    Base-inner convention: W outermost, C innermost.
    """
    use_jax = tree_contains_jax((WCo, Cio))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Cio.shape[:-2]
    i_shape = (Cio.shape[-2],)
    o_shape = (Cio.shape[-1],)
    W_shape = WCo.shape[:-(len(C_shape) + 1)]

    size_W = math.prod(W_shape)
    size_C = math.prod(C_shape)

    Cio = Cio.reshape((size_C,) + i_shape + o_shape)
    WCo = WCo.reshape((size_W,) + (size_C,) + o_shape)

    WCi = xnp.einsum('WCo,Cio->WCi', WCo, Cio)

    WCi = WCi.reshape(W_shape + C_shape + i_shape)
    return WCi


def WCo_WCa_to_Cao(
        WCo: NDArray,
        WCa: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack W (kept on both operands, dropped
    from the output). Capital letters indicate grouped indices, which may be empty. n_probe is the
    number of leading (probe-stack) batch axes to sum over (base-inner: W outermost, C innermost).
    """
    use_jax = tree_contains_jax((WCo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = WCo.shape[:-1]
    W_shape = prefix[:n_probe]
    C_shape = prefix[n_probe:]
    o_shape = (WCo.shape[-1],)
    a_shape = (WCa.shape[-1],)

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    WCo = WCo.reshape((size_W,) + (size_C,) + o_shape)
    WCa = WCa.reshape((size_W,) + (size_C,) + a_shape)

    Cao = xnp.einsum('WCo,WCa->Cao', WCo, WCa)

    Cao = Cao.reshape(C_shape + a_shape + o_shape)
    return Cao


def Wo_WCa_to_Cao(
        Wo: NDArray,
        WCa: NDArray,
) -> NDArray:
    """Computes named contraction, summing over the probe stack W (with Wo broadcast over the T3
    stack C). Capital letters indicate grouped indices, which may be empty (base-inner: W outer, C inner).
    """
    use_jax = tree_contains_jax((Wo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    o_shape = (Wo.shape[-1],)
    a_shape = (WCa.shape[-1],)
    C_shape = WCa.shape[len(W_shape):-1]

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    Wo = Wo.reshape((size_W,) + o_shape)
    WCa = WCa.reshape((size_W,) + (size_C,) + a_shape)

    Cao = xnp.einsum('Wo,WCa->Cao', Wo, WCa)

    Cao = Cao.reshape(C_shape + a_shape + o_shape)
    return Cao


def WCi_WCa_WCj_to_Ciaj(
        WCi: NDArray,
        WCa: NDArray,
        WCj: NDArray,
        n_probe: int,
) -> NDArray:
    """Computes named contraction, summing over the probe stack W (kept on all operands, dropped from
    the output). Capital letters indicate grouped indices, which may be empty. n_probe is the number
    of leading (probe-stack) batch axes to sum over (base-inner: W outermost, C innermost).
    """
    use_jax = tree_contains_jax((WCi, WCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    prefix = WCi.shape[:-1]
    W_shape = prefix[:n_probe]
    C_shape = prefix[n_probe:]
    i_shape = (WCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WCj.shape[-1],)

    size_C = math.prod(C_shape)
    size_W = math.prod(W_shape)

    WCi = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WCa = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCj = WCj.reshape((size_W,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        Ciaj = xnp.einsum('WCi,WCa,WCj->Ciaj', WCi, WCa, WCj)
    else:
        Ciaj = xnp.einsum('WCi,WCa,WCj->Ciaj', WCi, WCa, WCj, optimize=path)

    Ciaj = Ciaj.reshape(C_shape + i_shape + a_shape + j_shape)
    return Ciaj


###############################################################################
# Three-group contractions (probing a K-stacked tangent).
#
# A third independent batch block K (a stack of tangent vectors sharing one base
# point) joins the probe stack W and base stack C. Base-inner output order is
# W + K + C (W outer, K middle, C inner -- see docs/batching_and_stacking.md).
#
# The split is recovered from shapes, never passed from the frontend. A function
# self-infers when its operands include a C-only base core (pins len(C)) and an
# W+C edge variable (pins len(W)). When the only "core" operand is a variation
# core (K+C) with no C-only operand present, len(C) is underdetermined by the
# operands alone, so n_base is supplied (computed locally in the sweep _func from
# a C-only base core it already holds -- the same precedent as n_probe above).
###############################################################################


def WKCa_Caib_WCi_to_WKCb(
        WKCa: NDArray,  # W + K + C + (a,)   -- e.g. sigma (perturbation left edge var)
        Caib: NDArray,  # C + (a, i, b)      -- e.g. Q base core (C-only -> pins len(C))
        WCi:  NDArray,  # W + C + (i,)       -- e.g. xi-hat base edge var (W+C -> pins len(W))
) -> NDArray:           # W + K + C + (b,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCi (W+C) pins len(W), and K is the remainder of WKCa. K rides on WKCa and the output and
    broadcasts over the operands that lack it.
    """
    use_jax = tree_contains_jax((WKCa, Caib, WCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCi.shape[:-(len(C_shape) + 1)]
    K_shape = WKCa.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCb = xnp.einsum('WKCa,Caib,WCi->WKCb', WKCa, Caib, WCi)
    else:
        WKCb = xnp.einsum('WKCa,Caib,WCi->WKCb', WKCa, Caib, WCi, optimize=path)

    WKCb = WKCb.reshape(W_shape + K_shape + C_shape + b_shape)
    return WKCb


def WCa_Caib_WKCi_to_WKCb(
        WCa:  NDArray,  # W + C + (a,)       -- e.g. mu-hat base edge var (W+C -> pins len(W))
        Caib: NDArray,  # C + (a, i, b)      -- e.g. O base core (C-only -> pins len(C))
        WKCi: NDArray,  # W + K + C + (i,)   -- e.g. delta-xi (perturbation up edge var)
) -> NDArray:           # W + K + C + (b,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCa (W+C) pins len(W), and K is the remainder of WKCi.
    """
    use_jax = tree_contains_jax((WCa, Caib, WKCi))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCa.shape[:-(len(C_shape) + 1)]
    K_shape = WKCi.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WKCi = WKCi.reshape((size_W,) + (size_K,) + (size_C,) + i_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCb = xnp.einsum('WCa,Caib,WKCi->WKCb', WCa, Caib, WKCi)
    else:
        WKCb = xnp.einsum('WCa,Caib,WKCi->WKCb', WCa, Caib, WKCi, optimize=path)

    WKCb = WKCb.reshape(W_shape + K_shape + C_shape + b_shape)
    return WKCb


def WKCa_Caib_WCb_to_WKCi(
        WKCa: NDArray,  # W + K + C + (a,)   -- e.g. sigma (perturbation left edge var)
        Caib: NDArray,  # C + (a, i, b)      -- e.g. Q base core (C-only -> pins len(C))
        WCb:  NDArray,  # W + C + (b,)       -- e.g. nu-hat base edge var (W+C -> pins len(W))
) -> NDArray:           # W + K + C + (i,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCb (W+C) pins len(W), and K is the remainder of WKCa.
    """
    use_jax = tree_contains_jax((WKCa, Caib, WCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCb.shape[:-(len(C_shape) + 1)]
    K_shape = WKCa.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WCb  = WCb.reshape((size_W,) + (size_C,) + b_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCi = xnp.einsum('WKCa,Caib,WCb->WKCi', WKCa, Caib, WCb)
    else:
        WKCi = xnp.einsum('WKCa,Caib,WCb->WKCi', WKCa, Caib, WCb, optimize=path)

    WKCi = WKCi.reshape(W_shape + K_shape + C_shape + i_shape)
    return WKCi


def WCa_Caib_WKCb_to_WKCi(
        WCa:  NDArray,  # W + C + (a,)       -- e.g. mu-hat base edge var (W+C -> pins len(W))
        Caib: NDArray,  # C + (a, i, b)      -- e.g. P base core (C-only -> pins len(C))
        WKCb: NDArray,  # W + K + C + (b,)   -- e.g. tau (perturbation right edge var)
) -> NDArray:           # W + K + C + (i,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction. Self-infers the split: Caib (C-only) pins len(C),
    WCa (W+C) pins len(W), and K is the remainder of WKCb.
    """
    use_jax = tree_contains_jax((WCa, Caib, WKCb))
    xnp, _, _ = get_backend(True, use_jax)

    C_shape = Caib.shape[:-3]
    W_shape = WCa.shape[:-(len(C_shape) + 1)]
    K_shape = WKCb.shape[len(W_shape):-(len(C_shape) + 1)]

    a_shape = (Caib.shape[-3],)
    i_shape = (Caib.shape[-2],)
    b_shape = (Caib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    Caib = Caib.reshape((size_C,) + a_shape + i_shape + b_shape)
    WKCb = WKCb.reshape((size_W,) + (size_K,) + (size_C,) + b_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCi = xnp.einsum('WCa,Caib,WKCb->WKCi', WCa, Caib, WKCb)
    else:
        WKCi = xnp.einsum('WCa,Caib,WKCb->WKCi', WCa, Caib, WKCb, optimize=path)

    WKCi = WKCi.reshape(W_shape + K_shape + C_shape + i_shape)
    return WKCi


def WCa_KCaib_WCi_to_WKCb(
        WCa:   NDArray,  # W + C + (a,)        -- e.g. mu-hat base edge var (W+C)
        KCaib: NDArray,  # K + C + (a, i, b)   -- e.g. delta-C variation tt core (K+C)
        WCi:   NDArray,  # W + C + (i,)        -- e.g. xi-hat base edge var (W+C)
        n_base: int,     # len(C). The only core operand (KCaib) is K+C, so len(C) cannot be
                         # recovered from these operands -- it is supplied (the n_probe precedent).
) -> NDArray:            # W + K + C + (b,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction whose only core operand is a variation core (K+C).
    The operands {W+C, K+C, W+C} do not pin len(C), so it is passed as ``n_base``.
    """
    use_jax = tree_contains_jax((WCa, KCaib, WCi))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCaib.shape[:-3]
    C_shape = KC_shape[len(KC_shape) - n_base:]
    K_shape = KC_shape[:len(KC_shape) - n_base]
    W_shape = WCa.shape[:len(WCa.shape) - 1 - n_base]

    a_shape = (KCaib.shape[-3],)
    i_shape = (KCaib.shape[-2],)
    b_shape = (KCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa   = WCa.reshape((size_W,) + (size_C,) + a_shape)
    KCaib = KCaib.reshape((size_K,) + (size_C,) + a_shape + i_shape + b_shape)
    WCi   = WCi.reshape((size_W,) + (size_C,) + i_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCb = xnp.einsum('WCa,KCaib,WCi->WKCb', WCa, KCaib, WCi)
    else:
        WKCb = xnp.einsum('WCa,KCaib,WCi->WKCb', WCa, KCaib, WCi, optimize=path)

    WKCb = WKCb.reshape(W_shape + K_shape + C_shape + b_shape)
    return WKCb


def WCa_KCaib_WCb_to_WKCi(
        WCa:   NDArray,  # W + C + (a,)        -- e.g. mu-hat base edge var (W+C)
        KCaib: NDArray,  # K + C + (a, i, b)   -- e.g. delta-C variation tt core (K+C)
        WCb:   NDArray,  # W + C + (b,)        -- e.g. nu-hat base edge var (W+C)
        n_base: int,     # len(C) (supplied; KCaib is K+C with no C-only operand to pin it).
) -> NDArray:            # W + K + C + (i,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction whose only core operand is a variation core (K+C).
    The operands {W+C, K+C, W+C} do not pin len(C), so it is passed as ``n_base``.
    """
    use_jax = tree_contains_jax((WCa, KCaib, WCb))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCaib.shape[:-3]
    C_shape = KC_shape[len(KC_shape) - n_base:]
    K_shape = KC_shape[:len(KC_shape) - n_base]
    W_shape = WCa.shape[:len(WCa.shape) - 1 - n_base]

    a_shape = (KCaib.shape[-3],)
    i_shape = (KCaib.shape[-2],)
    b_shape = (KCaib.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCa   = WCa.reshape((size_W,) + (size_C,) + a_shape)
    KCaib = KCaib.reshape((size_K,) + (size_C,) + a_shape + i_shape + b_shape)
    WCb   = WCb.reshape((size_W,) + (size_C,) + b_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCi = xnp.einsum('WCa,KCaib,WCb->WKCi', WCa, KCaib, WCb)
    else:
        WKCi = xnp.einsum('WCa,KCaib,WCb->WKCi', WCa, KCaib, WCb, optimize=path)

    WKCi = WKCi.reshape(W_shape + K_shape + C_shape + i_shape)
    return WKCi


def WCi_KCio_to_WKCo(
        WCi:   NDArray,  # W + C + (i,)        -- e.g. eta-hat base edge var (W+C)
        KCio:  NDArray,  # K + C + (i, o)      -- e.g. delta-U variation tucker core (K+C)
        n_base: int,     # len(C) (supplied; KCio is K+C with no C-only operand to pin it).
) -> NDArray:            # W + K + C + (o,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group (W, K, C) base-inner contraction whose only core operand is a variation core (K+C).
    The operands {W+C, K+C} do not pin len(C), so it is passed as ``n_base``.
    """
    use_jax = tree_contains_jax((WCi, KCio))
    xnp, _, _ = get_backend(True, use_jax)

    KC_shape = KCio.shape[:-2]
    C_shape = KC_shape[len(KC_shape) - n_base:]
    K_shape = KC_shape[:len(KC_shape) - n_base]
    W_shape = WCi.shape[:len(WCi.shape) - 1 - n_base]

    i_shape = (KCio.shape[-2],)
    o_shape = (KCio.shape[-1],)

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    KCio = KCio.reshape((size_K,) + (size_C,) + i_shape + o_shape)

    WKCo = xnp.einsum('WCi,KCio->WKCo', WCi, KCio)

    WKCo = WKCo.reshape(W_shape + K_shape + C_shape + o_shape)
    return WKCo


def WKCi_Cio_to_WKCo(
        WKCi: NDArray,  # W + K + C + (i,)   -- e.g. delta-eta (perturbation down edge var)
        Cio:  NDArray,  # C + (i, o)         -- e.g. U base tucker core (C-only)
) -> NDArray:           # W + K + C + (o,)
    """Computes named contraction. Capital letters indicate grouped indices, which may be empty.

    Three-group name for readability; W and K fuse into one outer block and Cio is C-only, so this
    is exactly the two-group ``WCi_Cio_to_WCo`` with the outer block being W+K. Delegates to it.
    """
    return WCi_Cio_to_WCo(WKCi, Cio)


###############################################################################
# Transpose-assemble three-group contractions (K-stacked tangent transpose).
#
# These build variation cores by OUTER-PRODUCTINC edge variables (the indices a/i/j/o are free, not
# contracted), the adjoint analogue of the forward assembly. The tangent stack K rides on the
# residual-derived operand; the base edge vars stay W+C. Each comes in a keep-W (output W+K+C+...)
# and a sum-W (output K+C+..., the probe stack summed) form; the sum-W forms generalize the existing
# WCo_WCa_to_Cao / Wo_WCa_to_Cao / WCi_WCa_WCj_to_Ciaj (their K=() case).
#
# len(W) (n_probe) is supplied where no operand pins it; the w-bearing ones self-infer W from the
# W-only probe vector. See docs/batching_and_stacking.md and docs/probing_section6_notes.md.
###############################################################################


def WKCo_WCa_to_WKCao(
        WKCo:   NDArray,  # W + K + C + (o,)   -- z-tilde residual (carries K)
        WCa:    NDArray,  # W + C + (a,)       -- eta-hat base edge var (W+C -> pins C given n_probe)
        n_probe: int,     # len(W); {W+K+C, W+C} do not pin it, so it is supplied
) -> NDArray:             # W + K + C + (a, o)
    """Computes named contraction (outer product over a, o). Capitals are grouped indices, may be empty.

    Transpose-assemble (z-tilde (x) eta-hat), keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WKCo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    o_shape = (WKCo.shape[-1],)
    a_shape = (WCa.shape[-1],)
    K_shape = WKCo.shape[n_probe:len(WKCo.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCo = WKCo.reshape((size_W,) + (size_K,) + (size_C,) + o_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)

    WKCao = xnp.einsum('WKCo,WCa->WKCao', WKCo, WCa)

    WKCao = WKCao.reshape(W_shape + K_shape + C_shape + a_shape + o_shape)
    return WKCao


def WKCo_WCa_to_KCao(
        WKCo:   NDArray,  # W + K + C + (o,)   -- z-tilde residual (carries K)
        WCa:    NDArray,  # W + C + (a,)       -- eta-hat base edge var
        n_probe: int,     # len(W), summed out
) -> NDArray:             # K + C + (a, o)
    """Computes named contraction (outer product over a, o; probe stack W summed out).

    Transpose-assemble (z-tilde (x) eta-hat), summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WKCo, WCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    o_shape = (WKCo.shape[-1],)
    a_shape = (WCa.shape[-1],)
    K_shape = WKCo.shape[n_probe:len(WKCo.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCo = WKCo.reshape((size_W,) + (size_K,) + (size_C,) + o_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)

    KCao = xnp.einsum('WKCo,WCa->KCao', WKCo, WCa)

    KCao = KCao.reshape(K_shape + C_shape + a_shape + o_shape)
    return KCao


def Wo_WKCa_to_WKCao(
        Wo:   NDArray,  # W + (o,)           -- probe vector w (W-only -> self-pins len(W))
        WKCa: NDArray,  # W + K + C + (a,)   -- delta-xi-tilde (carries K)
) -> NDArray:           # W + K + C + (a, o)
    """Computes named contraction (outer product over a, o). Capitals are grouped indices, may be empty.

    Transpose-assemble (w (x) delta-xi-tilde), keeping the probe stack W. Self-infers len(W) from the
    W-only probe vector Wo; K and C never need separating here (no operand carries C without K), so
    they ride as one combined block.
    """
    use_jax = tree_contains_jax((Wo, WKCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    o_shape = (Wo.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    KC_shape = WKCa.shape[len(W_shape):-1]

    size_W = math.prod(W_shape)
    size_KC = math.prod(KC_shape)

    Wo   = Wo.reshape((size_W,) + o_shape)
    WKCa = WKCa.reshape((size_W,) + (size_KC,) + a_shape)

    WKCao = xnp.einsum('Wo,WXa->WXao', Wo, WKCa)

    WKCao = WKCao.reshape(W_shape + KC_shape + a_shape + o_shape)
    return WKCao


def Wo_WKCa_to_KCao(
        Wo:   NDArray,  # W + (o,)           -- probe vector w (W-only -> self-pins len(W))
        WKCa: NDArray,  # W + K + C + (a,)   -- delta-xi-tilde (carries K)
) -> NDArray:           # K + C + (a, o)
    """Computes named contraction (outer product over a, o; probe stack W summed out).

    Transpose-assemble (w (x) delta-xi-tilde), summing over the probe stack W. Self-infers len(W);
    K and C ride combined.
    """
    use_jax = tree_contains_jax((Wo, WKCa))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = Wo.shape[:-1]
    o_shape = (Wo.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    KC_shape = WKCa.shape[len(W_shape):-1]

    size_W = math.prod(W_shape)
    size_KC = math.prod(KC_shape)

    Wo   = Wo.reshape((size_W,) + o_shape)
    WKCa = WKCa.reshape((size_W,) + (size_KC,) + a_shape)

    KCao = xnp.einsum('Wo,WXa->Xao', Wo, WKCa)

    KCao = KCao.reshape(KC_shape + a_shape + o_shape)
    return KCao


def WCi_WCa_WKCj_to_WKCiaj(
        WCi:    NDArray,  # W + C + (i,)       -- base edge var (W+C -> pins C given n_probe)
        WCa:    NDArray,  # W + C + (a,)       -- base edge var
        WKCj:   NDArray,  # W + K + C + (j,)   -- residual-derived edge var (carries K)
        n_probe: int,     # len(W); {W+C, W+C, W+K+C} do not pin it, so it is supplied
) -> NDArray:             # W + K + C + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with K on the third (j) edge var, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WCa, WKCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WKCj.shape[-1],)
    K_shape = WKCj.shape[n_probe:len(WKCj.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WKCj = WKCj.reshape((size_W,) + (size_K,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCiaj = xnp.einsum('WCi,WCa,WKCj->WKCiaj', WCi, WCa, WKCj)
    else:
        WKCiaj = xnp.einsum('WCi,WCa,WKCj->WKCiaj', WCi, WCa, WKCj, optimize=path)

    WKCiaj = WKCiaj.reshape(W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return WKCiaj


def WCi_WCa_WKCj_to_KCiaj(
        WCi:    NDArray,  # W + C + (i,)
        WCa:    NDArray,  # W + C + (a,)
        WKCj:   NDArray,  # W + K + C + (j,)
        n_probe: int,     # len(W), summed out
) -> NDArray:             # K + C + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j; probe stack W summed out).

    Transpose tt-assemble term with K on the third (j) edge var, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WCa, WKCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WKCj.shape[-1],)
    K_shape = WKCj.shape[n_probe:len(WKCj.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WKCj = WKCj.reshape((size_W,) + (size_K,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        KCiaj = xnp.einsum('WCi,WCa,WKCj->KCiaj', WCi, WCa, WKCj)
    else:
        KCiaj = xnp.einsum('WCi,WCa,WKCj->KCiaj', WCi, WCa, WKCj, optimize=path)

    KCiaj = KCiaj.reshape(K_shape + C_shape + i_shape + a_shape + j_shape)
    return KCiaj


def WKCi_WCa_WCj_to_WKCiaj(
        WKCi:   NDArray,  # W + K + C + (i,)   -- residual-derived edge var (carries K)
        WCa:    NDArray,  # W + C + (a,)       -- base edge var (W+C -> pins C given n_probe)
        WCj:    NDArray,  # W + C + (j,)       -- base edge var
        n_probe: int,     # len(W); supplied
) -> NDArray:             # W + K + C + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with K on the first (i) edge var, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WKCi, WCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    i_shape = (WKCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCi.shape[n_probe:len(WKCi.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCi = WKCi.reshape((size_W,) + (size_K,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCiaj = xnp.einsum('WKCi,WCa,WCj->WKCiaj', WKCi, WCa, WCj)
    else:
        WKCiaj = xnp.einsum('WKCi,WCa,WCj->WKCiaj', WKCi, WCa, WCj, optimize=path)

    WKCiaj = WKCiaj.reshape(W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return WKCiaj


def WKCi_WCa_WCj_to_KCiaj(
        WKCi:   NDArray,  # W + K + C + (i,)
        WCa:    NDArray,  # W + C + (a,)
        WCj:    NDArray,  # W + C + (j,)
        n_probe: int,     # len(W), summed out
) -> NDArray:             # K + C + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j; probe stack W summed out).

    Transpose tt-assemble term with K on the first (i) edge var, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WKCi, WCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCa.shape[:n_probe]
    C_shape = WCa.shape[n_probe:-1]
    i_shape = (WKCi.shape[-1],)
    a_shape = (WCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCi.shape[n_probe:len(WKCi.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WKCi = WKCi.reshape((size_W,) + (size_K,) + (size_C,) + i_shape)
    WCa  = WCa.reshape((size_W,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        KCiaj = xnp.einsum('WKCi,WCa,WCj->KCiaj', WKCi, WCa, WCj)
    else:
        KCiaj = xnp.einsum('WKCi,WCa,WCj->KCiaj', WKCi, WCa, WCj, optimize=path)

    KCiaj = KCiaj.reshape(K_shape + C_shape + i_shape + a_shape + j_shape)
    return KCiaj


def WCi_WKCa_WCj_to_WKCiaj(
        WCi:    NDArray,  # W + C + (i,)       -- base edge var (W+C -> pins C given n_probe)
        WKCa:   NDArray,  # W + K + C + (a,)   -- residual-derived edge var (carries K)
        WCj:    NDArray,  # W + C + (j,)       -- base edge var
        n_probe: int,     # len(W); supplied
) -> NDArray:             # W + K + C + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j). Capitals may be empty.

    Transpose tt-assemble term with K on the middle (a) edge var, keeping the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WKCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCa.shape[n_probe:len(WKCa.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        WKCiaj = xnp.einsum('WCi,WKCa,WCj->WKCiaj', WCi, WKCa, WCj)
    else:
        WKCiaj = xnp.einsum('WCi,WKCa,WCj->WKCiaj', WCi, WKCa, WCj, optimize=path)

    WKCiaj = WKCiaj.reshape(W_shape + K_shape + C_shape + i_shape + a_shape + j_shape)
    return WKCiaj


def WCi_WKCa_WCj_to_KCiaj(
        WCi:    NDArray,  # W + C + (i,)
        WKCa:   NDArray,  # W + K + C + (a,)
        WCj:    NDArray,  # W + C + (j,)
        n_probe: int,     # len(W), summed out
) -> NDArray:             # K + C + (i, a, j)
    """Computes named contraction (triple outer product over i, a, j; probe stack W summed out).

    Transpose tt-assemble term with K on the middle (a) edge var, summing over the probe stack W.
    """
    use_jax = tree_contains_jax((WCi, WKCa, WCj))
    xnp, _, _ = get_backend(True, use_jax)

    W_shape = WCi.shape[:n_probe]
    C_shape = WCi.shape[n_probe:-1]
    i_shape = (WCi.shape[-1],)
    a_shape = (WKCa.shape[-1],)
    j_shape = (WCj.shape[-1],)
    K_shape = WKCa.shape[n_probe:len(WKCa.shape) - 1 - len(C_shape)]

    size_W = math.prod(W_shape)
    size_K = math.prod(K_shape)
    size_C = math.prod(C_shape)

    WCi  = WCi.reshape((size_W,) + (size_C,) + i_shape)
    WKCa = WKCa.reshape((size_W,) + (size_K,) + (size_C,) + a_shape)
    WCj  = WCj.reshape((size_W,) + (size_C,) + j_shape)

    path = ['einsum_path', (0, 1), (0, 1)]
    if use_jax:
        KCiaj = xnp.einsum('WCi,WKCa,WCj->KCiaj', WCi, WKCa, WCj)
    else:
        KCiaj = xnp.einsum('WCi,WKCa,WCj->KCiaj', WCi, WKCa, WCj, optimize=path)

    KCiaj = KCiaj.reshape(K_shape + C_shape + i_shape + a_shape + j_shape)
    return KCiaj

