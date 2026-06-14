# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
import numpy as np
import typing as typ

import t3toolbox.backend.contractions as contractions
import t3toolbox.backend.t3_operations as t3_operations
from t3toolbox.backend.common import *

__all__ = [
    'tucker_tensor_train_apply',
    'tucker_tensor_train_apply_transpose',
]

def tucker_tensor_train_apply(
        x: typ.Union[
            typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]], # (tucker_cores, tt_cores)
            typ.Tuple[NDArray, NDArray], # (tucker_supercore, tt_supercore)
        ],
        vecs: typ.Union[
            typ.Sequence[NDArray],  # len=d, elm_shape=vsw+(Ni,), ragged
            NDArray, # shape=(d,) + vsw +(Ni,), uniform (NOT IMPLEMENTED YET)
        ],
) -> NDArray:
    '''Contract a Tucker tensor train with vectors in all indices.
    '''
    use_jax = tree_contains_jax((x, vecs))
    xnp, _, xscan = get_backend(False, use_jax)

    #
    tucker_cores, tt_cores = x

    #

    vsc = tucker_cores[0].shape[:-2] # core/base stack C (the batch of T3s)
    vsw = vecs[0].shape[:-1]         # vec stack W (the probe-like vectors), base-inner: W outer, C inner

    def _func(mu_WCa, v_B_G):
        v_Wo, B_Cpo, G_Capb = v_B_G
        mu_WCb = contractions.WCa_Caib_Wo_Cio_to_WCb(
            mu_WCa, G_Capb, v_Wo, B_Cpo,
        )
        return mu_WCb, (0,)

    mu_WCa = xnp.ones(vsw + vsc + (tt_cores[0].shape[-3],))   # W + C
    v_B_G = (vecs, tucker_cores, tt_cores)
    mu_WCz, _ = xscan(_func, mu_WCa, v_B_G)

    result = xnp.sum(mu_WCz, axis=-1)
    return result


def tucker_tensor_train_apply_transpose(
        c:                  NDArray,                # residual, shape=W+C
        ww:                 typ.Sequence[NDArray],  # apply vectors, len=d, elm_shape=W+(Ni,)
        sum_over_probes:    bool = False,           # True: sum the apply stack W (Gauss-Newton J^T r)
) -> typ.Tuple[
    typ.Tuple[NDArray, ...],  # tucker_cores
    typ.Tuple[NDArray, ...],  # tt_cores
]:
    '''Transpose of :py:func:`tucker_tensor_train_apply`: back-project a residual ``c`` into a tensor.

    The Jacobian-transpose of the all-modes apply  ``X -> ( <X, w0^W (x) ... (x) w_{d-1}^W> )_W``.
    The atomic (single-probe) adjoint is the rank-1 tensor ``c * (w0 (x) ... (x) w_{d-1})``; the
    W-stacked operator is its stacking-lift, so:

    - ``sum_over_probes=False`` (primary): ``W`` is a passthrough stacking axis -- a ``W (+ C)`` stack
      of rank-1 tensors (canonical rank 1).
    - ``sum_over_probes=True``: contract ``W`` (``= sum_W`` of the primary) -- the rank-``|W|``
      back-projection ``sum_W c_W * (w0^W (x) ...)``, the Gauss-Newton ``J^T r`` (canonical rank ``|W|``).

    Built as a canonical (CP) decomposition fed to :py:func:`t3_operations.t3_from_canonical`, with
    ``c`` folded into the first factor. Returns ``(tucker_cores, tt_cores)``.
    '''
    use_jax = tree_contains_jax((c, ww))
    xnp, _, _ = get_backend(False, use_jax)
    c = xnp.asarray(c)

    nW = ww[0].ndim - 1     # probe stack rank (ww[i] is W + (Ni,))
    W  = ww[0].shape[:nW]   # probe stack
    C  = c.shape[nW:]       # base stack (c is W + C)
    nC = len(C)

    if sum_over_probes:
        # canonical rank |W|, stack C: w_i flattened over W into the rank axis, broadcast over C;
        # c folded into F_0 as  F_0[C, s, n] = c_flat[s, C] * w0_flat[s, n].
        m = int(np.prod(W, dtype=int))                       # |W|
        c_flat = xnp.moveaxis(c.reshape((m,) + C), 0, nC)    # (m,) + C  ->  C + (m,)
        factors = []
        for i, w in enumerate(ww):
            w_flat = w.reshape((1,) * nC + (m, w.shape[-1]))  # broadcastable to C + (m, Ni)
            if i == 0:
                factors.append(c_flat[..., None] * w_flat)               # C + (m, N0)
            else:
                factors.append(w_flat * xnp.ones(C + (1, 1)))            # materialize C + (m, Ni)
    else:
        # canonical rank 1, stack W + C: c folded into F_0 as  F_0[W, C, 0, n] = c[W, C] * w0[W, n].
        c_exp = c.reshape(W + C + (1, 1))                    # W + C + (1, 1)
        factors = []
        for i, w in enumerate(ww):
            w_exp = w.reshape(W + (1,) * nC + (1, w.shape[-1]))  # broadcastable to W + C + (1, Ni)
            if i == 0:
                factors.append(c_exp * w_exp)                            # W + C + (1, N0)
            else:
                factors.append(w_exp * xnp.ones(C + (1, 1)))             # materialize W + C + (1, Ni)

    return t3_operations.t3_from_canonical(factors)

