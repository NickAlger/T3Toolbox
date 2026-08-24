# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Left/right orthogonalization of bare tt chains -- polymorphic over ragged and uniform.

The ragged name is the polymorphic name (no ``utt_`` twins); dispatch is inferred from the input
arrays.
"""
import numpy as np
import typing as typ

import t3toolbox.backend.tt_operations as tt_operations
import t3toolbox.backend.linalg as linalg
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.ut3_operations as uniform_operations
from t3toolbox.backend.common import *

__all__ = [
    'tt_left_orthogonalize',
    'tt_right_orthogonalize',
]


def _tt_left_step(
        H0: NDArray,               # carry: running right factor, stack_shape+(rL,n,rR)
        x:  typ.Tuple[NDArray],    # (G1,) the next TT core
) -> typ.Tuple[NDArray, typ.Tuple[NDArray, NDArray]]:   # (next carry, (L0, H0))
    '''One core of the left-orthogonalizing sweep of :py:func:`tt_left_orthogonalize`. Closure-free
    scan body -- ``docs/contributor/scan_body_principles.md``.'''
    G1 = x[0]
    L0, H1, _ = linalg.left_svd_pair(H0, G1)
    return H1, (L0, H0)


def _tt_left_step_pad_safe(
        H0: NDArray,                # carry: running right factor, stack_shape+(rL,n,rR)
        x:  typ.Tuple[NDArray, NDArray, NDArray, NDArray],   # (G1, row_mask, col_mask, out_mask)
) -> typ.Tuple[NDArray, typ.Tuple[NDArray, NDArray]]:        # (next carry, (L0, H0))
    """Pad-safe twin of :py:func:`_tt_left_step` for the uniform representation (review S1b): the
    left unfolding's SVD goes through :py:func:`~t3toolbox.backend.linalg.pad_safe_svd`, so at a
    numerically rank-deficient point the sigma~0 completion columns stay OFF the padded slots (a
    black-box SVD may place them there; the masks then erase them -- a lost tangent direction).
    ``out_mask`` zeroes the don't-care completion columns beyond the recurrence rank, so the pushed
    chain stays bitwise-clean for the next step's pad-safe SVD. Closure-free scan body; the masks
    ride the scan ``xs`` as HOST arrays (constants under jit)."""
    G1, row_mask, col_mask, out_mask = x
    use_jax = tree_contains_jax((H0, G1))
    xnp, _, _ = get_backend(True, use_jax)
    lead = H0.shape[:-3]
    rL, n, rR = H0.shape[-3:]
    U, ss, Vt = linalg.pad_safe_svd(H0.reshape(lead + (rL * n, rR)), row_mask, col_mask)
    L0 = (U * out_mask[..., None, :]).reshape(lead + (rL, n, rR))    # K = rR statically (rL*n >= rR)
    H1 = xnp.einsum('...x,...xj,...jbk->...xbk', ss, Vt, G1)
    return H1, (L0, H0)


def tt_left_orthogonalize(
        tt_cores: typ.Union[
            typ.Sequence[NDArray], # ragged. len=d, elm_shape=stack_shape+(ri,ni,r(i+1))
            NDArray, # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
        return_variation_cores: bool = False,

        pad_masks: typ.Optional[typ.Tuple[NDArray, NDArray, NDArray]] = None,
                   # uniform only. Per-step HOST masks (rows, cols, outs), each stacked (d-1,)+stack+(...),
                   # from ut3_orthogonalization._tt_left_sweep_pad_masks -> the sweep SVDs are pad-safe.
) -> typ.Union[
    typ.Tuple[NDArray,...], # left_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # left_tt_cores, var_tt_cores
]:
    """Left-orthogonalize a Tensor train (no Tucker).

    ``pad_masks`` (uniform only): run each step's SVD through
    :py:func:`~t3toolbox.backend.linalg.pad_safe_svd` with the given per-step masks, so sigma~0
    completion columns stay off the padded slots (review S1b) and the output chain is bitwise-clean.
    """
    is_uniform = is_ndarray(tt_cores)
    use_jax = tree_contains_jax(tt_cores)
    xnp, xmap, xscan = get_backend(is_uniform, use_jax)

    if pad_masks is not None and not is_uniform:
        raise ValueError('pad_masks is only meaningful for the uniform (supercore) representation')

    init = tt_cores[0]
    if pad_masks is None:
        step, xs = _tt_left_step, (tt_cores[1:],)
    else:
        step, xs = _tt_left_step_pad_safe, (tt_cores[1:],) + tuple(pad_masks)
    if len(xs[0]) > 0:  # >1 core to sweep; len() works for a ragged tuple and a uniform supercore alike
        Hf, (LL0, HH0) = xscan(step, init, xs)
    else:
        Hf = init
        LL0 = ()
        HH0 = ()

    if is_uniform:
        Hf1 = Hf.reshape((1,) + Hf.shape)
        if len(xs[0]) > 0:
            left_tt_cores = xnp.concatenate([LL0, Hf1])
            var_tt_cores  = xnp.concatenate([HH0, Hf1])
        else:
            left_tt_cores = Hf1   # d = 1: nothing was swept; the single core is the remainder (no concat of ())
            var_tt_cores  = Hf1
    else:
        left_tt_cores = tuple(LL0) + (Hf,)
        var_tt_cores  = tuple(HH0) + (Hf,)

    if return_variation_cores:
        return left_tt_cores, var_tt_cores
    else:
        return left_tt_cores


def tt_right_orthogonalize(
        tt_cores: typ.Union[
            typ.Sequence[NDArray],  # ragged. len=d, elm_shape=stack_shape+(ri,ni,r(i+1))
            NDArray,  # uniform. shape=(d,)+stack_shape+(r,n,r)
        ],
        return_variation_cores: bool = False,

        pad_masks: typ.Optional[typ.Tuple[NDArray, NDArray, NDArray]] = None,
                   # uniform only; per-step masks IN THE EXECUTED (reversed) ORIENTATION -- build with
                   # ut3_orthogonalization._tt_left_sweep_pad_masks on [::-1]-reversed inputs.
) -> typ.Union[
    typ.Tuple[NDArray,...], # right_tt_cores
    typ.Tuple[typ.Tuple[NDArray,...], typ.Tuple[NDArray,...]], # right_tt_cores, var_tt_cores
]:
    reverse = tt_operations.tt_reverse   # polymorphic (ragged core tuple or uniform supercore)

    result = tt_left_orthogonalize(
        reverse(tt_cores), return_variation_cores=return_variation_cores, pad_masks=pad_masks,
    )
    if return_variation_cores:
        return reverse(result[0]), reverse(result[1])
    else:
        return reverse(result)


