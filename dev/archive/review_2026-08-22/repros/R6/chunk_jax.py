"""R6: chunking under jax (eager + jit) vs dense for every engaging (n_probe, sum_over_probes) combo and
chunk sizes around W; memory_analysis of the concat vs sum paths; jax.linear_transpose of the tangent
forward vs the hand-written transposes (ragged jax arrays, x64)."""
import itertools
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import t3toolbox.backend.sampling_derivatives as pd
from common_r6 import *

fails = []
def check(tag, err, tol=1e-11):
    if not (err <= tol):
        fails.append((tag, err)); print('FAIL', tag, err)

rng = np.random.default_rng(7)
R = lambda *s: jnp.asarray(rng.standard_normal(s))

# ---------------------------------------------------------------- 1. chunked == dense, uniform+jax
d, r, nO, N, K = 3, 3, 4, 5, 2
for n_probe, Wsh in [(1, (7,)), (2, (3, 4)), (2, (2, 2))]:
    W = int(np.prod(Wsh))
    for order in (0, 1, 3):
        sig, tau, deta = R(d, order + 1, *Wsh, K, r), R(d, order + 1, *Wsh, K, r), R(d, order + 1, *Wsh, K, r)
        xi, mu, nu = R(d, 2, *Wsh, r), R(d, order + 1, *Wsh, r), R(d, order + 1, *Wsh, r)
        zt, dxt = R(d, order + 1, *Wsh, K, N), R(d, order + 1, *Wsh, K, nO)
        ww, pp, eta = R(d, *Wsh, N), R(d, *Wsh, N), R(d, order + 1, *Wsh, nO)
        trs = pd.binomial_combine_tensor(order)
        for sop in (True, False):
            refG = pd.assemble_tt_variation_jets_trs(sig, tau, deta, xi, mu, nu, trs, n_probe, sop)
            refU = pd.assemble_tucker_variation_jets_trs(zt, dxt, ww, pp, eta, n_probe, sop)
            for cs in [1, 2, 3, W - 1, W, W + 1, None]:
                fG = lambda *a, cs=cs: pd.assemble_tt_variation_jets(*a, trs, n_probe, sop, chunk_size=cs)
                fU = lambda *a, cs=cs: pd.assemble_tucker_variation_jets(*a, n_probe, sop, chunk_size=cs)
                for mode, gG, gU in [('eager', fG, fU), ('jit', jax.jit(fG), jax.jit(fU))]:
                    tag = f'{mode} n_probe={n_probe} W={Wsh} o={order} sop={sop} cs={cs}'
                    a = gG(sig, tau, deta, xi, mu, nu)
                    assert a.shape == refG.shape, (tag, a.shape, refG.shape)
                    check('G ' + tag, relerr(a, refG))
                    b = gU(zt, dxt, ww, pp, eta)
                    assert b.shape == refU.shape, (tag, b.shape, refU.shape)
                    check('U ' + tag, relerr(b, refU))
print('1 done; fails so far', len(fails))

# ---------------------------------------------------------------- 2. memory: does the concat path bound the peak?
d, r, K, W = 4, 24, 1, 256
order = 3
S = lambda *s: jax.ShapeDtypeStruct(s, jnp.float64)
args = (S(d, order + 1, W, K, r), S(d, order + 1, W, K, r), S(d, order + 1, W, K, r),
        S(d, 2, W, r), S(d, order + 1, W, r), S(d, order + 1, W, r))
trs = pd.binomial_combine_tensor(order)
def temp(sop, cs):
    f = lambda *a: pd.assemble_tt_variation_jets(*a, trs, 1, sop, chunk_size=cs)
    return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes
for sop in (True, False):
    row = {cs: temp(sop, cs) for cs in (None, 128, 32, 8)}
    print(f'memory_analysis temp bytes, sum_over_probes={sop}:', row)
print('2 done')

# ---------------------------------------------------------------- 3. linear_transpose vs hand transposes (ragged jax, asym)
rngn = np.random.default_rng(11)
for dd in (1, 2, 3, 4):
    Nn = ASYM[dd][0]
    for W, Kk, C in [((), (), ()), ((3,), (2,), (2,))]:
        frame = jax.tree.map(jnp.asarray, make_frame(dd, C, rngn))
        var = jax.tree.map(jnp.asarray, make_var(dd, Kk, C, rngn))
        ww = [R(*W, n) for n in Nn]; pp = [R(*W, n) for n in Nn]
        idx = np.stack([rngn.integers(0, n, size=W) for n in Nn], axis=0)
        for order in (0, 2, 4):
            # probe
            fwd = lambda v: pd.tv_probe_derivatives(ww, pp, v, frame, order)
            Jv = fwd(var)
            rr = [R(*np.asarray(z).shape) for z in Jv]
            gU_lt, gG_lt = jax.linear_transpose(fwd, var)(tuple(rr))[0]
            gU, gG = pd.tv_probe_derivatives_transpose(rr, ww, pp, frame, order, sum_over_probes=True, chunk_size=None)
            for i in range(dd):
                check(f'LT probe U d={dd} W={W} K={Kk} C={C} o={order} i={i}', relerr(gU[i], gU_lt[i]), 1e-10)
                check(f'LT probe G d={dd} W={W} K={Kk} C={C} o={order} i={i}', relerr(gG[i], gG_lt[i]), 1e-10)
            # apply
            fwd = lambda v: pd.tv_apply_derivatives(ww, pp, v, frame, order)
            c = R(*fwd(var).shape)
            gU_lt, gG_lt = jax.linear_transpose(fwd, var)(c)[0]
            gU, gG = pd.tv_apply_derivatives_transpose(c, ww, pp, frame, order, sum_over_probes=True)
            for i in range(dd):
                check(f'LT apply U d={dd} W={W} K={Kk} C={C} o={order} i={i}', relerr(gU[i], gU_lt[i]), 1e-10)
                check(f'LT apply G d={dd} W={W} K={Kk} C={C} o={order} i={i}', relerr(gG[i], gG_lt[i]), 1e-10)
            # entries
            fwd = lambda v: pd.tv_entries_derivatives(idx, pp, v, frame, order)
            c = R(*fwd(var).shape)
            gU_lt, gG_lt = jax.linear_transpose(fwd, var)(c)[0]
            gU, gG = pd.tv_entries_derivatives_transpose(c, idx, pp, frame, order, sum_over_probes=True)
            for i in range(dd):
                check(f'LT entries U d={dd} W={W} K={Kk} C={C} o={order} i={i}', relerr(gU[i], gU_lt[i]), 1e-10)
                check(f'LT entries G d={dd} W={W} K={Kk} C={C} o={order} i={i}', relerr(gG[i], gG_lt[i]), 1e-10)
print('3 done')

# ---------------------------------------------------------------- 4. jit the whole ragged transpose + forward (tracer safety)
dd = 3; Nn = ASYM[dd][0]
frame = jax.tree.map(jnp.asarray, make_frame(dd, (), rngn)); var = jax.tree.map(jnp.asarray, make_var(dd, (), (), rngn))
ww = [R(4, n) for n in Nn]; pp = [R(4, n) for n in Nn]
for order in (0, 3):
    Jv = jax.jit(lambda v: pd.tv_probe_derivatives(ww, pp, v, frame, order))(var)
    rr = [R(*np.asarray(z).shape) for z in Jv]
    g1 = jax.jit(lambda r: pd.tv_probe_derivatives_transpose(r, ww, pp, frame, order, sum_over_probes=True))(rr)
    g2 = pd.tv_probe_derivatives_transpose(rr, ww, pp, frame, order, sum_over_probes=True)
    for i in range(dd):
        check(f'jit probe transpose o={order} U{i}', relerr(g1[0][i], g2[0][i]))
        check(f'jit probe transpose o={order} G{i}', relerr(g1[1][i], g2[1][i]))
print('4 done')
print('TOTAL FAILS', len(fails))
for f in fails[:40]: print(f)
