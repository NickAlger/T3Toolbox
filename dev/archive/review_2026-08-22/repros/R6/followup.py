"""R6 follow-ups: (a) sum-path memory vs chunk count incl. the two-chunk case; (b) does chunking engage on
uniform+NumPy (docs/chunking.md:61 says it falls back to dense); (c) frontend smoke, ragged + uniform,
chunked vs dense vs ragged through the public methods."""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import t3toolbox.backend.sampling_derivatives as pd
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_manifold as ut3m
from common_r6 import relerr

# (a) memory vs chunk count on the summed path, d=4 r=24 W=256 order=3
d, r, K, W, order = 4, 24, 1, 256, 3
S = lambda *s: jax.ShapeDtypeStruct(s, jnp.float64)
args = (S(d, order + 1, W, K, r), S(d, order + 1, W, K, r), S(d, order + 1, W, K, r),
        S(d, 2, W, r), S(d, order + 1, W, r), S(d, order + 1, W, r))
trs = pd.binomial_combine_tensor(order)
def temp(sop, cs):
    f = lambda *a: pd.assemble_tt_variation_jets(*a, trs, 1, sop, chunk_size=cs)
    return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes
print('(a) summed path temp bytes by chunk_size (W=256):',
      {cs: temp(True, cs) for cs in (None, 255, 128, 100, 85, 64, 32)})

# (b) uniform + NumPy: does the chunk loop run?
calls = {'n': 0}
orig = pd.assemble_tt_variation_jets_trs
def counting(*a, **k):
    calls['n'] += 1
    return orig(*a, **k)
pd.assemble_tt_variation_jets_trs = counting
rng = np.random.default_rng(0)
R = lambda *s: rng.standard_normal(s)
Wn = 7
sig, tau, deta = R(3, 3, Wn, 1, 2), R(3, 3, Wn, 1, 2), R(3, 3, Wn, 1, 2)
xi, mu, nu = R(3, 2, Wn, 2), R(3, 3, Wn, 2), R(3, 3, Wn, 2)
pd.assemble_tt_variation_jets(sig, tau, deta, xi, mu, nu, pd.binomial_combine_tensor(2), 1, True, chunk_size=2)
print('(b) uniform+NumPy, W=7, chunk_size=2: dense-assembly calls =', calls['n'],
      '(docs/chunking.md:61-62 says NumPy "falls back to the dense assembly regardless of chunk_size")')
pd.assemble_tt_variation_jets_trs = orig

# (c) frontend smoke, ragged vs uniform, chunked vs dense
np.random.seed(0)
shape, tr, tt = (6, 7, 5), (3, 4, 2), (1, 2, 3, 1)
x = t3.TuckerTensorTrain.randn(shape, tr, tt)
frame, _ = bvf.t3_orthogonal_representations(x)
v = t3m.COREWISE.randn(frame)
Wf = 5
ww = [np.random.randn(Wf, n) for n in shape]; pp = [np.random.randn(Wf, n) for n in shape]
index = np.stack([np.random.randint(0, n, size=(Wf,)) for n in shape], 0)
for order in (0, 2, 3):
    zj = v.probe_derivatives(ww, pp, order); yj = v.apply_derivatives(ww, pp, order); ej = v.entries_derivatives(index, pp, order)
    r = [np.random.randn(*z.shape) for z in zj]
    JTr = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order, sum_over_probes=True, chunk_size=2)
    JTr0 = t3m.T3Tangent.probe_derivatives_transpose(r, ww, pp, frame, order, sum_over_probes=True, chunk_size=None)
    lhs = sum(float(np.sum(ri * zi)) for ri, zi in zip(r, zj))
    print(f'(c) ragged o={order}: adjoint |lhs-<JTr,v>|/|lhs| =', abs(lhs - float(JTr.corewise_inner(v))) / abs(lhs),
          ' chunk2 vs None:', float((JTr - JTr0).corewise_norm()))
    c = np.random.randn(*yj.shape)
    JTc = t3m.T3Tangent.apply_derivatives_transpose(c, ww, pp, frame, order, sum_over_probes=True)
    print(f'    apply adjoint rel:', abs(float(np.sum(c * yj)) - float(JTc.corewise_inner(v))) / abs(float(np.sum(c * yj))))
    ce = np.random.randn(*ej.shape)
    JTe = t3m.T3Tangent.entries_derivatives_transpose(ce, index, pp, frame, order, sum_over_probes=True)
    print(f'    entries adjoint rel:', abs(float(np.sum(ce * ej)) - float(JTe.corewise_inner(v))) / abs(float(np.sum(ce * ej))))
    gU, gG = x.probe_corewise_derivatives_transpose(r, ww, pp, order, sum_over_probes=True)
    gU2, gG2 = x.apply_corewise_derivatives_transpose(c, ww, pp, order, sum_over_probes=True)
    gU3, gG3 = x.entries_corewise_derivatives_transpose(ce, index, pp, order, sum_over_probes=True)
    # uniform, jax: chunked vs dense vs ragged
    xu = ut3.UniformTuckerTensorTrain.from_t3(x)
    fu = ut3m.UNIFORM_MANIFOLD.frame(jax.tree.map(jnp.asarray, xu))
    wwj = [jnp.asarray(a) for a in ww]; ppj = [jnp.asarray(a) for a in pp]; rj = [jnp.asarray(a) for a in r]
    fr_j = jax.tree.map(jnp.asarray, frame)
    JTr_u2 = ut3m.UT3Tangent.probe_derivatives_transpose(rj, wwj, ppj, fu, order, sum_over_probes=True, chunk_size=2)
    JTr_u0 = ut3m.UT3Tangent.probe_derivatives_transpose(rj, wwj, ppj, fu, order, sum_over_probes=True, chunk_size=None)
    JTr_rj = t3m.T3Tangent.probe_derivatives_transpose(rj, wwj, ppj, fr_j, order, sum_over_probes=True, chunk_size=None)
    # compare on dense tangents (gauge-free): the uniform frame is the packed ragged frame, so the dense tangents agree
    du2, du0, dr = np.asarray(JTr_u2.to_dense()), np.asarray(JTr_u0.to_dense()), np.asarray(JTr_rj.to_dense())
    print(f'    uniform jax o={order}: chunk2 vs None rel', relerr(du2, du0), ' uniform vs ragged rel', relerr(du0, dr))
print('done')
