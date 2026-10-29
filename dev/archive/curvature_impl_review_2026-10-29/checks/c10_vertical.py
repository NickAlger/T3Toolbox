"""Read-only plan check: what 'vertical null space' means once Pi is the ORTHOGONAL gauge projection.
(a) Pi(frame-coordinate vertical) != 0 in general  => hvp(vertical) = H(Pi vertical) != 0 (DP 4.6 wording is off);
(b) L(frame vertical) is a raw-core vertical: DPhi[L v] = 0, and K_N L v = 0 (Lagrangian null space), so the
    correct test is on the un-projected curvature block L^T K_N L."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv
import t3toolbox.backend.apply as ap
import t3toolbox.backend.t3_conversions as t3c
from t3toolbox.backend.common import get_backend
import importlib.util
spec = importlib.util.spec_from_file_location('c4', __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'c4_hessian_oracle.py'))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    c4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c4)
rng = np.random.default_rng(21)
frame, d, tr, ttr, ww, r, Zg = c4.frame, c4.d, c4.tr, c4.ttr, c4.ww, c4.r, c4.Zg
U, O, L, R = frame
cf = fvc.t3_corewise_frame((U, L))
VV = [np.zeros(o.shape[-2:-1] + u.shape[-1:]) for o, u in zip(O, U)]; HH = [np.zeros(l.shape) for l in L]
for i in range(d):
    X = rng.standard_normal((O[i].shape[-2], U[i].shape[-2]))
    VV[i] += X @ U[i]; HH[i] -= np.einsum('aib,ij->ajb', O[i], X)
for i in range(d - 1):
    Y = rng.standard_normal((L[i].shape[-1], R[i + 1].shape[-3]))
    HH[i] += np.einsum('iaj,jk->iak', L[i], Y); HH[i + 1] -= np.einsum('jk,kbl->jbl', Y, R[i + 1])
pv = (tuple(VV), tuple(HH))
nrm = lambda a: np.sqrt(sum(np.sum(x * x) for x in a[0] + a[1]))
print('(a) |E p_vert| = %.1e ; |Pi p_vert| / |p_vert| = %.2f ; |E Pi p_vert| = %.2e' % (
    np.linalg.norm(tv.tv_to_dense(frame, pv)), nrm(tv.tv_orthogonal_gauge_projection(frame, pv)) / nrm(pv),
    np.linalg.norm(tv.tv_to_dense(frame, tv.tv_orthogonal_gauge_projection(frame, pv)))))
th = c4.lift(frame, pv)
Y = tv.tv_to_t3(cf, th)
Kr = c4.blocksum(ap.t3_apply_corewise_transpose(r, ww, Y, sum_over_probes=True), d, tr, ttr)
Kg = c4.blocksum(c4.ungauged_inner_grad(Y, Zg), d, tr, ttr)
KN = np.concatenate([(a - b).ravel() for a, b in zip(Kr[0] + Kr[1], Kg[0] + Kg[1])])
print('(b) |DPhi[L p_vert]| = %.1e ; |K_N L p_vert| = %.1e (vs |K_r L p_vert| = %.1e)' % (
    np.linalg.norm(tv.tv_to_dense(cf, th)), np.linalg.norm(KN), np.linalg.norm(c4.flat(Kr))))
