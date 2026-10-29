"""Read-only plan check (v2): the SHARED exact Hessian (T3Polynomial fits with sharing).
(a) the tied lift Udot_g = M_g^+ [V_i1; V_i2] (one array per group) satisfies DPhi[L v] = E v for a TIED gauged v;
(b) exact quadratic form GN + <th,K_r th> - <th,K_gsh th> (th replicated over the group, grad_sh = the tied
    gradient) vs FD of f along the SHARED retraction (whose second order is unproven -- the FD ladder tells);
(c) the probe_derivatives kind (T3Polynomial's kind) through the block-sum oracle, so (b) covers it."""
import sys; sys.path.insert(0, __import__("os").getcwd())
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.fv_conversions as fvc
import t3toolbox.backend.tt_operations as tto
import t3toolbox.backend.tv_operations as tv; import t3toolbox.backend.t3_linalg as t3_linalg
import t3toolbox.backend.sampling_derivatives as sd
import t3toolbox.backend.t3_conversions as t3c
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.sharing as sh
from t3toolbox.backend.common import get_backend
np.random.seed(5); rng = np.random.default_rng(5)

def blocksum(gY, d, tr, ttr):
    gYU, gYG = gY
    gU = tuple(gYU[i][..., :tr[i], :] for i in range(d)); gG = []
    for i in range(d):
        r, n, rn = ttr[i], tr[i], ttr[i + 1]; g = gYG[i]
        if i == 0:   gG.append(g[..., :, :n, rn:] + g[..., :, n:, :rn])
        elif i == d - 1: gG.append(g[..., :r, :n, :] + g[..., r:, n:, :])
        else:        gG.append(g[..., :r, :n, :rn] + g[..., r:, :n, rn:] + g[..., r:, n:, :rn])
    return gU, tuple(gG)
flat = lambda a: np.concatenate([x.ravel() for x in a[0] + a[1]])
def ungauged_inner_grad(th, z):
    up, down, left, right = fvc.t3_corewise_frame(th)
    ztk, ztt = z; ztt = tto.tt_squash_tails(ztt)
    _, xmap, _ = get_backend(False, False)
    (o2,) = xmap(t3_linalg._inner_product_rebasis_step, (ztt, ztk, up))
    zl = tto.tt_zipper_left_to_right(o2[:-1], left[:-1]); zr = tto.tt_zipper_right_to_left(o2[1:], right[1:])
    dG, dB = xmap(t3_linalg._inner_product_variations_step, (zl, zr, ztt, ztk, down, up))
    return (tuple(dB), tuple(dG))

shape, sharing = (5, 5, 6), (0, 0, 1)
x0 = t3.TuckerTensorTrain.randn(shape, (3, 3, 3), (1, 3, 3, 1))
x = x0.share(sharing, max_tucker_ranks=(2, 2, 2), max_tt_ranks=(1, 2, 2, 1))
x = x.rank_adjustment_sweep('right_to_left', sharing=sharing)
print('tied:', bool(np.all(x.has_shared_tucker_factors(sharing))), 'ranks', x.tucker_ranks, x.tt_ranks)
geom = bgeo.ManifoldGeometryOps().with_sharing(sharing, shape)
frame = geom.frame(x.data)
aux = geom.precompute(frame)
U, O, L, R = frame
d = 3; tr = tuple(u.shape[-2] for u in U); ttr = (1,) + tuple(l.shape[-1] for l in L)
cf = fvc.t3_corewise_frame((U, L))
order, Wn = 2, 60
ww = [rng.standard_normal((Wn, N)) for N in shape]; pp = [rng.standard_normal((Wn, N)) for N in shape]
kind = bfit.probe_derivatives_kind(order, None)
data = [z + 0.3 * rng.standard_normal(z.shape) for z in kind.point_forward(x.data, (ww, pp))]
r = [a - b for a, b in zip(kind.point_forward(x.data, (ww, pp)), data)]
sweep = kind.precompute(frame, (ww, pp))
g = geom.project(frame, kind.transpose(r, (ww, pp), frame, sweep), aux)          # tied misfit gradient
Zg = tv.tv_to_t3(frame, g)                                                       # untied embedding: same ambient tensor
vs = ([o.shape[-2:-1] + u.shape[-1:] for o, u in zip(O, U)], [l.shape for l in L])
v = geom.project(frame, (tuple(rng.standard_normal(s) for s in vs[0]), tuple(rng.standard_normal(s) for s in vs[1])), aux)
xi = tv.tv_to_dense(frame, v); nv = np.linalg.norm(xi)
v = (tuple(a / nv for a in v[0]), tuple(a / nv for a in v[1])); xi = xi / nv
# tied lift: group modes take Udot_g = M_g^+ [V_i...] (the companion's clipped solve); singletons per-mode S^+; TT by T^+
Z = tto.tt_zipper_right_to_left(L, R)
udots = sh.fv_tied_ambient_directions(v, aux)
Ud = []
for i in range(d):
    grp = [gi for gi, gr in enumerate(sh.nontrivial_groups(aux.groups)) if i in gr]
    if grp:
        Ud.append(udots[grp[0]])
    else:
        ST = np.einsum('axb,aub->xu', O[i], np.einsum('aib,bc->aic', L[i], Z[i + 1]))
        Ud.append(np.linalg.pinv(ST) @ v[0][i])
Gd = [np.einsum('aib,bc->aic', v[1][i], np.linalg.pinv(Z[i + 1])) if i < d - 1 else v[1][i] for i in range(d)]
th = (tuple(Ud), tuple(Gd))
print('(a) tied lift identity |DPhi[Lv]-Ev|/|Ev| = %.1e ; replicated group factor: %s' % (
    np.linalg.norm(tv.tv_to_dense(cf, th) - xi), np.allclose(Ud[0], Ud[1])))
Y = tv.tv_to_t3(cf, th)
Kr = blocksum(sd.t3_probe_corewise_derivatives_transpose(r, ww, pp, Y, order, sum_over_probes=True, chunk_size=None), d, tr, ttr)
Kg = blocksum(ungauged_inner_grad(Y, Zg), d, tr, ttr)
Jv = kind.forward(v, (ww, pp), frame, sweep)
gn = float(kind.sumsq(Jv, 1))
exact = gn + flat(th) @ flat(Kr) - flat(th) @ flat(Kg)
print('(b) GN %.8f   exact (shared) %.8f' % (gn, exact))
f = lambda xc: 0.5 * float(kind.sumsq([a - b for a, b in zip(kind.point_forward(xc, (ww, pp)), data)], 1))
f0 = f(x.data)
for hh in (2e-2, 1e-2, 5e-3, 2.5e-3):
    fp = f(geom.retract(frame, (tuple(hh * a for a in v[0]), tuple(hh * a for a in v[1])), aux))
    fm = f(geom.retract(frame, (tuple(-hh * a for a in v[0]), tuple(-hh * a for a in v[1])), aux))
    print('    FD along SHARED retraction h=%.1e: %.8f' % (hh, (fp - 2 * f0 + fm) / hh ** 2))
