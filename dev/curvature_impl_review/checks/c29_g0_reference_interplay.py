"""Read-only plan check (v5): the g0_reference='zero' interplay on F19's problem (E2 on probe_derivatives, order 2,
omega weight, zero-padded restart rank 2 -> 3, target 4, 5% noise) and the I-7 change (cg_forcing_power 0.5 for
exact mode too, was 1.0 in v4's F19 numbers).

Measures:
  (1) the zero-tensor reference ||g(0)|| at the level's ranks vs the warm-start ||g(x2)|| (manifold, ragged);
  (2) GN + Armijo (the library's backend newton_cg) with g0norm_newton = warm (default) / zero, gtol_rel 1e-8 / 1e-3;
  (3) exact + Steihaug/TR (c24's loop) with the reference warm / zero and forcing power 0.5 / 1.0;
  (4) the corewise zero reference (0 exactly) and what today's newton_cg does with g0norm_newton=0.0.
"""
import sys, os, math, io, contextlib
sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(SCR, 'check_restart_tr2.py')).read()
ns = {}
exec(compile(src[:src.index('def scenario(')], 'check_restart_tr2_defs', 'exec'), ns)
np, t3, t3m, topt, bopt, bgeo, bfit, cw, tv, fvc = (ns[k] for k in ('np', 't3', 't3m', 'topt', 'bopt', 'bgeo', 'bfit', 'cw', 'tv', 'fvc'))
Lift, steihaug, blocksum, ungauged = ns['Lift'], ns['steihaug'], ns['blocksum'], ns['ungauged_inner_grad']
from c22_probe_derivs_v3_recipe import two_hole_v3

ORDER = 2


def curvature(lm, g_mis, lift_obj, kind):
    frame = lm.frame; U, O, L, R = frame; d = len(U)
    tr = [u.shape[0] for u in U]; ttr = [L[0].shape[0]] + [l.shape[2] for l in L]
    cframe = fvc.t3_corewise_frame((tuple(U), tuple(L))); Zg = tv.tv_to_t3(frame, g_mis)
    ww, pp = lm.sample
    zt = kind._apply_weight(lm.residual, 2)
    def curv(p):
        Pp = tv.tv_orthogonal_gauge_projection(frame, p)
        th = lift_obj.lift(Pp)
        Kr = two_hole_v3(frame, lm.sweep, tuple(a[None] for a in th[0]), tuple(a[None] for a in th[1]),
                         ww, pp, zt, ORDER, (1,), ())
        Kr = (tuple(a[0] for a in Kr[0]), tuple(a[0] for a in Kr[1]))
        Kg = blocksum(ungauged(tv.tv_to_t3(cframe, th), Zg), d, tr, ttr)
        return tv.tv_orthogonal_gauge_projection(frame, lift_obj.lift_T(cw.corewise_sub(Kr, Kg)))
    return curv


def run(problem, x, g0_ref, power, gtol, eps_rel=1e-6, max_outer=80):
    """c24's exact Steihaug/TR loop; g0_ref None -> the warm-start ||g||, else the given reference."""
    radius = None; n_rej = n_hvp = it = 0; acc_last = True; g0 = g0_ref
    while it < max_outer:
        if acc_last:
            lm = problem.local_model(x); g = lm.gradient; gnorm = math.sqrt(cw.corewise_dot(g, g))
            g0 = gnorm if g0 is None else g0; f = float(lm.objective)
            Xn = math.sqrt(float(cw.corewise_dot(lm.frame[2][-1], lm.frame[2][-1])))
            curv = curvature(lm, g, Lift(lm.frame, eps_rel * Xn), problem.kind)
            hvp = lambda p: cw.corewise_add(lm.hvp(p), curv(p))
        if gnorm <= gtol * g0:
            break
        if radius is None:
            radius = gnorm ** 3 / max(float(lm.gn_quadratic(g)), 1e-300); r0 = radius
        cnt = [0]
        def hvp_c(p):
            cnt[0] += 1; return hvp(p)
        eta = min(0.5, (gnorm / g0) ** power)
        p, cgi, hit, negc = steihaug(hvp_c, cw.corewise_scale(g, -1.0), eta * gnorm, 80, radius)
        n_hvp += cnt[0]
        pred = cw.corewise_dot(g, p) + 0.5 * cw.corewise_dot(p, hvp(p))
        xt = lm.retract(p); fn = float(problem.objective(xt))
        rho = (fn - f) / pred if pred < 0 else float('nan'); pn = math.sqrt(cw.corewise_dot(p, p))
        accept = math.isfinite(rho) and rho > 0.1
        if (not math.isfinite(rho)) or rho < 0.25:
            radius = 0.25 * pn
        elif rho > 0.75 and hit:
            radius = min(2 * radius, 1e4 * r0)
        it += 1
        if accept:
            x = xt; acc_last = True
        else:
            n_rej += 1; acc_last = False
    return dict(it=it, rej=n_rej, hvps=n_hvp, conv=gnorm <= gtol * g0, rel=gnorm / g0, f=f)


rng = np.random.default_rng(7); np.random.seed(7)
shape, W, noise = (6, 6, 6), 120, 0.05
target = t3.TuckerTensorTrain.randn(shape, (4, 4, 4), (1, 4, 4, 1))
ww = [rng.standard_normal((W, N)) for N in shape]; pp = [rng.standard_normal((W, N)) for N in shape]
omega = np.array([1.0, 0.5, 0.25])
kind = bfit.probe_derivatives_kind(ORDER, omega)
clean = kind.point_forward(target.data, (ww, pp))
data = [z + noise * np.linalg.norm(z) / math.sqrt(z.size) * rng.standard_normal(z.shape) for z in clean]
x1 = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
with contextlib.redirect_stdout(io.StringIO()):
    x1, _ = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), data, x1, order=ORDER,
                           weight=omega, max_newton=80, gtol_rel=1e-10)
x2 = x1.resize(shape, (3, 3, 3), (1, 3, 3, 1))
geom = bgeo.ManifoldGeometryOps()
problem = bopt.least_squares_problem(geom, kind, (ww, pp), data)

# (1) the two references, computed exactly as v5's resolver will: on the level's own Problem, in its geometry
gz = problem.local_model(cw.corewise_zeros_like(x2.data)).gradient
zref = float(geom.inner(gz, gz)) ** 0.5
gw = problem.local_model(x2.data).gradient
wref = float(geom.inner(gw, gw)) ** 0.5
print(f"F19 problem, zero-padded restart rank 2 -> 3: ||g(zero tensor)|| = {zref:.4e}, warm-start ||g0|| = {wref:.4e}, "
      f"warm/zero = {wref / zref:.3e}")

# (2) GN + Armijo, library backend newton_cg
for label, ref in (('warm (today)', None), ('zero', zref)):
    for gtol in (1e-8, 1e-3):
        _, st = bopt.newton_cg(problem, x2.data, max_newton=80, gtol_rel=gtol, g0norm_newton=ref)
        h = st['history']
        cg = sum(r['cg_iters'] or 0 for r in h)
        print(f"  GN+Armijo  ref={label:12s} gtol_rel={gtol:g}: newton {st['newton']:3d}, cg iters {cg:4d}, "
              f"final ||g||/ref {h[-1]['gnorm'] / h[-1]['g0norm']:.1e}, converged {h[-1]['converged']}, f {h[-1]['objective']:.6e}")

# (3) exact + TR, forcing power 0.5 (v5, I-7) vs 1.0 (v4's F19 numbers), reference warm vs zero
for power in (1.0, 0.5):
    for label, ref in (('warm', None), ('zero', zref)):
        for gtol in (1e-8, 1e-3):
            r = run(problem, x2.data, ref, power, gtol)
            print(f"  exact+TR   power={power:g} ref={label:5s} gtol_rel={gtol:g}: outer {r['it']:3d} (rejected {r['rej']:2d}), "
                  f"hvps {r['hvps']:4d}, converged {r['conv']}, final ||g||/ref {r['rel']:.1e}, f {r['f']:.6e}")

# (4) COREWISE: the zero tensor is a critical point of every corewise objective (Phi multilinear in the cores)
cgeom = bgeo.CorewiseGeometryOps()
cprob = bopt.least_squares_problem(cgeom, kind, (ww, pp), data)
gcz = cprob.local_model(cw.corewise_zeros_like(x2.data)).gradient
print(f"COREWISE: ||g(zero tensor)|| = {float(cgeom.inner(gcz, gcz)) ** 0.5:.3e}")
try:
    bopt.newton_cg(cprob, x2.data, max_newton=2, g0norm_newton=0.0)
    print("  today's newton_cg(g0norm_newton=0.0): ran")
except ZeroDivisionError as e:
    print(f"  today's newton_cg(g0norm_newton=0.0): ZeroDivisionError ({e}) -- the forcing term divides by the reference")
