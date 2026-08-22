"""R7: (a) _cg_core vs a dense solve of the GN system; (b) compile counts ('compiles once'); (c) display table math;
(d) gradient_descent use_jit; (e) stacked rejections; (f) forcing-term semantics."""
import numpy as np, itertools
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.optimizer_display as bdisp
import t3toolbox.corewise as cw
import t3toolbox.shared_geometry as sg

np.random.seed(0)
shape, tr, rr = (3, 4, 5), (2, 2, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(shape, tr, rr)
ww = [np.random.randn(60, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
x = t3.TuckerTensorTrain.randn(shape, tr, rr)

# (a) CG vs dense: H on the manifold tangent coordinates; g in range(H); CG from 0 gives the min-norm solution
lam = 0.05
P = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b, regularizer=bopt.IdentityRegularizer(lam))
lm = P.local_model(x.data)
g = lm.gradient
leaves = list(g[0]) + list(g[1])
sizes = [l.size for l in leaves]; ntot = sum(sizes)
def to_vec(t): return np.concatenate([np.ravel(l) for l in list(t[0]) + list(t[1])])
def from_vec(v):
    out, k = [], 0
    for l in leaves:
        out.append(v[k:k + l.size].reshape(l.shape)); k += l.size
    return (tuple(out[:len(g[0])]), tuple(out[len(g[0]):]))
H = np.zeros((ntot, ntot))
for j in range(ntot):
    e = np.zeros(ntot); e[j] = 1.0
    H[:, j] = to_vec(lm.hvp(from_vec(e)))
print("dense H: n=%d, symmetric to %.1e, rank=%d" % (ntot, np.max(np.abs(H - H.T)), np.linalg.matrix_rank(H)))
gv = to_vec(g)
p_dense = np.linalg.pinv(H, rcond=1e-10) @ (-gv)
p_cg, i, rs, ok = bopt._cg_core(lm, cw.corewise_scale(g, -1.0), 1e-12, 500, False)
print("CG: %d iters, resid %.1e, ok=%s ; ||p_cg - p_dense||/||p_dense|| = %.2e ; ||H p_cg + g|| = %.2e"
      % (int(i), float(rs) ** 0.5, bool(ok), np.linalg.norm(to_vec(p_cg) - p_dense) / np.linalg.norm(p_dense),
         np.linalg.norm(H @ to_vec(p_cg) + gv)))

# (f) forcing term semantics: eta = min(0.5, (gnorm/ref)**power), cg_tol = eta*gnorm; refs chained
x0 = t3.TuckerTensorTrain.zeros(shape, tr, rr)
_, st = bopt.newton_cg(bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b), x0.data,
                       max_newton=4, g0norm_newton=7.0, cg_forcing_power=0.75)
ok_all = True
for h in st['history']:
    if h['forcing_eta'] is None: continue
    eta = min(0.5, (h['gnorm'] / 7.0) ** 0.75)
    ok_all &= np.isclose(eta, h['forcing_eta']) and np.isclose(h['cg_tol'], eta * h['gnorm']) and h['g0norm'] == 7.0
print("forcing term: eta/cg_tol/g0norm match the docs (g0norm_newton feeds CG): %s" % ok_all)
_, st2 = bopt.newton_cg(bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, ww, b), x0.data,
                        max_newton=2, g0norm_cg=3.0)
h = st2['history'][0]
print("g0norm_cg alone: eta uses 3.0 -> %s ; NewtonInfo.g0norm is the initial gnorm -> %s"
      % (np.isclose(h['forcing_eta'], min(0.5, (h['gnorm'] / 3.0) ** 0.5)), np.isclose(h['g0norm'], h['gnorm'])))

# (c) display table math: probe_derivatives, per-(mode, order) ||r_ij||/||y_ij|| by hand
pp = [np.random.randn(60, N) for N in shape]
K = 2
jets = A.probe_derivatives(ww, pp, K)
kind = bfit.probe_derivatives_kind(K, np.array([1.0, 0.5, 0.2]))
Pd = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), kind, (ww, pp), jets)
cb, recs = bdisp.make_newton_display(Pd, print_fn=None)
bopt.newton_cg(Pd, x.data, max_newton=1, callback=cb)
r = [zi - yi for zi, yi in zip(x.probe_derivatives(ww, pp, K), jets)]
hand = np.array([[np.linalg.norm(r[i][t]) / np.linalg.norm(jets[i][t]) for t in range(K + 1)] for i in range(len(shape))])
print("display train_err vs hand (d x order+1): max abs diff %.1e ; unweighted objective in record? obj=%.4f misfit(weighted)=%.4f"
      % (np.max(np.abs(recs[0]['train_err'] - hand)), recs[0]['objective'], recs[0]['misfit']))
# val column math
vww = [w[:10] for w in ww]; vpp = [p[:10] for p in pp]; vj = [j[:, :10] for j in jets]
cb, recs = bdisp.make_newton_display(Pd, val_sample=(vww, vpp), val_data=vj, print_fn=None)
bopt.newton_cg(Pd, x.data, max_newton=1, callback=cb)
rv = [zi - yi for zi, yi in zip(x.probe_derivatives(vww, vpp, K), vj)]
handv = np.array([[np.linalg.norm(rv[i][t]) / np.linalg.norm(vj[i][t]) for t in range(K + 1)] for i in range(len(shape))])
print("display val_err vs hand: max abs diff %.1e" % np.max(np.abs(recs[0]['val_err'] - handv)))

# (d) gradient_descent has no use_jit
try:
    topt.gradient_descent(t3m.MANIFOLD, 'apply', ww, b, x0, use_jit=True, n_iter=1)
except TypeError as e:
    print("gradient_descent(use_jit=True) -> TypeError:", str(e)[:80])

# (e) stacked rejections
xs = t3.TuckerTensorTrain.randn(shape, tr, rr, stack_shape=(2,))
bws = [np.random.randn(20, N) for N in shape]; bb = A.apply(bws)
for name, fn in (("gradient_descent", lambda: topt.gradient_descent(t3m.MANIFOLD, 'apply', bws, bb, xs)),
                 ("newton_cg", lambda: topt.newton_cg(t3m.MANIFOLD, 'apply', bws, bb, xs)),
                 ("mc_sgd", lambda: topt.mc_sgd(t3m.MANIFOLD, 'apply', bws, bb, xs, np.random.default_rng(0), 5)),
                 ("adam", lambda: topt.adam(t3m.COREWISE, 'apply', bws, bb, xs, np.random.default_rng(0), 5))):
    try:
        fn(); print(name, "stacked x0: no error")
    except Exception as e:
        print("%s stacked x0 -> %s (%s...)" % (name, type(e).__name__, str(e)[:50]))

# (b) compile counts: _cg_core compiled once across a regularized + shared newton run under jit
import jax
jax.config.update("jax_enable_x64", True)
bopt._jitted.cache_clear()
sharing = (0, 0, 1)
As = t3.TuckerTensorTrain.randn((5, 5, 4), (2, 2, 2), (1, 2, 2, 1)).share(sharing)
wws = [np.random.randn(60, N) for N in As.shape]; bs = As.apply(wws)
xs0 = t3.TuckerTensorTrain.zeros(As.shape, (2, 2, 2), (1, 2, 2, 1))
_, st = topt.newton_cg(sg.shared_manifold(sharing), 'apply', wws, bs, xs0, regularizer=topt.IdentityRegularizer(1e-3),
                       use_jit=True, max_newton=6)
jf = bopt._jitted(bopt._cg_core, (4,))
print("jit(_cg_core) cache size after a %d-step shared+regularized newton run: %d" % (st['newton'], jf._cache_size()))
# a second run at the same shapes: still 1
_, st = topt.newton_cg(sg.shared_manifold(sharing), 'apply', wws, bs, xs0, regularizer=topt.IdentityRegularizer(1e-3),
                       use_jit=True, max_newton=3)
print("after a second run (rebuilt geometry/kind/regularizer objects): %d" % jf._cache_size())
# mc_sgd step kernel
_, st = topt.mc_sgd(t3m.MANIFOLD, 'apply', wws, bs, xs0, np.random.default_rng(0), 10, use_jit=True, max_iter=30, check_every=10,
                    regularizer=topt.IdentityRegularizer(1e-3))
jm = bopt._jitted(bopt._mc_sgd_step)
print("jit(_mc_sgd_step) cache size after 30 regularized steps: %d" % jm._cache_size())
jax.config.update("jax_enable_x64", False)
