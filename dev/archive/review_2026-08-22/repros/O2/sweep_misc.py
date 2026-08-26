"""O2 part 3: jit == eager (float64), chunk_size variants, stacked x0 rejection, x0 variants
(orthogonalized / zero / non-minimal), verbose display smoke, d=1 ragged, Problem.objective at raw x0."""
import sys, time, traceback
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, '.')
import oracle as O
import sweep_models as SM
import sweep_optimizers as SO
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.shared_geometry as sg
import t3toolbox.optimizers as topt
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.geometry as bgeo
from t3toolbox.backend.regularization import IdentityRegularizer

out = []
def log(*a):
    s = ' '.join(str(v) for v in a); print(s); out.append(s)

# ------------------------------------------------------------------ jit == eager
log('== jit == eager (x64) ==')
for rep in ('ragged', 'uniform'):
    for geom_name, shared in [('manifold', False), ('corewise', False), ('manifold', True)]:
        for kind, order, wspec in [('apply', None, 'none'), ('probe', None, 'mode'), ('probe_derivatives', 2, 'matrix'), ('entries_derivatives', 1, 'order')]:
            for regspec in (False, True):
                x, sharing, sample, weight, y = SO.setup(3, shared, kind, order, wspec, 5)
                geom = SO.geometry_for(rep, geom_name, shared, sharing)
                x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
                reg = IdentityRegularizer(0.3) if regspec else None
                kw = dict(order=order, weight=weight, regularizer=reg)
                try:
                    xe, se = topt.newton_cg(geom, kind, sample, y, x0, max_newton=3, **kw)
                    xj, sj = topt.newton_cg(geom, kind, sample, y, x0, max_newton=3, use_jit=True, **kw)
                    e_nt = max(SM.rel(se['losses'], sj['losses']), SM.rel(xe.to_dense(), np.asarray(xj.to_dense())))
                    dt = str(np.asarray(xj.data[0][0] if rep == 'ragged' else xj.data[0]).dtype)
                    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
                    xe2, se2 = topt.mc_sgd(geom, kind, sample, y, x0, rng1, 6, max_iter=4, check_every=2, **kw)
                    xj2, sj2 = topt.mc_sgd(geom, kind, sample, y, x0, rng2, 6, max_iter=4, check_every=2, use_jit=True, **kw)
                    e_mc = max(SM.rel(se2['losses'], sj2['losses']), SM.rel(xe2.to_dense(), np.asarray(xj2.to_dense())))
                    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(1)
                    xe3, se3 = topt.adam(geom, kind, sample, y, x0, rng1, 6, max_iter=50, **kw)
                    xj3, sj3 = topt.adam(geom, kind, sample, y, x0, rng2, 6, max_iter=50, use_jit=True, **kw)
                    e_ad = max(SM.rel(se3['losses'], sj3['losses']), SM.rel(xe3.to_dense(), np.asarray(xj3.to_dense())))
                    flag = 'OK' if max(e_nt, e_mc, e_ad) < 1e-9 else 'MISMATCH'
                    log(f'{flag} {rep:7s} {geom_name:8s} shared={shared!s:5s} {kind:19s} w={wspec:6s} reg={regspec!s:5s} '
                        f'newton={e_nt:.1e} mc_sgd={e_mc:.1e} adam={e_ad:.1e} jit_dtype={dt}')
                except Exception as e:
                    log(f'EXC {rep} {geom_name} shared={shared} {kind} w={wspec} reg={regspec}: {type(e).__name__}: {str(e)[:200]}')

# ------------------------------------------------------------------ chunk_size
log('== chunk_size (probe_derivatives, order 2, W=12) ==')
for rep in ('ragged', 'uniform'):
    for use_jit in (False, True):
        x, sharing, sample, weight, y = SO.setup(3, False, 'probe_derivatives', 2, 'matrix', 5)
        geom = SO.geometry_for(rep, 'manifold', False, None)
        x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
        results = {}
        for cs in (None, 'auto', 5, 1, 100):
            try:
                xr, st = topt.newton_cg(geom, 'probe_derivatives', sample, y, x0, order=2, weight=weight, max_newton=3,
                                        chunk_size=cs, use_jit=use_jit)
                results[cs] = (np.asarray(xr.to_dense()), list(st['losses']))
            except Exception as e:
                log(f'EXC {rep} jit={use_jit} chunk={cs}: {type(e).__name__}: {str(e)[:200]}')
        ref = results.get(None)
        for cs, (xd, ls) in results.items():
            if ref is not None:
                e = max(SM.rel(xd, ref[0]), SM.rel(ls, ref[1]))
                log(f'{"OK" if e < 1e-9 else "MISMATCH"} {rep:7s} jit={use_jit!s:5s} chunk={cs!s:5s} vs None: {e:.1e}')

# ------------------------------------------------------------------ stacked x0 -> NotImplementedError at entry
log('== stacked x0 ==')
x, sharing, sample, weight, y = SO.setup(3, False, 'apply', None, 'none', 5)
xs = t3.TuckerTensorTrain.stack([x, x])
bad_sample = [w[:5] for w in sample]            # wrong W: any real work on it would raise a shape error
for name, fn, extra in [('gradient_descent', topt.gradient_descent, ()), ('newton_cg', topt.newton_cg, ()),
                        ('mc_sgd', topt.mc_sgd, (np.random.default_rng(0), 6)), ('adam', topt.adam, (np.random.default_rng(0), 6))]:
    for rep in ('ragged', 'uniform'):
        x0s = xs if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(xs)
        try:
            fn(t3m.MANIFOLD if rep == 'ragged' else ut3m.UNIFORM_MANIFOLD, 'apply', bad_sample, y, x0s, *extra)
            log(f'NO-RAISE {name} {rep}')
        except NotImplementedError as e:
            log(f'OK {name:16s} {rep:7s} NotImplementedError at entry (before touching the mismatched sample): {str(e)[:60]}...')
        except Exception as e:
            log(f'WRONG-ERROR {name} {rep}: {type(e).__name__}: {str(e)[:160]}')
# regularizer on a stacked point, model path
try:
    import t3toolbox.fitting as fitting
    fitting.apply_model(t3m.MANIFOLD, xs, sample, np.zeros((12, 2)), regularizer=IdentityRegularizer(0.1))
    log('NO-RAISE stacked regularized model')
except NotImplementedError as e:
    log('OK stacked regularized model raises NotImplementedError')
except Exception as e:
    log(f'WRONG-ERROR stacked regularized model: {type(e).__name__}: {str(e)[:160]}')

# ------------------------------------------------------------------ x0 variants
log('== x0 variants (manifold, apply) ==')
x, sharing, sample, weight, y = SO.setup(3, False, 'apply', None, 'none', 5)
x_orth = t3.TuckerTensorTrain(*SM.t3.TuckerTensorTrain.left_orthogonalize_tt_cores(x).data)
x_zero = t3.TuckerTensorTrain.zeros(x.shape, x.tucker_ranks, x.tt_ranks)
x_nonmin = t3.TuckerTensorTrain.randn(x.shape, (2, 3, 2), (1, 2, 1, 1))      # n_1 = 3 > r_1 r_2 = 2
for label, x0 in [('random', x), ('orthogonalized', x_orth), ('zero', x_zero), ('nonminimal', x_nonmin)]:
    for rep in ('ragged', 'uniform'):
        geom = t3m.MANIFOLD if rep == 'ragged' else ut3m.UNIFORM_MANIFOLD
        x0r = x0 if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x0)
        for name, fn, extra, kw in [('newton_cg', topt.newton_cg, (), dict(max_newton=4)),
                                    ('gradient_descent', topt.gradient_descent, (), dict(n_iter=4)),
                                    ('mc_sgd', topt.mc_sgd, (np.random.default_rng(0), 6), dict(max_iter=4, check_every=4)),
                                    ('adam', topt.adam, (np.random.default_rng(0), 6), dict(max_iter=4))]:
            try:
                xr, st = fn(geom, 'apply', sample, y, x0r, *extra, **kw)
                m, r = SO.hand_objective(xr, 'apply', sample, y, None, None, 'manifold', None)
                log(f'OK  {label:14s} {rep:7s} {name:16s} final_obj={m:.4e} losses={[f"{v:.3e}" for v in st["losses"]]}')
            except Exception as e:
                log(f'EXC {label:14s} {rep:7s} {name:16s}: {type(e).__name__}: {str(e)[:120]}')

# ------------------------------------------------------------------ verbose display smoke (+ validation)
log('== verbose display smoke ==')
for rep in ('ragged', 'uniform'):
    for kind, order, wspec in [('probe', None, 'mode'), ('probe_derivatives', 2, 'matrix'), ('apply', None, 'none')]:
        x, sharing, sample, weight, y = SO.setup(3, False, kind, order, wspec, 5)
        xv, _, vsample, _, vy = SO.setup(3, False, kind, order, wspec, 9)
        geom = t3m.MANIFOLD if rep == 'ragged' else ut3m.UNIFORM_MANIFOLD
        x0 = x if rep == 'ragged' else ut3.UniformTuckerTensorTrain.from_t3(x)
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                xr, st = topt.newton_cg(geom, kind, sample, y, x0, order=order, weight=weight, max_newton=2, verbose=True,
                                        val_sample=vsample, val_data=vy, regularizer=IdentityRegularizer(0.1))
            recs = st['diagnostics']
            log(f'OK  {rep:7s} {kind:19s} verbose: {len(recs)} records, {len(buf.getvalue().splitlines())} lines printed; '
                f'history keys ok={all(k in st["history"][0] for k in ("objective", "misfit", "regularization"))}')
        except Exception as e:
            log(f'EXC {rep} {kind} verbose: {type(e).__name__}: {str(e)[:200]}')

# ------------------------------------------------------------------ d = 1 ragged
log('== d=1 ragged ==')
for kind, order in [('apply', None), ('entries', None), ('probe', None), ('apply_derivatives', 1), ('probe_derivatives', 1)]:
    x, sharing, sample, weight, y = SO.setup(1, False, kind, order, 'none', 5)
    for geom in (t3m.MANIFOLD, t3m.COREWISE):
        try:
            xr, st = topt.newton_cg(geom, kind, sample, y, x, order=order, max_newton=3)
            m, r = SO.hand_objective(xr, kind, sample, y, None, order, 'manifold', None)
            log(f'OK  d=1 {kind:19s} {type(geom).__name__:18s} losses[0]={st["losses"][0]:.3e} hand_final={m:.3e}')
        except Exception as e:
            log(f'EXC d=1 {kind} {type(geom).__name__}: {type(e).__name__}: {str(e)[:160]}')

# ------------------------------------------------------------------ Problem.objective at a raw x0 (known S) -- consequence
log('== Problem.objective (backend) at a raw non-left-orthogonal x0 with a regularizer (known S) ==')
x, sharing, sample, weight, y = SO.setup(3, False, 'apply', None, 'none', 5)
prob = bopt.least_squares_problem(bgeo.ManifoldGeometryOps(), bfit.APPLY, sample, y, regularizer=IdentityRegularizer(0.3))
m, r = SO.hand_objective(x, 'apply', sample, y, None, None, 'manifold', True)
lm = prob.local_model(x.data)
log(f'Problem.objective(raw x0) = {float(prob.objective(x.data)):.6f}   hand = {m + r:.6f}   LocalModel.objective = {float(lm.objective):.6f}')
x1, st = bopt.gradient_descent(prob, x.data, n_iter=2)
m1, r1 = SO.hand_objective(t3.TuckerTensorTrain(*x1), 'apply', sample, y, None, None, 'manifold', True)
log(f'gradient_descent losses = {[f"{v:.6f}" for v in st["losses"]]}, hand at returned = {m1 + r1:.6f} (the loop itself uses LocalModel.objective + retracted points: unaffected)')
open('sweep_misc.out', 'w').write('\n'.join(out))
