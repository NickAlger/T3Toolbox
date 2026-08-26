"""H1: count TRACES (== compiles) of the memoized kernels across rebuilds, to confirm the compile-once
claims in CHANGELOG [2026.2.0] and parameters_not_closures.md, and that different objects do NOT share."""
import dataclasses as dc, functools
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as um
import t3toolbox.shared_geometry as sg
import t3toolbox.optimizers as opt
import t3toolbox.fitting as fitting
from t3toolbox.backend import optimizers as bopt, fitting as bfit

# --- instrument the module-level kernels with trace counters (the memo keys on the function object,
#     so we swap the module globals BEFORE first use; the wrapper is then the stable memo key).
counts = {}
def counted(name, fn):
    @functools.wraps(fn)
    def w(*a, **k):
        counts[name] = counts.get(name, 0) + 1
        return fn(*a, **k)
    return w
bopt._cg_core = counted('cg_core', bopt._cg_core)
bopt._mc_sgd_step = counted('mc_sgd_step', bopt._mc_sgd_step)
bopt._adam_step = counted('adam_step', bopt._adam_step)

np.random.seed(0)
SH, TK, TT = (6, 6, 5), (2, 3, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(SH, TK, TT)
W = 40
ww = [np.random.randn(W, n) for n in SH]
pp = [np.random.randn(W, n) for n in SH]
b_apply = A.apply(ww)
x0 = t3.TuckerTensorTrain.randn(SH, TK, TT)

print('--- ragged newton_cg(use_jit) : cg_core traces over a whole run')
counts.clear()
x, stats = opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b_apply, x0, max_newton=4, use_jit=True)
print('    newton iterations:', stats['newton'], '; cg_core traces:', counts.get('cg_core'))
counts.clear()
x, stats = opt.newton_cg(t3m.MANIFOLD, 'apply', ww, b_apply, x0, max_newton=4, use_jit=True)
print('    second identical run -> cg_core traces:', counts.get('cg_core', 0), '(process-wide memo)')

print('--- ragged probe_derivatives order=1 (parameterized kind), then order=2: separate programs?')
b_pd1 = A.probe_derivatives(ww, pp, 1); b_pd2 = A.probe_derivatives(ww, pp, 2)
counts.clear()
opt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), b_pd1, x0, order=1, max_newton=3, use_jit=True)
c1 = counts.get('cg_core', 0)
opt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), b_pd2, x0, order=2, max_newton=3, use_jit=True)
print('    traces after order=1 run:', c1, '; after order=2 run:', counts.get('cg_core', 0), '(expect +1)')
counts.clear()
opt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (ww, pp), b_pd1, x0, order=1, weight=[1.0, 0.5], max_newton=3, use_jit=True)
print('    weighted order=1 run -> new traces:', counts.get('cg_core', 0), '(expect 1: different weight != unweighted key)')

print('--- uniform probe_derivatives(use_jit): cg_core traces')
ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
counts.clear()
x, stats = opt.newton_cg(um.UNIFORM_MANIFOLD, 'probe_derivatives', (ww, pp), b_pd1, ux0, order=1, max_newton=4, use_jit=True)
print('    newton iterations:', stats['newton'], '; cg_core traces:', counts.get('cg_core'))

print('--- shared uniform (use_jit): cg_core traces')
xs0 = t3.TuckerTensorTrain.randn(SH, (2, 2, 2), TT).share((0, 0, 1))
counts.clear()
x, stats = opt.newton_cg(sg.shared(um.UNIFORM_MANIFOLD, (0, 0, 1)), 'apply', ww, b_apply,
                         ut3.UniformTuckerTensorTrain.from_t3(xs0), max_newton=4, use_jit=True)
print('    newton iterations:', stats['newton'], '; cg_core traces:', counts.get('cg_core'))

print('--- mc_sgd / adam: per-step kernel compiles across two optimizer CALLS')
counts.clear()
opt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b_apply, x0, np.random.default_rng(0), batch=10, max_iter=3, use_jit=True)
opt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b_apply, x0, np.random.default_rng(0), batch=10, max_iter=3, use_jit=True)
print('    mc_sgd two calls, same shapes -> traces:', counts.get('mc_sgd_step'))
opt.mc_sgd(t3m.MANIFOLD, 'apply', ww, b_apply, x0, np.random.default_rng(0), batch=20, max_iter=3, use_jit=True)
print('    third call, batch=20 -> traces:', counts.get('mc_sgd_step'), '(expect +1: new shape signature)')
counts.clear()
opt.adam(t3m.MANIFOLD, 'apply', ww, b_apply, x0, np.random.default_rng(0), batch=10, max_iter=3, use_jit=True)
opt.adam(t3m.COREWISE, 'apply', ww, b_apply, x0, np.random.default_rng(0), batch=10, max_iter=3, use_jit=True)
print('    adam manifold then corewise -> traces:', counts.get('adam_step'), '(expect 2: geometry is in the key)')

print('--- user-defined kind rebuilt 5x as jit aux')
@dc.dataclass(frozen=True, eq=False)
class ScaledApply(bfit.ApplyKind):
    scale: float = 1.0
    def forward(self, v, ww_, frame, sweep):
        return self.scale * super().forward(v, ww_, frame, sweep)
    def transpose(self, r, ww_, frame, sweep):
        return super().transpose(self.scale * r, ww_, frame, sweep)
xj = t3.TuckerTensorTrain(tuple(jnp.asarray(c) for c in x0.data[0]), tuple(jnp.asarray(c) for c in x0.data[1]))
wwj = [jnp.asarray(w) for w in ww]; rj = jnp.asarray(np.random.randn(W))
tr = [0]
@jax.jit
def quad(m, p):
    tr[0] += 1
    return m.gn_quadratic(p)
base = fitting.apply_model(t3m.MANIFOLD, xj, wwj, rj)
p0 = t3m.MANIFOLD.randn(base.frame)
for _ in range(5):
    m = dc.replace(base, kind=ScaledApply(scale=2.0))
    q2 = float(quad(m, p0))
print('    5 rebuilds of ScaledApply(2.0) -> traces:', tr[0])
q3 = float(quad(dc.replace(base, kind=ScaledApply(scale=3.0)), p0))
print('    scale=3.0 -> traces:', tr[0], '; jit q(2.0)=%.6f q(3.0)=%.6f ; eager q(3.0)=%.6f' % (
    q2, q3, float(dc.replace(base, kind=ScaledApply(scale=3.0)).gn_quadratic(p0))))
print('    ratio q3/q2 = %.4f (expect 2.25)' % (q3 / q2))
