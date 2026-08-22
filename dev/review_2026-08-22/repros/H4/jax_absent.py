"""Simulate jax absence (sys.modules['jax']=None) and exercise the numpy-only paths the README promises."""
import sys
for m in list(sys.modules):
    if m == 'jax' or m.startswith('jax.'):
        del sys.modules[m]
sys.modules['jax'] = None          # `import jax` now raises ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox as tb
from t3toolbox.backend import common
print('jax_available =', common.jax_available)
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD, COREWISE, T3Frame, T3Weights, T3FrameWeights
from t3toolbox.frame_variations_format import t3_orthogonal_representations
import t3toolbox.optimizers as opt
import t3toolbox.fitting as fit
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3, UT3Weights
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD, UNIFORM_COREWISE
from t3toolbox.uniform_frame_variations_format import UT3Frame, ut3_orthogonal_representations, UT3FrameWeights
import t3toolbox.shared_geometry as sg
import t3toolbox.backend.regularization as reg

def case(name, f):
    try:
        r = f()
        print('%-48s OK  %s' % (name, type(r).__name__))
    except Exception as e:
        print('%-48s FAIL %s: %s' % (name, type(e).__name__, (str(e).splitlines() or [''])[0][:160]))

np.random.seed(0)
x = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
ww = tuple(np.random.randn(8, N) for N in x.shape)
pp = tuple(np.random.randn(8, N) for N in x.shape)
idx = np.stack([np.random.randint(0, N, size=8) for N in x.shape])
case('T3.randn/to_dense/norm', lambda: x.norm())
case('T3.randn(use_jax=True) [expect clear error]', lambda: T3.randn((4,), (2,), (1, 1), use_jax=True))
case('get_backend(False, True) [known NameError]', lambda: common.get_backend(False, True))
case('to_jax()', lambda: x.to_jax())
case('contains_jax', lambda: x.contains_jax)
case('t3svd', lambda: x.t3svd())
case('probe / apply / entries', lambda: (x.probe(ww), x.apply(ww), x.entries(idx)))
case('probe_derivatives', lambda: x.probe_derivatives(ww, pp, 2))
f, v = t3_orthogonal_representations(x)
t = T3Tangent(f, v)
case('MANIFOLD.project/inner/retract', lambda: MANIFOLD.retract(MANIFOLD.project(t)))
case('MANIFOLD.norm safe-mode check', lambda: MANIFOLD.norm(MANIFOLD.project(t)))
case('safety.is_tracing', lambda: tb.safety.is_tracing(x.data))
case('safety.checks_active', lambda: tb.safety.checks_active(x.data))
case('T3Tangent.probe_derivatives_transpose (chunk)', lambda: T3Tangent.probe_derivatives_transpose(tuple(np.random.randn(3, 8, N) for N in x.shape), ww, pp, f, 2))
case('T3Weights.from_t3svd + absorb', lambda: tb.tucker_tensor_train.t3_absorb_weights(x.t3svd()[0], T3Weights.from_t3svd(x)))
# fitting / optimizers
zz = x.probe(ww)
case('probe_model', lambda: fit.probe_model(x, ww, zz))
case('newton_cg (ragged, no jit)', lambda: opt.newton_cg(x, MANIFOLD, fit.probe_model, ww, zz, max_iter=2) if False else None)
try:
    import inspect
    print('newton_cg sig:', str(inspect.signature(opt.newton_cg))[:300])
except Exception as e:
    print('sig fail', e)
case('probe_derivatives_model chunk_size=auto', lambda: fit.probe_derivatives_model(x, ww, pp, x.probe_derivatives(ww, pp, 2), 2, chunk_size='auto') if 'chunk_size' in str(inspect.signature(fit.probe_derivatives_model)) else 'n/a')
# uniform
ux = UT3.from_t3(x)
case('UT3.from_t3/to_dense/norm', lambda: ux.norm())
case('UT3.randn(use_jax=True) [expect clear error]', lambda: UT3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1), use_jax=True))
case('UT3.probe/apply/entries', lambda: (ux.probe(ww), ux.apply(ww), ux.entries(idx)))
case('UT3.probe_derivatives', lambda: ux.probe_derivatives(ww, pp, 2))
uf, uv = ut3_orthogonal_representations(ux)
ut = UT3Tangent(uf, uv)
case('UNIFORM_MANIFOLD.project/retract', lambda: UNIFORM_MANIFOLD.retract(UNIFORM_MANIFOLD.project(ut)))
case('UT3Tangent.probe_derivatives_transpose', lambda: UT3Tangent.probe_derivatives_transpose(tuple(np.random.randn(3, 8, N) for N in x.shape), ww, pp, uf, 2))
case('ut3 t3svd', lambda: ux.t3svd())
case('UT3Weights.from_ut3svd', lambda: UT3Weights.from_ut3svd(ux))
case('shared_manifold', lambda: sg.shared_manifold([0, 0, 0]) if hasattr(sg, 'shared_manifold') else None)
case('IdentityRegularizer', lambda: reg.IdentityRegularizer(0.1))
case('xwhile (use_jit=True, jax absent -> eager)', lambda: common.xwhile(lambda s: s < 3, lambda s: s + 1, 0, use_jit=True))
case('save/load', lambda: (x.save('/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a6ed361-8c79-489c-87ff-713bf71ecb11/scratchpad/repros/H4/x.npz'), T3.load('/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a6ed361-8c79-489c-87ff-713bf71ecb11/scratchpad/repros/H4/x.npz')))
case('T3.load(use_jax=True) [expect clear error]', lambda: T3.load('/tmp/claude-1000/-home-nick-repos-T3Toolbox/7a6ed361-8c79-489c-87ff-713bf71ecb11/scratchpad/repros/H4/x.npz', use_jax=True))
