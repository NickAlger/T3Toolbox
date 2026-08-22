import sys
for m in list(sys.modules):
    if m == 'jax' or m.startswith('jax.'):
        del sys.modules[m]
sys.modules['jax'] = None; sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, MANIFOLD, COREWISE
import t3toolbox.optimizers as opt
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_manifold import UNIFORM_MANIFOLD
np.random.seed(0)
x = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1))
ww = tuple(np.random.randn(16, N) for N in x.shape)
zz = x.probe(ww)
x0 = T3.randn((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)) * 0.1
def case(name, f):
    try:
        r = f(); print('%-46s OK' % name)
    except Exception as e:
        print('%-46s FAIL %s: %s' % (name, type(e).__name__, (str(e).splitlines() or [''])[0][:200]))
case('newton_cg ragged (jax absent)', lambda: opt.newton_cg(MANIFOLD, 'probe', ww, zz, x0, max_newton=2))
case('newton_cg use_jit=True (jax absent)', lambda: opt.newton_cg(MANIFOLD, 'probe', ww, zz, x0, max_newton=2, use_jit=True))
case('adam ragged (jax absent)', lambda: opt.adam(COREWISE, 'probe', ww, zz, x0, max_newton=2) if 'max_iter' in str(__import__('inspect').signature(opt.adam)) else opt.adam(COREWISE, 'probe', ww, zz, x0, num_iter=2))
ux0 = UT3.from_t3(x0)
case('newton_cg uniform (jax absent)', lambda: opt.newton_cg(UNIFORM_MANIFOLD, 'probe', ww, zz, ux0, max_newton=2))
case('newton_cg uniform chunk auto', lambda: opt.newton_cg(UNIFORM_MANIFOLD, 'probe_derivatives', (ww, ww), x.probe_derivatives(ww, ww, 1), ux0, order=1, max_newton=1, chunk_size='auto'))
