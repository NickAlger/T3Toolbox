"""Rank-continuation x0 (zero-padded resize, the documented uniform continuation pattern,
docs/rank_continuation.md 'On the uniform layer') on the uniform manifold."""
import numpy as np
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, MANIFOLD
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_manifold import UNIFORM_MANIFOLD as UM
import t3toolbox.optimizers as opt
np.random.seed(0)
xt = T3.randn((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))
ww = tuple(np.random.randn(40, N) for N in xt.shape); zz = xt.probe(ww)
x0 = T3.randn((5, 6, 7), (2, 2, 2), (1, 2, 2, 1)).resize((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))   # zero-padded rank growth
print('x0 has_minimal_ranks =', x0.has_minimal_ranks, '| ragged frame residual = %.1e | uniform frame residual = %.1e' % (
    float(MANIFOLD.frame(x0).orthogonality_residual), float(UM.frame(UT3.from_t3(x0)).orthogonality_residual)))
def hist(st):
    h = st['history']
    return [float(v) for v in (h['objective'] if isinstance(h, dict) else [e['objective'] for e in h])][:5]
xr, st = opt.newton_cg(MANIFOLD, 'probe', ww, zz, x0, max_newton=4)
print('ragged  newton_cg (safe):  objective history', ['%.3g' % v for v in hist(st)])
try:
    xu, st = opt.newton_cg(UM, 'probe', ww, zz, UT3.from_t3(x0), max_newton=4)
    print('uniform newton_cg (safe):  objective history', ['%.3g' % v for v in hist(st)])
except Exception as e:
    print('uniform newton_cg (safe) RAISED:', type(e).__name__, str(e).splitlines()[0][:100])
with tb.safety.unsafe():
    xu, st = opt.newton_cg(UM, 'probe', ww, zz, UT3.from_t3(x0), max_newton=4)
    print('uniform newton_cg (unsafe): objective history', ['%.3g' % v for v in hist(st)])
