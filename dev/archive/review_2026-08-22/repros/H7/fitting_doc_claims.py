import numpy as np, warnings, inspect, dataclasses as dc; warnings.filterwarnings('ignore')
import t3toolbox as t3t
from t3toolbox.backend import optimizers as bo, fitting as bf, uniform_fitting as uf, ranks
import t3toolbox.optimizers as topt
print('LocalModel members:', [m for m in dir(bo.LocalModel) if not m.startswith('_')])
print('uniform_minimal sig:', inspect.signature(uf.uniform_minimal))
print('frame_has_minimal_ranks sig:', inspect.signature(ranks.frame_has_minimal_ranks))
np.random.seed(0)
shape=(4,5,3); x = t3t.TuckerTensorTrain.randn(shape,(2,2,2),(1,2,2,1))
ww=[np.random.randn(20,N) for N in shape]; b=x.apply(ww)
xs = t3t.TuckerTensorTrain.zeros(shape,(2,2,2),(1,2,2,1), stack_shape=(2,))
try:
    m = t3t.apply_model(t3t.MANIFOLD, xs, ww, np.stack([b,b]), regularizer=topt.IdentityRegularizer(1e-3)); print('stacked+regularizer factory: NO RAISE; objective', np.shape(m.objective()) if callable(getattr(m,'objective',None)) else '?')
except Exception as e: print('stacked+regularizer factory RAISED', type(e).__name__, str(e)[:100])
try:
    m = t3t.apply_model(t3t.MANIFOLD, xs, ww, np.stack([b,b])); print('stacked no-reg factory ok; objective shape', np.shape(m.objective()) if callable(getattr(m,'objective',None)) else [a for a in dir(m) if 'obj' in a])
except Exception as e: print('stacked no-reg factory RAISED', type(e).__name__, str(e)[:100])
# non-field parameter -> TypeError on hash/compare (fitting doc :224)
class Bad(bf.ApplyKind):
    def __init__(self, p): super().__init__(); object.__setattr__(self, 'p', p)
try:
    k = Bad(3); hash(k); print('non-field param kind: hash OK (doc says TypeError)')
except TypeError as e: print('non-field param kind: TypeError:', str(e)[:100])
except Exception as e: print('non-field param kind: OTHER', type(e).__name__, str(e)[:100])
class Bad2(bf.ApplyKind):  # forgot @dataclass, class attr
    pass
try:
    k = Bad2(); k2 = Bad2(); print('forgot-dataclass subclass eq:', k == k2, hash(k)==hash(k2))
except Exception as e: print('forgot-dataclass: ', type(e).__name__, str(e)[:100])
# frame_variations.md random_orthogonal example
fr = t3t.T3Frame.random_orthogonal((5,5,5),(4,4,4),(1,2,2,1))
print('random_orthogonal((5,5,5),(4,4,4),(1,2,2,1)) up_ranks', fr.up_ranks, 'down_ranks', fr.down_ranks, 'orthogonal', fr.is_orthogonal())
