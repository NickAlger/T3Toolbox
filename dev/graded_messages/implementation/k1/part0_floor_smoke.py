import copy, dataclasses as dc, pickle, sys
import numpy as np, jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3, t3toolbox.manifold as t3m
from t3toolbox.backend.fitting import LeastSquaresBatch, LeastSquaresOracle
import t3toolbox.backend.optimizers as bopt
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
ww = [np.random.randn(7, n) for n in x.shape]
b = LeastSquaresBatch('probe', (np.int64(0),)[:0], ww, list(x.probe(ww)), weight=None)
print(sys.version.split()[0], b.kind, b.order, b.n_measurements)
for bad in ((ww, list(x.probe(ww))), (ww, list(x.probe(ww)), None)):
    try: LeastSquaresBatch(*bad)
    except TypeError as e: print('TypeError', str(e)[:70])
b2 = dc.replace(b, minibatch_scale=2.0); print(b2.minibatch_scale, copy.deepcopy(b2).kind, pickle.loads(pickle.dumps(b2)).order)
leaves, td = jax.tree_util.tree_flatten(b2); print(jax.tree_util.tree_unflatten(td, [object()] * len(leaves)).kind)
frame = t3m.MANIFOLD.frame(x).data
(f,), _ = LeastSquaresOracle().query(frame, b.take(np.array([0, 3])), ('f',)); print('f', float(f))
n = []
@jax.jit
def q(batch):
    n.append(1); return jnp.sum(batch.y[0])
q(jax.tree_util.tree_map(jnp.asarray, b)); q(jax.tree_util.tree_map(jnp.asarray, b)); print('traces', len(n))
print(bopt.flat_draw(b, 3)(np.random.default_rng(0)).drawn_indices)
