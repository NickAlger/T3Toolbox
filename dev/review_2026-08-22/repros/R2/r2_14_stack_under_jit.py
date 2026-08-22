"""stacking.stack inside jit: moveaxis(source=jnp.arange(...)) is a tracer."""
import numpy as np, jax, jax.numpy as jnp, t3toolbox as t3
from t3toolbox.backend import stacking
x = t3.TuckerTensorTrain.randn((3, 4), (2, 2), (1, 2, 1)).to_jax()
y = t3.TuckerTensorTrain.randn((3, 4), (2, 2), (1, 2, 1)).to_jax()
def show(label, f):
    try:
        r = f(); print('%-55s -> %s' % (label, r))
    except Exception as e:
        print('%-55s -> %s: %s' % (label, type(e).__name__, str(e).splitlines()[0][:100]))
show('eager TuckerTensorTrain.stack((x, y))', lambda: t3.TuckerTensorTrain.stack((x, y)).stack_shape)
show('jit   TuckerTensorTrain.stack((x, y))', lambda: jax.jit(lambda u, v: t3.TuckerTensorTrain.stack((u, v)).data)(x, y)[0][0].shape)
show('jit   stacking.stack axes=()', lambda: jax.jit(lambda u: stacking.stack(u, ()))((jnp.ones(2), jnp.ones(3)))[0].shape)
show('jit   stacking.stack axes=(0,)', lambda: jax.jit(lambda u: stacking.stack(u, (0,)))(((jnp.ones(2),), (jnp.ones(2),)))[0].shape)
show('jit   stacking.unstack axes=(0,)', lambda: len(jax.jit(lambda s: stacking.unstack(s, (0,)))((jnp.ones((2, 3)),))))
fr = t3.T3Frame.randn((3, 4), (2, 2), (1, 2, 1)) if hasattr(t3.T3Frame, 'randn') else None
import t3toolbox.manifold as m
frame = t3.T3Frame.from_tucker_tensor_train(x) if hasattr(t3.T3Frame, 'from_tucker_tensor_train') else None
