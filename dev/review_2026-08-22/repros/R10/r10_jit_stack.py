import numpy as np, jax, jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
J = lambda o: jax.tree_util.tree_map(jnp.asarray, o)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (2, 3, 3), (1, 2, 3, 1), stack_shape=(2,))
W = t3.T3Weights.from_t3svd(x)
for label, f, arg in (('TuckerTensorTrain.stack(unstack)', lambda a: t3.TuckerTensorTrain.stack(a.unstack()).data, x),
                      ('T3Weights.stack(unstack)', lambda w: t3.T3Weights.stack(w.unstack()).data, W),
                      ('T3Weights.unstack only', lambda w: w.unstack(), W)):
    try: jax.jit(f)(J(arg)); print('%-36s jit OK' % label)
    except Exception as e: print('%-36s jit RAISED %s: %s' % (label, type(e).__name__, str(e).splitlines()[0][:80]))
