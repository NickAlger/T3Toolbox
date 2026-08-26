"""H6: a STRUCTURAL mismatch (tangents of different rank structure) is only caught by the NUMERICAL same-frame guard,
so under safety.unsafe() / jit it is skipped and broadcasting yields a silent wrong answer; the safe-mode message
tells the user to use unsafe mode to 'skip this numerical check'."""
import numpy as np, jax, jax.numpy as jnp
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fit
np.random.seed(0)
shape = (4, 5, 3)
frA = bvf.T3Frame.random_orthogonal(shape, (2, 3, 2), (1, 2, 2, 1))
frB = bvf.T3Frame.random_orthogonal(shape, (1, 3, 2), (1, 1, 2, 1))   # different ranks -> broadcastable holes
a, b = t3m.MANIFOLD.randn(frA), t3m.MANIFOLD.randn(frB)
print('structures:', a.frame.variation_shapes[0][0], 'vs', b.frame.variation_shapes[0][0], '| H0', a.frame.variation_shapes[1][0], 'vs', b.frame.variation_shapes[1][0])
try:
    a + b
except ValueError as e:
    print('safe mode a+b ->', str(e).splitlines()[0]); print('   advice:', str(e).splitlines()[-1])
with safety.unsafe():
    try:
        c = a + b
        print('unsafe mode a+b -> NO ERROR; result tucker_variations[0].shape =', c.variations.tucker_variations[0].shape, '(broadcast of', a.variations.tucker_variations[0].shape, '+', b.variations.tucker_variations[0].shape, ')')
        print('   corewise_inner in unsafe:', float(a.corewise_inner(b)) if False else 'n/a')
    except Exception as e:
        print('unsafe mode a+b ->', type(e).__name__, str(e)[:100])
try:
    out = jax.jit(lambda p, q: (p + q).variations.data)(jax.tree_util.tree_map(jnp.asarray, a), jax.tree_util.tree_map(jnp.asarray, b))
    print('jit a+b -> NO ERROR; V0 shape', out[0][0].shape)
except Exception as e:
    print('jit a+b ->', type(e).__name__, str(e)[:100])

# GaussNewtonModel with p of a different structure
x = t3.TuckerTensorTrain.randn(shape, (2, 3, 2), (1, 2, 2, 1))
ww = [np.random.randn(6, n) for n in shape]
m = fit.apply_model(t3m.MANIFOLD, x, ww, x.apply(ww))
try:
    m.gn_quadratic(b)
except ValueError as e:
    print('safe gn_quadratic(p of other structure) ->', str(e)[:120])
with safety.unsafe():
    try:
        q = m.gn_quadratic(b); print('unsafe gn_quadratic(p of other structure) -> NO ERROR, value', float(q))
    except Exception as e:
        print('unsafe gn_quadratic(p of other structure) ->', type(e).__name__, str(e)[:100])

# set_default_safety(None, None): accepted, then every check site crashes with a TypeError
safety.set_default_safety(None, None)
print('current_safety after set_default_safety(None, None):', safety.current_safety())
try:
    t3m.MANIFOLD.norm(a); print('MANIFOLD.norm -> OK')
except Exception as e:
    print('MANIFOLD.norm ->', type(e).__name__, str(e)[:100])
safety.set_default_safety()
