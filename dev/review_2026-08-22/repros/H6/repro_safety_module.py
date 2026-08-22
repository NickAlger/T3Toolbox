"""H6: the safety module itself -- trace detection, thread inheritance, nesting/restoration, rtol selection."""
import threading, contextvars
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
from t3toolbox.backend.common import tree_to_jax

print('jax', jax.__version__, 'has trace_state_clean:', hasattr(jax.core, 'trace_state_clean'), 'has Tracer:', hasattr(jax.core, 'Tracer'))

# 1. checks_active under transforms with closed-over CONCRETE operands (numpy and jax)
npx = np.zeros(3); jx = jnp.zeros(3)
print('eager numpy   checks_active:', safety.checks_active(npx))
print('eager jax     checks_active:', safety.checks_active(jx))
print('jit, closed-over numpy  :', jax.jit(lambda: safety.checks_active(npx))() is False or 'RAN (returned %r)' % jax.jit(lambda: safety.checks_active(npx))())
print('jit, closed-over jax    :', jax.jit(lambda: safety.checks_active(jx))())
print('jit, traced arg         :', jax.jit(lambda a: safety.checks_active(a))(jx))
print('vmap, closed-over numpy :', jax.vmap(lambda a: jnp.float32(safety.checks_active(npx)))(jnp.zeros(2)))
print('grad, closed-over jax   :', jax.grad(lambda a: a.sum() + jnp.float32(safety.checks_active(jx)))(jnp.zeros(2)))
print('_inside_jax_trace eager :', safety._inside_jax_trace())
print('_inside_jax_trace in jit:', jax.jit(lambda: safety._inside_jax_trace())())

# 2. a real check site with a closed-over CONCRETE JAX frame inside jit (the documented hard case)
np.random.seed(0)
frame_np = bvf.T3Frame.random_orthogonal((4, 5, 3), (2, 3, 2), (1, 2, 2, 1))
frame_j = bvf.T3Frame(*tree_to_jax(frame_np.data))
v_np = t3m.MANIFOLD.randn(frame_np)
def f(vdata):
    v = t3m.T3Tangent(frame_j, bvf.T3Variations(*vdata))
    return t3m.MANIFOLD.project(v).variations.data
try:
    out = jax.jit(f)(tree_to_jax(v_np.variations.data))
    print('jit project with closed-over concrete jax frame: OK')
except Exception as e:
    print('jit project with closed-over concrete jax frame: FAILED', type(e).__name__, str(e)[:200])
def g(vdata):
    v = t3m.T3Tangent(frame_j, bvf.T3Variations(*vdata))
    return t3m.MANIFOLD.norm(v)
try:
    out = jax.jit(g)(tree_to_jax(v_np.variations.data))
    print('jit norm with closed-over concrete jax frame: OK', float(out))
except Exception as e:
    print('jit norm with closed-over concrete jax frame: FAILED', type(e).__name__, str(e)[:200])

# 3. set_default_safety: current context only -- worker threads do NOT see it
safety.set_default_safety(rtol_numpy=1e-3, rtol_jax=1e-2)
print('main thread after set_default_safety:', safety.current_safety())
res = {}
t = threading.Thread(target=lambda: res.__setitem__('thr', safety.current_safety())); t.start(); t.join()
print('worker thread sees                  :', res['thr'])
# and there is no script-level UNSAFE default at all
print('set_default_safety can express unsafe?', 'no -- signature is (rtol_numpy, rtol_jax); None is not accepted:')
try:
    safety.set_default_safety(None, None); print('  set_default_safety(None, None) ->', safety.current_safety(), 'checks_active:', safety.checks_active(npx))
except Exception as e:
    print('  ', type(e).__name__, e)
safety.set_default_safety()  # restore

# 4. nesting + restoration on exception
print('default:', safety.current_safety())
try:
    with safety.unsafe():
        with safety.safe(1e-3, 1e-2):
            print(' inner safe:', safety.current_safety())
            raise RuntimeError('boom')
except RuntimeError:
    pass
print('after exception:', safety.current_safety())

# 5. effective_rtol / frames_equal selection on MIXED inputs
print('effective_rtol(numpy):', safety.effective_rtol(frame_np.data))
print('effective_rtol(jax)  :', safety.effective_rtol(frame_j.data))
print('effective_rtol(mixed numpy frame, jax variations):', safety.effective_rtol(frame_np.data, tree_to_jax(v_np.variations.data)))
print('frames_equal(np, jax-float32 copy of same frame):', safety.frames_equal(frame_np.data, frame_j.data))
with safety.unsafe():
    print('frames_equal in unsafe mode still compares:', safety.frames_equal(frame_np.data, bvf.T3Frame.random_orthogonal_like(frame_np).data))

# 6. frames_equal is RELATIVE with atol=0: an exact-zero entry must match exactly
a = (np.array([[0.0, 1.0]]),); b = (np.array([[1e-300, 1.0]]),)
print('frames_equal([0,1],[1e-300,1]):', safety.frames_equal(a, b), ' (atol=0 -> any nonzero vs exact zero differs)')
