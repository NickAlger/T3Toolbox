"""R3: does the np.minimum / xnp mix in ranks.py matter with jax inputs (eager and under jit)?"""
import numpy as np
import jax, jax.numpy as jnp
import t3toolbox.backend.ranks as ranks

shape = (5, 6, 7)
tk = jnp.array([4, 9, 6]); tt = jnp.array([1, 4, 20, 1])
out = ranks.compute_minimal_ranks(shape, tk, tt, use_jax=True)
print('eager jax input: types', type(out[0]).__name__, type(out[1]).__name__, 'values', np.asarray(out[0]), np.asarray(out[1]))
out2 = ranks.compute_orthogonal_representation_ranks(shape, tk, tt, use_jax=True)
print('eager orth-rep jax input types:', [type(o).__name__ for o in out2])
out3 = ranks.compute_raw_sweep_ranks(shape, tk, tt, tk, tt, use_jax=True)
print('eager raw-sweep jax input types:', [type(o).__name__ for o in out3])
for name, f in [
    ('compute_minimal_ranks', lambda a, b: ranks.compute_minimal_ranks(shape, a, b, use_jax=True)),
    ('compute_raw_sweep_ranks', lambda a, b: ranks.compute_raw_sweep_ranks(shape, a, b, a, b, use_jax=True)),
    ('compute_orthogonal_representation_ranks', lambda a, b: ranks.compute_orthogonal_representation_ranks(shape, a, b, use_jax=True)),
]:
    try:
        jax.jit(f)(tk, tt)
        print('jit', name, ': OK')
    except Exception as e:
        print('jit', name, ': FAIL', type(e).__name__, str(e).splitlines()[0][:100])
# is any in-library caller passing traced ranks? (answer by grep, documented in the report)
