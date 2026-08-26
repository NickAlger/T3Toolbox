"""R6: the W // chunk_size == 2 case -- lax.scan over a single chunk index (length-1 trip count) is
simplified away by XLA, so chunk 0, chunk 1 (and the remainder) are co-resident: no memory reduction."""
import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import t3toolbox.backend.sampling_derivatives as pd
d, r, K, order = 4, 24, 1, 3
trs = pd.binomial_combine_tensor(order)
S = lambda *s: jax.ShapeDtypeStruct(s, jnp.float64)
def temp(W, cs):
    args = (S(d, order + 1, W, K, r), S(d, order + 1, W, K, r), S(d, order + 1, W, K, r),
            S(d, 2, W, r), S(d, order + 1, W, r), S(d, order + 1, W, r))
    f = lambda *a: pd.assemble_tt_variation_jets(*a, trs, 1, True, chunk_size=cs)
    return jax.jit(f).lower(*args).compile().memory_analysis().temp_size_in_bytes / 2**20
for W in (200, 300):
    dense = temp(W, None)
    print(f'W={W}: dense {dense:.1f} MiB;', {cs: f'{temp(W, cs):.1f} MiB (n_full={W // cs}, rem={W % cs}, expected ~{dense * cs / W:.1f})' for cs in (100, 75, 50)})
