"""adam on MANIFOLD is gauge-dependent: its moment trees m/v live in the coordinates of the frame, and the
orthonormal frame is re-chosen every step (by an SVD whose sign/gauge convention differs between numpy and jax,
and between the ragged and uniform layers). The same call therefore gives different iterates numpy-eager vs
jax-eager vs jit, and ragged vs uniform -- while adam on COREWISE and every other optimizer agree to ~1e-15."""
import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.optimizers as topt
np.random.seed(0)
shape, tk, tt = (4, 5, 6), (2, 3, 2), (1, 2, 2, 1)
A = t3.TuckerTensorTrain.randn(shape, tk, tt)
x0 = t3.TuckerTensorTrain.randn(shape, tk, tt)
ww = [np.random.randn(12, N) for N in shape]
ww = [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]
b = A.apply(ww)
def run(geom, x, jit=False, to_jax=False, n=50):
    s, d = ww, b
    if to_jax:
        s, d, x = [jnp.asarray(w) for w in ww], jnp.asarray(b), x.to_jax()
    xr, st = topt.adam(geom, 'apply', s, d, x, np.random.default_rng(0), 6, max_iter=n, use_jit=jit)
    return np.asarray(xr.to_dense())
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
for gname, rg, ug in [('MANIFOLD', t3m.MANIFOLD, ut3m.UNIFORM_MANIFOLD), ('COREWISE', t3m.COREWISE, ut3m.UNIFORM_COREWISE)]:
    ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)
    r_np = run(rg, x0)
    r_jx = run(rg, x0, to_jax=True)          # jax arrays, no jit
    r_jit = run(rg, x0, jit=True)
    u_np = run(ug, ux0)
    print(f'{gname}: ragged numpy-eager vs jax-eager {rel(r_jx, r_np):.1e} | vs jit {rel(r_jit, r_np):.1e} | ragged vs uniform (numpy) {rel(u_np, r_np):.1e}')
# the same comparison for newton_cg on MANIFOLD (frame also re-chosen each step, but nothing is carried in coordinates)
def run_nt(geom, x, jit=False):
    xr, st = topt.newton_cg(geom, 'apply', ww, b, x, max_newton=5, use_jit=jit)
    return np.asarray(xr.to_dense())
print(f'newton_cg MANIFOLD: eager vs jit {rel(run_nt(t3m.MANIFOLD, x0, True), run_nt(t3m.MANIFOLD, x0)):.1e}, '
      f'ragged vs uniform {rel(run_nt(ut3m.UNIFORM_MANIFOLD, ut3.UniformTuckerTensorTrain.from_t3(x0)), run_nt(t3m.MANIFOLD, x0)):.1e}')
