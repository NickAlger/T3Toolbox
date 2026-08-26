"""R6: uniform frontend adjoint identity (chunked), same frame for forward and transpose."""
import numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import t3toolbox.tucker_tensor_train as t3, t3toolbox.uniform_tucker_tensor_train as ut3, t3toolbox.uniform_manifold as ut3m
np.random.seed(0)
shape, tr, tt = (6, 7, 5), (3, 4, 2), (1, 2, 3, 1)
xu = jax.tree.map(jnp.asarray, ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tr, tt)))
fu = ut3m.UNIFORM_MANIFOLD.frame(xu)
vu = ut3m.UNIFORM_COREWISE.randn(fu)
W = 7
ww = [jnp.asarray(np.random.randn(W, n)) for n in shape]; pp = [jnp.asarray(np.random.randn(W, n)) for n in shape]
for order in (0, 3):
    Jv = vu.probe_derivatives(ww, pp, order)
    r = [jnp.asarray(np.random.randn(*np.asarray(z).shape)) for z in Jv]
    lhs = sum(float(jnp.sum(ri * zi)) for ri, zi in zip(r, Jv))
    for cs in (None, 2, 100):
        JTr = ut3m.UT3Tangent.probe_derivatives_transpose(r, ww, pp, fu, order, sum_over_probes=True, chunk_size=cs)
        print(f'order={order} chunk_size={cs}: adjoint rel err', abs(lhs - float(JTr.corewise_inner(vu))) / abs(lhs))
