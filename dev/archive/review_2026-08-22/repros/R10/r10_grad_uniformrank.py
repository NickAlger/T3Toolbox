"""R10: is the NaN grad of the uniform (orthogonalized) norm only a padding artifact? Fully uniform-rank, no padding."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
J = lambda o: jax.tree_util.tree_map(jnp.asarray, o)
fin = lambda g: all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(g))
np.random.seed(0)
x = t3.TuckerTensorTrain.randn((5, 6, 7), (3, 3, 3), (1, 3, 3, 1))   # every interior edge rank 3 -> only the boundary bonds are padded
ux = ut3.UniformTuckerTensorTrain.from_t3(x)
print('masks all real (interior):', bool(ux.masks.tucker_edge_mask.all()), '; tt mask:', ux.masks.tt_edge_mask.astype(int).tolist())
print('grad ux.norm() finite:', fin(jax.grad(lambda a: a.norm())(J(ux))))
print('grad x.norm() (ragged) finite:', fin(jax.grad(lambda a: a.norm())(J(x))))
W = ut3.UT3Weights.from_ut3svd(ux)
print('grad ut3_weighted_norm default finite:', fin(jax.grad(lambda w: ut3.ut3_weighted_norm(J(ux), w))(J(W))),
      '| no-orth:', fin(jax.grad(lambda w: ut3.ut3_weighted_norm(J(ux), w, use_orthogonalization=False))(J(W))))
# where does the NaN come from? grad of the left-orthogonalization alone
from t3toolbox.backend import ut3_linalg
g = jax.grad(lambda a: sum(jnp.sum(l ** 2) for l in ut3_linalg._ut3_left_orthogonalized(a.data)[:2]))(J(ux))
print('grad through _ut3_left_orthogonalized finite:', fin(g))
