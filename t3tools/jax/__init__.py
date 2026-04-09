import typing as typ
import numpy as np
import jax
import jax as jnp


import t3tools.corewise                     as corewise
import t3tools.linalg                       as linalg
import t3tools.base_variation_format        as base_variation_format
import t3tools.manifold                     as manifold
import t3tools.orthogonalization            as orthogonalization
import t3tools.probing                      as probing
import t3tools.tucker_tensor_train          as tucker_tensor_train
import t3tools.uniform                      as uniform
import t3tools.uniform_manifold             as uniform_manifold
import t3tools.uniform_orthogonalization    as uniform_orthogonalization
import t3tools.uniform_probing              as uniform_probing
import t3tools.uniform_t3svd                as uniform_t3svd


for m in [
    corewise,
    linalg,
    base_variation_format,
    manifold,
    orthogonalization,
    probing,
    tucker_tensor_train,
    uniform,
    uniform_manifold,
    uniform_orthogonalization,
    uniform_probing,
    uniform_t3svd,
]:
    m.xnp = jnp
    m.randn = lambda s: jnp.array(np.random.randn(s))
    m.scan = jax.lax.scan
    m.NDArray = typ.Union[np.ndarray, jnp.ndarray]
