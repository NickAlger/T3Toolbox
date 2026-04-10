import typing as typ
import numpy as np
import jax
import jax.numpy as jnp
import importlib.util

__all__ = [
    "corewise",
    "linalg",
    "base_variation_format",
    "manifold",
    "orthogonalization",
    "probing",
    "probing",
    "tucker_tensor_train",
    "uniform",
    "uniform_manifold",
    "uniform_orthogonalization",
    "uniform_probing",
    "uniform_t3svd",
]

def load_isolated(module_name):
    spec = importlib.util.find_spec(module_name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    m.xnp = jnp
    m.randn = lambda *args, **kwargs: jnp.array(np.random.randn(*args, **kwargs))
    m.scan = jax.lax.scan
    m.NDArray = typ.Union[np.ndarray, jnp.ndarray]

    return m


corewise                    = load_isolated("t3tools.corewise")
linalg                      = load_isolated("t3tools.linalg")
base_variation_format       = load_isolated("t3tools.base_variation_format")
manifold                    = load_isolated("t3tools.manifold")
orthogonalization           = load_isolated("t3tools.orthogonalization")
probing                     = load_isolated("t3tools.probing")
tucker_tensor_train         = load_isolated("t3tools.tucker_tensor_train")
uniform                     = load_isolated("t3tools.uniform")
uniform_manifold            = load_isolated("t3tools.uniform_manifold")
uniform_orthogonalization   = load_isolated("t3tools.uniform_orthogonalization")
uniform_probing             = load_isolated("t3tools.uniform_probing")
uniform_t3svd               = load_isolated("t3tools.uniform_t3svd")
