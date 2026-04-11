import numpy as np
import jax

from t3toolbox.jax import tucker_tensor_train as t3_jax
from t3toolbox import tucker_tensor_train as t3

from t3toolbox.jax import corewise as cw

jax.config.update("jax_enable_x64", True) # enable double precision for finite difference
get_entry_123 = lambda x: t3_jax.t3_entry(x, (1,2,3))
A0 = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1))) # random 10x10x10 Tucker tensor train
f0 = get_entry_123(A0)
G0 = jax.grad(get_entry_123)(A0) # gradient using automatic differentiation
dA = t3_jax.t3_corewise_randn(((10,10,10),(5,5,5),(1,4,4,1)))
df = cw.corewise_dot(dA, G0) # sensitivity in direction dA
print(df)

s = 1e-7
A1 = cw.corewise_add(A0, cw.corewise_scale(dA, s)) # A1 = A0 + s*dA
f1 = get_entry_123(A1)
df_diff = (f1 - f0) / s # finite difference
print(df_diff)

# from t3toolbox import tucker_tensor_train as t3
