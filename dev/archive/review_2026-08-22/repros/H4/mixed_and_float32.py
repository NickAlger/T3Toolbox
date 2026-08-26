"""Mixed numpy/jax operands, jax-default float32 (no x64) tolerances, to_jax() silent downcast."""
import numpy as np, jax, jax.numpy as jnp
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
np.random.seed(0)
def case(name, f):
    try:
        r = f(); print('%-52s OK %s' % (name, r))
    except Exception as e:
        print('%-52s FAIL %s: %s' % (name, type(e).__name__, (str(e).splitlines() or [''])[0][:160]))
x = T3.randn((4, 5, 6), (2, 3, 2), (1, 2, 3, 1))
ww = tuple(np.random.randn(5, N) for N in x.shape)
xj = x.to_jax()
print('to_jax dtype (no x64):', xj.tucker_cores[0].dtype)
case('numpy T3 + jax T3', lambda: type((x + xj).tucker_cores[0]).__name__)
case('numpy T3 .probe(jax ww)', lambda: type(x.probe(tuple(jnp.asarray(w) for w in ww))[0]).__name__)
case('jax T3 .probe(numpy ww)', lambda: type(xj.probe(ww)[0]).__name__)
case('numpy T3 .inner(jax T3)', lambda: type(x.inner(xj)).__name__)
f, v = t3_orthogonal_representations(x)
fj, vj = t3_orthogonal_representations(xj)
case('T3Tangent(np frame, jax variations)', lambda: type(T3Tangent(f, vj).to_dense()).__name__)
case('np tangent + jax tangent (same frame numerically)', lambda: type((T3Tangent(f, v) + T3Tangent(fj, vj)).variations.tucker_variations[0]).__name__)
case('MANIFOLD.norm jax f32 safe-mode (orthogonality check at rtol_jax)', lambda: float(MANIFOLD.norm(MANIFOLD.project(T3Tangent(fj, vj)))))
case('is_orthogonal jax f32 default atol', lambda: fj.is_orthogonal())
case('has_numerically_minimal_ranks jax f32 (rtol=1e-9)', lambda: xj.has_numerically_minimal_ranks())
case('has_numerically_minimal_ranks numpy (rtol=1e-9)', lambda: x.has_numerically_minimal_ranks())
# a numerically rank-deficient tensor: does f32 hide the redundancy?
xr = T3.randn((4, 5, 6), (2, 3, 2), (1, 2, 3, 1)).resize((4, 5, 6), (3, 3, 2), (1, 2, 3, 1))   # tucker_0 padded with zeros
xr = T3(tuple(B + 1e-8 * np.random.randn(*B.shape) for B in xr.tucker_cores), xr.tt_cores)     # tiny noise -> numerically redundant
case('numerically-redundant numpy has_num_minimal', lambda: xr.has_numerically_minimal_ranks())
case('numerically-redundant jax f32 has_num_minimal', lambda: xr.to_jax().has_numerically_minimal_ranks())
case('t3svd(rtol=1e-9) numpy tucker ranks', lambda: xr.t3svd(rtol=1e-9)[0].tucker_ranks)
case('t3svd(rtol=1e-9) jax f32 tucker ranks', lambda: xr.to_jax().t3svd(rtol=1e-9)[0].tucker_ranks)
case('continuation_ranks numpy', lambda: x.continuation_ranks()[0])
case('continuation_ranks jax f32', lambda: xj.continuation_ranks()[0])
# uniform
ux = UT3.from_t3(x); uxj = ux.to_jax()
case('uniform np + jax', lambda: type((ux + uxj).tucker_supercore).__name__)
case('uniform jax .probe(np ww) supercore type', lambda: type(uxj.probe(ww)[0]).__name__)
case('uniform np .probe(jax ww)', lambda: type(ux.probe(tuple(jnp.asarray(w) for w in ww))[0]).__name__)
case('uniform jax masks still numpy', lambda: type(uxj.masks.tucker_mask).__name__)
