"""Safe-mode precondition checks on jax default float32 (no x64): do they false-fail?"""
import sys, numpy as np, jax, jax.numpy as jnp
if len(sys.argv) > 1 and sys.argv[1] == 'x64':
    jax.config.update("jax_enable_x64", True)
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD, COREWISE
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3
from t3toolbox.uniform_frame_variations_format import ut3_orthogonal_representations
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD as UM
np.random.seed(0)
for struct in [((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)), ((6, 7, 8, 5), (3, 3, 3, 3), (1, 3, 4, 3, 1))]:
    x = T3.randn(*struct)
    xj = x.to_jax()
    print('=== struct', struct, 'dtype', xj.tucker_cores[0].dtype)
    fj, vj = t3_orthogonal_representations(xj)
    print('frame.orthogonality_residual =', float(fj.orthogonality_residual), ' is_orthogonal(default atol) =', fj.is_orthogonal(),
          ' effective_rtol =', tb.safety.effective_rtol(fj.data))
    t = T3Tangent(fj, vj)
    try:
        p = MANIFOLD.project(t)
        print('project ok; gauge_residual =', float(p.gauge_residual), ' is_gauged(default) =', p.is_gauged(), ' is_gauged(atol=rtol_jax) =', p.is_gauged(atol=tb.safety.effective_rtol(p.data)) if 'atol' in p.is_gauged.__code__.co_varnames else 'n/a')
        n = MANIFOLD.norm(p); print('MANIFOLD.norm ok', float(n))
    except Exception as e:
        print('MANIFOLD.norm FAILED:', type(e).__name__, str(e).splitlines()[0][:120])
    try:
        g = COREWISE.randn(fj)
        print('COREWISE.randn ok; MANIFOLD.inner(project(g), project(g)) =', float(MANIFOLD.inner(MANIFOLD.project(g), MANIFOLD.project(g))))
    except Exception as e:
        print('MANIFOLD.inner FAILED:', type(e).__name__, str(e).splitlines()[0][:120])
    try:
        r = MANIFOLD.retract(MANIFOLD.project(t)); print('retract ok')
    except Exception as e:
        print('retract FAILED:', type(e).__name__, str(e).splitlines()[0][:120])
    # uniform
    uxj = UT3.from_t3(x).to_jax()
    ufj, uvj = ut3_orthogonal_representations(uxj)
    try:
        up = UM.project(UT3Tangent(ufj, uvj)); print('uniform: frame residual', float(ufj.orthogonality_residual), 'gauge residual', float(up.gauge_residual)); print('UM.norm ok', float(UM.norm(up)))
    except Exception as e:
        print('UNIFORM FAILED:', type(e).__name__, str(e).splitlines()[0][:120])
