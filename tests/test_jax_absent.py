"""The jax-absent policy (review 2026-08-22, C10): a request for jax on a machine without jax runs on numpy
and WARNS -- never a bare NameError, never a silent numpy result, never an error from use_jit=True.
jax IS installed in the test environment, so the check runs in a subprocess that hides it.

T28 (2026.3.0, the exact Hessian): every class the exact mode adds registers its jax pytree only under
``if jax_available:``, so in the jax-hidden subprocess each one is CONSTRUCTED (not merely imported) --
``LiftFactors`` (``fv_lift_factors``), ``ManifoldCurvatureData`` (``curvature_precompute``), the backend
``CurvatureState`` and an exact ``fitting.LocalModel`` (``probe_derivatives_model(..., hessian='exact')``),
whose ``hessian(p)`` is applied -- and ``newton_cg(hessian='exact', use_jit=True)`` runs to completion on numpy
with the jax-absent warning."""
import os
import subprocess
import sys
import unittest

_SCRIPT = r'''
import sys, warnings
sys.modules['jax'] = None            # make `import jax` raise ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as common
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.optimizers as topt
import t3toolbox.manifold as t3m
assert not common.jax_available
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 2, 2), (1, 2, 2, 1), use_jax=True)
    ux = ut3.UniformTuckerTensorTrain.from_t3(x).to_jax()
    xj = x.to_jax()
    xnp, _, _ = common.get_backend(False, True)
    ww = tuple(np.random.randn(6, n) for n in x.shape)
    x_opt, stats = topt.newton_cg(t3m.MANIFOLD, 'apply', ww, x.apply(ww), x, max_newton=2, use_jit=True)
assert all(isinstance(c, np.ndarray) for c in x.tucker_cores), 'randn(use_jax=True) must fall back to numpy'
assert isinstance(xj.tucker_cores[0], np.ndarray) and isinstance(ux.tucker_supercore, np.ndarray)
assert xnp is np
assert isinstance(x_opt, t3.TuckerTensorTrain) and isinstance(x_opt.tucker_cores[0], np.ndarray)
msgs = [str(m.message) for m in w if issubclass(m.category, RuntimeWarning)]
assert any('randn(use_jax=True)' in m for m in msgs), msgs
assert any('to_jax' in m for m in msgs), msgs
assert any('get_backend' in m for m in msgs), msgs
assert any('use_jit=True' in m for m in msgs), msgs

# T28: the exact mode's classes, constructed without jax
import t3toolbox.fitting as fitting
import t3toolbox.frame_variations_format as bvf
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.geometry as bgeo
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit
rng = np.random.default_rng(0)
np.random.seed(0)
shape, order = (4, 5, 3), 1
xe = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
target = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
wwe = [rng.standard_normal((8, n)) for n in shape]
ppe = [rng.standard_normal((8, n)) for n in shape]
data = target.probe_derivatives(wwe, ppe, order)
frame, _ = bvf.t3_orthogonal_representations(xe)
lift = fvo.fv_lift_factors(frame.data, 1e-6)
assert type(lift) is fvo.LiftFactors
geom = bgeo.ManifoldGeometryOps()
prob = bopt.least_squares_problem(geom, bfit.probe_derivatives_kind(order), (wwe, ppe), tuple(data))
lm = prob.local_model(xe.data, hessian='exact')
assert type(lm.curvature) is bopt.CurvatureState and lm.is_exact
assert type(lm.curvature.geometry_data) is bgeo.ManifoldCurvatureData
cdata = geom.curvature_precompute(lm.frame, lm.curvature.misfit_gradient, lm.geom_aux, None)
assert type(cdata) is bgeo.ManifoldCurvatureData
state = bopt.CurvatureState(lm.curvature.misfit_gradient, cdata)
assert type(state) is bopt.CurvatureState
residual = [z - y for z, y in zip(xe.probe_derivatives(wwe, ppe, order), data)]
model = fitting.probe_derivatives_model(t3m.MANIFOLD, xe, wwe, ppe, order, residual, hessian='exact')
assert type(model) is fitting.LocalModel and model.is_exact
p = t3m.MANIFOLD.randn(model.frame)
Hp = model.hessian(p)
assert isinstance(Hp.variations.data[0][0], np.ndarray)
assert abs(float(p.corewise_inner(Hp)) - float(model.quadratic(p))) <= 1e-10 * abs(float(model.quadratic(p)))
common._JAX_WARNED.clear()          # the warning is once per call site; re-arm it for the exact run
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter('always')
    xe_opt, st = topt.newton_cg(t3m.MANIFOLD, 'probe_derivatives', (wwe, ppe), data, xe, order=order,
                                hessian='exact', use_jit=True, max_newton=4)
assert isinstance(xe_opt, t3.TuckerTensorTrain) and isinstance(xe_opt.tucker_cores[0], np.ndarray)
assert st['history'][0]['hessian'] == 'exact' and st['history'][0]['step_control'] == 'trust_region'
assert st['stop_reason'] is not None and st['losses'][-1] <= st['losses'][0]
assert any('use_jit=True' in str(m.message) for m in w2 if issubclass(m.category, RuntimeWarning))
print('JAX_ABSENT_OK')
'''


class TestJaxAbsentPolicy(unittest.TestCase):
    def test_requests_for_jax_run_on_numpy_with_a_warning(self):
        out = subprocess.run([sys.executable, '-c', _SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
