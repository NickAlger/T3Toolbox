"""The jax-absent policy (review 2026-08-22, C10): a request for jax on a machine without jax runs on numpy
and WARNS -- never a bare NameError, never a silent numpy result, never an error from use_jit=True.
jax IS installed in the test environment, so the check runs in a subprocess that hides it.

T28 (2026.3.0, the exact Hessian): every class the exact mode adds registers its jax pytree only under
``if jax_available:``, so in the jax-hidden subprocess each one is CONSTRUCTED (not merely imported) --
``LiftFactors`` (``fv_lift_factors``), ``ManifoldCurvatureData`` (``curvature_precompute``), the backend
``CurvatureState`` and an exact ``fitting.LocalModel`` (``probe_derivatives_model(..., hessian='exact')``),
whose ``hessian(p)`` is applied -- and ``newton_cg(hessian='exact', use_jit=True)`` runs to completion on numpy
with the jax-absent warning.

The Newton timing plan's T6 (2026.5.0): the timers' synchronization helper ``common.tree_block_until_ready`` is a no-op
without jax (it must not reach for ``jax``), on a mixed tree and on a point's cores."""
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

# the Newton timing plan's T6: the synchronization helper without jax (a missing guard raises a NameError)
assert common.tree_block_until_ready((np.ones(2), 3, None, 'D')) is None
assert common.tree_block_until_ready(x.data) is None

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


# The reference module of the sampling variables and the three dense sampling functions (graded messages, P2):
# they import and run with no jax. (Kp of numpy arrays is taken by finite differences, with or without jax; its
# other route, automatic differentiation, is for jax arrays, and the module must not reach for jax to import.)
_DEFINITIONS_SCRIPT = r'''
import sys
sys.modules['jax'] = None            # make `import jax` raise ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as common
import t3toolbox.backend.sampling_definitions as sdef
import t3toolbox.backend.probing as probing
import t3toolbox.backend.apply as apply
import t3toolbox.backend.entries as entries
assert not common.jax_available

rng = np.random.default_rng(0)
N, nD, nU, rL, rR = (4, 5, 6), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
D = [rng.standard_normal((nD[i], N[i])) for i in range(3)]
U = [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)]
L = [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)]
R = [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)]
ww = [rng.standard_normal((5, n)) for n in N]
pp = [rng.standard_normal((5, n)) for n in N]
index = np.stack([rng.integers(0, n, size=5) for n in N])
arguments = dict(order=2, variations=([rng.standard_normal((nU[i], N[i])) for i in range(3)],
                                      [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]),
                 core_directions=([rng.standard_normal(B.shape) for B in D], [rng.standard_normal(G.shape) for G in L]))
leaves = lambda value: sum((leaves(v) for v in value), []) if isinstance(value, (tuple, list)) else [value]
for kind, first in (('probe', ww), ('apply', ww), ('entries', index)):
    y = [rng.standard_normal((3, 5, n)) for n in N] if kind == 'probe' else rng.standard_normal((3, 5))
    for name in sdef.__all__:
        value = getattr(sdef, name)(kind, (D, U, L, R), (first, pp), y=y, **arguments)
        assert all(isinstance(leaf, np.ndarray) or np.isscalar(leaf) for leaf in leaves(value)), (kind, name)
T = rng.standard_normal(N)
assert probing.dense_probe(T, ww, pp, 2)[0].shape == (3, 5, 4)
assert apply.dense_apply(T, ww, pp, 2).shape == (3, 5) and entries.dense_entries(T, index, pp, 2).shape == (3, 5)
print('JAX_ABSENT_OK')
'''


# Jets, the checkers and data preparation (graded messages, P3): the two backend modules, their reach from
# t3toolbox.fitting (the frontend dual_jet_to_jet with its safe-mode check included) and unscale_coordinates on
# both tensor classes import and run with no jax, on numpy arrays.
_JETS_SCRIPT = r'''
import sys, warnings
sys.modules['jax'] = None            # make `import jax` raise ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as common
import t3toolbox.backend.jets as jets
import t3toolbox.backend.data_preparation as prep
import t3toolbox.fitting as fitting
import t3toolbox.safety as safety
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
assert not common.jax_available

rng = np.random.default_rng(0)
np.random.seed(0)
shape, W, order = (4, 5, 6), (7,), 2
leaves = lambda value: sum((leaves(v) for v in value), []) if isinstance(value, (tuple, list)) else [value]
def on_numpy(value):
    assert all(isinstance(leaf, np.ndarray) for leaf in leaves(value) if leaf is not None), value
    return value
ww = [rng.standard_normal(W + (n,)) for n in shape]
pp = [rng.standard_normal(W + (n,)) for n in shape]
index = np.stack([rng.integers(0, n, size=W) for n in shape])
scales = [0.5 + rng.random(n) for n in shape]
with warnings.catch_warnings():
    warnings.simplefilter('error')       # nothing here may ask for jax (the jax-absent warning would be an error)
    for kind in ('probe', 'apply', 'entries'):
        first = index if kind == 'entries' else ww
        y = [rng.standard_normal((order + 1,) + W + (n,)) for n in shape] if kind == 'probe'             else rng.standard_normal((order + 1,) + W)
        weight = 0.5 + rng.random(((3,) if kind == 'probe' else ()) + (order + 1,))
        jets.check_sample(kind, order, (first, pp), shape)
        jets.check_data(kind, order, (first, pp), y)
        jets.check_weight(kind, order, weight, 3)
        dual = on_numpy(jets.jet_to_dual_jet(kind, order, y, weight))
        on_numpy(jets.dual_jet_to_jet(kind, order, dual, weight))
        on_numpy(fitting.dual_jet_to_jet(kind, order, dual, weight))
        on_numpy(jets.taylor_to_derivatives(kind, order, jets.derivatives_to_taylor(kind, order, y)))
        on_numpy(jets.prepare_derivative_data(kind, order, y, validation=y))
        on_numpy(fitting.prepare_derivative_data(kind, order, y, weight))
        on_numpy(prep.equalizing_weight(prep.data_block_norms(kind, order, y)))
        on_numpy(prep.rescale_direction(kind, order, (first, pp), y, 2.0))
        if kind != 'entries':
            on_numpy(prep.rescale_coordinates(kind, order, (first, pp), y, scales))
    zero = np.array([1.0, 0.0, 1.0])
    try:
        fitting.dual_jet_to_jet('apply', order, np.ones((3, 7)), zero)
        raise AssertionError('the frontend dual_jet_to_jet must raise in safe mode where a weight is zero')
    except ValueError:
        pass
    with safety.unsafe():
        assert not np.isfinite(fitting.dual_jet_to_jet('apply', order, np.ones((3, 7)), zero)[1]).any()
    x = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    prep.check_coordinate_scales(scales, shape)
    on_numpy(prep.t3_unscale_coordinates(x.data, scales))
    on_numpy(prep.ut3_unscale_coordinates(ux.data, scales)[:2])
    assert isinstance(x.unscale_coordinates(scales).tucker_cores[0], np.ndarray)
    assert isinstance(ux.unscale_coordinates(scales).tucker_supercore, np.ndarray)
    assert np.allclose(ux.unscale_coordinates(scales).to_dense(), x.unscale_coordinates(scales).to_dense())
print('JAX_ABSENT_OK')
'''


class TestJaxAbsentPolicy(unittest.TestCase):
    def test_requests_for_jax_run_on_numpy_with_a_warning(self):
        out = subprocess.run([sys.executable, '-c', _SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])

    def test_the_sampling_definitions_and_the_dense_functions_run_without_jax(self):
        out = subprocess.run([sys.executable, '-c', _DEFINITIONS_SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])

    def test_jets_the_checkers_and_data_preparation_run_without_jax(self):
        out = subprocess.run([sys.executable, '-c', _JETS_SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
