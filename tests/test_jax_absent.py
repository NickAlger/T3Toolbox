"""The jax-absent policy (review 2026-08-22, C10): a request for jax on a machine without jax runs on numpy
and WARNS -- never a bare NameError, never a silent numpy result, never an error from use_jit=True.
jax IS installed in the test environment, so the check runs in a subprocess that hides it.

T28 (2026.3.0, the exact Hessian): every class the exact mode adds registers its jax pytree only under
``if jax_available:``, so in the jax-hidden subprocess each one is CONSTRUCTED (not merely imported) --
``LiftFactors`` (``fv_lift_factors``), ``ManifoldCurvatureData`` (``curvature_precompute``), the backend
``CurvatureState`` and an exact ``fitting.LocalModel`` (``local_model(..., hessian='exact')``),
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
import t3toolbox.backend.jets as jets_
derivatives = lambda y: jets_.taylor_to_derivatives('probe', (order,), y.probe(jets_.affine_jet(wwe, ppe), (order,)))
data = derivatives(target)
frame, _ = bvf.t3_orthogonal_representations(xe)
lift = fvo.fv_lift_factors(frame.data, 1e-6)
assert type(lift) is fvo.LiftFactors
geom = bgeo.ManifoldGeometryOps()
line = jets_.affine_jet(wwe, ppe)
oracle = bfit.LeastSquaresOracle()
batch = bfit.LeastSquaresBatch('probe', (order,), line, target.probe(line, (order,)))     # the Taylor coefficients
lm = bopt.local_model(geom, oracle, batch, xe.data, hessian='exact')
assert type(lm.curvature) is bopt.CurvatureState and lm.is_exact
assert type(lm.curvature.geometry_data) is bgeo.ManifoldCurvatureData
cdata = geom.curvature_precompute(lm.frame, lm.curvature.misfit_gradient, lm.geom_aux, None)
assert type(cdata) is bgeo.ManifoldCurvatureData
state = bopt.CurvatureState(lm.curvature.misfit_gradient, cdata)
assert type(state) is bopt.CurvatureState
model = fitting.local_model(t3m.MANIFOLD, oracle, batch, xe, hessian='exact')
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
line = [np.stack([w, p]) for w, p in zip(ww, pp)]      # the input jet of w + s p, built by hand: jets is not imported
arguments = dict(variations=([rng.standard_normal((nU[i], N[i])) for i in range(3)],
                                      [rng.standard_normal((rL[i], nD[i], rR[i + 1])) for i in range(3)]),
                 core_directions=([rng.standard_normal(B.shape) for B in D], [rng.standard_normal(G.shape) for G in L]))
leaves = lambda value: sum((leaves(v) for v in value), []) if isinstance(value, (tuple, list)) else [value]
for kind, sample, order in (('probe', line, (2,)), ('apply', line, (2,)), ('entries', index, ())):
    F = tuple(n + 1 for n in order)
    y = [rng.standard_normal(F + (5, n)) for n in N] if kind == 'probe' else rng.standard_normal(F + (5,))
    for name in sdef.__all__:
        value = getattr(sdef, name)(kind, (D, U, L, R), sample, order=order, y=y, **arguments)
        assert all(isinstance(leaf, np.ndarray) or np.isscalar(leaf) for leaf in leaves(value)), (kind, name)
T = rng.standard_normal(N)
assert probing.dense_probe(T, line, (2,))[0].shape == (3, 5, 4)
assert apply.dense_apply(T, line, (2,)).shape == (3, 5) and entries.dense_entries(T, index).shape == (5,)
print('JAX_ABSENT_OK')
'''


# Jets, the checkers and data preparation (graded messages, P3; in the jet shape since P4d): the two backend
# modules, their reach from t3toolbox.fitting and t3toolbox, the data metric (M1), and unscale_coordinates on both
# tensor classes import and run with no jax, on numpy arrays.
_JETS_SCRIPT = r'''
import sys, warnings
sys.modules['jax'] = None            # make `import jax` raise ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as common
import t3toolbox.backend.jets as jets
import t3toolbox.backend.data_preparation as prep
import t3toolbox.fitting as fitting
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
assert not common.jax_available

rng = np.random.default_rng(0)
np.random.seed(0)
shape, W = (4, 5, 6), (7,)
leaves = lambda value: sum((leaves(v) for v in value), []) if isinstance(value, (tuple, list)) else [value]
def on_numpy(value):
    assert all(isinstance(leaf, np.ndarray) for leaf in leaves(value) if leaf is not None), value
    return value
ww = [rng.standard_normal(W + (n,)) for n in shape]
pp = [rng.standard_normal(W + (n,)) for n in shape]
qq = [rng.standard_normal(W + (n,)) for n in shape]
index = np.stack([rng.integers(0, n, size=W) for n in shape])
scales = [0.5 + rng.random(n) for n in shape]
with warnings.catch_warnings():
    warnings.simplefilter('error')       # nothing here may ask for jax (the jax-absent warning would be an error)
    for kind, order in (('probe', (2,)), ('apply', (2,)), ('probe', (1, 2)), ('apply', (1, 2)), ('entries', ())):
        F = tuple(n + 1 for n in order)
        sample = index if kind == 'entries' else (jets.affine_jet(ww, pp) if len(order) == 1 else jets.affine_jet(ww, pp, qq))
        y = [rng.standard_normal(F + W + (n,)) for n in shape] if kind == 'probe' else rng.standard_normal(F + W)
        blocks = 0.5 + rng.random(((3,) if kind == 'probe' else ()) + F)
        weight = on_numpy(fitting.blocks_to_weight(kind, order, blocks, y))     # of the type of z
        jets.check_sample(kind, order, sample, shape)
        jets.check_data(kind, order, sample, y)
        jets.check_weight(kind, order, sample, weight)
        dual = on_numpy(jets.jet_to_dual_jet(kind, order, y, weight))
        on_numpy(jets.dual_jet_to_jet(kind, order, dual, weight))
        on_numpy(fitting.data_metric(y, weight, 2.0))
        on_numpy(jets.taylor_to_derivatives(kind, order, jets.derivatives_to_taylor(kind, order, y)))
        on_numpy(jets.prepare_derivative_data(kind, order, y, validation=y))
        on_numpy(fitting.prepare_derivative_data(kind, order, y, weight))
        on_numpy(prep.equalizing_weight(prep.data_block_norms(kind, order, y)))
        if kind != 'entries':
            on_numpy(prep.rescale_direction(kind, order, sample, y, 2.0))
            on_numpy(prep.rescale_coordinates(kind, order, sample, y, scales))
            on_numpy(fitting.rescale_coordinates(kind, order, sample, y, scales))
    zero = np.array([[1.0], [0.0], [1.0]])                  # per slot, of the type of z
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # the backend divides by zero silently by design
        assert not np.isfinite(jets.dual_jet_to_jet('apply', (2,), np.ones((3, 7)), zero)[1]).any()
    x = t3.TuckerTensorTrain.randn(shape, (2, 2, 2), (1, 2, 2, 1))
    ux = ut3.UniformTuckerTensorTrain.from_t3(x)
    prep.check_coordinate_scales(scales, shape)
    on_numpy(prep.t3_unscale_coordinates(x.data, scales))
    on_numpy(prep.ut3_unscale_coordinates(ux.data, scales)[:2])
    assert isinstance(x.unscale_coordinates(scales).tucker_cores[0], np.ndarray)
    assert isinstance(ux.unscale_coordinates(scales).tucker_supercore, np.ndarray)
    # the input jets (graded messages, P4b): numpy in, numpy out, with no jax
    line = on_numpy(jets.affine_jet(ww, pp))
    assert [x.shape for x in line] == [(2,) + W + (n,) for n in shape]
    assert on_numpy(jets.affine_jet(np.stack(ww[:1] * 2), np.stack(pp[:1] * 2))).shape == (2, 2) + W + (4,)
    one_hot = on_numpy(jets.index_to_one_hot(index, shape))
    assert [x.dtype for x in one_hot] == [np.float64] * 3
    import t3toolbox
    assert t3toolbox.affine_jet is jets.affine_jet and t3toolbox.index_to_one_hot is jets.index_to_one_hot
    assert np.allclose(ux.unscale_coordinates(scales).to_dense(), x.unscale_coordinates(scales).to_dense())
print('JAX_ABSENT_OK')
'''

# Graded messages, P4: the checkers of a frame (ragged and uniform, backend and frontend) run with no jax, on numpy
# arrays, and ask for no jax.
_FRAME_CHECKERS_SCRIPT = r'''
import sys, warnings
sys.modules['jax'] = None            # make `import jax` raise ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as common
import t3toolbox.backend.fv_operations as fvo
import t3toolbox.backend.ufv_operations as ufvo
import t3toolbox.frame_variations_format as bvf
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
assert not common.jax_available

np.random.seed(0)
with warnings.catch_warnings():
    warnings.simplefilter('error')       # nothing here may ask for jax (the jax-absent warning would be an error)
    x = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1), stack_shape=(2,))
    frame, centers = bvf.t3_orthogonal_representations(x)
    uframe, ucenters = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(x))
    residuals = [fvo.fv_frame_consistency_residual(frame.data),
                 fvo.fv_centers_consistency_residual(frame.data, centers.data),
                 ufvo.ufv_frame_consistency_residual(uframe.data),
                 ufvo.ufv_centers_consistency_residual(uframe.data, tuple(ucenters.supercores)),
                 frame.consistency_residual, uframe.consistency_residual]
    for residual in residuals:
        assert isinstance(residual, np.ndarray) and residual.shape == (2,) and np.all(residual < 1e-12), residual
    assert frame.is_consistent().all() and uframe.is_consistent().all()
print('JAX_ABSENT_OK')
'''

# Graded messages, P5: the sampling engine and HeldVariables import and run with no jax, on numpy arrays, and ask
# for no jax (HeldVariables is then an ordinary object: no pytree registration).
_ENGINE_SCRIPT = r'''
import sys, warnings
sys.modules['jax'] = None            # make `import jax` raise ImportError
sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as common
import t3toolbox.backend.jets as jets
import t3toolbox.backend.sampling_variables as sv
assert not common.jax_available

rng = np.random.default_rng(0)
N, nD, nU, rL, rR = (4, 5, 3), (2, 3, 2), (3, 2, 3), (1, 2, 3, 1), (1, 3, 2, 1)
frame = ([rng.standard_normal((nD[i], N[i])) for i in range(3)],
         [rng.standard_normal((rL[i], nU[i], rR[i + 1])) for i in range(3)],
         [rng.standard_normal((rL[i], nD[i], rL[i + 1])) for i in range(3)],
         [rng.standard_normal((rR[i], nD[i], rR[i + 1])) for i in range(3)])
ww = [rng.standard_normal((2, n)) for n in N]
pp = [rng.standard_normal((2, n)) for n in N]
y = [rng.standard_normal((3, 2, n)) for n in N]
with warnings.catch_warnings():
    warnings.simplefilter('error')       # nothing here may ask for jax (the jax-absent warning would be an error)
    line = jets.affine_jet(ww, pp)
    mu, z, f, g = sv.sampling_variables('probe', frame, line, ('mu', 'z', 'f', 'g'), order=(2,), y=y)
    assert all(isinstance(a, np.ndarray) for a in mu + z + g[0] + g[1]) and isinstance(f, np.ndarray), (mu, z, f)
    variations = ([rng.standard_normal((2, nU[i], N[i])) for i in range(3)],
                  [rng.standard_normal((2, rL[i], nD[i], rR[i + 1])) for i in range(3)])
    Jp, JtJp = sv.sampling_variables('probe', frame, line, ('Jp', 'JtJp'), order=(2,), variations=variations)
    assert all(isinstance(a, np.ndarray) for a in Jp + JtJp[0] + JtJp[1]) and Jp[0].shape == (3, 2, 2, 4)
    core_directions = ([rng.standard_normal((2, nD[i], N[i])) for i in range(3)],
                       [rng.standard_normal((2, rL[i], nD[i], rL[i + 1])) for i in range(3)])
    (Kp,) = sv.sampling_variables('probe', frame, line, ('Kp',), order=(2,), y=y, core_directions=core_directions)
    nu, g = sv.sampling_variables('probe', (frame[0], frame[2]), line, ('nu', 'g'), order=(2,), y=y)
    assert all(isinstance(a, np.ndarray) for a in Kp[0] + Kp[1] + nu + g[0] + g[1]) and Kp[0][0].shape == (2, 2, 4)
    y_apply = rng.standard_normal((3, 2))
    z_a, Kp_a, JtJp_a = sv.sampling_variables('apply', frame, line, ('z', 'Kp', 'JtJp'), order=(2,), y=y_apply,
                                              variations=variations, core_directions=core_directions)
    assert isinstance(z_a, np.ndarray) and z_a.shape == (3, 2), z_a
    assert all(isinstance(a, np.ndarray) for a in Kp_a[0] + Kp_a[1] + JtJp_a[0] + JtJp_a[1])
    index = np.array([[0, 3], [1, 1], [2, 2]])
    z_e, g_e = sv.sampling_variables('entries', frame, index, ('z', 'g'), y=rng.standard_normal(2))
    assert isinstance(z_e, np.ndarray) and z_e.shape == (2,) and all(isinstance(a, np.ndarray) for a in g_e[0])
    held = sv.HeldVariables(mu=mu)
    assert held.keys() == ('mu',) and 'mu' in held and len(held) == 1
    assert held.with_values(z=z).keys() == ('mu', 'z') and held.keys() == ('mu',)
    (f2,) = sv.sampling_variables('probe', frame, line, ('f',), order=(2,), y=y, held=held.with_values(z=z))
    assert float(f2) == float(f), (f2, f)
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

    def test_the_frame_checkers_run_without_jax(self):
        out = subprocess.run([sys.executable, '-c', _FRAME_CHECKERS_SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])

    def test_the_sampling_engine_and_held_variables_run_without_jax(self):
        out = subprocess.run([sys.executable, '-c', _ENGINE_SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
