"""The jax-absent policy (review 2026-08-22, C10): a request for jax on a machine without jax runs on numpy
and WARNS -- never a bare NameError, never a silent numpy result, never an error from use_jit=True.
jax IS installed in the test environment, so the check runs in a subprocess that hides it."""
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
print('JAX_ABSENT_OK')
'''


class TestJaxAbsentPolicy(unittest.TestCase):
    def test_requests_for_jax_run_on_numpy_with_a_warning(self):
        out = subprocess.run([sys.executable, '-c', _SCRIPT], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONPATH': os.getcwd()})
        self.assertIn('JAX_ABSENT_OK', out.stdout, out.stderr[-2000:])


if __name__ == '__main__':
    unittest.main()
