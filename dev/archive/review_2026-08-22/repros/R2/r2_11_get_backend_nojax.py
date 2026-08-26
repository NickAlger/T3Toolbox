"""Characterize behaviour when jax is absent (simulated by blocking the import)."""
import sys
sys.modules['jax'] = None; sys.modules['jax.numpy'] = None
import numpy as np
import t3toolbox.backend.common as c
import t3toolbox as t3
print('jax_available:', c.jax_available)
def show(label, f):
    try:
        r = f(); print('%-60s -> %s' % (label, r))
    except Exception as e:
        print('%-60s -> %s: %s' % (label, type(e).__name__, e))
show('get_backend(False, True)', lambda: c.get_backend(False, True))
show('get_backend(True, True)', lambda: c.get_backend(True, True))
show('TuckerTensorTrain.randn(use_jax=True)', lambda: t3.TuckerTensorTrain.randn((3, 4), (2, 2), (1, 2, 1), use_jax=True))
show('TuckerTensorTrain.zeros(use_jax=True)', lambda: t3.TuckerTensorTrain.zeros((3, 4), (2, 2), (1, 2, 1), use_jax=True))
show('common.randn(3, use_jax=True)', lambda: c.randn(3, use_jax=True))
show('common.to_jax(np.ones(2)) (silent numpy)', lambda: type(c.to_jax(np.ones(2))).__name__)
show('common.tree_to_jax((np.ones(2),))', lambda: c.tree_to_jax((np.ones(2),)))
show('common.xcat on numpy arrays', lambda: c.xcat(np.ones(2), np.ones(3)).shape)
show('jax_map is numpy_map', lambda: c.jax_map is c.numpy_map)
from t3toolbox.backend import t3_constructors as K
show('t3_corewise_randn(use_jax=True)', lambda: K.t3_corewise_randn((3,), (2,), (1, 1), use_jax=True))
