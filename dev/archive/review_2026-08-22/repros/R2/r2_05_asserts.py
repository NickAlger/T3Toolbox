"""Structural checks implemented with assert: behaviour with and without python -O."""
import sys, numpy as np, t3toolbox as t3
from t3toolbox.backend import t3_linalg as L, t3_operations as Op
print('optimize flag (-O):', sys.flags.optimize)
x = t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1))
for label, f in [
    ('__mul__ by array of shape (1,5) (doc: must equal stack_shape+shape)', lambda: (x * np.ones((1, 5))).shape),
    ('t3_add stack (2,) + stack (3,)', lambda: L.t3_add(t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1), stack_shape=(2,)).data, t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1), stack_shape=(3,)).data)[0][0].shape),
    ('t3_inner_product stack (2,) vs (3,)', lambda: L.t3_inner_product(t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1), stack_shape=(2,)).data, t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1), stack_shape=(3,)).data).shape),
    ('t3_sum axis=5 on d=2', lambda: Op.t3_sum(x.data, axis=5)),
    ('t3_sum_stack axis=3 on stack (2,)', lambda: L.t3_sum_stack(t3.TuckerTensorTrain.randn((4, 5), (2, 3), (1, 2, 1), stack_shape=(2,)).data, axis=3)),
    ('x * ones(5) (broadcast-compatible wrong shape)', lambda: (x * np.ones(5)).shape),
]:
    try:
        r = f(); print('  %-70s -> returned %s' % (label, r))
    except Exception as e:
        print('  %-70s -> %s: %s' % (label, type(e).__name__, (str(e).splitlines() or ['(empty message)'])[0][:80]))
