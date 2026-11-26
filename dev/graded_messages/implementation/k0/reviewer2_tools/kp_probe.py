import numpy as np, sys
import t3toolbox.backend.probing as probing
from t3toolbox.backend.t3_constructors import t3_corewise_randn
np.random.seed(0)
x = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1)); dd = t3_corewise_randn((4, 5, 6), (2, 3, 2), (1, 2, 2, 1))
ww = [np.random.randn(3, n) for n in (4, 5, 6)]; c = [np.random.randn(3, n) for n in (4, 5, 6)]
KB, KG = probing.t3_probe_corewise_hessian(x, ww, c, dd)
print(probing.__file__.split('/')[-4], repr(float(sum(np.sum(a) for a in KB + KG))))
