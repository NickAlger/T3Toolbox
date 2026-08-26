"""Doc says rtol/atol remove singular values with sigma < max(atol, rtol*sigma1). Code uses a tail-Frobenius criterion."""
import numpy as np
from t3toolbox.backend.linalg import truncated_svd, left_svd
np.random.seed(0)
Q1, _ = np.linalg.qr(np.random.randn(7, 5)); Q2, _ = np.linalg.qr(np.random.randn(9, 5))
s = np.array([1.0, 0.08, 0.08, 0.08, 0.08])
A = (Q1 * s) @ Q2.T
U, ss, Vt = truncated_svd(A, rtol=0.1)
print('sigma =', s)
print('rtol=0.1: doc criterion (sigma < rtol*sigma1) would keep', int(np.sum(s >= 0.1 * s[0])), '; code keeps', ss.shape[-1])
print('  code criterion: tail Frobenius norms', np.sqrt(np.cumsum(s[::-1]**2))[::-1].round(4), '>= rtol*||A||_F =', round(0.1*np.linalg.norm(s), 4))
s2 = np.array([1.0, 0.3, 0.3])
Q1, _ = np.linalg.qr(np.random.randn(6, 3)); Q2, _ = np.linalg.qr(np.random.randn(4, 3))
B = (Q1 * s2) @ Q2.T
U, ss, Vt = truncated_svd(B, atol=0.35)
print('sigma =', s2, 'atol=0.35: doc criterion would keep', int(np.sum(s2 >= 0.35)), '; code keeps', ss.shape[-1])
# Directional wrappers inherit the same signature comment
U, ss, Vt = left_svd(A.reshape(7, 1, 9), rtol=0.1)
print('left_svd rtol=0.1 keeps', ss.shape[-1], '(same tail-Frobenius rule)')
# min_rank larger than min(N,M): silently fewer than min_rank
U, ss, Vt = truncated_svd(np.random.randn(3, 4), min_rank=6)
print('min_rank=6 on a 3x4 matrix -> kept rank', ss.shape[-1], '(no error)')
