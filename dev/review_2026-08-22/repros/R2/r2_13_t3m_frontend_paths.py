"""Frontend-reachable paths into t3m with unsquashed tails, and max_tt_ranks given as a numpy array to method='swap'."""
import numpy as np, t3toolbox as t3
np.random.seed(0)
A = t3.TuckerTensorTrain.randn((3, 4, 5, 6), (2, 2, 3, 2), (1, 2, 2, 2, 1))
B = t3.TuckerTensorTrain.randn((3, 4, 5, 6), (2, 3, 2, 2), (1, 2, 2, 2, 1))
a, b = A.segment(1, 3), B.segment(1, 3)         # documented API; segments carry boundary ranks 2
print('segment tt_ranks:', a.tt_ranks, b.tt_ranks)
ref = a.to_dense() * b.to_dense()
for m in ('form_then_round', 'inplace_fused', 'swap'):
    P = a.t3m(b, method=m, max_tt_ranks=50, max_tucker_ranks=50)
    print('  %-16s relerr vs dense product: %.2e' % (m, np.linalg.norm(P.to_dense() - ref) / np.linalg.norm(ref)))
P = (a * b)
print('  a * b (form_then_round, no truncation) relerr: %.2e' % (np.linalg.norm(P.to_dense() - ref) / np.linalg.norm(ref)))
# max_tt_ranks as numpy array for swap (need_cleanup uses isinstance(..., Sequence))
seq = [1, 2, 3, 2, 1]
P1 = A.t3m(B, method='swap', max_tt_ranks=seq)
try:
    P2 = A.t3m(B, method='swap', max_tt_ranks=np.array(seq))
    print('swap max_tt_ranks list  -> tt_ranks', P1.tt_ranks)
    print('swap max_tt_ranks array -> tt_ranks', P2.tt_ranks, '(cap was %s)' % seq)
except Exception as e:
    print('swap max_tt_ranks array raised', type(e).__name__, str(e)[:100])
try:
    P3 = A.t3m(B, method='inplace_fused', max_tt_ranks=np.array(seq)); print('fused max_tt_ranks array -> tt_ranks', P3.tt_ranks)
except Exception as e:
    print('fused max_tt_ranks array raised', type(e).__name__, str(e)[:100])
