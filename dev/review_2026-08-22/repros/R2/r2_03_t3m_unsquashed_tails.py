"""The three t3m strategies on inputs whose boundary TT ranks r0, rd != 1 (allowed by validate; squash_tails is optional)."""
import numpy as np, t3toolbox as t3
np.random.seed(0)
def run(ttrA, ttrB, label):
    A = t3.TuckerTensorTrain.randn((4, 5, 6), (2, 3, 2), ttrA)
    B = t3.TuckerTensorTrain.randn((4, 5, 6), (3, 2, 2), ttrB)
    ref = A.to_dense() * B.to_dense()
    print('---', label, 'A.tt_ranks', A.tt_ranks, 'B.tt_ranks', B.tt_ranks)
    for m in ('form_then_round', 'inplace_fused', 'swap'):
        for kw in (dict(max_tucker_ranks=100, max_tt_ranks=100), dict(rtol=1e-12)):
            try:
                P = A.t3m(B, method=m, **kw)
                err = np.linalg.norm(P.to_dense() - ref) / np.linalg.norm(ref)
                print('  %-16s %-42s tt_ranks=%s  relerr=%.2e %s' % (m, kw, P.tt_ranks, err, '<-- WRONG' if err > 1e-8 else ''))
            except Exception as e:
                print('  %-16s %-42s RAISED %s: %s' % (m, kw, type(e).__name__, str(e).splitlines()[0][:90]))
run((2, 3, 2, 2), (2, 2, 3, 2), 'r0=rd=2 on both (equal boundary ranks)')
run((1, 3, 2, 2), (1, 2, 3, 3), 'r0=1, rd differs (2 vs 3)')
run((3, 3, 2, 1), (2, 2, 3, 1), 'rd=1, r0 differs (3 vs 2)')
run((1, 3, 2, 1), (1, 2, 3, 1), 'squashed (control)')
