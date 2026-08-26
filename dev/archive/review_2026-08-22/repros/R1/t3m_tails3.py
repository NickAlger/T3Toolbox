import numpy as np
from t3toolbox.tucker_tensor_train import TuckerTensorTrain as T3
np.random.seed(0)
def rel(a, b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
cases = {'both r0!=1 (2 vs 3), rd=1': ((2,2,2,1),(3,3,2,1)), 'both rd!=1 (2 vs 3), r0=1': ((1,2,2,2),(1,3,2,3)),
         'both r0=2 equal, rd=1': ((2,2,2,1),(2,3,2,1)), 'x r0=2 only': ((2,2,2,1),(1,3,2,1)), 'y r0=3 only': ((1,2,2,1),(3,3,2,1))}
for label, (rx, ry) in cases.items():
    x = T3.randn((5,6,7), (2,3,2), rx); y = T3.randn((5,6,7), (3,2,2), ry); ref = x.to_dense()*y.to_dense()
    for m in ['form_then_round','inplace_fused','swap']:
        try:
            z = x.t3m(y, method=m, max_tt_ranks=3, max_tucker_ranks=3)
            print(f'  {label:28s} {m:16s} ranks {z.ranks} relerr {rel(z.to_dense(), ref):.2e}')
        except Exception as e:
            print(f'  {label:28s} {m:16s} RAISED {type(e).__name__}: {str(e)[:80]}')
