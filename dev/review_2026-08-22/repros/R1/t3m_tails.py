import numpy as np
from t3toolbox.tucker_tensor_train import TuckerTensorTrain as T3
np.random.seed(0)
def rel(a, b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
for stack in [(), (2,)]:
    x = T3.randn((5,6,7), (2,3,2), (2,2,2,3), stack_shape=stack)
    y = T3.randn((5,6,7), (3,2,2), (3,3,2,2), stack_shape=stack)
    ref = x.to_dense() * y.to_dense()
    print(f'--- stack={stack}  x.tt_ranks={x.tt_ranks} y.tt_ranks={y.tt_ranks}')
    z = x * y
    print('  x*y            ranks', z.ranks, 'relerr vs dense product', rel(z.to_dense(), ref))
    for m in ['form_then_round', 'inplace_fused', 'swap']:
        z = x.t3m(y, method=m)
        print(f'  t3m {m:16s} no-trunc  ranks', z.ranks, 'relerr', rel(z.to_dense(), ref))
    for m in ['form_then_round', 'inplace_fused', 'swap']:
        z = x.t3m(y, method=m, max_tt_ranks=30, max_tucker_ranks=30)   # caps that do NOT bind
        print(f'  t3m {m:16s} loose-cap ranks', z.ranks, 'relerr', rel(z.to_dense(), ref))
    for m in ['form_then_round', 'inplace_fused', 'swap']:
        z = x.t3m(y, method=m, max_tt_ranks=3, max_tucker_ranks=3)
        print(f'  t3m {m:16s} cap=3     ranks', z.ranks, 'relerr', rel(z.to_dense(), ref), ' (x.squash*y.squash ref)')
    if stack == ():
        for m in ['form_then_round', 'inplace_fused', 'swap']:
            z = x.t3m(y, method=m, rtol=1e-12)
            print(f'  t3m {m:16s} rtol=1e-12 ranks', z.ranks, 'relerr', rel(z.to_dense(), ref))
# same, squashed tails first
print('--- squashed tails first')
x = T3.randn((5,6,7), (2,3,2), (2,2,2,3)).squash_tails(); y = T3.randn((5,6,7), (3,2,2), (3,3,2,2)).squash_tails()
ref = x.to_dense() * y.to_dense()
for m in ['form_then_round', 'inplace_fused', 'swap']:
    z = x.t3m(y, method=m, max_tt_ranks=30, max_tucker_ranks=30)
    print(f'  t3m {m:16s} loose-cap ranks', z.ranks, 'relerr', rel(z.to_dense(), ref))
