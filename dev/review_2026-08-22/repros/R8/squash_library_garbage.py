"""R8: can a LIBRARY-produced UT3 (no synthetic corruption) carry garbage in the padded boundary-bond
slots that ut3_squash_tails / + / - / sum_stack then sum into the real slot?  The SVD completion case:
structural rank > numerical rank (e.g. x + x, then t3svd / rank_adjustment_sweep / up_orthogonalize)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3

np.random.seed(0)
def relerr(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

x = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1))
y = t3.TuckerTensorTrain.randn((4, 5, 3), (3, 2, 2), (1, 3, 2, 1))
ux, uy = ut3.UniformTuckerTensorTrain.from_t3(x), ut3.UniformTuckerTensorTrain.from_t3(y)

def padded_boundary_garbage(u):
    tkm, ttm = u.masks.data
    G0, Gf = u.tt_supercore[0], u.tt_supercore[-1]
    return float(max(np.abs(G0[~ttm[0]]).max(initial=0.0), np.abs(Gf[..., ~ttm[-1]]).max(initial=0.0)))

z = ux + ux                                   # structural ranks double, numerical ranks do not
zr = x + x
for name, uz, rz in [
    ('t3svd(x+x)',                 z.t3svd()[0],                         zr.t3svd()[0]),
    ('rank_adjustment_sweep(t3svd)', z.t3svd()[0].rank_adjustment_sweep(), zr.t3svd()[0].rank_adjustment_sweep()),
    ('up_orthogonalize(x+x)',      z.up_orthogonalize_tt_cores(),        zr.up_orthogonalize_tt_cores()),
    ('left_orthogonalize(x+x)',    z.left_orthogonalize_tt_cores(),      zr.left_orthogonalize_tt_cores()),
    ('right_orthogonalize(x+x)',   z.right_orthogonalize_tt_cores(),     zr.right_orthogonalize_tt_cores()),
]:
    g = padded_boundary_garbage(uz)
    e_dense  = relerr(uz.to_dense(), rz.to_dense())                  # the object itself is fine...
    e_squash = relerr(uz.squash_tails().to_dense(), rz.to_dense())   # ...until a summing op eats the padding
    e_add    = relerr((uz + uy).to_dense(), (rz + y).to_dense())
    e_sub    = relerr((uz - uy).to_dense(), (rz - y).to_dense())
    e_norm   = abs(float(uz.norm()) - float(rz.norm())) / float(rz.norm())
    print('%-32s boundary-pad garbage max=%.3g | to_dense %.1e | squash_tails %.1e | + %.1e | - %.1e | norm %.1e'
          % (name, g, e_dense, e_squash, e_add, e_sub, e_norm))

# stacked version: sum_stack after t3svd of a stack with rank-deficient elements
xs = t3.TuckerTensorTrain.randn((4, 5, 3), (2, 3, 2), (1, 2, 3, 1), stack_shape=(2,))
us = ut3.UniformTuckerTensorTrain.from_t3(xs)
zs = (us + us).t3svd()[0]
zsr = (xs + xs).t3svd()[0]
print('%-32s boundary-pad garbage max=%.3g | sum_stack rel err %.1e' % (
    'sum_stack(t3svd(xs+xs))', padded_boundary_garbage(zs),
    relerr(zs.sum_stack().to_dense(), zsr.to_dense().sum(axis=0))))
