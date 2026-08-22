"""R3: t3_share_tucker_factors -- (i) on tied input equals grouped t3svd; (ii) quasi-optimality
err(share) <= C(d) * best, checked against a provable LOWER bound on best (tail of the group
concatenated-matricization spectrum / sqrt(k), tails of TT unfoldings) and against a refined upper
estimate of best (Riemannian refinement from share's output is too slow here; we use ALS-free lower bound only)."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.sharing as sh
import t3toolbox.safety as safety

np.random.seed(0)
def C(d): return np.sqrt(d) + np.sqrt(d) * np.sqrt(d - 1) + np.sqrt(d - 1)

viol = 0; tot = 0; worst = 0.0
for trial in range(40):
    d = 3
    shape = (6, 6, 5); labels = (0, 0, 1)
    x = t3.TuckerTensorTrain.randn(shape, (4, 4, 3), (1, 4, 3, 1))
    # make a near-shared tensor + noise so the best shared approx is nontrivial
    T = np.asarray(x.to_dense())
    caps_tk = (2, 2, 2); caps_tt = (1, 2, 2, 1)
    xs = x.share(labels, max_tucker_ranks=caps_tk, max_tt_ranks=caps_tt)
    err = np.linalg.norm(np.asarray(xs.to_dense()) - T)
    # lower bound on the best shared approximation error at these ranks
    lb = 0.0
    groups = sh.validate_sharing(labels, shape)
    for g in groups:
        mats = [np.moveaxis(T, i, 0).reshape(T.shape[i], -1) for i in g]
        s = np.linalg.svd(np.concatenate(mats, axis=1), compute_uv=False)
        lb = max(lb, np.linalg.norm(s[caps_tk[g[0]]:]) / np.sqrt(len(g)))
    for k in range(1, d):
        s = np.linalg.svd(T.reshape(int(np.prod(shape[:k])), -1), compute_uv=False)
        lb = max(lb, np.linalg.norm(s[caps_tt[k]:]))
    tot += 1
    worst = max(worst, err / lb)
    if err > C(d) * lb:
        viol += 1
print('quasi-optimality vs lower bound: violations %d/%d ; worst err/LB = %.3f ; C(3) = %.3f' % (viol, tot, worst, C(3)))

# (i) tied input: share == grouped t3svd spectra and tensor
x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 3, 2, 1))
tk, tt = x.data
xt = t3.TuckerTensorTrain((tk[0], tk[0], tk[2]), tt)
import t3toolbox.backend.t3_svd as bsvd
a, ska, sta = bsvd.t3_share_tucker_factors(xt.data, (0, 0, 1), max_tucker_ranks=2, max_tt_ranks=2)
with safety.unsafe():
    b, skb, stb = xt.t3svd(sharing=(0, 0, 1), max_tucker_ranks=2, max_tt_ranks=2)
print('tied input: share tensor == grouped t3svd tensor?', np.allclose(t3.TuckerTensorTrain(*a).to_dense(), b.to_dense()),
      '; spectra equal?', all(np.allclose(p, q) for p, q in zip(ska, skb)), all(np.allclose(p, q) for p, q in zip(sta, stb)))

# stacked share with caps vs per element
xs = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 3, 2, 1), stack_shape=(2,))
ys = xs.share((0, 0, 1), max_tucker_ranks=2, max_tt_ranks=2)
ok = all(np.allclose(np.asarray(ys.to_dense())[k], np.asarray(xs.unstack()[k].share((0, 0, 1), max_tucker_ranks=2, max_tt_ranks=2).to_dense())) for k in range(2))
print('stacked share == per-element share:', ok)
