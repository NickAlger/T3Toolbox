import numpy as np, itertools, traceback
import t3toolbox.tucker_tensor_train as t3
np.random.seed(0)
def idxs(C): return list(itertools.product(*[range(c) for c in C]))
def sl(x, i): return t3.TuckerTensorTrain(tuple(B[i] for B in x.tucker_cores), tuple(G[i] for G in x.tt_cores))
def slw(W, i): return t3.T3Weights(tuple(a[i] for a in W.tucker_weights), tuple(a[i] for a in W.tt_weights))
for C in [(), (1,), (3,), (2,3)]:
    x = t3.TuckerTensorTrain.randn((5,6,7),(2,3,4),(1,2,3,1), stack_shape=C).t3svd()[0]
    y = t3.TuckerTensorTrain.randn((5,6,7),(2,3,3),(1,2,3,1), stack_shape=C)
    W = t3.T3Weights.from_t3svd(x)
    print(C, 'W stack', W.stack_shape, 'consistent', W.is_consistent_with(x))
    xw = t3.t3_absorb_weights(x, W)
    for i in idxs(C):
        Wi = slw(W, i)
        assert np.allclose(t3.t3_absorb_weights(sl(x,i), Wi).to_dense(), sl(xw,i).to_dense()), ('absorb', i)
        assert np.allclose(t3.t3_weighted_norm(sl(x,i), Wi), np.asarray(t3.t3_weighted_norm(x, W))[i]), ('wnorm', i)
        assert np.allclose(t3.t3_weighted_inner(sl(x,i), Wi, sl(y,i), Wi), np.asarray(t3.t3_weighted_inner(x, W, y, W))[i]), ('winner', i)
        assert np.allclose(np.asarray(t3.t3_weighted_norm(x, W))[i], np.linalg.norm(t3.t3_absorb_weights(sl(x,i), Wi).to_dense())), 'wnorm dense'
    # concat / kron
    Wc = W.concatenate(W); Wk = W.kronecker(W)
    print('  concat stack', Wc.stack_shape, 'kron stack', Wk.stack_shape, 'rev', W.reverse().stack_shape)
    try:
        print('  Wc consistent with x+x:', Wc.is_consistent_with(x + x), '  Wk consistent with x*x:', Wk.is_consistent_with(x * x), 'x*x ranks', (x*x).ranks, 'Wk ranks', Wk.tucker_ranks, Wk.tt_ranks)
        for i in idxs(C):
            Wci = slw(W,i).concatenate(slw(W,i))
            assert np.allclose(slw(Wc,i).tucker_weights[0], Wci.tucker_weights[0]) and np.allclose(slw(Wc,i).tt_weights[1], Wci.tt_weights[1])
            Wki = slw(W,i).kronecker(slw(W,i))
            assert np.allclose(slw(Wk,i).tucker_weights[0], Wki.tucker_weights[0]) and np.allclose(slw(Wk,i).tt_weights[1], Wki.tt_weights[1])
            assert np.allclose(slw(W.reverse(), i).tucker_weights[0], slw(W,i).reverse().tucker_weights[0])
            assert np.allclose(slw(W.reciprocal(), i).tt_weights[1], slw(W,i).reciprocal().tt_weights[1])
        print('  per-element concat/kron/reverse OK')
    except Exception as e:
        print('  EXC', repr(e)); traceback.print_exc()
    Wt = t3.T3Weights.stack(W.unstack())
    print('  unstack/stack roundtrip:', Wt.stack_shape == C and all(np.allclose(a,b) for a,b in zip(Wt.tucker_weights, W.tucker_weights)))
