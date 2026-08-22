"""R3: ut3svd(sharing=) exact output masks vs ragged grouped t3svd ranks, incl. varying-rank stacks and
per-element caps; plus real parts vs ragged; plus spectra masked correctly."""
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.sharing as sh
import t3toolbox.safety as safety

rng = np.random.default_rng(7)
def tied(shape, tk, tt, labels):
    x = t3.TuckerTensorTrain.randn(shape, tk, tt)
    groups = sh.validate_sharing(labels, shape)
    tkc = list(x.data[0])
    for g in groups:
        for i in g[1:]:
            tkc[i] = tkc[g[0]]
    return t3.TuckerTensorTrain(tuple(tkc), x.data[1])

bad_mask = 0; bad_val = 0; bad_spec = 0; tot = 0
for trial in range(60):
    d = int(rng.integers(2, 5))
    while True:
        labels = tuple(int(v) for v in rng.integers(0, max(1, d - 1), size=d))
        groups = sh._groups_from_labels(labels, d)
        if any(len(g) > 1 for g in groups):
            break
    shape = [0] * d
    for g in groups:
        N = int(rng.integers(3, 7))
        for i in g: shape[i] = N
    shape = tuple(shape)
    xs = []
    for k in range(2):   # two stack elements with DIFFERENT ranks
        tk = [0] * d
        for g in groups:
            n = int(rng.integers(1, 5))
            for i in g: tk[i] = n
        tt = (1,) + tuple(int(v) for v in rng.integers(1, 6, size=d - 1)) + (1,)
        xs.append(tied(shape, tuple(tk), tt, labels))
    u = ut3.UniformTuckerTensorTrain.stack([ut3.UniformTuckerTensorTrain.from_t3(x, n=5, r=6) for x in xs])
    # per-element caps: array (d,)+(2,) and (d+1,)+(2,)
    cap_tk = np.zeros((d, 2), dtype=int); cap_tt = np.ones((d + 1, 2), dtype=int)
    for k in range(2):
        for g in groups:
            c = int(rng.integers(1, 5))
            for i in g: cap_tk[i, k] = c
        cap_tt[1:d, k] = rng.integers(1, 6, size=d - 1)
    with safety.unsafe():
        yu, su_tk, su_tt = u.t3svd(max_tucker_ranks=cap_tk, max_tt_ranks=cap_tt, sharing=labels)
    for k in range(2):
        with safety.unsafe():
            yk, sk_tk, sk_tt = xs[k].t3svd(max_tucker_ranks=tuple(int(v) for v in cap_tk[:, k]),
                                           max_tt_ranks=tuple(int(v) for v in cap_tt[:, k]), sharing=labels)
        tot += 1
        mk = (tuple(int(v) for v in yu.data[3][0][:, k].sum(-1)), tuple(int(v) for v in yu.data[3][1][:, k].sum(-1)))
        if mk != (yk.tucker_ranks, yk.tt_ranks):
            bad_mask += 1
            if bad_mask <= 5:
                print('MASK MISMATCH', shape, xs[k].ranks, labels, 'caps', tuple(cap_tk[:, k]), tuple(cap_tt[:, k]), 'uniform', mk, 'ragged', (yk.tucker_ranks, yk.tt_ranks))
        yk_u = yu.unstack()[k].to_t3()
        if not np.allclose(np.asarray(yk_u.to_dense()), np.asarray(yk.to_dense()), atol=1e-8):
            bad_val += 1
            if bad_val <= 5:
                print('VALUE MISMATCH', shape, xs[k].ranks, labels, 'caps', tuple(cap_tk[:, k]), tuple(cap_tt[:, k]),
                      'rel err %.2e' % (np.linalg.norm(np.asarray(yk_u.to_dense()) - np.asarray(yk.to_dense())) / np.linalg.norm(np.asarray(yk.to_dense()))))
        # spectra: real prefix equals ragged, padding zero
        for i in range(d):
            s_u = np.asarray(su_tk[i, k]); s_r = np.asarray(sk_tk[i])
            if not (np.allclose(s_u[:len(s_r)], s_r, atol=1e-8) and np.all(s_u[len(s_r):] == 0)):
                bad_spec += 1; break
        for i in range(d + 1):
            s_u = np.asarray(su_tt[i, k]); s_r = np.asarray(sk_tt[i])
            if not (np.allclose(s_u[:len(s_r)], s_r, atol=1e-8) and np.all(s_u[len(s_r):] == 0)):
                bad_spec += 1
                if bad_spec <= 3:
                    print('SPECTRUM MISMATCH tt', i, shape, xs[k].ranks, labels, 'caps', tuple(cap_tk[:, k]), tuple(cap_tt[:, k]), 'uniform', s_u, 'ragged', s_r)
                break
print('elements: %d ; mask mismatches %d ; value mismatches %d ; spectrum mismatches %d' % (tot, bad_mask, bad_val, bad_spec))
