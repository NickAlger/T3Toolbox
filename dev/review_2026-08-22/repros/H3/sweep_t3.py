"""H3 sweep 1: TuckerTensorTrain ops over stack_shape in {(), (1,), (3,), (2,3)} vs per-element unstacked.
Asymmetric structure. Compare to_dense / norms, never cores.
"""
import itertools, traceback, sys
import numpy as np
import t3toolbox as t3t
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.corewise as cw

np.random.seed(0)
STRUCTS = [
    ((5, 6, 7), (2, 3, 4), (1, 2, 3, 1)),
    ((5, 6, 7, 3), (2, 3, 4, 2), (1, 2, 3, 2, 1)),
    ((6,), (3,), (1, 1)),
    ((5, 4), (3, 2), (1, 2, 1)),
]
STACKS = [(), (1,), (3,), (2, 3)]
WS = [(), (4,), (2, 2)]
FAILS = []
OK = 0


def report(name, ok, detail=''):
    global OK
    if ok:
        OK += 1
    else:
        FAILS.append((name, detail))
        print('FAIL', name, detail)


def idxs(C):
    return list(itertools.product(*[range(c) for c in C]))


def slice_t3(x, idx):
    return t3.TuckerTensorTrain(tuple(B[idx] for B in x.tucker_cores), tuple(G[idx] for G in x.tt_cores))


def slice_arr(a, idx, n_lead=0):
    """slice a stack-C axis group that sits after n_lead leading axes"""
    sl = (slice(None),) * n_lead + tuple(idx)
    return a[sl]


def close(a, b, tol=1e-9):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return False, 'shape %s vs %s' % (a.shape, b.shape)
    den = max(np.linalg.norm(b), 1.0)
    err = np.linalg.norm(a - b) / den
    return err < tol, 'relerr=%.2e' % err


def check_per_element(name, C, stacked, per_elem, n_lead=0):
    """stacked: array with C at axes [n_lead: n_lead+len(C)]; per_elem(idx) -> unstacked array"""
    try:
        stacked = np.asarray(stacked)
        for idx in idxs(C):
            ref = np.asarray(per_elem(idx))
            got = slice_arr(stacked, idx, n_lead)
            ok, d = close(got, ref)
            if not ok:
                report(name + ' C=%s idx=%s' % (C, idx), False, d)
                return
        report(name + ' C=%s' % (C,), True)
    except Exception as e:
        report(name + ' C=%s' % (C,), False, 'EXC ' + repr(e))


def check_t3_per_element(name, C, stacked_t3, per_elem_t3):
    try:
        if stacked_t3.stack_shape != tuple(C):
            report(name + ' C=%s' % (C,), False, 'stack_shape %s != %s' % (stacked_t3.stack_shape, C))
            return
        for idx in idxs(C):
            ref = per_elem_t3(idx).to_dense()
            got = slice_t3(stacked_t3, idx).to_dense()
            ok, d = close(got, ref)
            if not ok:
                report(name + ' C=%s idx=%s' % (C, idx), False, d)
                return
        report(name + ' C=%s' % (C,), True)
    except Exception as e:
        report(name + ' C=%s' % (C,), False, 'EXC ' + repr(e))


def randvecs(shape, W):
    return tuple(np.random.randn(*(W + (N,))) for N in shape)


for (shape, nn, rr) in STRUCTS:
    d = len(shape)
    for C in STACKS:
        tag = 'd=%d ' % d
        x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
        y = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
        xs = {idx: slice_t3(x, idx) for idx in idxs(C)}
        ys = {idx: slice_t3(y, idx) for idx in idxs(C)}

        # to_dense
        check_per_element(tag + 'to_dense', C, x.to_dense(), lambda i: xs[i].to_dense())
        # arithmetic
        check_t3_per_element(tag + 'add', C, x + y, lambda i: xs[i] + ys[i])
        check_t3_per_element(tag + 'sub', C, x - y, lambda i: xs[i] - ys[i])
        check_t3_per_element(tag + 'neg', C, -x, lambda i: -xs[i])
        check_t3_per_element(tag + 'scale', C, x * 2.5, lambda i: xs[i] * 2.5)
        pass
        check_t3_per_element(tag + 'mul_t3', C, x * y, lambda i: xs[i] * ys[i])
        check_per_element(tag + 'mul_dense', C, x * y.to_dense(), lambda i: xs[i] * ys[i].to_dense())
        for method in ['form_then_round', 'inplace_fused', 'swap']:
            try:
                check_t3_per_element(tag + 't3m[%s]' % method, C, x.t3m(y, method=method),
                                     lambda i: xs[i].t3m(ys[i], method=method))
            except Exception as e:
                report(tag + 't3m[%s] C=%s' % (method, C), False, 'EXC ' + repr(e))
        # inner / norm
        for uo in [True, False]:
            check_per_element(tag + 'inner[orth=%s]' % uo, C, x.inner(y, use_orthogonalization=uo),
                              lambda i: xs[i].inner(ys[i], use_orthogonalization=uo))
            check_per_element(tag + 'norm[orth=%s]' % uo, C, x.norm(use_orthogonalization=uo),
                              lambda i: xs[i].norm(use_orthogonalization=uo))
        check_per_element(tag + 'inner_dense', C, x.inner(y.to_dense()), lambda i: xs[i].inner(ys[i].to_dense()))
        # sum
        check_per_element(tag + 'sum_all', C, x.sum(), lambda i: xs[i].sum())
        if d >= 2:
            check_t3_per_element(tag + 'sum_axis0', C, x.sum(axis=0), lambda i: xs[i].sum(axis=0))
            check_t3_per_element(tag + 'sum_axis_last', C, x.sum(axis=d - 1), lambda i: xs[i].sum(axis=d - 1))
            if d >= 3:
                check_t3_per_element(tag + 'sum_axis(0,2)', C, x.sum(axis=(0, 2)), lambda i: xs[i].sum(axis=(0, 2)))
        # sum_stack
        if C:
            try:
                ss = x.sum_stack()
                ref = sum(xs[i].to_dense() for i in idxs(C))
                ok, dd = close(ss.to_dense(), ref)
                report(tag + 'sum_stack C=%s' % (C,), ok and ss.stack_shape == (), dd)
                if len(C) == 2:
                    ss0 = x.sum_stack(axis=0)
                    ref0 = x.to_dense().sum(axis=0)
                    ok, dd = close(ss0.to_dense(), ref0)
                    report(tag + 'sum_stack(axis=0) C=%s' % (C,), ok and ss0.stack_shape == (C[1],), dd + ' ss=%s' % (ss0.stack_shape,))
                    ss1 = x.sum_stack(axis=1)
                    ok, dd = close(ss1.to_dense(), x.to_dense().sum(axis=1))
                    report(tag + 'sum_stack(axis=1) C=%s' % (C,), ok and ss1.stack_shape == (C[0],), dd + ' ss=%s' % (ss1.stack_shape,))
                    ssc = x.sum_stack_corewise(axis=1)
                    ok = ssc.stack_shape == (C[0],) and all(np.allclose(a, b.sum(axis=1)) for a, b in zip(ssc.tucker_cores, x.tucker_cores))
                    report(tag + 'sum_stack_corewise(axis=1) C=%s' % (C,), ok)
            except Exception as e:
                report(tag + 'sum_stack C=%s' % (C,), False, 'EXC ' + repr(e))
        # t3svd
        try:
            xsvd, st, stt = x.t3svd()
            check_t3_per_element(tag + 't3svd', C, xsvd, lambda i: xs[i].t3svd()[0])
            # singular values per element
            for k in range(d):
                check_per_element(tag + 't3svd tucker sv[%d]' % k, C, st[k], lambda i: xs[i].t3svd()[1][k])
            for k in range(d + 1):
                check_per_element(tag + 't3svd tt sv[%d]' % k, C, stt[k], lambda i: xs[i].t3svd()[2][k])
            xsvd2, _, _ = (x + y).t3svd(max_tt_ranks=2, max_tucker_ranks=2)
            check_t3_per_element(tag + 't3svd trunc', C, xsvd2,
                                 lambda i: (xs[i] + ys[i]).t3svd(max_tt_ranks=2, max_tucker_ranks=2)[0])
        except Exception as e:
            report(tag + 't3svd C=%s' % (C,), False, 'EXC ' + repr(e))
        # rank_adjustment_sweep
        try:
            z = x + y
            for dirn in ['right_to_left', 'left_to_right']:
                check_t3_per_element(tag + 'rank_adjustment_sweep[%s]' % dirn, C, z.rank_adjustment_sweep(dirn),
                                     lambda i: (xs[i] + ys[i]).rank_adjustment_sweep(dirn))
        except Exception as e:
            report(tag + 'rank_adjustment_sweep C=%s' % (C,), False, 'EXC ' + repr(e))
        # orthogonalizations
        for meth in ['down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores',
                     'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores']:
            try:
                check_t3_per_element(tag + meth, C, getattr(x, meth)(), lambda i: getattr(xs[i], meth)())
            except Exception as e:
                report(tag + meth + ' C=%s' % (C,), False, 'EXC ' + repr(e))
        try:
            xl = x.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
            r = xl.is_left_orthogonal()
            report(tag + 'is_left_orthogonal shape C=%s' % (C,), np.shape(r) == C and np.all(r), 'shape=%s val=%s' % (np.shape(r), r))
            xr = x.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
            r = xr.is_right_orthogonal()
            report(tag + 'is_right_orthogonal shape C=%s' % (C,), np.shape(r) == C and np.all(r), 'shape=%s' % (np.shape(r),))
        except Exception as e:
            report(tag + 'is_*_orthogonal C=%s' % (C,), False, 'EXC ' + repr(e))
        # single core svd/orth
        try:
            check_t3_per_element(tag + 'down_svd_tucker_core', C, x.down_svd_tucker_core(0)[0], lambda i: xs[i].down_svd_tucker_core(0)[0])
            check_t3_per_element(tag + 'up_svd_tt_core', C, x.up_svd_tt_core(0)[0], lambda i: xs[i].up_svd_tt_core(0)[0])
            if d >= 2:
                check_t3_per_element(tag + 'left_svd_tt_core', C, x.left_svd_tt_core(0)[0], lambda i: xs[i].left_svd_tt_core(0)[0])
                check_t3_per_element(tag + 'right_svd_tt_core', C, x.right_svd_tt_core(d - 1)[0], lambda i: xs[i].right_svd_tt_core(d - 1)[0])
                check_t3_per_element(tag + 'orth_rel_tucker', C, x.orthogonalize_relative_to_tucker_core(1), lambda i: xs[i].orthogonalize_relative_to_tucker_core(1))
                check_t3_per_element(tag + 'orth_rel_tt', C, x.orthogonalize_relative_to_tt_core(1), lambda i: xs[i].orthogonalize_relative_to_tt_core(1))
        except Exception as e:
            report(tag + 'single core svd C=%s' % (C,), False, 'EXC ' + repr(e))
        # structural
        try:
            check_t3_per_element(tag + 'reverse', C, x.reverse(), lambda i: xs[i].reverse())
            check_t3_per_element(tag + 'squash_tails', C, x.squash_tails(), lambda i: xs[i].squash_tails())
            if d >= 2:
                check_t3_per_element(tag + 'segment', C, x.segment(0, d - 1), lambda i: xs[i].segment(0, d - 1))
                check_t3_per_element(tag + 'concatenate', C, t3.TuckerTensorTrain.concatenate([x.segment(0, 1), x.segment(1, d)]),
                                     lambda i: t3.TuckerTensorTrain.concatenate([xs[i].segment(0, 1), xs[i].segment(1, d)]))
            new_nn = tuple(n + 1 for n in nn); new_rr = (1,) + tuple(r + 1 for r in rr[1:-1]) + (1,)
            check_t3_per_element(tag + 'resize_pad', C, x.resize(shape, new_nn, new_rr), lambda i: xs[i].resize(shape, new_nn, new_rr))
            v = x.to_vector()
            x2 = t3.TuckerTensorTrain.from_vector(v, shape, nn, rr, stack_shape=C)
            report(tag + 'to/from_vector C=%s' % (C,), np.allclose(x2.to_dense(), x.to_dense()) and v.shape == (x.data_size,), 'vshape=%s data_size=%s' % (v.shape, x.data_size))
            # unstack/stack
            tree = x.unstack()
            x3 = t3.TuckerTensorTrain.stack(tree)
            report(tag + 'unstack/stack roundtrip C=%s' % (C,), x3.stack_shape == C and np.allclose(x3.to_dense(), x.to_dense()))
            if C:
                leaf = tree
                for i in idxs(C)[-1]:
                    leaf = leaf[i]
                report(tag + 'unstack last leaf C=%s' % (C,), np.allclose(leaf.to_dense(), xs[idxs(C)[-1]].to_dense()))
            # from_canonical
            R = 3
            fac = tuple(np.random.randn(*(C + (R, N))) for N in shape)
            xc = t3.TuckerTensorTrain.from_canonical(fac)
            check_t3_per_element(tag + 'from_canonical', C, xc, lambda i: t3.TuckerTensorTrain.from_canonical(tuple(f[i] for f in fac)))
            # to/from tensor train
            tt = x.to_tensor_train()
            xtt = t3.TuckerTensorTrain.from_tensor_train(tt)
            report(tag + 'to/from_tensor_train C=%s' % (C,), xtt.stack_shape == C and np.allclose(xtt.to_dense(), x.to_dense()), 'ss=%s' % (xtt.stack_shape,))
        except Exception as e:
            report(tag + 'structural C=%s' % (C,), False, 'EXC ' + repr(e) + traceback.format_exc()[-400:])
        # continuation_ranks / share / has_shared
        try:
            cr = x.continuation_ranks()
            report(tag + 'continuation_ranks C=%s' % (C,), True)
        except Exception as e:
            report(tag + 'continuation_ranks C=%s' % (C,), False, 'EXC ' + repr(e))
        if d >= 2 and shape[0] == shape[1] or (d == 4):
            pass
        # weights
        try:
            xsv = x.t3svd()[0]; x = xsv; xs = {i: slice_t3(x, i) for i in idxs(C)}
            W = t3.T3Weights.from_t3svd(x)
            xw = t3.t3_absorb_weights(x, W)
            check_t3_per_element(tag + 'absorb_weights', C, xw, lambda i: t3.t3_absorb_weights(xs[i], t3.T3Weights.from_t3svd(xs[i])))
            check_per_element(tag + 'weighted_norm', C, t3.t3_weighted_norm(x, W), lambda i: t3.t3_weighted_norm(xs[i], t3.T3Weights.from_t3svd(xs[i])))
            check_per_element(tag + 'weighted_inner', C, t3.t3_weighted_inner(x, W, y, W),
                              lambda i: t3.t3_weighted_inner(xs[i], t3.T3Weights.from_t3svd(xs[i]), ys[i], t3.T3Weights.from_t3svd(xs[i])))
            Wc = W.concatenate(W); Wk = W.kronecker(W)
            report(tag + 'weights concat/kron stack C=%s' % (C,), Wc.stack_shape == C and Wk.stack_shape == C)
            check_t3_per_element(tag + 'absorb_weights_kron', C, t3.t3_absorb_weights(x * x, Wk),
                                 lambda i: t3.t3_absorb_weights(xs[i] * xs[i], t3.T3Weights.from_t3svd(xs[i]).kronecker(t3.T3Weights.from_t3svd(xs[i]))))
            check_t3_per_element(tag + 'absorb_weights_concat', C, t3.t3_absorb_weights(x + x, Wc),
                                 lambda i: t3.t3_absorb_weights(xs[i] + xs[i], t3.T3Weights.from_t3svd(xs[i]).concatenate(t3.T3Weights.from_t3svd(xs[i]))))
            Wr = W.reverse()
            report(tag + 'weights reverse stack C=%s' % (C,), Wr.stack_shape == C and Wr.is_consistent_with(x.reverse()))
            Wt = t3.T3Weights.stack(W.unstack())
            report(tag + 'weights unstack/stack C=%s' % (C,), Wt.stack_shape == C and all(np.allclose(a, b) for a, b in zip(Wt.tucker_weights, W.tucker_weights)))
        except Exception as e:
            report(tag + 'weights C=%s' % (C,), False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])

        # ---------------- sampling ops with W ----------------
        for Wsh in WS:
            wtag = tag + 'W=%s ' % (Wsh,)
            ww = randvecs(shape, Wsh)
            pp = randvecs(shape, Wsh)
            widx = idxs(Wsh)

            def wslice(vv, wi):
                return tuple(v[wi] for v in vv)

            # entries
            index = np.stack([np.random.randint(0, N, size=Wsh) for N in shape])  # (d,)+W
            try:
                ent = x.entries(index)
                ok = True; msg = ''
                for wi in widx:
                    for ci in idxs(C):
                        ref = xs[ci].to_dense()[tuple(index[(slice(None),) + wi])]
                        got = np.asarray(ent)[wi + ci]
                        if not np.allclose(got, ref):
                            ok = False; msg = 'w=%s c=%s' % (wi, ci)
                report(wtag + 'entries C=%s' % (C,), ok and np.shape(ent) == Wsh + C, msg + ' shape=%s' % (np.shape(ent),))
            except Exception as e:
                report(wtag + 'entries C=%s' % (C,), False, 'EXC ' + repr(e))
            # apply
            try:
                ap = x.apply(ww)
                ok = True; msg = ''
                for wi in widx:
                    for ci in idxs(C):
                        dense = xs[ci].to_dense()
                        ref = dense
                        for k in range(d):
                            ref = np.tensordot(ref, ww[k][wi], axes=([0], [0]))
                        got = np.asarray(ap)[wi + ci]
                        if not np.allclose(got, ref):
                            ok = False; msg = 'w=%s c=%s' % (wi, ci)
                report(wtag + 'apply C=%s' % (C,), ok and np.shape(ap) == Wsh + C, msg + ' shape=%s' % (np.shape(ap),))
            except Exception as e:
                report(wtag + 'apply C=%s' % (C,), False, 'EXC ' + repr(e))
            # probe
            try:
                zz = x.probe(ww)
                ok = True; msg = ''
                for k in range(d):
                    for wi in widx:
                        for ci in idxs(C):
                            ref = xs[ci].to_dense()
                            for m in range(d):
                                if m == k:
                                    continue
                                ref = np.tensordot(ref, ww[m][wi], axes=([0 if m < k else 1], [0]))
                            got = np.asarray(zz[k])[wi + ci]
                            if not np.allclose(got, ref):
                                ok = False; msg = 'k=%d w=%s c=%s' % (k, wi, ci)
                    if np.shape(zz[k]) != Wsh + C + (shape[k],):
                        ok = False; msg += ' shape=%s' % (np.shape(zz[k]),)
                report(wtag + 'probe C=%s' % (C,), ok, msg)
            except Exception as e:
                report(wtag + 'probe C=%s' % (C,), False, 'EXC ' + repr(e))
            # transposes (per element vs unstacked)
            c_res = np.asarray(np.random.randn(*(Wsh + C)))
            zt = tuple(np.random.randn(*(Wsh + C + (N,))) for N in shape)
            for sop in [False, True]:
                stag = wtag + 'sop=%s ' % sop
                try:
                    fac = t3.TuckerTensorTrain.apply_ambient_transpose(c_res, ww, sum_over_probes=sop)
                    xa = t3.TuckerTensorTrain.from_canonical(fac)
                    if sop:
                        # stack C; value = sum_W c * outer(ww)
                        def ref_fn(ci):
                            out = 0
                            for wi in widx:
                                out = out + c_res[wi + ci] * t3.TuckerTensorTrain.from_canonical(tuple(w[wi][None] for w in ww)).to_dense()
                            return out
                        check_per_element(stag + 'apply_ambient_transpose', C, xa.to_dense(), ref_fn)
                    else:
                        ok = True; msg = ''
                        for wi in widx:
                            for ci in idxs(C):
                                ref = c_res[wi + ci] * t3.TuckerTensorTrain.from_canonical(tuple(w[wi][None] for w in ww)).to_dense()
                                got = xa.to_dense()[wi + ci]
                                if not np.allclose(got, ref):
                                    ok = False; msg = 'w=%s c=%s' % (wi, ci)
                        report(stag + 'apply_ambient_transpose C=%s' % (C,), ok and xa.stack_shape == Wsh + C, msg + ' ss=%s' % (xa.stack_shape,))
                except Exception as e:
                    report(stag + 'apply_ambient_transpose C=%s' % (C,), False, 'EXC ' + repr(e))
                try:
                    fac = t3.TuckerTensorTrain.entries_ambient_transpose(c_res, index, shape, sum_over_probes=sop) if 'shape' in t3.TuckerTensorTrain.entries_ambient_transpose.__code__.co_varnames else t3.TuckerTensorTrain.entries_ambient_transpose(c_res, index, sum_over_probes=sop)
                    report(stag + 'entries_ambient_transpose runs C=%s' % (C,), True)
                except Exception as e:
                    report(stag + 'entries_ambient_transpose C=%s' % (C,), False, 'EXC ' + repr(e))
                # corewise transposes: compare per element with unstacked call
                try:
                    g = x.apply_corewise_transpose(c_res, ww, sum_over_probes=sop)
                    ok = True; msg = ''
                    if sop:
                        for ci in idxs(C):
                            ref = xs[ci].apply_corewise_transpose(c_res[(...,) + ci] if C else c_res, ww, sum_over_probes=True)
                            for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                if not np.allclose(a[ci], b):
                                    ok = False; msg = 'c=%s' % (ci,)
                        report(stag + 'apply_corewise_transpose C=%s' % (C,), ok and g[0][0].shape == x.tucker_cores[0].shape, msg + ' shape=%s' % (g[0][0].shape,))
                    else:
                        for wi in widx:
                            for ci in idxs(C):
                                ref = xs[ci].apply_corewise_transpose(np.asarray(c_res[wi + ci]), wslice(ww, wi), sum_over_probes=False)
                                for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                    if not np.allclose(a[wi + ci], b):
                                        ok = False; msg = 'w=%s c=%s' % (wi, ci)
                        report(stag + 'apply_corewise_transpose C=%s' % (C,), ok and g[0][0].shape == Wsh + x.tucker_cores[0].shape, msg + ' shape=%s' % (g[0][0].shape,))
                except Exception as e:
                    report(stag + 'apply_corewise_transpose C=%s' % (C,), False, 'EXC ' + repr(e))
                try:
                    g = x.probe_corewise_transpose(zt, ww, sum_over_probes=sop)
                    ok = True; msg = ''
                    if sop:
                        for ci in idxs(C):
                            ref = xs[ci].probe_corewise_transpose(tuple(z[(...,) + ci + (slice(None),)] if C else z for z in zt), ww, sum_over_probes=True)
                            for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                if not np.allclose(a[ci], b):
                                    ok = False; msg = 'c=%s' % (ci,)
                        report(stag + 'probe_corewise_transpose C=%s' % (C,), ok and g[0][0].shape == x.tucker_cores[0].shape, msg + ' shape=%s' % (g[0][0].shape,))
                    else:
                        for wi in widx:
                            for ci in idxs(C):
                                ref = xs[ci].probe_corewise_transpose(tuple(z[wi + ci] for z in zt), wslice(ww, wi), sum_over_probes=False)
                                for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                    if not np.allclose(a[wi + ci], b):
                                        ok = False; msg = 'w=%s c=%s' % (wi, ci)
                        report(stag + 'probe_corewise_transpose C=%s' % (C,), ok and g[0][0].shape == Wsh + x.tucker_cores[0].shape, msg + ' shape=%s' % (g[0][0].shape,))
                except Exception as e:
                    report(stag + 'probe_corewise_transpose C=%s' % (C,), False, 'EXC ' + repr(e))
                try:
                    g = x.entries_corewise_transpose(c_res, index, sum_over_probes=sop)
                    ok = True; msg = ''
                    if sop:
                        for ci in idxs(C):
                            ref = xs[ci].entries_corewise_transpose(c_res[(...,) + ci] if C else c_res, index, sum_over_probes=True)
                            for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                if not np.allclose(a[ci], b):
                                    ok = False; msg = 'c=%s' % (ci,)
                        report(stag + 'entries_corewise_transpose C=%s' % (C,), ok, msg)
                    else:
                        for wi in widx:
                            for ci in idxs(C):
                                ref = xs[ci].entries_corewise_transpose(np.asarray(c_res[wi + ci]), index[(slice(None),) + wi], sum_over_probes=False)
                                for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                    if not np.allclose(a[wi + ci], b):
                                        ok = False; msg = 'w=%s c=%s' % (wi, ci)
                        report(stag + 'entries_corewise_transpose C=%s' % (C,), ok, msg)
                except Exception as e:
                    report(stag + 'entries_corewise_transpose C=%s' % (C,), False, 'EXC ' + repr(e))
                try:
                    fac = t3.TuckerTensorTrain.probe_ambient_transpose(zt, ww, sum_over_probes=sop)
                    xa = t3.TuckerTensorTrain.from_canonical(fac)
                    ok = True; msg = ''
                    for ci in idxs(C):
                        if sop:
                            ref = 0
                            for wi in widx:
                                f1 = t3.TuckerTensorTrain.probe_ambient_transpose(tuple(z[wi + ci] for z in zt), wslice(ww, wi), sum_over_probes=False)
                                ref = ref + t3.TuckerTensorTrain.from_canonical(f1).to_dense()
                            got = xa.to_dense()[ci]
                            if not np.allclose(got, ref):
                                ok = False; msg = 'c=%s' % (ci,)
                        else:
                            for wi in widx:
                                f1 = t3.TuckerTensorTrain.probe_ambient_transpose(tuple(z[wi + ci] for z in zt), wslice(ww, wi), sum_over_probes=False)
                                ref = t3.TuckerTensorTrain.from_canonical(f1).to_dense()
                                # also check against manual: sum_i w0 x ... x zt_i x ... x w_{d-1}
                                man = 0
                                for k in range(d):
                                    fs = [ww[m][wi] if m != k else zt[k][wi + ci] for m in range(d)]
                                    t = fs[0]
                                    for f in fs[1:]:
                                        t = np.multiply.outer(t, f)
                                    man = man + t
                                if not np.allclose(ref, man):
                                    ok = False; msg = 'manual mismatch w=%s' % (wi,)
                                if not np.allclose(xa.to_dense()[wi + ci], ref):
                                    ok = False; msg = 'w=%s c=%s' % (wi, ci)
                    report(stag + 'probe_ambient_transpose C=%s' % (C,), ok, msg + ' ss=%s' % (xa.stack_shape,))
                except Exception as e:
                    report(stag + 'probe_ambient_transpose C=%s' % (C,), False, 'EXC ' + repr(e))
            # derivatives (order 2) per element
            order = 2
            try:
                dz = x.probe_derivatives(ww, pp, order)
                ok = True; msg = ''
                for k in range(d):
                    if np.shape(dz[k]) != (order + 1,) + Wsh + C + (shape[k],):
                        ok = False; msg = 'shape %s' % (np.shape(dz[k]),)
                for wi in widx:
                    for ci in idxs(C):
                        ref = xs[ci].probe_derivatives(wslice(ww, wi), wslice(pp, wi), order)
                        for k in range(d):
                            if not np.allclose(dz[k][(slice(None),) + wi + ci], ref[k]):
                                ok = False; msg = 'k=%d w=%s c=%s' % (k, wi, ci)
                report(wtag + 'probe_derivatives C=%s' % (C,), ok, msg)
                da = x.apply_derivatives(ww, pp, order)
                ok = np.shape(da) == (order + 1,) + Wsh + C; msg = 'shape=%s' % (np.shape(da),)
                for wi in widx:
                    for ci in idxs(C):
                        ref = xs[ci].apply_derivatives(wslice(ww, wi), wslice(pp, wi), order)
                        if not np.allclose(da[(slice(None),) + wi + ci], ref):
                            ok = False; msg = 'w=%s c=%s' % (wi, ci)
                report(wtag + 'apply_derivatives C=%s' % (C,), ok, msg)
                de = x.entries_derivatives(index, pp, order)
                ok = np.shape(de) == (order + 1,) + Wsh + C; msg = 'shape=%s' % (np.shape(de),)
                for wi in widx:
                    for ci in idxs(C):
                        ref = xs[ci].entries_derivatives(index[(slice(None),) + wi], wslice(pp, wi), order)
                        if not np.allclose(de[(slice(None),) + wi + ci], ref):
                            ok = False; msg = 'w=%s c=%s' % (wi, ci)
                report(wtag + 'entries_derivatives C=%s' % (C,), ok, msg)
            except Exception as e:
                report(wtag + 'derivatives C=%s' % (C,), False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])
            # corewise derivative transposes, sum_over_probes True, per element
            try:
                ztj = tuple(np.random.randn(*((order + 1,) + Wsh + C + (N,))) for N in shape)
                cj = np.random.randn(*((order + 1,) + Wsh + C))
                for sop in [True, False]:
                    g = x.probe_corewise_derivatives_transpose(ztj, ww, pp, order, sum_over_probes=sop)
                    ok = True; msg = ''
                    for ci in idxs(C):
                        if sop:
                            ref = xs[ci].probe_corewise_derivatives_transpose(tuple(z[(slice(None),) + (...,) + ci + (slice(None),)] if C else z for z in ztj), ww, pp, order, sum_over_probes=True)
                            for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                if not np.allclose(a[ci], b):
                                    ok = False; msg = 'c=%s' % (ci,)
                        else:
                            for wi in widx:
                                ref = xs[ci].probe_corewise_derivatives_transpose(tuple(z[(slice(None),) + wi + ci] for z in ztj), wslice(ww, wi), wslice(pp, wi), order, sum_over_probes=False)
                                for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                    if not np.allclose(a[wi + ci], b):
                                        ok = False; msg = 'w=%s c=%s shape=%s' % (wi, ci, a.shape)
                    report(wtag + 'probe_corewise_derivatives_transpose sop=%s C=%s' % (sop, C), ok, msg)
                    g = x.apply_corewise_derivatives_transpose(cj, ww, pp, order, sum_over_probes=sop)
                    ok = True; msg = ''
                    for ci in idxs(C):
                        if sop:
                            ref = xs[ci].apply_corewise_derivatives_transpose(cj[(slice(None),) + (...,) + ci] if C else cj, ww, pp, order, sum_over_probes=True)
                            for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                if not np.allclose(a[ci], b):
                                    ok = False; msg = 'c=%s' % (ci,)
                        else:
                            for wi in widx:
                                ref = xs[ci].apply_corewise_derivatives_transpose(cj[(slice(None),) + wi + ci], wslice(ww, wi), wslice(pp, wi), order, sum_over_probes=False)
                                for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                    if not np.allclose(a[wi + ci], b):
                                        ok = False; msg = 'w=%s c=%s shape=%s' % (wi, ci, a.shape)
                    report(wtag + 'apply_corewise_derivatives_transpose sop=%s C=%s' % (sop, C), ok, msg)
                    g = x.entries_corewise_derivatives_transpose(cj, index, pp, order, sum_over_probes=sop)
                    ok = True; msg = ''
                    for ci in idxs(C):
                        if sop:
                            ref = xs[ci].entries_corewise_derivatives_transpose(cj[(slice(None),) + (...,) + ci] if C else cj, index, pp, order, sum_over_probes=True)
                            for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                if not np.allclose(a[ci], b):
                                    ok = False; msg = 'c=%s' % (ci,)
                        else:
                            for wi in widx:
                                ref = xs[ci].entries_corewise_derivatives_transpose(cj[(slice(None),) + wi + ci], index[(slice(None),) + wi], wslice(pp, wi), order, sum_over_probes=False)
                                for a, b in zip(g[0] + g[1], ref[0] + ref[1]):
                                    if not np.allclose(a[wi + ci], b):
                                        ok = False; msg = 'w=%s c=%s shape=%s' % (wi, ci, a.shape)
                    report(wtag + 'entries_corewise_derivatives_transpose sop=%s C=%s' % (sop, C), ok, msg)
            except Exception as e:
                report(wtag + 'corewise_derivatives_transpose C=%s' % (C,), False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])

print('\n=== OK:', OK, ' FAILS:', len(FAILS))
for f in FAILS:
    print('  ', f)
