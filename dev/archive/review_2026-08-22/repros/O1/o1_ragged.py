"""Phase 1: ragged sweep -- dense + adjoint oracles over the full matrix."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from o1_common import *

TOL = 1e-9
ORDER = 3

def sweep_structure(sname, struct, sh):
    shape, tr, ttr = struct; d = len(shape)
    for C in CS:
        np.random.seed(1)
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
        if sh is not None:
            x = x.share(sh)
        X = np.asarray(x.to_dense())
        # ---- arithmetic / conversions (C only)
        y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
        Y = np.asarray(y.to_dense())
        check('add', sname, 'ragged', C, (), (), sh, lambda: relerr((x + y).to_dense(), X + Y))
        check('sub', sname, 'ragged', C, (), (), sh, lambda: relerr((x - y).to_dense(), X - Y))
        check('hadamard', sname, 'ragged', C, (), (), sh, lambda: relerr((x * y).to_dense(), X * Y))
        check('scalar_mul', sname, 'ragged', C, (), (), sh, lambda: relerr((x * 2.5).to_dense(), 2.5 * X))
        check('neg', sname, 'ragged', C, (), (), sh, lambda: relerr((-x).to_dense(), -X))
        check('inner', sname, 'ragged', C, (), (), sh,
              lambda: relerr(x.inner(y), np.sum((X * Y).reshape(C + (-1,)), axis=-1)))
        check('norm', sname, 'ragged', C, (), (), sh,
              lambda: relerr(x.norm(), np.linalg.norm(X.reshape(C + (-1,)), axis=-1)))
        check('reverse', sname, 'ragged', C, (), (), sh,
              lambda: relerr(x.reverse().to_dense(), np.transpose(X, tuple(range(len(C))) + tuple(range(X.ndim - 1, len(C) - 1, -1)))))
        def _vec():
            x2 = t3.TuckerTensorTrain.from_vector(x.to_vector(), x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape)
            return relerr(x2.to_dense(), X)
        check('to_from_vector', sname, 'ragged', C, (), (), sh, _vec)
        # ---- t3svd / rank sweeps
        def _svd():
            xs, stk, stt = x.t3svd(sharing=sh)
            e = relerr(xs.to_dense(), X)
            if not bool(np.all(xs.is_left_orthogonal())): e = max(e, 1.0)
            # last TT sval norm == tensor norm
            e = max(e, relerr(np.linalg.norm(np.asarray(stt[-1]), axis=-1), x.norm()))
            if sh is not None and not bool(np.all(xs.has_shared_tucker_factors(sh))): e = max(e, 1.0)
            return e
        check('t3svd_lossless', sname, 'ragged', C, (), (), sh, _svd)
        def _svd_trunc():
            caps_tk = tuple(max(1, r - 1) for r in tr)
            if sh is not None:   # equal within groups already
                pass
            xs, stk, stt = x.t3svd(max_tucker_ranks=caps_tk, max_tt_ranks=tuple(max(1, r - 1) if 0 < i < d else 1 for i, r in enumerate(ttr)), sharing=sh)
            ok = all(a <= b for a, b in zip(xs.tucker_ranks, caps_tk))
            e = 0.0 if ok else 1.0
            if not bool(np.all(xs.is_left_orthogonal())): e = max(e, 1.0)
            if sh is not None and not bool(np.all(xs.has_shared_tucker_factors(sh))): e = max(e, 1.0)
            # truncation error must not exceed the (loose) bound sqrt(sum of dropped svals^2) * (d+1)... just sanity: err <= norm
            e = max(e, 0.0 if relerr(xs.to_dense(), X) <= 1.0 + 1e-12 else 1.0)
            return e
        check('t3svd_trunc_struct', sname, 'ragged', C, (), (), sh, _svd_trunc)
        def _ras():
            xs = x.t3svd(sharing=sh)[0].rank_adjustment_sweep('right_to_left', sharing=sh)
            e = relerr(xs.to_dense(), X)
            if not bool(np.all(xs.is_right_orthogonal())): e = max(e, 1.0)
            if sh is None and not xs.has_minimal_ranks: e = max(e, 1.0)
            if sh is not None and not bool(np.all(xs.has_shared_tucker_factors(sh))): e = max(e, 1.0)
            xs2 = xs.rank_adjustment_sweep('left_to_right', sharing=sh)
            e = max(e, relerr(xs2.to_dense(), X))
            return e
        check('rank_adjustment_sweep', sname, 'ragged', C, (), (), sh, _ras)
        # ---- orthogonal representations
        frame, var = bvf.t3_orthogonal_representations(x)
        def _orep():
            e = relerr(frame.to_dense(), X)
            if not bool(np.all(frame.is_orthogonal())): e = max(e, 1.0)
            tv = t3m.T3Tangent(frame, var)
            e = max(e, relerr(tv.to_dense(), 2 * d * X))   # each of the 2d single-core terms represents X
            return e
        check('orthogonal_representations', sname, 'ragged', C, (), (), sh, _orep)
        Bd = np.asarray(frame.to_dense())
        for W in WS:
            ww = rand_ww(shape, W, 7); pp = rand_ww(shape, W, 8); index = rand_index(shape, W, 9)
            ww_n = [w / np.linalg.norm(w, axis=-1, keepdims=True) for w in ww]
            # ---- forward sampling on TuckerTensorTrain
            check('t3_apply', sname, 'ragged', C, W, (), sh, lambda: relerr(x.apply(ww), dense_apply(X, ww, d)))
            check('t3_entries', sname, 'ragged', C, W, (), sh, lambda: relerr(x.entries(index), dense_entries(X, index, d, shape)))
            check('t3_probe', sname, 'ragged', C, W, (), sh,
                  lambda: max(relerr(a, b) for a, b in zip(x.probe(ww), dense_probe(X, ww, d))))
            # ---- derivative jets on TuckerTensorTrain
            check('t3_apply_derivatives', sname, 'ragged', C, W, (), sh,
                  lambda: relerr(x.apply_derivatives(ww, pp, ORDER), dense_apply_jets(X, ww, pp, d, ORDER)))
            check('t3_entries_derivatives', sname, 'ragged', C, W, (), sh,
                  lambda: relerr(x.entries_derivatives(index, pp, ORDER), dense_apply_jets(X, onehots(index, shape), pp, d, ORDER)))
            check('t3_probe_derivatives', sname, 'ragged', C, W, (), sh,
                  lambda: max(relerr(a, b) for a, b in zip(x.probe_derivatives(ww, pp, ORDER), dense_probe_jets(X, ww, pp, d, ORDER))))
            # ---- ambient transposes (dense oracle)
            rng = np.random.RandomState(11)
            c = np.asarray(rng.randn(*(W + C)))
            zt = [rng.randn(*(W + C + (N,))) for N in shape]
            def cp_dense(factors):
                d_ = len(factors)
                s = ','.join('...r' + LET[i] for i in range(d_)) + '->...' + LET[:d_]
                return np.einsum(s, *[np.asarray(f) for f in factors])
            def outer_w(vecs):  # vecs: list of W+(Ni,) -> W + shape
                s = ','.join('w' + LET[i] for i in range(d)) + '->w' + LET[:d]
                vf, W_ = flatW(vecs)
                return np.einsum(s, *vf).reshape(W_ + shape)
            def _amb_apply(sum_):
                F = t3.TuckerTensorTrain.apply_ambient_transpose(c, ww, sum_over_probes=sum_)
                D = cp_dense(F)
                ref = c.reshape(W + C + (1,) * d) * outer_w(ww).reshape(W + (1,) * len(C) + shape)
                if sum_: ref = ref.reshape((-1,) + C + shape).sum(axis=0)
                return relerr(D, ref)
            def _amb_entries(sum_):
                F = t3.TuckerTensorTrain.entries_ambient_transpose(c, index, shape, sum_over_probes=sum_)
                D = cp_dense(F)
                ref = c.reshape(W + C + (1,) * d) * outer_w(onehots(index, shape)).reshape(W + (1,) * len(C) + shape)
                if sum_: ref = ref.reshape((-1,) + C + shape).sum(axis=0)
                return relerr(D, ref)
            def _amb_probe(sum_):
                F = t3.TuckerTensorTrain.probe_ambient_transpose(zt, ww, sum_over_probes=sum_)
                D = cp_dense(F)
                ref = 0
                for m in range(d):
                    vecs = [zt[m].reshape(W + C + (shape[m],)) if i == m else np.asarray(ww[i]).reshape(W + (1,) * len(C) + (shape[i],)) for i in range(d)]
                    # outer product with broadcasting over W+C
                    term = np.ones(W + C + shape)
                    for i in range(d):
                        vi = np.broadcast_to(vecs[i], W + C + (shape[i],))
                        term = term * vi.reshape(W + C + (1,) * i + (shape[i],) + (1,) * (d - 1 - i))
                    ref = ref + term
                if sum_: ref = ref.reshape((-1,) + C + shape).sum(axis=0)
                return relerr(D, ref)
            for sum_ in (False, True):
                check('apply_ambient_transpose(sum=%s)' % sum_, sname, 'ragged', C, W, (), sh, lambda: _amb_apply(sum_))
                check('entries_ambient_transpose(sum=%s)' % sum_, sname, 'ragged', C, W, (), sh, lambda: _amb_entries(sum_))
                check('probe_ambient_transpose(sum=%s)' % sum_, sname, 'ragged', C, W, (), sh, lambda: _amb_probe(sum_))
            # ---- corewise transposes: adjoint vs the COREWISE tangent forward (which is checked vs dense below)
            cframe = t3m.COREWISE.frame(x)
            np.random.seed(5)
            dl = bvf.T3Variations.randn(cframe.variation_shapes, C)
            tdl = t3m.T3Tangent(cframe, dl)
            def _cw(kind, sum_):
                if kind == 'apply':
                    Jd = tdl.apply(ww); g = x.apply_corewise_transpose(c, ww, sum_over_probes=sum_); r = c
                elif kind == 'entries':
                    Jd = tdl.entries(index); g = x.entries_corewise_transpose(c, index, sum_over_probes=sum_); r = c
                else:
                    Jd = tdl.probe(ww); g = x.probe_corewise_transpose(zt, ww, sum_over_probes=sum_); r = zt
                lhs = sum(tdot(a, b) for a, b in zip(r, Jd)) if kind == 'probe' else tdot(r, Jd)
                rhs = var_dot(g, dl.data)
                return abs(lhs - rhs) / max(abs(lhs), 1e-300)
            for sum_ in (False, True):
                for kind in ('apply', 'entries', 'probe'):
                    check('%s_corewise_transpose(sum=%s)' % (kind, sum_), sname, 'ragged', C, W, (), sh, lambda: _cw(kind, sum_))
            # corewise derivative transposes
            cj = np.random.RandomState(12).randn(*((ORDER + 1,) + W + C))
            ztj = [np.random.RandomState(13 + i).randn(*((ORDER + 1,) + W + C + (N,))) for i, N in enumerate(shape)]
            def _cwd(kind, sum_):
                if kind == 'apply':
                    Jd = tdl.apply_derivatives(ww, pp, ORDER); g = x.apply_corewise_derivatives_transpose(cj, ww, pp, ORDER, sum_over_probes=sum_); r = cj
                elif kind == 'entries':
                    Jd = tdl.entries_derivatives(index, pp, ORDER); g = x.entries_corewise_derivatives_transpose(cj, index, pp, ORDER, sum_over_probes=sum_); r = cj
                else:
                    Jd = tdl.probe_derivatives(ww, pp, ORDER); g = x.probe_corewise_derivatives_transpose(ztj, ww, pp, ORDER, sum_over_probes=sum_); r = ztj
                lhs = sum(tdot(a, b) for a, b in zip(r, Jd)) if kind == 'probe' else tdot(r, Jd)
                rhs = var_dot(g, dl.data)
                return abs(lhs - rhs) / max(abs(lhs), 1e-300)
            for sum_ in (False, True):
                for kind in ('apply', 'entries', 'probe'):
                    check('%s_corewise_derivatives_transpose(sum=%s)' % (kind, sum_), sname, 'ragged', C, W, (), sh, lambda: _cwd(kind, sum_))
            # corewise tangent forward vs dense (anchors the corewise transposes): tangent to_dense
            Dd = np.asarray(tdl.to_dense())
            check('corewise_tangent_apply', sname, 'ragged', C, W, (), sh, lambda: relerr(tdl.apply(ww), dense_apply(Dd, ww, d)))
            # ---- tangent ops at the orthogonal frame
            for K in KS:
                np.random.seed(3)
                v = t3m.MANIFOLD.randn(frame, stack_shape=K)
                Vd = np.asarray(v.to_dense())   # K + C + shape
                check('tv_apply', sname, 'ragged', C, W, K, sh, lambda: relerr(v.apply(ww), dense_apply(Vd, ww, d)))
                check('tv_entries', sname, 'ragged', C, W, K, sh, lambda: relerr(v.entries(index), dense_entries(Vd, index, d, shape)))
                check('tv_probe', sname, 'ragged', C, W, K, sh,
                      lambda: max(relerr(a, b) for a, b in zip(v.probe(ww), dense_probe(Vd, ww, d))))
                check('tv_apply_derivatives', sname, 'ragged', C, W, K, sh,
                      lambda: relerr(v.apply_derivatives(ww, pp, ORDER), dense_apply_jets(Vd, ww, pp, d, ORDER)))
                check('tv_entries_derivatives', sname, 'ragged', C, W, K, sh,
                      lambda: relerr(v.entries_derivatives(index, pp, ORDER), dense_apply_jets(Vd, onehots(index, shape), pp, d, ORDER)))
                check('tv_probe_derivatives', sname, 'ragged', C, W, K, sh,
                      lambda: max(relerr(a, b) for a, b in zip(v.probe_derivatives(ww, pp, ORDER), dense_probe_jets(Vd, ww, pp, d, ORDER))))
                # adjoint identities (bare J, coordinate inner product), residual carries K
                rng = np.random.RandomState(21)
                rk = np.asarray(rng.randn(*(W + K + C)))
                ztk = [rng.randn(*(W + K + C + (N,))) for N in shape]
                rkj = rng.randn(*((ORDER + 1,) + W + K + C))
                ztkj = [rng.randn(*((ORDER + 1,) + W + K + C + (N,))) for N in shape]
                def _adj(kind, sum_):
                    if kind == 'apply':
                        Jv = v.apply(ww); t = t3m.T3Tangent.apply_transpose(rk, ww, frame, sum_over_probes=sum_); r = rk
                    elif kind == 'entries':
                        Jv = v.entries(index); t = t3m.T3Tangent.entries_transpose(rk, index, frame, sum_over_probes=sum_); r = rk
                    else:
                        Jv = v.probe(ww); t = t3m.T3Tangent.probe_transpose(ztk, ww, frame, sum_over_probes=sum_); r = ztk
                    lhs = sum(tdot(a, b) for a, b in zip(r, Jv)) if kind == 'probe' else tdot(r, Jv)
                    rhs = var_dot(t.variations.data, v.variations.data)
                    exp_stack = (W if not sum_ else ()) + K + C
                    e = abs(lhs - rhs) / max(abs(lhs), 1e-300)
                    if t.variations.stack_shape != exp_stack:
                        return (1.0, 'stack %s != expected %s' % (t.variations.stack_shape, exp_stack))
                    return e
                def _adjd(kind, sum_):
                    if kind == 'apply':
                        Jv = v.apply_derivatives(ww, pp, ORDER); t = t3m.T3Tangent.apply_derivatives_transpose(rkj, ww, pp, frame, ORDER, sum_over_probes=sum_); r = rkj
                    elif kind == 'entries':
                        Jv = v.entries_derivatives(index, pp, ORDER); t = t3m.T3Tangent.entries_derivatives_transpose(rkj, index, pp, frame, ORDER, sum_over_probes=sum_); r = rkj
                    else:
                        Jv = v.probe_derivatives(ww, pp, ORDER); t = t3m.T3Tangent.probe_derivatives_transpose(ztkj, ww, pp, frame, ORDER, sum_over_probes=sum_); r = ztkj
                    lhs = sum(tdot(a, b) for a, b in zip(r, Jv)) if kind == 'probe' else tdot(r, Jv)
                    rhs = var_dot(t.variations.data, v.variations.data)
                    exp_stack = (W if not sum_ else ()) + K + C
                    e = abs(lhs - rhs) / max(abs(lhs), 1e-300)
                    if t.variations.stack_shape != exp_stack:
                        return (1.0, 'stack %s != expected %s' % (t.variations.stack_shape, exp_stack))
                    return e
                for sum_ in (False, True):
                    for kind in ('apply', 'entries', 'probe'):
                        check('tv_%s_transpose_adjoint(sum=%s)' % (kind, sum_), sname, 'ragged', C, W, K, sh, lambda: _adj(kind, sum_))
                        check('tv_%s_derivatives_transpose_adjoint(sum=%s)' % (kind, sum_), sname, 'ragged', C, W, K, sh, lambda: _adjd(kind, sum_))
        # ---- geometry ops (C, K)
        for K in KS:
            np.random.seed(4)
            v = t3m.MANIFOLD.randn(frame, stack_shape=K)
            u = t3m.MANIFOLD.randn(frame, stack_shape=K)
            raw = t3m.T3Tangent(frame, bvf.T3Variations.randn(frame.variation_shapes, K + C))
            Vd = np.asarray(v.to_dense()); Ud = np.asarray(u.to_dense())
            check('gauge_project_idempotent', sname, 'ragged', C, (), K, sh,
                  lambda: relerr(t3m.MANIFOLD.project(v).to_vector(), v.to_vector()))
            def _proj_orth():
                pv = t3m.MANIFOLD.project(raw)
                e = 0.0 if bool(np.all(pv.is_gauged())) else 1.0
                # orthogonal projection in coordinates: <u, P raw> == <u, raw> for gauged u
                lhs = var_dot(u.variations.data, pv.variations.data); rhs = var_dot(u.variations.data, raw.variations.data)
                return max(e, abs(lhs - rhs) / max(abs(lhs), 1e-300))
            check('gauge_project_orthogonal', sname, 'ragged', C, (), K, sh, _proj_orth)
            def _proj_obl():
                pv = t3m.MANIFOLD.project_oblique(raw)
                e = 0.0 if bool(np.all(pv.is_gauged())) else 1.0
                return max(e, relerr(pv.to_dense(), raw.to_dense()))
            check('gauge_project_oblique', sname, 'ragged', C, (), K, sh, _proj_obl)
            check('manifold_inner', sname, 'ragged', C, (), K, sh,
                  lambda: relerr(t3m.MANIFOLD.inner(u, v), np.sum((Ud * Vd).reshape(K + C + (-1,)), axis=-1)))
            check('manifold_norm', sname, 'ragged', C, (), K, sh,
                  lambda: relerr(t3m.MANIFOLD.norm(v), np.linalg.norm(Vd.reshape(K + C + (-1,)), axis=-1)))
            def _tvec():
                v2 = t3m.T3Tangent.from_vector(v.to_vector(), frame, tangent_stack_shape=K)
                return relerr(v2.to_dense(), Vd)
            check('tangent_to_from_vector', sname, 'ragged', C, (), K, sh, _tvec)
            check('tangent_reverse', sname, 'ragged', C, (), K, sh,
                  lambda: relerr(v.reverse().to_dense(), np.transpose(Vd, tuple(range(len(K + C))) + tuple(range(Vd.ndim - 1, len(K + C) - 1, -1)))))
            check('tangent_add_scale', sname, 'ragged', C, (), K, sh,
                  lambda: relerr((v + u * 0.5 - v * 2.0).to_dense(), Vd + 0.5 * Ud - 2.0 * Vd))
            # project_ambient dense & T3
            Z = np.random.RandomState(31).randn(*(K + C + shape))
            def _pa_dense():
                P = t3m.MANIFOLD.project_ambient(frame, Z)
                e = 0.0 if bool(np.all(P.is_gauged())) else 1.0
                Pd = np.asarray(P.to_dense())
                # <P Z, u> == <Z, u> for u in the tangent space
                lhs = np.sum((Pd * Ud).reshape(K + C + (-1,)), -1); rhs = np.sum((Z * Ud).reshape(K + C + (-1,)), -1)
                e = max(e, relerr(lhs, rhs))
                # idempotent on its own output
                e = max(e, relerr(t3m.MANIFOLD.project_ambient(frame, Pd).to_dense(), Pd))
                return e
            check('project_ambient_dense', sname, 'ragged', C, (), K, sh, _pa_dense)
            if K == ():
                g = t3.TuckerTensorTrain.randn(shape, tuple(min(N, r + 1) for N, r in zip(shape, tr)), tuple(r + 1 if 0 < i < d else 1 for i, r in enumerate(ttr)), stack_shape=C)
                check('project_ambient_t3_vs_dense', sname, 'ragged', C, (), K, sh,
                      lambda: relerr(t3m.MANIFOLD.project_ambient(frame, g).to_dense(), t3m.MANIFOLD.project_ambient(frame, np.asarray(g.to_dense())).to_dense()))
                check('project_ambient_t3svd_method', sname, 'ragged', C, (), K, sh,
                      lambda: relerr(t3m.MANIFOLD.project_ambient(frame, Z, method='t3svd').to_dense(), t3m.MANIFOLD.project_ambient(frame, Z).to_dense()))
            # retract
            def _retr0():
                return relerr(t3m.MANIFOLD.retract(t3m.T3Tangent.zeros(frame, stack_shape=K)).to_dense(), np.broadcast_to(Bd, K + C + shape))
            check('retract_zero', sname, 'ragged', C, (), K, sh, _retr0)
            def _retr_fd():
                # central difference of retract(h v) around 0 vs dense(v); h^2 scaling
                errs = []
                for h in (1e-3, 5e-4):
                    rp = np.asarray(t3m.MANIFOLD.retract(v * h).to_dense()); rm = np.asarray(t3m.MANIFOLD.retract(v * (-h)).to_dense())
                    errs.append(np.linalg.norm(((rp - rm) / (2 * h) - Vd).reshape(-1)) / np.linalg.norm(Vd.reshape(-1)))
                ratio = errs[0] / max(errs[1], 1e-300)
                # accept if small error and ratio ~4 (2..8), or error already at roundoff
                if errs[1] < 1e-8: return errs[1]
                return (errs[1], 'ratio=%.2f errs=%s' % (ratio, errs)) if 2.5 < ratio < 6 else (1.0, 'ratio=%.2f errs=%s' % (ratio, errs))
            check('retract_fd_jacobian', sname, 'ragged', C, (), K, sh, _retr_fd, tol=1e-2)
            if K == () and C == ():
                def _retr_svd():
                    shifted = Bd + Vd
                    ref, _, _ = t3.TuckerTensorTrain.t3svd_dense(shifted, max_tucker_ranks=tuple(frame.up_ranks), max_tt_ranks=tuple(frame.left_ranks))
                    return relerr(t3m.MANIFOLD.retract(v).to_dense(), ref.to_dense())
                check('retract_vs_t3svd_dense', sname, 'ragged', C, (), K, sh, _retr_svd, tol=1e-6)
            # transport
            frame2, _ = bvf.t3_orthogonal_representations(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C))
            check('transport_identity', sname, 'ragged', C, (), K, sh,
                  lambda: relerr(t3m.MANIFOLD.transport(v, frame).to_dense(), Vd))
            check('transport_vs_project_ambient', sname, 'ragged', C, (), K, sh,
                  lambda: relerr(t3m.MANIFOLD.transport(v, frame2).to_dense(), t3m.MANIFOLD.project_ambient(frame2, Vd).to_dense()))
            # corewise geometry retract: additive
            cf = t3m.COREWISE.frame(x)
            cv = t3m.COREWISE.randn(cf, stack_shape=K) if K == () else None
            if cv is not None:
                def _cwr():
                    B, G = x.data; V, H = cv.variations.data
                    ref = t3.TuckerTensorTrain(tuple(np.asarray(b) + np.asarray(vv) for b, vv in zip(B, V)),
                                               tuple(np.asarray(g) + np.asarray(h) for g, h in zip(G, H))).to_dense()
                    return relerr(t3m.COREWISE.retract(cv).to_dense(), ref)
                check('corewise_retract', sname, 'ragged', C, (), K, sh, _cwr)

if __name__ == '__main__':
    names = sys.argv[1:] or list(STRUCTS)
    for sname in names:
        sweep_structure(sname, STRUCTS[sname], SHARING.get(sname))
        print('done', sname, len(RESULTS), flush=True)
    dump(os.path.join(os.path.dirname(__file__), 'results_ragged.md'))
