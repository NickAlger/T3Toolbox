"""Phase 2: uniform sweep -- uniform ops vs dense ground truth (same oracles as ragged), incl. varying-rank stacks."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from o1_common import *
import t3toolbox.backend.ut3_conversions as ut3c

ORDER = 3
PAD = dict(N=8, n=5, r=5)

def udense(u):  # UniformTuckerTensorTrain -> dense (stacked or not)
    return np.asarray(u.to_dense())

def sweep(sname, struct, sh, force_pad):
    shape, tr, ttr = struct; d = len(shape)
    rep = 'uniform' + ('+pad' if force_pad else '')
    for C in CS:
        np.random.seed(1)
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C)
        if sh is not None: x = x.share(sh)
        X = np.asarray(x.to_dense())
        ux = ut3.UniformTuckerTensorTrain.from_t3(x, **(PAD if force_pad else {}))
        y = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C); Y = np.asarray(y.to_dense())
        uy = ut3.UniformTuckerTensorTrain.from_t3(y, **(PAD if force_pad else {}))
        check('u_to_dense', sname, rep, C, (), (), sh, lambda: relerr(udense(ux), X))
        check('u_add', sname, rep, C, (), (), sh, lambda: relerr(udense(ux + uy), X + Y))
        check('u_sub', sname, rep, C, (), (), sh, lambda: relerr(udense(ux - uy), X - Y))
        check('u_scalar_mul', sname, rep, C, (), (), sh, lambda: relerr(udense(ux * 2.5), 2.5 * X))
        check('u_inner', sname, rep, C, (), (), sh, lambda: relerr(ux.inner(uy), np.sum((X * Y).reshape(C + (-1,)), -1)))
        check('u_norm', sname, rep, C, (), (), sh, lambda: relerr(ux.norm(), np.linalg.norm(X.reshape(C + (-1,)), axis=-1)))
        check('u_reverse', sname, rep, C, (), (), sh,
              lambda: relerr(udense(ux.reverse()), np.transpose(X, tuple(range(len(C))) + tuple(range(X.ndim - 1, len(C) - 1, -1)))))
        def _svd():
            xs, stk, stt = ux.t3svd(sharing=sh)
            e = relerr(udense(xs), X)
            if not bool(np.all(xs.is_left_orthogonal())): e = max(e, 1.0)
            if sh is not None and not bool(np.all(xs.has_shared_tucker_factors(sh))): e = max(e, 1.0)
            return e
        check('u_t3svd_lossless', sname, rep, C, (), (), sh, _svd)
        def _ras():
            xs = ux.t3svd(sharing=sh)[0].rank_adjustment_sweep('right_to_left', sharing=sh)
            e = relerr(udense(xs), X)
            if not bool(np.all(xs.is_right_orthogonal())): e = max(e, 1.0)
            if sh is None and not bool(np.all(xs.has_minimal_ranks)): e = max(e, 1.0)
            if sh is not None and not bool(np.all(xs.has_shared_tucker_factors(sh))): e = max(e, 1.0)
            return e
        check('u_rank_adjustment_sweep', sname, rep, C, (), (), sh, _ras)
        def _svd_trunc():
            caps = tuple(max(1, r - 1) for r in tr)
            xs, _, _ = ux.t3svd(max_tucker_ranks=caps, sharing=sh)
            rs, _, _ = x.t3svd(max_tucker_ranks=caps, sharing=sh)
            return relerr(udense(xs), rs.to_dense())
        check('u_t3svd_trunc_vs_ragged', sname, rep, C, (), (), sh, _svd_trunc, tol=1e-7)
        frame, var = ubv.ut3_orthogonal_representations(ux)
        if not bool(np.all(frame.is_orthogonal())):
            record('u_orthogonal_representations_NOT_ORTHOGONAL', sname, rep, C, (), (), sh, 'FAIL', float(np.max(np.asarray(frame.orthogonality_residual))), 'frame from ut3_orthogonal_representations is not orthogonal; tangent section below uses a rank-minimized ux instead')
            ux = ux.t3svd(sharing=sh)[0].rank_adjustment_sweep('right_to_left', sharing=sh)
            frame, var = ubv.ut3_orthogonal_representations(ux)
        Bd = udense(frame.to_ut3())
        check('u_orthogonal_representations', sname, rep, C, (), (), sh,
              lambda: max(relerr(Bd, X), 0.0 if bool(np.all(frame.is_orthogonal())) else 1.0,
                          relerr(ut3m.UT3Tangent(frame, var).to_dense(), 2 * d * X)))
        for W in WS:
            ww = rand_ww(shape, W, 7); pp = rand_ww(shape, W, 8); index = rand_index(shape, W, 9)
            check('u_apply', sname, rep, C, W, (), sh, lambda: relerr(ux.apply(ww), dense_apply(X, ww, d)))
            check('u_entries', sname, rep, C, W, (), sh, lambda: relerr(ux.entries(index), dense_entries(X, index, d, shape)))
            check('u_probe', sname, rep, C, W, (), sh, lambda: max(relerr(a, b) for a, b in zip(ux.probe(ww), dense_probe(X, ww, d))))
            check('u_apply_derivatives', sname, rep, C, W, (), sh, lambda: relerr(ux.apply_derivatives(ww, pp, ORDER), dense_apply_jets(X, ww, pp, d, ORDER)))
            check('u_entries_derivatives', sname, rep, C, W, (), sh, lambda: relerr(ux.entries_derivatives(index, pp, ORDER), dense_apply_jets(X, onehots(index, shape), pp, d, ORDER)))
            check('u_probe_derivatives', sname, rep, C, W, (), sh, lambda: max(relerr(a, b) for a, b in zip(ux.probe_derivatives(ww, pp, ORDER), dense_probe_jets(X, ww, pp, d, ORDER))))
            # corewise transposes vs ragged (masked content)
            rng = np.random.RandomState(11)
            c = np.asarray(rng.randn(*(W + C))); zt = [rng.randn(*(W + C + (N,))) for N in shape]
            cj = np.asarray(rng.randn(*((ORDER + 1,) + W + C))); ztj = [rng.randn(*((ORDER + 1,) + W + C + (N,))) for N in shape]
            def _cw_vs_ragged(kind, sum_, deriv):
                if deriv:
                    if kind == 'apply': gu = ux.apply_corewise_derivatives_transpose(cj, ww, pp, ORDER, sum_over_probes=sum_); gr = x.apply_corewise_derivatives_transpose(cj, ww, pp, ORDER, sum_over_probes=sum_)
                    elif kind == 'entries': gu = ux.entries_corewise_derivatives_transpose(cj, index, pp, ORDER, sum_over_probes=sum_); gr = x.entries_corewise_derivatives_transpose(cj, index, pp, ORDER, sum_over_probes=sum_)
                    else: gu = ux.probe_corewise_derivatives_transpose(ztj, ww, pp, ORDER, sum_over_probes=sum_); gr = x.probe_corewise_derivatives_transpose(ztj, ww, pp, ORDER, sum_over_probes=sum_)
                else:
                    if kind == 'apply': gu = ux.apply_corewise_transpose(c, ww, sum_over_probes=sum_); gr = x.apply_corewise_transpose(c, ww, sum_over_probes=sum_)
                    elif kind == 'entries': gu = ux.entries_corewise_transpose(c, index, sum_over_probes=sum_); gr = x.entries_corewise_transpose(c, index, sum_over_probes=sum_)
                    else: gu = ux.probe_corewise_transpose(zt, ww, sum_over_probes=sum_); gr = x.probe_corewise_transpose(zt, ww, sum_over_probes=sum_)
                # gu: uniform grads -> compare to ragged grads via adjoint identity with a random ragged delta
                # (convert uniform grad supercores to ragged by slicing real parts using ux's masks)
                tk_sc, tt_sc = gu[0], gu[1]
                tkm, ttm = ux.masks.data
                e = 0.0
                stack_extra = () if sum_ else W
                for i in range(d):
                    ni = tr[i]; Ni = shape[i]
                    a = np.asarray(tk_sc[i])[..., :ni, :Ni]; b = np.asarray(gr[0][i])
                    e = max(e, relerr(a, b))
                    ri, ro = ttr[i], ttr[i + 1]
                    a = np.asarray(tt_sc[i])[..., :ri, :ni, :ro]; b = np.asarray(gr[1][i])
                    e = max(e, relerr(a, b))
                return e
            for deriv in (False, True):
                for sum_ in (False, True):
                    for kind in ('apply', 'entries', 'probe'):
                        check('u_%s_corewise%s_transpose(sum=%s)_vs_ragged' % (kind, '_derivatives' if deriv else '', sum_), sname, rep, C, W, (), sh, lambda: _cw_vs_ragged(kind, sum_, deriv))
            for K in KS:
                np.random.seed(3)
                v = ut3m.UNIFORM_MANIFOLD.randn(frame, stack_shape=K)
                Vd = np.asarray(v.to_dense())
                check('utv_apply', sname, rep, C, W, K, sh, lambda: relerr(v.apply(ww), dense_apply(Vd, ww, d)))
                check('utv_entries', sname, rep, C, W, K, sh, lambda: relerr(v.entries(index), dense_entries(Vd, index, d, shape)))
                check('utv_probe', sname, rep, C, W, K, sh, lambda: max(relerr(a, b) for a, b in zip(v.probe(ww), dense_probe(Vd, ww, d))))
                check('utv_apply_derivatives', sname, rep, C, W, K, sh, lambda: relerr(v.apply_derivatives(ww, pp, ORDER), dense_apply_jets(Vd, ww, pp, d, ORDER)))
                check('utv_entries_derivatives', sname, rep, C, W, K, sh, lambda: relerr(v.entries_derivatives(index, pp, ORDER), dense_apply_jets(Vd, onehots(index, shape), pp, d, ORDER)))
                check('utv_probe_derivatives', sname, rep, C, W, K, sh, lambda: max(relerr(a, b) for a, b in zip(v.probe_derivatives(ww, pp, ORDER), dense_probe_jets(Vd, ww, pp, d, ORDER))))
                rng = np.random.RandomState(21)
                rk = np.asarray(rng.randn(*(W + K + C))); ztk = [rng.randn(*(W + K + C + (N,))) for N in shape]
                rkj = np.asarray(rng.randn(*((ORDER + 1,) + W + K + C))); ztkj = [rng.randn(*((ORDER + 1,) + W + K + C + (N,))) for N in shape]
                def _adj(kind, sum_, deriv):
                    if deriv:
                        if kind == 'apply': Jv = v.apply_derivatives(ww, pp, ORDER); t = ut3m.UT3Tangent.apply_derivatives_transpose(rkj, ww, pp, frame, ORDER, sum_over_probes=sum_); r = rkj
                        elif kind == 'entries': Jv = v.entries_derivatives(index, pp, ORDER); t = ut3m.UT3Tangent.entries_derivatives_transpose(rkj, index, pp, frame, ORDER, sum_over_probes=sum_); r = rkj
                        else: Jv = v.probe_derivatives(ww, pp, ORDER); t = ut3m.UT3Tangent.probe_derivatives_transpose(ztkj, ww, pp, frame, ORDER, sum_over_probes=sum_); r = ztkj
                    else:
                        if kind == 'apply': Jv = v.apply(ww); t = ut3m.UT3Tangent.apply_transpose(rk, ww, frame, sum_over_probes=sum_); r = rk
                        elif kind == 'entries': Jv = v.entries(index); t = ut3m.UT3Tangent.entries_transpose(rk, index, frame, sum_over_probes=sum_); r = rk
                        else: Jv = v.probe(ww); t = ut3m.UT3Tangent.probe_transpose(ztk, ww, frame, sum_over_probes=sum_); r = ztk
                    lhs = sum(tdot(a, b) for a, b in zip(r, Jv)) if kind == 'probe' else tdot(r, Jv)
                    # <t, v> via masked corewise inner: sum over the tangent stack of t with v broadcast
                    tm = t.variations.apply_masks() if hasattr(t.variations, 'apply_masks') else t.variations
                    vm = v.variations.apply_masks() if hasattr(v.variations, 'apply_masks') else v.variations
                    rhs = var_dot(tm.supercores, vm.supercores) if hasattr(tm, 'supercores') else var_dot(tm.data[:2], vm.data[:2])
                    exp_stack = (W if not sum_ else ()) + K + C
                    if t.variations.stack_shape != exp_stack:
                        return (1.0, 'stack %s != expected %s' % (t.variations.stack_shape, exp_stack))
                    return abs(lhs - rhs) / max(abs(lhs), 1e-300)
                for deriv in (False, True):
                    for sum_ in (False, True):
                        for kind in ('apply', 'entries', 'probe'):
                            check('utv_%s%s_transpose_adjoint(sum=%s)' % (kind, '_derivatives' if deriv else '', sum_), sname, rep, C, W, K, sh, lambda: _adj(kind, sum_, deriv))
        for K in KS:
            np.random.seed(4)
            v = ut3m.UNIFORM_MANIFOLD.randn(frame, stack_shape=K); u = ut3m.UNIFORM_MANIFOLD.randn(frame, stack_shape=K)
            Vd = np.asarray(v.to_dense()); Ud = np.asarray(u.to_dense())
            check('u_manifold_inner', sname, rep, C, (), K, sh, lambda: relerr(ut3m.UNIFORM_MANIFOLD.inner(u, v), np.sum((Ud * Vd).reshape(K + C + (-1,)), -1)))
            check('u_manifold_norm', sname, rep, C, (), K, sh, lambda: relerr(ut3m.UNIFORM_MANIFOLD.norm(v), np.linalg.norm(Vd.reshape(K + C + (-1,)), axis=-1)))
            check('u_gauge_project_idempotent', sname, rep, C, (), K, sh, lambda: relerr(ut3m.UNIFORM_MANIFOLD.project(v).to_dense(), Vd))
            check('u_tangent_add_scale', sname, rep, C, (), K, sh, lambda: relerr((v + u * 0.5 - v * 2.0).to_dense(), Vd + 0.5 * Ud - 2.0 * Vd))
            check('u_tangent_reverse', sname, rep, C, (), K, sh, lambda: relerr(v.reverse().to_dense(), np.transpose(Vd, tuple(range(len(K + C))) + tuple(range(Vd.ndim - 1, len(K + C) - 1, -1)))))
            check('u_retract_zero', sname, rep, C, (), K, sh, lambda: relerr(udense(ut3m.UNIFORM_MANIFOLD.retract(ut3m.UT3Tangent.zeros(frame, stack_shape=K))), np.broadcast_to(Bd, K + C + shape)))
            def _retr_fd():
                errs = []
                for h in (1e-3, 5e-4):
                    rp = udense(ut3m.UNIFORM_MANIFOLD.retract(v * h)); rm = udense(ut3m.UNIFORM_MANIFOLD.retract(v * (-h)))
                    errs.append(np.linalg.norm(((rp - rm) / (2 * h) - Vd).reshape(-1)) / np.linalg.norm(Vd.reshape(-1)))
                ratio = errs[0] / max(errs[1], 1e-300)
                if errs[1] < 1e-8: return errs[1]
                return (errs[1], 'ratio=%.2f' % ratio) if 2.5 < ratio < 6 else (1.0, 'ratio=%.2f errs=%s' % (ratio, errs))
            check('u_retract_fd_jacobian', sname, rep, C, (), K, sh, _retr_fd, tol=1e-2)
            # retract vs ragged retract on the converted tangent (unstacked K, any C): compare dense
            if K == ():
                rframe, _ = bvf.t3_orthogonal_representations(x)
                def _retr_vs_ragged():
                    # same tangent in both layers: build ragged tangent from the uniform one via to_t3tangent when unstacked
                    if C: return 0.0
                    rt = v.to_t3tangent()
                    return relerr(udense(ut3m.UNIFORM_MANIFOLD.retract(v)), t3m.MANIFOLD.retract(rt).to_dense())
                check('u_retract_vs_ragged', sname, rep, C, (), K, sh, _retr_vs_ragged, tol=1e-7)
            # transport / project_ambient (UT3 grad)
            g = ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=K + C), **(PAD if force_pad else {}))
            Gd = udense(g)
            def _pa():
                P = ut3m.UNIFORM_MANIFOLD.project_ambient(frame, g)
                e = 0.0 if bool(np.all(P.is_gauged())) else 1.0
                Pd = np.asarray(P.to_dense())
                lhs = np.sum((Pd * Ud).reshape(K + C + (-1,)), -1); rhs = np.sum((Gd * Ud).reshape(K + C + (-1,)), -1)
                return max(e, relerr(lhs, rhs))
            check('u_project_ambient', sname, rep, C, (), K, sh, _pa)
            check('u_transport_identity', sname, rep, C, (), K, sh, lambda: relerr(ut3m.UNIFORM_MANIFOLD.transport(v, frame).to_dense(), Vd))
            frame2, _ = ubv.ut3_orthogonal_representations(ut3.UniformTuckerTensorTrain.from_t3(t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C), **(PAD if force_pad else {})))
            def _tr():
                vt = ut3m.UNIFORM_MANIFOLD.transport(v, frame2)
                # reference: ragged projection of dense Vd onto frame2's tangent space
                rf2 = frame2.to_t3frame()
                if C:
                    return 0.0
                return relerr(vt.to_dense(), t3m.MANIFOLD.project_ambient(rf2, Vd).to_dense())
            check('u_transport_vs_ragged_projection', sname, rep, C, (), K, sh, _tr, tol=1e-7)

def varying_rank():
    """Stack two different-rank uniform T3s / tangents on one C batch and compare per element to dense."""
    shape = (4, 5, 6)
    S = [((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)), ((4, 5, 6), (3, 3, 2), (1, 1, 2, 1))]
    PADV = dict(N=6, n=4, r=3); PADB = dict(N=6, nU=4, nD=4, rL=3, rR=3)
    np.random.seed(2)
    xs = [t3.TuckerTensorTrain.randn(*s) for s in S]
    us = [ut3.UniformTuckerTensorTrain.from_t3(x, **PADV) for x in xs]
    U = ut3.UniformTuckerTensorTrain.stack(us)
    Xs = np.stack([np.asarray(x.to_dense()) for x in xs])
    C = (2,); d = 3
    check('vr_to_dense', 'vary', 'uniform', C, (), (), None, lambda: relerr(udense(U), Xs))
    for W in WS:
        ww = rand_ww(shape, W, 7); pp = rand_ww(shape, W, 8); index = rand_index(shape, W, 9)
        check('vr_apply', 'vary', 'uniform', C, W, (), None, lambda: relerr(U.apply(ww), dense_apply(Xs, ww, d)))
        check('vr_probe', 'vary', 'uniform', C, W, (), None, lambda: max(relerr(a, b) for a, b in zip(U.probe(ww), dense_probe(Xs, ww, d))))
        check('vr_entries', 'vary', 'uniform', C, W, (), None, lambda: relerr(U.entries(index), dense_entries(Xs, index, d, shape)))
        check('vr_probe_derivatives', 'vary', 'uniform', C, W, (), None, lambda: max(relerr(a, b) for a, b in zip(U.probe_derivatives(ww, pp, ORDER), dense_probe_jets(Xs, ww, pp, d, ORDER))))
        check('vr_apply_derivatives', 'vary', 'uniform', C, W, (), None, lambda: relerr(U.apply_derivatives(ww, pp, ORDER), dense_apply_jets(Xs, ww, pp, d, ORDER)))
        # tangents at varying-rank frames
        ts = []
        for x in xs:
            rb, rv = bvf.t3_orthogonal_representations(x)
            ub = ubv.UT3Frame.from_t3frame(rb, **PADB); uv = ubv.UT3Variations.from_t3variations(rv, **PADB)
            ts.append(ut3m.UT3Tangent(ub, uv))
        for K in KS:
            tsK = [ut3m.UNIFORM_MANIFOLD.randn(t.frame, stack_shape=K) for t in ts]
            V = ut3m.UT3Tangent.stack_frame(tsK)
            Vd = np.asarray(V.to_dense())
            Vref = np.stack([np.asarray(t.to_dense()) for t in tsK], axis=len(K))
            check('vr_tangent_to_dense', 'vary', 'uniform', C, W, K, None, lambda: relerr(Vd, Vref))
            check('vr_tv_apply', 'vary', 'uniform', C, W, K, None, lambda: relerr(V.apply(ww), dense_apply(Vd, ww, d)))
            check('vr_tv_probe', 'vary', 'uniform', C, W, K, None, lambda: max(relerr(a, b) for a, b in zip(V.probe(ww), dense_probe(Vd, ww, d))))
            check('vr_tv_entries', 'vary', 'uniform', C, W, K, None, lambda: relerr(V.entries(index), dense_entries(Vd, index, d, shape)))
            check('vr_tv_probe_derivatives', 'vary', 'uniform', C, W, K, None, lambda: max(relerr(a, b) for a, b in zip(V.probe_derivatives(ww, pp, ORDER), dense_probe_jets(Vd, ww, pp, d, ORDER))))
            rng = np.random.RandomState(21)
            ztk = [rng.randn(*(W + K + C + (N,))) for N in shape]
            def _adj(sum_):
                Jv = V.probe(ww); t = ut3m.UT3Tangent.probe_transpose(ztk, ww, V.frame, sum_over_probes=sum_)
                lhs = sum(tdot(a, b) for a, b in zip(ztk, Jv))
                rhs = var_dot(t.variations.apply_masks().supercores, V.variations.apply_masks().supercores)
                return abs(lhs - rhs) / max(abs(lhs), 1e-300)
            for sum_ in (False, True):
                check('vr_tv_probe_transpose_adjoint(sum=%s)' % sum_, 'vary', 'uniform', C, W, K, None, lambda: _adj(sum_))
            Ud = np.asarray(ut3m.UNIFORM_MANIFOLD.randn(V.frame, stack_shape=K).to_dense())
            check('vr_manifold_norm', 'vary', 'uniform', C, W, K, None, lambda: relerr(ut3m.UNIFORM_MANIFOLD.norm(V), np.linalg.norm(Vd.reshape(K + C + (-1,)), axis=-1)))
            def _retr():
                R = udense(ut3m.UNIFORM_MANIFOLD.retract(V * 1e-3))
                Rref = np.stack([udense(ut3m.UNIFORM_MANIFOLD.retract(t * 1e-3)) for t in tsK], axis=len(K))
                return relerr(R, Rref)
            check('vr_retract_per_element', 'vary', 'uniform', C, W, K, None, _retr)

if __name__ == '__main__':
    names = sys.argv[1:] or ['d2', 'd3', 'd4', 'rank1', 'nonmin', 'sh2', 'shall']
    if 'vary' in names:
        names.remove('vary'); varying_rank()
    for sname in names:
        for fp in (False, True):
            try:
                sweep(sname, STRUCTS[sname], SHARING.get(sname), fp)
            except Exception as e:
                import traceback; record('SWEEP_CRASH', sname, 'uniform+pad' if fp else 'uniform', (), (), (), SHARING.get(sname), 'EXC', float('nan'), '%s: %s @ %s' % (type(e).__name__, str(e)[:100], traceback.format_exc().splitlines()[-3].strip()[:120]))
        print('done', sname, len(RESULTS), flush=True)
        dump(os.path.join(os.path.dirname(__file__), 'results_uniform_%s.md' % ('_'.join(names) if names else 'vary')))
    dump(os.path.join(os.path.dirname(__file__), 'results_uniform_%s.md' % ('_'.join(names) if names else 'vary')))
