"""H3 sweep 2: T3Frame / T3Variations / T3Tangent / MANIFOLD / COREWISE ops over C x K x W vs per-element."""
import itertools, traceback
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
from t3toolbox import safety

np.random.seed(0)
STRUCTS = [
    ((5, 6, 7), (2, 3, 4), (1, 2, 3, 1)),
    ((5, 6, 7, 3), (2, 3, 4, 2), (1, 2, 3, 2, 1)),
    ((5, 4), (3, 2), (1, 2, 1)),
    ((6,), (3,), (1, 1)),
]
CASES = [((), ()), ((1,), ()), ((3,), ()), ((2, 3), ()), ((), (2,)), ((3,), (2,)), ((1,), (2,)), ((2, 3), (4,))]
WS = [(), (4,), (2, 2)]
FAILS = []
OK = 0


def report(name, ok, detail=''):
    global OK
    if ok:
        OK += 1
    else:
        FAILS.append((name, detail))
        print('FAIL', name, detail[:600])


def idxs(S):
    return list(itertools.product(*[range(c) for c in S]))


def close(a, b, tol=1e-8):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape:
        return False, 'shape %s vs %s' % (a.shape, b.shape)
    den = max(np.linalg.norm(b), 1.0)
    err = np.linalg.norm(a - b) / den
    return err < tol, 'relerr=%.2e' % err


def slice_frame(f, c):
    return bvf.T3Frame(*[tuple(A[c] for A in fam) for fam in f.data])


def slice_vars(v, kc):
    return bvf.T3Variations(*[tuple(A[kc] for A in fam) for fam in v.data])


def slice_t3(x, c):
    return t3.TuckerTensorTrain(tuple(B[c] for B in x.tucker_cores), tuple(G[c] for G in x.tt_cores))


def slice_tan(t, k, c):
    return t3m.T3Tangent(slice_frame(t.frame, c), slice_vars(t.variations, k + c))


def per_elem_check(name, K, C, stacked_arr, ref_fn, n_lead=0):
    """stacked_arr has K+C at axes n_lead.. ; ref_fn(k, c) -> scalar/array"""
    try:
        A = np.asarray(stacked_arr)
        for k in idxs(K):
            for c in idxs(C):
                got = A[(slice(None),) * n_lead + k + c]
                ok, dd = close(got, ref_fn(k, c))
                if not ok:
                    report(name, False, 'k=%s c=%s %s' % (k, c, dd))
                    return
        report(name, True)
    except Exception as e:
        report(name, False, 'EXC ' + repr(e) + traceback.format_exc()[-700:])


def tan_check(name, K, C, stacked_tan, ref_fn):
    """compare to_dense per element; ref_fn(k,c) -> T3Tangent or dense"""
    try:
        if stacked_tan.frame_stack_shape != C or stacked_tan.tangent_stack_shape != K:
            report(name, False, 'stacks C=%s K=%s got frame %s tangent %s' % (C, K, stacked_tan.frame_stack_shape, stacked_tan.tangent_stack_shape))
            return
        D = stacked_tan.to_dense()
        for k in idxs(K):
            for c in idxs(C):
                r = ref_fn(k, c)
                r = r.to_dense() if hasattr(r, 'to_dense') else r
                ok, dd = close(D[k + c], r)
                if not ok:
                    report(name, False, 'k=%s c=%s %s' % (k, c, dd))
                    return
        report(name, True)
    except Exception as e:
        report(name, False, 'EXC ' + repr(e) + traceback.format_exc()[-700:])


def t3_check(name, C, x, ref_fn):
    try:
        if x.stack_shape != C:
            report(name, False, 'stack %s != %s' % (x.stack_shape, C)); return
        D = x.to_dense()
        for c in idxs(C):
            r = ref_fn(c); r = r.to_dense() if hasattr(r, 'to_dense') else r
            ok, dd = close(D[c], r)
            if not ok:
                report(name, False, 'c=%s %s' % (c, dd)); return
        report(name, True)
    except Exception as e:
        report(name, False, 'EXC ' + repr(e) + traceback.format_exc()[-700:])


for (shape, nn, rr) in STRUCTS:
    d = len(shape)
    for (C, K) in CASES:
        tag = 'd=%d C=%s K=%s ' % (d, C, K)
        x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
        frame, vars0 = bvf.t3_orthogonal_representations(x)
        frames = {c: slice_frame(frame, c) for c in idxs(C)}
        # t3_orthogonal_representations per element: fv_to_t3 == x
        t3_check(tag + 'fv_to_t3(orth_reps, tt1)', C, bvf.fv_to_t3((True, min(1, d-1)), frame, vars0), lambda c: slice_t3(x, c))
        t3_check(tag + 'fv_to_t3(orth_reps, tucker0)', C, bvf.fv_to_t3((False, 0), frame, vars0), lambda c: slice_t3(x, c))
        for c in idxs(C):
            fc, vc = bvf.t3_orthogonal_representations(slice_t3(x, c))
            ok, dd = close(fc.to_dense(), frame.to_dense()[c])
            if not ok:
                report(tag + 'frame.to_dense per elem', False, dd)
            ok2 = np.allclose(bvf.fv_to_t3((False, 0), fc, vc).to_dense(), slice_t3(x, c).to_dense())
            if not ok2:
                report(tag + 'orth_reps per elem', False, 'mismatch')
        # frame checkers
        try:
            r = frame.is_orthogonal(); report(tag + 'frame.is_orthogonal shape', np.shape(r) == C and np.all(r), str(np.shape(r)))
            r = frame.is_consistent(); report(tag + 'frame.is_consistent shape', np.shape(r) == C and np.all(r), str(np.shape(r)))
            r = frame.orthogonality_residual; report(tag + 'frame.orthogonality_residual shape', np.shape(r) == C, str(np.shape(r)))
            r = frame.allclose(frame); report(tag + 'frame.allclose shape', np.shape(r) == C and np.all(r), str(np.shape(r)))
            r = frame.has_minimal_ranks; report(tag + 'frame.has_minimal_ranks', True, str(r))
            fr = frame.reverse()
            per_elem_check(tag + 'frame.reverse', (), C, fr.to_dense(), lambda k, c: frames[c].reverse().to_dense())
            f2 = bvf.T3Frame.stack(frame.unstack())
            report(tag + 'frame unstack/stack', f2.stack_shape == C and np.allclose(f2.to_dense(), frame.to_dense()))
            # orthogonalize a non-orthogonal frame per element
            nf = bvf.T3Frame(*[tuple(A + 0.3 * np.random.randn(*A.shape) for A in fam) for fam in frame.data])
            of = nf.orthogonalize()
            per_elem_check(tag + 'frame.orthogonalize', (), C, of.to_dense(), lambda k, c: slice_frame(nf, c).orthogonalize().to_dense())
            report(tag + 'frame.orthogonalize is_orthogonal', np.all(of.is_orthogonal()))
        except Exception as e:
            report(tag + 'frame ops', False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])

        # tangents
        v = t3m.MANIFOLD.randn(frame, stack_shape=K)
        w = t3m.MANIFOLD.randn(frame, stack_shape=K)
        vs = {(k, c): slice_tan(v, k, c) for k in idxs(K) for c in idxs(C)}
        ws = {(k, c): slice_tan(w, k, c) for k in idxs(K) for c in idxs(C)}
        report(tag + 'tangent stack attrs', v.frame_stack_shape == C and v.tangent_stack_shape == K and v.stack_shape == K + C,
               '%s %s %s' % (v.frame_stack_shape, v.tangent_stack_shape, v.stack_shape))
        tan_check(tag + 'to_dense', K, C, v, lambda k, c: vs[(k, c)])
        per_elem_check(tag + 'to_dense(shift)', K, C, v.to_dense(include_shift=True), lambda k, c: vs[(k, c)].to_dense(include_shift=True))
        try:
            vt3 = v.to_t3()
            report(tag + 'to_t3 stack', vt3.stack_shape == K + C, str(vt3.stack_shape))
            per_elem_check(tag + 'to_t3', K, C, vt3.to_dense(), lambda k, c: vs[(k, c)].to_dense())
            per_elem_check(tag + 'to_t3(shift)', K, C, v.to_t3(include_shift=True).to_dense(), lambda k, c: vs[(k, c)].to_dense(include_shift=True))
        except Exception as e:
            report(tag + 'to_t3', False, 'EXC ' + repr(e))
        # metrics
        per_elem_check(tag + 'corewise_inner', K, C, v.corewise_inner(w), lambda k, c: vs[(k, c)].corewise_inner(ws[(k, c)]))
        per_elem_check(tag + 'corewise_norm', K, C, v.corewise_norm(), lambda k, c: vs[(k, c)].corewise_norm())
        per_elem_check(tag + 'MANIFOLD.inner', K, C, t3m.MANIFOLD.inner(v, w), lambda k, c: np.sum(vs[(k, c)].to_dense() * ws[(k, c)].to_dense()))
        per_elem_check(tag + 'MANIFOLD.norm', K, C, t3m.MANIFOLD.norm(v), lambda k, c: np.linalg.norm(vs[(k, c)].to_dense()))
        per_elem_check(tag + 'COREWISE.inner', K, C, t3m.COREWISE.inner(v, w), lambda k, c: vs[(k, c)].corewise_inner(ws[(k, c)]))
        per_elem_check(tag + 'gauge_residual', K, C, v.gauge_residual, lambda k, c: vs[(k, c)].gauge_residual)
        try:
            r = v.is_gauged(); report(tag + 'is_gauged shape', np.shape(r) == K + C and np.all(r), str(np.shape(r)))
            r = v.is_orthogonal(); report(tag + 'tangent.is_orthogonal shape', np.shape(r) == C, str(np.shape(r)))
            r = v.allclose(v); report(tag + 'tangent.allclose shape', np.shape(r) == K + C and np.all(r), str(np.shape(r)))
            r = v.allclose(w); report(tag + 'tangent.allclose(w) false', np.shape(r) == K + C and not np.any(r), str(r))
        except Exception as e:
            report(tag + 'checkers', False, 'EXC ' + repr(e))
        # algebra
        tan_check(tag + 'add', K, C, v + w, lambda k, c: vs[(k, c)] + ws[(k, c)])
        tan_check(tag + 'sub', K, C, v - w, lambda k, c: vs[(k, c)] - ws[(k, c)])
        tan_check(tag + 'scale', K, C, v * 2.5, lambda k, c: vs[(k, c)] * 2.5)
        tan_check(tag + 'neg', K, C, -v, lambda k, c: -vs[(k, c)])
        tan_check(tag + 'normalized', K, C, v.normalized(), lambda k, c: vs[(k, c)].normalized())
        # projections / retract
        raw = t3m.T3Tangent(frame, bvf.T3Variations.randn(frame.variation_shapes, K + C))
        raws = {(k, c): slice_tan(raw, k, c) for k in idxs(K) for c in idxs(C)}
        tan_check(tag + 'project', K, C, t3m.MANIFOLD.project(raw), lambda k, c: t3m.MANIFOLD.project(raws[(k, c)]))
        tan_check(tag + 'project_oblique', K, C, t3m.MANIFOLD.project_oblique(raw), lambda k, c: t3m.MANIFOLD.project_oblique(raws[(k, c)]))
        tan_check(tag + 'project_oblique preserves', K, C, t3m.MANIFOLD.project_oblique(raw), lambda k, c: raws[(k, c)])
        try:
            ret = t3m.MANIFOLD.retract(v)
            report(tag + 'retract stack', ret.stack_shape == K + C, str(ret.stack_shape))
            per_elem_check(tag + 'retract', K, C, ret.to_dense(), lambda k, c: t3m.MANIFOLD.retract(vs[(k, c)]).to_dense())
            cfr = t3m.COREWISE.frame(x); cv = t3m.COREWISE.randn(cfr, stack_shape=K) if 'stack_shape' in t3m.COREWISE.randn.__code__.co_varnames else t3m.T3Tangent(cfr, bvf.T3Variations.randn(cfr.variation_shapes, K + C))
            cvs = {(k, c): slice_tan(cv, k, c) for k in idxs(K) for c in idxs(C)}
            retc = t3m.COREWISE.retract(cv)
            report(tag + 'COREWISE.retract stack', retc.stack_shape == K + C, str(retc.stack_shape))
            per_elem_check(tag + 'COREWISE.retract', K, C, retc.to_dense(), lambda k, c: t3m.COREWISE.retract(cvs[(k, c)]).to_dense())
            per_elem_check(tag + 'COREWISE.retract dense', K, C, retc.to_dense(), lambda k, c: slice_t3(x, c).to_dense() + cvs[(k, c)].to_dense())
        except Exception as e:
            report(tag + 'retract', False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])
        # project_ambient with a C-stacked T3 grad and with dense
        try:
            g = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
            pa = t3m.MANIFOLD.project_ambient(frame, g)
            tan_check(tag + 'project_ambient(T3,C)', (), C, pa, lambda k, c: t3m.MANIFOLD.project_ambient(frames[c], slice_t3(g, c)))
            pad = t3m.MANIFOLD.project_ambient(frame, g.to_dense())
            tan_check(tag + 'project_ambient(dense,C)', (), C, pad, lambda k, c: t3m.MANIFOLD.project_ambient(frames[c], slice_t3(g, c).to_dense()))
            tan_check(tag + 'project_ambient(dense)==project_ambient(T3)', (), C, pad, lambda k, c: pa.to_dense()[c])
            if C:
                pads = t3m.MANIFOLD.project_ambient(frame, g.to_dense(), method='t3svd')
                tan_check(tag + 'project_ambient(dense,t3svd,C)', (), C, pads, lambda k, c: pa.to_dense()[c])
            # K-stacked dense grad against C-frame
            gk = np.random.randn(*(K + C + shape))
            pak = t3m.MANIFOLD.project_ambient(frame, gk)
            tan_check(tag + 'project_ambient(dense,K+C)', K, C, pak, lambda k, c: t3m.MANIFOLD.project_ambient(frames[c], gk[k + c]))
            # K-stacked T3 grad vs C-frame (heterogeneous broadcast): not a documented path; record behaviour
            if K:
                try:
                    gkt = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=K + C)
                    pakt = t3m.MANIFOLD.project_ambient(frame, gkt)
                    tan_check(tag + 'project_ambient(T3,K+C)', K, C, pakt, lambda k, c: t3m.MANIFOLD.project_ambient(frames[c], slice_t3(gkt, k + c)))
                except Exception as e:
                    report(tag + 'project_ambient(T3,K+C) [undocumented]', False, 'EXC ' + repr(e)[:200])
            # transport
            y = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
            nframe = bvf.t3_orthogonal_representations(y)[0]
            tr = t3m.MANIFOLD.transport(v, nframe)
            tan_check(tag + 'transport', K, C, tr, lambda k, c: t3m.MANIFOLD.transport(vs[(k, c)], slice_frame(nframe, c)))
        except Exception as e:
            report(tag + 'project_ambient/transport', False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])
        # sum_tangents, stack/unstack
        try:
            if K:
                st = v.sum_tangents()
                tan_check(tag + 'sum_tangents', (), C, st, lambda k, c: sum(vs[(kk, c)].to_dense() for kk in idxs(K)))
                tree = v.unstack_tangents()
                v2 = t3m.T3Tangent.stack_tangents(tree)
                report(tag + 'unstack/stack_tangents', v2.tangent_stack_shape == K and v2.frame_stack_shape == C and np.allclose(v2.to_dense(), v.to_dense()))
                leaf = tree
                for i in idxs(K)[-1]:
                    leaf = leaf[i]
                report(tag + 'unstack_tangents leaf', leaf.frame is v.frame and np.allclose(leaf.to_dense(), v.to_dense()[idxs(K)[-1]]))
            if C:
                tree = v.unstack_frame()
                v3 = t3m.T3Tangent.stack_frame(tree)
                report(tag + 'unstack/stack_frame', v3.tangent_stack_shape == K and v3.frame_stack_shape == C and np.allclose(v3.to_dense(), v.to_dense()))
                leaf = tree
                for i in idxs(C)[-1]:
                    leaf = leaf[i]
                report(tag + 'unstack_frame leaf', leaf.frame_stack_shape == () and leaf.tangent_stack_shape == K and np.allclose(leaf.to_dense(), v.to_dense()[(slice(None),) * len(K) + idxs(C)[-1]]))
            vec = v.to_vector()
            v4 = t3m.T3Tangent.from_vector(vec, frame, tangent_stack_shape=K)
            report(tag + 'to/from_vector', np.allclose(v4.to_dense(), v.to_dense()))
            vr = v.reverse()
            per_elem_check(tag + 'tangent.reverse', K, C, vr.to_dense(), lambda k, c: vs[(k, c)].reverse().to_dense())
            # variations
            vv = v.variations
            r = vv.allclose(vv); report(tag + 'variations.allclose shape', np.shape(r) == K + C, str(np.shape(r)))
            vv2 = bvf.T3Variations.stack(vv.unstack())
            report(tag + 'variations unstack/stack', vv2.stack_shape == K + C and all(np.allclose(a, b) for a, b in zip(vv2.tucker_variations, vv.tucker_variations)))
            if K + C:
                vss = vv.sum_stack(axis=0)
                report(tag + 'variations.sum_stack(0)', vss.stack_shape == (K + C)[1:] and np.allclose(vss.tucker_variations[0], vv.tucker_variations[0].sum(axis=0)))
        except Exception as e:
            report(tag + 'stack ops', False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])
        # weights
        try:
            xsv = x.t3svd()[0]
            Wt = t3.T3Weights.from_t3svd(xsv)
            fw = bvf.T3FrameWeights.from_t3weights(Wt)
            report(tag + 'T3FrameWeights stack', fw.stack_shape == C, str(fw.stack_shape))
            frame2 = bvf.t3_orthogonal_representations(xsv)[0]
            v2 = t3m.MANIFOLD.randn(frame2, stack_shape=K)
            v2s = {(k, c): slice_tan(v2, k, c) for k in idxs(K) for c in idxs(C)}
            fws = {c: bvf.T3FrameWeights(*[tuple(A[c] for A in fam) for fam in fw.data]) for c in idxs(C)}
            aw = v2.absorb_weights(fw)
            per_elem_check(tag + 'absorb_weights', K, C, aw.to_dense(), lambda k, c: v2s[(k, c)].absorb_weights(fws[c]).to_dense())
            per_elem_check(tag + 'weighted_norm', K, C, v2.weighted_norm(fw), lambda k, c: v2s[(k, c)].weighted_norm(fws[c]))
            per_elem_check(tag + 'weighted_inner', K, C, v2.weighted_inner(v2 * 0.5 + v2, fw), lambda k, c: v2s[(k, c)].weighted_inner(v2s[(k, c)] * 1.5, fws[c]))
            fw2 = bvf.T3FrameWeights.stack(fw.unstack())
            report(tag + 'frameweights unstack/stack', fw2.stack_shape == C)
            report(tag + 'frameweights reverse stack', fw.reverse().stack_shape == C)
            report(tag + 'frameweights concat/kron stack', fw.concatenate(fw).stack_shape == C and fw.kronecker(fw).stack_shape == C)
            if K:
                # a K+C weight must be rejected by the tangent methods
                fwk = bvf.T3FrameWeights(*[tuple(np.broadcast_to(A, K + A.shape) for A in fam) for fam in fw.data])
                try:
                    v2.weighted_norm(fwk); report(tag + 'K+C weight rejected', False, 'no raise')
                except ValueError:
                    report(tag + 'K+C weight rejected', True)
        except Exception as e:
            report(tag + 'weights', False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])

        # ---------- sampling with W ----------
        for Wsh in WS:
            wtag = tag + 'W=%s ' % (Wsh,)
            ww = tuple(np.random.randn(*(Wsh + (N,))) for N in shape)
            pp = tuple(np.random.randn(*(Wsh + (N,))) for N in shape)
            index = np.stack([np.random.randint(0, N, size=Wsh) for N in shape])
            widx = idxs(Wsh)
            wsl = lambda vv, wi: tuple(a[wi] for a in vv)
            # forward
            try:
                zz = v.probe(ww)
                ok = True; msg = ''
                for k_ in range(d):
                    if np.shape(zz[k_]) != Wsh + K + C + (shape[k_],):
                        ok = False; msg = 'shape %s' % (np.shape(zz[k_]),)
                for wi in widx:
                    for k in idxs(K):
                        for c in idxs(C):
                            ref = vs[(k, c)].probe(wsl(ww, wi))
                            for k_ in range(d):
                                if not np.allclose(zz[k_][wi + k + c], ref[k_]):
                                    ok = False; msg = 'mismatch w=%s k=%s c=%s' % (wi, k, c)
                            # also vs dense
                            Dv = vs[(k, c)].to_dense()
                            for k_ in range(d):
                                r = Dv
                                for m in range(d):
                                    if m == k_: continue
                                    r = np.tensordot(r, ww[m][wi], axes=([0 if m < k_ else 1], [0]))
                                if not np.allclose(zz[k_][wi + k + c], r):
                                    ok = False; msg = 'dense mismatch'
                report(wtag + 'probe', ok, msg)
                ap = v.apply(ww)
                ok = np.shape(ap) == Wsh + K + C; msg = str(np.shape(ap))
                for wi in widx:
                    for k in idxs(K):
                        for c in idxs(C):
                            r = vs[(k, c)].to_dense()
                            for m in range(d):
                                r = np.tensordot(r, ww[m][wi], axes=([0], [0]))
                            if not np.allclose(ap[wi + k + c], r):
                                ok = False; msg = 'mismatch'
                report(wtag + 'apply', ok, msg)
                en = v.entries(index)
                ok = np.shape(en) == Wsh + K + C; msg = str(np.shape(en))
                for wi in widx:
                    for k in idxs(K):
                        for c in idxs(C):
                            r = vs[(k, c)].to_dense()[tuple(index[(slice(None),) + wi])]
                            if not np.allclose(en[wi + k + c], r):
                                ok = False; msg = 'mismatch'
                report(wtag + 'entries', ok, msg)
            except Exception as e:
                report(wtag + 'forward', False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])
            # transposes: adjoint identity per element + per-element vs unstacked
            try:
                zt = tuple(np.random.randn(*(Wsh + K + C + (N,))) for N in shape)
                for sop in [False, True]:
                    T = t3m.T3Tangent.probe_transpose(zt, ww, frame, sum_over_probes=sop)
                    expK = (K if sop else Wsh + K)
                    ok = T.tangent_stack_shape == expK and T.frame_stack_shape == C; msg = 'stacks K=%s C=%s' % (T.tangent_stack_shape, T.frame_stack_shape)
                    TD = T.to_dense()
                    for k in idxs(K):
                        for c in idxs(C):
                            if sop:
                                ref = 0
                                for wi in widx:
                                    ref = ref + t3m.T3Tangent.probe_transpose(tuple(z[wi + k + c] for z in zt), wsl(ww, wi), frames[c]).to_dense()
                                if not np.allclose(TD[k + c], ref): ok = False; msg += ' mismatch k=%s c=%s' % (k, c)
                            else:
                                for wi in widx:
                                    REF_T = t3m.T3Tangent.probe_transpose(tuple(z[wi + k + c] for z in zt), wsl(ww, wi), frames[c]); ref = REF_T.to_dense()
                                    if not np.allclose(TD[wi + k + c], ref): ok = False; msg += ' mismatch w=%s k=%s c=%s' % (wi, k, c)
                                    # adjoint identity vs v
                                    lhs = sum(np.sum(zt[m][wi + k + c] * vs[(k, c)].probe(wsl(ww, wi))[m]) for m in range(d))
                                    rhs = float(cw.corewise_dot(REF_T.variations.data, vs[(k, c)].variations.data))
                                    if not np.allclose(lhs, rhs): ok = False; msg += ' adjoint fail'
                    report(wtag + 'probe_transpose sop=%s' % sop, ok, msg)
                    cres = np.asarray(np.random.randn(*(Wsh + K + C)))
                    T = t3m.T3Tangent.apply_transpose(cres, ww, frame, sum_over_probes=sop)
                    ok = T.tangent_stack_shape == expK and T.frame_stack_shape == C; msg = 'stacks K=%s C=%s' % (T.tangent_stack_shape, T.frame_stack_shape)
                    TD = T.to_dense()
                    for k in idxs(K):
                        for c in idxs(C):
                            if sop:
                                ref = sum(t3m.T3Tangent.apply_transpose(np.asarray(cres[wi + k + c]), wsl(ww, wi), frames[c]).to_dense() for wi in widx)
                                if not np.allclose(TD[k + c], ref): ok = False; msg += ' mismatch'
                            else:
                                for wi in widx:
                                    REF_T = t3m.T3Tangent.apply_transpose(np.asarray(cres[wi + k + c]), wsl(ww, wi), frames[c]); ref = REF_T.to_dense()
                                    if not np.allclose(TD[wi + k + c], ref): ok = False; msg += ' mismatch w=%s k=%s c=%s' % (wi, k, c)
                                    lhs = cres[wi + k + c] * vs[(k, c)].apply(wsl(ww, wi))
                                    rhs = float(cw.corewise_dot(REF_T.variations.data, vs[(k, c)].variations.data))
                                    if not np.allclose(lhs, rhs): ok = False; msg += ' adjoint fail'
                    report(wtag + 'apply_transpose sop=%s' % sop, ok, msg)
                    T = t3m.T3Tangent.entries_transpose(cres, index, frame, sum_over_probes=sop)
                    ok = T.tangent_stack_shape == expK and T.frame_stack_shape == C; msg = 'stacks K=%s C=%s' % (T.tangent_stack_shape, T.frame_stack_shape)
                    TD = T.to_dense()
                    for k in idxs(K):
                        for c in idxs(C):
                            if sop:
                                ref = sum(t3m.T3Tangent.entries_transpose(np.asarray(cres[wi + k + c]), index[(slice(None),) + wi], frames[c]).to_dense() for wi in widx)
                                if not np.allclose(TD[k + c], ref): ok = False; msg += ' mismatch'
                            else:
                                for wi in widx:
                                    REF_T = t3m.T3Tangent.entries_transpose(np.asarray(cres[wi + k + c]), index[(slice(None),) + wi], frames[c]); ref = REF_T.to_dense()
                                    if not np.allclose(TD[wi + k + c], ref): ok = False; msg += ' mismatch'
                                    lhs = cres[wi + k + c] * vs[(k, c)].entries(index[(slice(None),) + wi])
                                    rhs = float(cw.corewise_dot(REF_T.variations.data, vs[(k, c)].variations.data))
                                    if not np.allclose(lhs, rhs): ok = False; msg += ' adjoint fail'
                    report(wtag + 'entries_transpose sop=%s' % sop, ok, msg)
            except Exception as e:
                report(wtag + 'transposes', False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])
            # derivatives forward + transpose
            order = 2
            try:
                dz = v.probe_derivatives(ww, pp, order)
                ok = all(np.shape(dz[m]) == (order + 1,) + Wsh + K + C + (shape[m],) for m in range(d)); msg = str(np.shape(dz[0]))
                for wi in widx:
                    for k in idxs(K):
                        for c in idxs(C):
                            ref = vs[(k, c)].probe_derivatives(wsl(ww, wi), wsl(pp, wi), order)
                            for m in range(d):
                                if not np.allclose(dz[m][(slice(None),) + wi + k + c], ref[m]): ok = False; msg += ' mismatch'
                report(wtag + 'probe_derivatives', ok, msg)
                da = v.apply_derivatives(ww, pp, order)
                ok = np.shape(da) == (order + 1,) + Wsh + K + C; msg = str(np.shape(da))
                for wi in widx:
                    for k in idxs(K):
                        for c in idxs(C):
                            ref = vs[(k, c)].apply_derivatives(wsl(ww, wi), wsl(pp, wi), order)
                            if not np.allclose(da[(slice(None),) + wi + k + c], ref): ok = False; msg += ' mismatch'
                report(wtag + 'apply_derivatives', ok, msg)
                de = v.entries_derivatives(index, pp, order)
                ok = np.shape(de) == (order + 1,) + Wsh + K + C; msg = str(np.shape(de))
                for wi in widx:
                    for k in idxs(K):
                        for c in idxs(C):
                            ref = vs[(k, c)].entries_derivatives(index[(slice(None),) + wi], wsl(pp, wi), order)
                            if not np.allclose(de[(slice(None),) + wi + k + c], ref): ok = False; msg += ' mismatch'
                report(wtag + 'entries_derivatives', ok, msg)
                ztj = tuple(np.random.randn(*((order + 1,) + Wsh + K + C + (N,))) for N in shape)
                cj = np.random.randn(*((order + 1,) + Wsh + K + C))
                for sop in [False, True]:
                    expK = (K if sop else Wsh + K)
                    for nm, fn, res, fw_ in [
                        ('probe_derivatives_transpose', lambda r, wi_, fr: t3m.T3Tangent.probe_derivatives_transpose(r, wsl(ww, wi_), wsl(pp, wi_), fr, order, sum_over_probes=sop), ztj, lambda k, c: dz),
                        ('apply_derivatives_transpose', lambda r, wi_, fr: t3m.T3Tangent.apply_derivatives_transpose(r, wsl(ww, wi_), wsl(pp, wi_), fr, order, sum_over_probes=sop), cj, None),
                        ('entries_derivatives_transpose', lambda r, wi_, fr: t3m.T3Tangent.entries_derivatives_transpose(r, index[(slice(None),) + wi_], wsl(pp, wi_), fr, order, sum_over_probes=sop), cj, None),
                    ]:
                        if nm.startswith('probe'):
                            T = t3m.T3Tangent.probe_derivatives_transpose(ztj, ww, pp, frame, order, sum_over_probes=sop)
                        elif nm.startswith('apply'):
                            T = t3m.T3Tangent.apply_derivatives_transpose(cj, ww, pp, frame, order, sum_over_probes=sop)
                        else:
                            T = t3m.T3Tangent.entries_derivatives_transpose(cj, index, pp, frame, order, sum_over_probes=sop)
                        ok = T.tangent_stack_shape == expK and T.frame_stack_shape == C; msg = 'stacks K=%s C=%s' % (T.tangent_stack_shape, T.frame_stack_shape)
                        TD = T.to_dense()
                        for k in idxs(K):
                            for c in idxs(C):
                                if nm.startswith('probe'):
                                    rs = lambda wi_: tuple(z[(slice(None),) + wi_ + k + c] for z in ztj)
                                else:
                                    rs = lambda wi_: cj[(slice(None),) + wi_ + k + c]
                                if sop:
                                    ref = sum(fn(rs(wi), wi, frames[c]).to_dense() for wi in widx)
                                    # per-element call with W=() on sum_over_probes=True
                                    if not np.allclose(TD[k + c], ref): ok = False; msg += ' mismatch k=%s c=%s' % (k, c)
                                else:
                                    for wi in widx:
                                        REF_T = fn(rs(wi), wi, frames[c]); ref = REF_T.to_dense()
                                        if not np.allclose(TD[wi + k + c], ref): ok = False; msg += ' mismatch w=%s k=%s c=%s' % (wi, k, c)
                                        # adjoint identity
                                        if nm.startswith('probe'):
                                            fwd = vs[(k, c)].probe_derivatives(wsl(ww, wi), wsl(pp, wi), order)
                                            lhs = sum(np.sum(rs(wi)[m] * fwd[m]) for m in range(d))
                                        elif nm.startswith('apply'):
                                            lhs = np.sum(rs(wi) * vs[(k, c)].apply_derivatives(wsl(ww, wi), wsl(pp, wi), order))
                                        else:
                                            lhs = np.sum(rs(wi) * vs[(k, c)].entries_derivatives(index[(slice(None),) + wi], wsl(pp, wi), order))
                                        rhs = float(cw.corewise_dot(REF_T.variations.data, vs[(k, c)].variations.data))
                                        if not np.allclose(lhs, rhs): ok = False; msg += ' adjoint fail'
                        report(wtag + nm + ' sop=%s' % sop, ok, msg)
            except Exception as e:
                report(wtag + 'derivatives', False, 'EXC ' + repr(e) + traceback.format_exc()[-700:])

print('\n=== OK:', OK, ' FAILS:', len(FAILS))
for f in FAILS:
    print('  ', f[0], '|', f[1][:300])
