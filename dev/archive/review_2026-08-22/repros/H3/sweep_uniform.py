"""H3 sweep 3: uniform twins over C x K x W vs the ragged per-element results (dense comparison)."""
import itertools, traceback
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m

np.random.seed(0)
STRUCTS = [
    ((5, 6, 7), (2, 3, 4), (1, 2, 3, 1)),
    ((5, 6, 7, 3), (2, 3, 4, 2), (1, 2, 3, 2, 1)),
    ((5, 4), (3, 2), (1, 2, 1)),
]
CASES = [((), ()), ((1,), ()), ((3,), ()), ((2, 3), ()), ((), (2,)), ((3,), (2,)), ((2, 3), (4,))]
WS = [(), (4,), (2, 2)]
FAILS = []; OK = 0


def report(name, ok, detail=''):
    global OK
    if ok: OK += 1
    else:
        FAILS.append((name, detail)); print('FAIL', name, detail[:500])


def idxs(S): return list(itertools.product(*[range(c) for c in S]))


def close(a, b, tol=1e-8):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape: return False, 'shape %s vs %s' % (a.shape, b.shape)
    err = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)
    return err < tol, 'relerr=%.2e' % err


def cmp(name, got, ref):
    try:
        ok, dd = close(got, ref); report(name, ok, dd)
    except Exception as e:
        report(name, False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])


def slice_t3(x, c): return t3.TuckerTensorTrain(tuple(B[c] for B in x.tucker_cores), tuple(G[c] for G in x.tt_cores))
def slice_frame(f, c): return bvf.T3Frame(*[tuple(A[c] for A in fam) for fam in f.data])
def slice_vars(v, kc): return bvf.T3Variations(*[tuple(A[kc] for A in fam) for fam in v.data])
def slice_tan(t, k, c): return t3m.T3Tangent(slice_frame(t.frame, c), slice_vars(t.variations, k + c))


for (shape, nn, rr) in STRUCTS:
    d = len(shape)
    for (C, K) in CASES:
        tag = 'd=%d C=%s K=%s ' % (d, C, K)
        x = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
        y = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x); uy = ut3.UniformTuckerTensorTrain.from_t3(y)
        report(tag + 'ut3 stack_shape', ux.stack_shape == C, str(ux.stack_shape))
        report(tag + 'ut3 supercore layout (d,)+C', ux.tucker_supercore.shape[:1 + len(C)] == (d,) + C, str(ux.tucker_supercore.shape))
        XD = x.to_dense(); YD = y.to_dense()
        cmp(tag + 'ut3.to_dense', ux.to_dense(), XD)
        cmp(tag + 'ut3 add', (ux + uy).to_dense(), XD + YD)
        cmp(tag + 'ut3 sub', (ux - uy).to_dense(), XD - YD)
        cmp(tag + 'ut3 scale', (ux * 2.5).to_dense(), 2.5 * XD)
        try:
            cmp(tag + 'ut3 mul', (ux * uy).to_dense(), XD * YD)
        except Exception as e:
            report(tag + 'ut3 mul', False, 'EXC ' + repr(e)[:100])
        for uo in [True, False]:
            cmp(tag + 'ut3 inner orth=%s' % uo, ux.inner(uy, use_orthogonalization=uo), x.inner(y, use_orthogonalization=uo))
            cmp(tag + 'ut3 norm orth=%s' % uo, ux.norm(use_orthogonalization=uo), x.norm(use_orthogonalization=uo))
        cmp(tag + 'ut3 sum', ux.sum(), x.sum())
        if C:
            cmp(tag + 'ut3 sum_stack', ux.sum_stack().to_dense(), XD.reshape((-1,) + shape).sum(axis=0))
        try:
            uxs = ux.t3svd()[0]; cmp(tag + 'ut3 t3svd', uxs.to_dense(), XD)
            cmp(tag + 'ut3 t3svd(max)', (ux + uy).t3svd(max_tt_ranks=2, max_tucker_ranks=2)[0].to_dense(), (x + y).t3svd(max_tt_ranks=2, max_tucker_ranks=2)[0].to_dense())
            for k in range(d):
                cmp(tag + 'ut3 t3svd tucker sv[%d]' % k, np.asarray(ux.t3svd()[1])[k][..., :nn[k]] if False else np.asarray(ux.t3svd()[1])[k], None) if False else None
            cmp(tag + 'ut3 ras r2l', uxs.rank_adjustment_sweep('right_to_left').to_dense(), XD)
            cmp(tag + 'ut3 ras l2r', uxs.rank_adjustment_sweep('right_to_left').rank_adjustment_sweep('left_to_right').to_dense(), XD)
            for meth in ['down_orthogonalize_tucker_cores', 'up_orthogonalize_tt_cores', 'left_orthogonalize_tt_cores', 'right_orthogonalize_tt_cores']:
                cmp(tag + 'ut3 ' + meth, getattr(ux, meth)().to_dense(), XD)
            r = ux.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores().is_left_orthogonal()
            report(tag + 'ut3 is_left_orthogonal', np.shape(r) == C and np.all(r), str(np.shape(r)))
            cmp(tag + 'ut3 reverse', ux.reverse().to_dense(), x.reverse().to_dense())
            cmp(tag + 'ut3 squash_tails', ux.squash_tails().to_dense(), XD)
            u2 = ut3.UniformTuckerTensorTrain.stack(ux.unstack())
            report(tag + 'ut3 unstack/stack', u2.stack_shape == C and np.allclose(u2.to_dense(), XD))
            tree = ux.to_t3()
            if C:
                leaf = tree
                for i in idxs(C)[-1]: leaf = leaf[i]
                cmp(tag + 'ut3 to_t3 leaf', leaf.to_dense(), XD[idxs(C)[-1]])
            else:
                cmp(tag + 'ut3 to_t3', tree.to_dense(), XD)
            r = ux.has_minimal_ranks; report(tag + 'ut3 has_minimal_ranks shape', np.shape(r) == C, str(np.shape(r)))
            # weights
            Wr = t3.T3Weights.from_t3svd(x.t3svd()[0]); xsv = x.t3svd()[0]
            uxsv = ut3.UniformTuckerTensorTrain.from_t3(xsv)
            UW = ut3.UT3Weights.from_ut3svd(uxsv)
            report(tag + 'UT3Weights stack', UW.stack_shape == C, str(UW.stack_shape))
            cmp(tag + 'ut3 absorb_weights', ut3.ut3_absorb_weights(uxsv, UW).to_dense(), t3.t3_absorb_weights(xsv, Wr).to_dense())
            cmp(tag + 'ut3 weighted_norm', ut3.ut3_weighted_norm(uxsv, UW), t3.t3_weighted_norm(xsv, Wr))
            cmp(tag + 'ut3 weighted_inner', ut3.ut3_weighted_inner(uxsv, UW, uxsv + uxsv, UW.concatenate(UW)) if False else ut3.ut3_weighted_inner(uxsv, UW, uxsv, UW), t3.t3_weighted_inner(xsv, Wr, xsv, Wr))
        except Exception as e:
            report(tag + 'ut3 misc', False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])

        # tangent layer
        try:
            frame, _ = bvf.t3_orthogonal_representations(x)
            uframe = ubv.UT3Frame.from_t3frame(frame)
            report(tag + 'uframe stack', uframe.stack_shape == C, str(uframe.stack_shape))
            cmp(tag + 'uframe to_dense', uframe.to_dense(), frame.to_dense())
            r = uframe.is_orthogonal(); report(tag + 'uframe is_orthogonal', np.shape(r) == C and np.all(r), str(np.shape(r)))
            uframe2 = ubv.UT3Frame.from_ut3(ux)
            cmp(tag + 'uframe from_ut3 to_dense', uframe2.to_dense(), frame.to_dense())
            cmp(tag + 'uframe reverse', uframe.reverse().to_dense(), frame.reverse().to_dense())
            v = t3m.MANIFOLD.randn(frame, stack_shape=K); w = t3m.MANIFOLD.randn(frame, stack_shape=K)
            uv = ut3m.UT3Tangent.from_t3tangent(v); uw = ut3m.UT3Tangent.from_t3tangent(w)
            report(tag + 'utangent stacks', uv.frame_stack_shape == C and uv.tangent_stack_shape == K, '%s %s' % (uv.frame_stack_shape, uv.tangent_stack_shape))
            VD = v.to_dense(); WD = w.to_dense()
            cmp(tag + 'utangent to_dense', uv.to_dense(), VD)
            cmp(tag + 'utangent to_dense(shift)', uv.to_dense(include_shift=True), v.to_dense(include_shift=True))
            cmp(tag + 'utangent to_ut3', uv.to_ut3().to_dense(), VD)
            cmp(tag + 'utangent corewise_inner', uv.corewise_inner(uw), v.corewise_inner(w))
            cmp(tag + 'utangent corewise_norm', uv.corewise_norm(), v.corewise_norm())
            cmp(tag + 'UNIFORM_MANIFOLD.inner', ut3m.UNIFORM_MANIFOLD.inner(uv, uw), t3m.MANIFOLD.inner(v, w))
            cmp(tag + 'UNIFORM_MANIFOLD.norm', ut3m.UNIFORM_MANIFOLD.norm(uv), t3m.MANIFOLD.norm(v))
            cmp(tag + 'UNIFORM_COREWISE.inner', ut3m.UNIFORM_COREWISE.inner(uv, uw), t3m.COREWISE.inner(v, w))
            cmp(tag + 'utangent gauge_residual shape', np.shape(uv.gauge_residual), np.shape(v.gauge_residual)) if False else report(tag + 'utangent gauge_residual shape', np.shape(uv.gauge_residual) == K + C, str(np.shape(uv.gauge_residual)))
            r = uv.is_gauged(); report(tag + 'utangent is_gauged', np.shape(r) == K + C and np.all(r), str(np.shape(r)))
            cmp(tag + 'utangent add', (uv + uw).to_dense(), VD + WD)
            cmp(tag + 'utangent scale', (uv * 2.5).to_dense(), 2.5 * VD)
            cmp(tag + 'utangent normalized', uv.normalized().to_dense(), v.normalized().to_dense())
            cmp(tag + 'utangent reverse', uv.reverse().to_dense(), v.reverse().to_dense())
            raw = t3m.T3Tangent(frame, bvf.T3Variations.randn(frame.variation_shapes, K + C))
            uraw = ut3m.UT3Tangent.from_t3tangent(raw)
            cmp(tag + 'UNIFORM_MANIFOLD.project', ut3m.UNIFORM_MANIFOLD.project(uraw).to_dense(), t3m.MANIFOLD.project(raw).to_dense())
            cmp(tag + 'UNIFORM_MANIFOLD.project_oblique', ut3m.UNIFORM_MANIFOLD.project_oblique(uraw).to_dense(), raw.to_dense())
            cmp(tag + 'UNIFORM_MANIFOLD.retract', ut3m.UNIFORM_MANIFOLD.retract(uv).to_dense(), t3m.MANIFOLD.retract(v).to_dense())
            cfr = t3m.COREWISE.frame(x); cv = t3m.T3Tangent(cfr, bvf.T3Variations.randn(cfr.variation_shapes, K + C)); ucv = ut3m.UT3Tangent.from_t3tangent(cv)
            cmp(tag + 'UNIFORM_COREWISE.retract', ut3m.UNIFORM_COREWISE.retract(ucv).to_dense(), t3m.COREWISE.retract(cv).to_dense())
            g = t3.TuckerTensorTrain.randn(shape, nn, rr, stack_shape=C)
            cmp(tag + 'UNIFORM_MANIFOLD.project_ambient', ut3m.UNIFORM_MANIFOLD.project_ambient(uframe, ut3.UniformTuckerTensorTrain.from_t3(g)).to_dense(), t3m.MANIFOLD.project_ambient(frame, g).to_dense())
            nframe = bvf.t3_orthogonal_representations(y)[0]
            cmp(tag + 'UNIFORM_MANIFOLD.transport', ut3m.UNIFORM_MANIFOLD.transport(uv, ubv.UT3Frame.from_t3frame(nframe)).to_dense(), t3m.MANIFOLD.transport(v, nframe).to_dense())
            if K:
                cmp(tag + 'utangent sum_tangents', uv.sum_tangents().to_dense(), v.sum_tangents().to_dense())
                u2 = ut3m.UT3Tangent.stack_tangents(uv.unstack_tangents())
                report(tag + 'utangent unstack/stack_tangents', u2.tangent_stack_shape == K and u2.frame_stack_shape == C and np.allclose(u2.to_dense(), VD))
            if C:
                u3 = ut3m.UT3Tangent.stack_frame(uv.unstack_frame())
                report(tag + 'utangent unstack/stack_frame', u3.tangent_stack_shape == K and u3.frame_stack_shape == C and np.allclose(u3.to_dense(), VD))
            tt = uv.to_t3tangent()
            if K + C:
                leaf = tt
                for i in idxs(K + C)[-1]: leaf = leaf[i]
                cmp(tag + 'utangent to_t3tangent leaf', leaf.to_dense(), VD[idxs(K + C)[-1]])
            else:
                cmp(tag + 'utangent to_t3tangent', tt.to_dense(), VD)
            # frame weights
            xsv = x.t3svd()[0]; frame2 = bvf.t3_orthogonal_representations(xsv)[0]
            fw = bvf.T3FrameWeights.from_t3weights(t3.T3Weights.from_t3svd(xsv))
            ufw = ubv.UT3FrameWeights.from_t3frameweights(fw, N=uframe.N if hasattr(uframe, 'N') else None) if False else ubv.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(ut3.UniformTuckerTensorTrain.from_t3(xsv)))
            v2 = t3m.MANIFOLD.randn(frame2, stack_shape=K)
            uframe3 = ubv.UT3Frame.from_t3frame(frame2)
            uv2 = ut3m.UT3Tangent.from_t3tangent(v2)
            report(tag + 'UT3FrameWeights stack', ufw.stack_shape == C, str(ufw.stack_shape))
            cmp(tag + 'utangent absorb_weights', uv2.absorb_weights(ufw).to_dense(), v2.absorb_weights(fw).to_dense())
            cmp(tag + 'utangent weighted_norm', uv2.weighted_norm(ufw), v2.weighted_norm(fw))
            cmp(tag + 'utangent weighted_inner', uv2.weighted_inner(uv2 * 1.5, ufw), v2.weighted_inner(v2 * 1.5, fw))
        except Exception as e:
            report(tag + 'utangent misc', False, 'EXC ' + repr(e) + traceback.format_exc()[-600:])

        # sampling
        for Wsh in WS:
            wtag = tag + 'W=%s ' % (Wsh,)
            ww = tuple(np.random.randn(*(Wsh + (N,))) for N in shape)
            pp = tuple(np.random.randn(*(Wsh + (N,))) for N in shape)
            index = np.stack([np.random.randint(0, N, size=Wsh) for N in shape])
            try:
                cmp(wtag + 'ut3.entries', ux.entries(index), x.entries(index))
                cmp(wtag + 'ut3.apply', ux.apply(ww), x.apply(ww))
                for m in range(d):
                    cmp(wtag + 'ut3.probe[%d]' % m, ux.probe(ww)[m], x.probe(ww)[m])
                cres = np.asarray(np.random.randn(*(Wsh + C)))
                zt = tuple(np.random.randn(*(Wsh + C + (N,))) for N in shape)
                order = 2
                cmp(wtag + 'ut3.apply_derivatives', ux.apply_derivatives(ww, pp, order), x.apply_derivatives(ww, pp, order))
                cmp(wtag + 'ut3.entries_derivatives', ux.entries_derivatives(index, pp, order), x.entries_derivatives(index, pp, order))
                for m in range(d):
                    cmp(wtag + 'ut3.probe_derivatives[%d]' % m, ux.probe_derivatives(ww, pp, order)[m], x.probe_derivatives(ww, pp, order)[m])
                # corewise transposes: compare via the dense gradient identity <grad, dcore> -- instead compare masked cores vs ragged
                for sop in [False, True]:
                    gu = ux.apply_corewise_transpose(cres, ww, sum_over_probes=sop)
                    gr = x.apply_corewise_transpose(cres, ww, sum_over_probes=sop)
                    # gu are supercores (d,)+W+C+(n,N) padded; compare real slots
                    ok = True
                    for m in range(d):
                        a = np.asarray(gu[0])[m][..., :nn[m], :shape[m]]
                        if not np.allclose(a, gr[0][m]): ok = False
                    report(wtag + 'ut3.apply_corewise_transpose sop=%s' % sop, ok)
                    gu = ux.probe_corewise_transpose(zt, ww, sum_over_probes=sop)
                    gr = x.probe_corewise_transpose(zt, ww, sum_over_probes=sop)
                    ok = True
                    for m in range(d):
                        a = np.asarray(gu[0])[m][..., :nn[m], :shape[m]]
                        if not np.allclose(a, gr[0][m]): ok = False
                        b = np.asarray(gu[1])[m][..., :rr[m], :nn[m], :rr[m + 1]]
                        if not np.allclose(b, gr[1][m]): ok = False
                    report(wtag + 'ut3.probe_corewise_transpose sop=%s' % sop, ok)
            except Exception as e:
                report(wtag + 'ut3 sampling', False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])
            try:
                zt = tuple(np.random.randn(*(Wsh + K + C + (N,))) for N in shape)
                cres = np.asarray(np.random.randn(*(Wsh + K + C)))
                for m in range(d):
                    cmp(wtag + 'utangent.probe[%d]' % m, uv.probe(ww)[m], v.probe(ww)[m])
                cmp(wtag + 'utangent.apply', uv.apply(ww), v.apply(ww))
                cmp(wtag + 'utangent.entries', uv.entries(index), v.entries(index))
                for sop in [False, True]:
                    T = ut3m.UT3Tangent.probe_transpose(zt, ww, uframe, sum_over_probes=sop)
                    R = t3m.T3Tangent.probe_transpose(zt, ww, frame, sum_over_probes=sop)
                    report(wtag + 'utangent.probe_transpose stacks sop=%s' % sop, T.tangent_stack_shape == R.tangent_stack_shape and T.frame_stack_shape == C, '%s vs %s' % (T.tangent_stack_shape, R.tangent_stack_shape))
                    cmp(wtag + 'utangent.probe_transpose sop=%s' % sop, T.to_dense(), R.to_dense())
                    T = ut3m.UT3Tangent.apply_transpose(cres, ww, uframe, sum_over_probes=sop)
                    R = t3m.T3Tangent.apply_transpose(cres, ww, frame, sum_over_probes=sop)
                    cmp(wtag + 'utangent.apply_transpose sop=%s' % sop, T.to_dense(), R.to_dense())
                    T = ut3m.UT3Tangent.entries_transpose(cres, index, uframe, sum_over_probes=sop)
                    R = t3m.T3Tangent.entries_transpose(cres, index, frame, sum_over_probes=sop)
                    cmp(wtag + 'utangent.entries_transpose sop=%s' % sop, T.to_dense(), R.to_dense())
                order = 2
                for m in range(d):
                    cmp(wtag + 'utangent.probe_derivatives[%d]' % m, uv.probe_derivatives(ww, pp, order)[m], v.probe_derivatives(ww, pp, order)[m])
                cmp(wtag + 'utangent.apply_derivatives', uv.apply_derivatives(ww, pp, order), v.apply_derivatives(ww, pp, order))
                cmp(wtag + 'utangent.entries_derivatives', uv.entries_derivatives(index, pp, order), v.entries_derivatives(index, pp, order))
                ztj = tuple(np.random.randn(*((order + 1,) + Wsh + K + C + (N,))) for N in shape)
                cj = np.random.randn(*((order + 1,) + Wsh + K + C))
                for sop in [False, True]:
                    T = ut3m.UT3Tangent.probe_derivatives_transpose(ztj, ww, pp, uframe, order, sum_over_probes=sop)
                    R = t3m.T3Tangent.probe_derivatives_transpose(ztj, ww, pp, frame, order, sum_over_probes=sop)
                    cmp(wtag + 'utangent.probe_derivatives_transpose sop=%s' % sop, T.to_dense(), R.to_dense())
                    T = ut3m.UT3Tangent.apply_derivatives_transpose(cj, ww, pp, uframe, order, sum_over_probes=sop)
                    R = t3m.T3Tangent.apply_derivatives_transpose(cj, ww, pp, frame, order, sum_over_probes=sop)
                    cmp(wtag + 'utangent.apply_derivatives_transpose sop=%s' % sop, T.to_dense(), R.to_dense())
                    T = ut3m.UT3Tangent.entries_derivatives_transpose(cj, index, pp, uframe, order, sum_over_probes=sop)
                    R = t3m.T3Tangent.entries_derivatives_transpose(cj, index, pp, frame, order, sum_over_probes=sop)
                    cmp(wtag + 'utangent.entries_derivatives_transpose sop=%s' % sop, T.to_dense(), R.to_dense())
            except Exception as e:
                report(wtag + 'utangent sampling', False, 'EXC ' + repr(e) + traceback.format_exc()[-500:])

print('\n=== OK:', OK, ' FAILS:', len(FAILS))
for f in FAILS: print('  ', f[0], '|', f[1][:300])
