"""R5 lane: sampling ops (probe/apply/entries) -- dense oracles, adjoint identities, from_sweep hooks,
W+K+C ordering, corewise + ambient transposes, project_ambient identity. Asymmetric shapes, d in 1..4."""
import itertools, math, sys
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.backend.probing as pr
import t3toolbox.backend.apply as ap
import t3toolbox.backend.entries as en

np.random.seed(0)
L = 'abcdefghijklmnopqrstuvwxyz'
FAIL = []
def check(name, a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        FAIL.append((name, 'SHAPE', a.shape, b.shape)); print('FAIL', name, 'shape', a.shape, b.shape); return
    err = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
    if not err < tol:
        FAIL.append((name, 'VALUE', err)); print('FAIL', name, 'relerr', err)

def dense_apply(ww, T, d):
    # T: S + N (S = any leading stack), ww[i]: W + (Ni,) -> W + S
    nW = ww[0].ndim - 1; nS = T.ndim - d
    Ws = L[:nW]; Ss = L[nW:nW+nS]; Ns = L[nW+nS:nW+nS+d]
    s = Ss + Ns + ',' + ','.join(Ws + Ns[i] for i in range(d)) + '->' + Ws + Ss
    return np.einsum(s, T, *ww)

def dense_entries(index, T, d):
    index = np.asarray(index); W = index.shape[1:]
    out = np.empty(W + T.shape[:T.ndim - d])
    for w in itertools.product(*[range(n) for n in W]):
        out[w] = T[(Ellipsis,) + tuple(int(index[(i,) + w]) for i in range(d))]
    return out

def cdot(A, B, n_lead):
    # sum over cores of <a,b> keeping n_lead leading axes
    return sum(np.sum(a * b, axis=tuple(range(n_lead, a.ndim))) for a, b in zip(A[0] + A[1], B[0] + B[1]))

STRUCTS = [
    ((7,),          (3,),        (1, 1)),
    ((7,),          (3,),        (2, 3)),          # boundary ranks != 1 (allowed: doctest uses (2,3,4,2))
    ((7, 5),        (3, 2),      (1, 2, 1)),
    ((7, 5, 6),     (3, 2, 4),   (1, 2, 3, 1)),
    ((7, 5, 6),     (3, 2, 4),   (2, 2, 3, 2)),
    ((6, 5, 7, 4),  (2, 3, 4, 2),(1, 2, 3, 2, 1)),
]
STACKS = [((), (), ()), ((1,), (), ()), ((2,), (), (3,)), ((2,), (3,), ()), ((2,), (3,), (4,)), ((2, 2), (3,), (4,))]  # (W, K, C)

for (shape, tr, rr), (W, K, C) in itertools.product(STRUCTS, STACKS):
    d = len(shape); nW, nK, nC = len(W), len(K), len(C)
    tag = f'd={d} shape={shape} tr={tr} rr={rr} W={W} K={K} C={C}'
    x = t3.TuckerTensorTrain.randn(shape, tr, rr, stack_shape=C)
    xd = x.to_dense()                                     # C + N
    ww = [np.random.randn(*(W + (N,))) for N in shape]
    index = np.array([np.random.randint(0, N, size=W) for N in shape])  # (d,)+W

    # ---- plain ops vs dense
    def dense_probe_safe(ww, T):
        try:
            return pr.dense_probe(ww, T)
        except Exception as e:
            FAIL.append(('dense_probe ' + tag, 'EXC', repr(e))); print('EXC dense_probe', tag, repr(e))
            # fallback oracle: explicit loop over modes via dense_apply with identity in the free mode
            out = []
            dd = len(ww); nWl = ww[0].ndim - 1
            for m in range(dd):
                Nm = ww[m].shape[-1]
                eye = np.broadcast_to(np.eye(Nm), ww[m].shape[:-1] + (Nm, Nm))  # W + (Nm, Nm)
                cols = []
                for jn in range(Nm):
                    wwm = list(ww); wwm[m] = eye[..., jn, :]
                    cols.append(dense_apply(wwm, T, dd))
                out.append(np.stack(cols, axis=-1))
            return out
    pd_ = dense_probe_safe(ww, xd)
    for i, (p, q) in enumerate(zip(x.probe(ww), pd_)):
        check(f'probe[{i}] ' + tag, p, q)
    check('apply ' + tag, x.apply(ww), dense_apply(ww, xd, d))
    check('entries ' + tag, x.entries(index), dense_entries(index, xd, d))

    # ---- tangent ops vs dense (bare J, no projector)
    frame, _ = bvf.t3_orthogonal_representations(x)
    v = t3m.COREWISE.randn(frame, stack_shape=K)          # raw variations, stack K + C
    vd = v.to_dense()                                     # K + C + N
    assert vd.shape == K + C + shape, (vd.shape, tag)
    Jp = v.probe(ww); Jp_d = dense_probe_safe(ww, vd)       # W + K + C + (Ni,)
    for i in range(d):
        assert Jp[i].shape == W + K + C + (shape[i],), (Jp[i].shape, tag)
        check(f'tv_probe[{i}] ' + tag, Jp[i], Jp_d[i])
    Ja = v.apply(ww); assert np.shape(Ja) == W + K + C, (np.shape(Ja), tag)
    check('tv_apply ' + tag, Ja, dense_apply(ww, vd, d))
    Je = v.entries(index); assert np.shape(Je) == W + K + C, (np.shape(Je), tag)
    check('tv_entries ' + tag, Je, dense_entries(index, vd, d))

    # ---- adjoint identities, both sum_over_probes, residual W+K+C
    z = [np.random.randn(*(W + K + C + (N,))) for N in shape]
    c = np.asarray(np.random.randn(*(W + K + C)))
    for sop in (True, False):
        n_lead = (0 if sop else nW)
        def lhs_of(zz, Jv):       # <z, Jv> summed over (K+C) and modes; keep W if not sop
            tot = 0
            for a, b in zip(zz, Jv):
                tot = tot + np.sum(a * b, axis=tuple(range(nW if not sop else 0, a.ndim)))
            return tot
        # probe
        JTz = t3m.T3Tangent.probe_transpose(z, ww, frame, sum_over_probes=sop)
        exp_tstack = (K if sop else W + K)
        assert JTz.tangent_stack_shape == exp_tstack and JTz.frame_stack_shape == C, (JTz.tangent_stack_shape, tag)
        rhs = cdot(JTz.variations.data, v.variations.data, n_lead)
        check(f'adj probe sop={sop} ' + tag, lhs_of(z, Jp), rhs)
        # apply
        ATc = t3m.T3Tangent.apply_transpose(c, ww, frame, sum_over_probes=sop)
        assert ATc.tangent_stack_shape == exp_tstack, (ATc.tangent_stack_shape, tag)
        lhs = np.sum(c * Ja, axis=tuple(range(nW if not sop else 0, c.ndim)))
        check(f'adj apply sop={sop} ' + tag, lhs, cdot(ATc.variations.data, v.variations.data, n_lead))
        # entries
        ETc = t3m.T3Tangent.entries_transpose(c, index, frame, sum_over_probes=sop)
        assert ETc.tangent_stack_shape == exp_tstack, (ETc.tangent_stack_shape, tag)
        lhs = np.sum(c * Je, axis=tuple(range(nW if not sop else 0, c.ndim)))
        check(f'adj entries sop={sop} ' + tag, lhs, cdot(ETc.variations.data, v.variations.data, n_lead))

        # ---- from_sweep hooks == one-shot
        fr = frame.data; vr = v.variations.data
        sw_p = pr.tv_precompute_probe_frame_sweep(fr, ww)
        sw_a = ap.tv_precompute_apply_frame_sweep(fr, ww)
        sw_e = en.tv_precompute_entries_frame_sweep(fr, index)
        for i, (a, b) in enumerate(zip(pr.tv_probe_jacobian_from_sweep(vr, ww, fr, sw_p), Jp)):
            check(f'probe_jac_from_sweep[{i}] ' + tag, a, b, 1e-13)
        check('apply_jac_from_sweep ' + tag, ap.tv_apply_jacobian_from_sweep(vr, ww, fr, sw_a), Ja, 1e-13)
        check('entries_jac_from_sweep ' + tag, en.tv_entries_jacobian_from_sweep(vr, index, fr, sw_e), Je, 1e-13)
        for nm, got, ref in [
            ('probe_T_from_sweep', pr.tv_probe_transpose_from_sweep(z, ww, fr, sw_p, sop), JTz.variations.data),
            ('apply_T_from_sweep', ap.tv_apply_transpose_from_sweep(c, ww, fr, sw_a, sop), ATc.variations.data),
            ('entries_T_from_sweep', en.tv_entries_transpose_from_sweep(c, index, fr, sw_e, sop), ETc.variations.data),
        ]:
            for j, (a, b) in enumerate(zip(got[0] + got[1], ref[0] + ref[1])):
                check(f'{nm}[{j}] sop={sop} ' + tag, a, b, 1e-13)

    # ---- corewise transposes vs EXACT corewise Jacobian (single-core replacement), residual W+C
    cc = np.asarray(np.random.randn(*(W + C))); zc = [np.random.randn(*(W + C + (N,))) for N in shape]
    tk, tt = [list(cs) for cs in x.data]
    dU = [np.random.randn(*u.shape) for u in tk]; dG = [np.random.randn(*g.shape) for g in tt]
    def replace(kind, i, new):
        a, b = list(tk), list(tt); (a if kind == 'U' else b)[i] = new
        return t3.TuckerTensorTrain(tuple(a), tuple(b))
    Jd_apply = sum(np.asarray(replace('U', i, dU[i]).apply(ww)) for i in range(d)) + sum(np.asarray(replace('G', i, dG[i]).apply(ww)) for i in range(d))
    Jd_entries = sum(np.asarray(replace('U', i, dU[i]).entries(index)) for i in range(d)) + sum(np.asarray(replace('G', i, dG[i]).entries(index)) for i in range(d))
    Jd_probe = [sum(np.asarray(replace('U', i, dU[i]).probe(ww)[m]) for i in range(d)) + sum(np.asarray(replace('G', i, dG[i]).probe(ww)[m]) for i in range(d)) for m in range(d)]
    for sop in (True, False):
        n_lead = 0 if sop else nW
        gA = x.apply_corewise_transpose(cc, ww, sum_over_probes=sop)
        check(f'corewise apply sop={sop} ' + tag, np.sum(cc * Jd_apply, axis=tuple(range(n_lead, cc.ndim))), cdot(gA, (dU, dG), n_lead))
        gE = x.entries_corewise_transpose(cc, index, sum_over_probes=sop)
        check(f'corewise entries sop={sop} ' + tag, np.sum(cc * Jd_entries, axis=tuple(range(n_lead, cc.ndim))), cdot(gE, (dU, dG), n_lead))
        gP = x.probe_corewise_transpose(zc, ww, sum_over_probes=sop)
        lhs = sum(np.sum(a * b, axis=tuple(range(n_lead, a.ndim))) for a, b in zip(zc, Jd_probe))
        check(f'corewise probe sop={sop} ' + tag, lhs, cdot(gP, (dU, dG), n_lead))
        if not sop:
            assert gA[0][0].shape == W + tk[0].shape, (gA[0][0].shape, tag)
            assert gP[1][0].shape == W + tt[0].shape, (gP[1][0].shape, tag)

    # ---- ambient transposes: <from_canonical(AT c), X>_F == <c, op(X)>, on an unstacked T3 (C=()) only
    if nC == 0:
        for sop in (True, False):
            fa = t3.TuckerTensorTrain.apply_ambient_transpose(cc, ww, sum_over_probes=sop)
            fe = t3.TuckerTensorTrain.entries_ambient_transpose(cc, index, shape, sum_over_probes=sop)
            fp = t3.TuckerTensorTrain.probe_ambient_transpose(zc, ww, sum_over_probes=sop)
            for nm, f, lhs_full in [('apply', fa, cc * np.asarray(x.apply(ww))), ('entries', fe, cc * np.asarray(x.entries(index))),
                                    ('probe', fp, sum(np.sum(a * b, axis=-1) for a, b in zip(zc, x.probe(ww))))]:
                T = t3.TuckerTensorTrain.from_canonical(f).to_dense()    # (W+) N
                if sop:
                    assert T.shape == shape, (T.shape, nm, tag)
                    check(f'ambient {nm} sop=True ' + tag, np.sum(T * xd), np.sum(lhs_full))
                else:
                    assert T.shape == W + shape, (T.shape, nm, tag)
                    check(f'ambient {nm} sop=False ' + tag, np.sum(T * xd, axis=tuple(range(nW, T.ndim))), lhs_full)

    # ---- docs/transposes.md: tangent = projection of the ambient back-projection (K=(), C=())
    if nC == 0 and nK == 0:
        ofr, _ = bvf.t3_orthogonal_representations(x)
        for nm, amb, tan in [
            ('apply', t3.TuckerTensorTrain.apply_ambient_transpose(cc, ww, sum_over_probes=True),
                      t3m.T3Tangent.apply_transpose(cc, ww, ofr, sum_over_probes=True)),
            ('entries', t3.TuckerTensorTrain.entries_ambient_transpose(cc, index, shape, sum_over_probes=True),
                        t3m.T3Tangent.entries_transpose(cc, index, ofr, sum_over_probes=True)),
            ('probe', t3.TuckerTensorTrain.probe_ambient_transpose(zc, ww, sum_over_probes=True),
                      t3m.T3Tangent.probe_transpose(zc, ww, ofr, sum_over_probes=True)),
        ]:
            try:
                pt = t3m.MANIFOLD.project(tan)
            except Exception as e:
                FAIL.append(('project(JT) ' + nm + ' ' + tag, 'EXC', repr(e))); print('EXC project', nm, tag, repr(e)); continue
            try:
                pa = t3m.MANIFOLD.project_ambient(ofr, t3.TuckerTensorTrain.from_canonical(amb))
                check(f'project_ambient==project(JT) {nm} ' + tag, pa.to_dense(), pt.to_dense(), 1e-8)
            except Exception as e:
                FAIL.append(('project_ambient(t3) ' + nm + ' ' + tag, 'EXC', repr(e))); print('EXC project_ambient(t3)', nm, tag, repr(e))
            try:
                pad = t3m.MANIFOLD.project_ambient(ofr, t3.TuckerTensorTrain.from_canonical(amb).to_dense())
                check(f'project_ambient(dense)==project(JT) {nm} ' + tag, pad.to_dense(), pt.to_dense(), 1e-8)
            except Exception as e:
                FAIL.append(('project_ambient(dense) ' + nm + ' ' + tag, 'EXC', repr(e))); print('EXC project_ambient(dense)', nm, tag, repr(e))

print('DONE. failures:', len(FAIL))
for f in FAIL[:40]:
    print(f)
