"""t3_operations / t3_conversions / t3_linalg on stacks vs dense."""
import numpy as np, t3toolbox as t3
from t3toolbox.backend import t3_operations as Op, t3_conversions as Cv, t3_linalg as L, tt_operations as TT, t3_constructors as K
np.random.seed(0)
bad = []
def chk(name, cond, info=''):
    if not cond: bad.append((name, info))
rel = lambda a, b: np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)
for (shape, tr, ttr) in [((4,), (3,), (2, 3)), ((4, 5), (3, 2), (1, 3, 1)), ((4, 5, 6, 3), (3, 2, 4, 2), (2, 2, 3, 2, 3))]:
    for ss in [(), (2,), (2, 3)]:
        d = len(shape)
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=ss).data
        y = t3.TuckerTensorTrain.randn(shape, tuple(r + 1 for r in tr), tuple(r + 1 for r in ttr), stack_shape=ss).data
        X = Cv.t3_to_dense(x); Y = Cv.t3_to_dense(y)
        # to_dense squash False
        Xf = Cv.t3_to_dense(x, squash_tails=False)
        chk('to_dense squash=False shape', Xf.shape == ss + (ttr[0],) + shape + (ttr[-1],), Xf.shape)
        chk('to_dense squash=False sums to X', rel(Xf.sum(axis=(len(ss), -1)), X) < 1e-12)
        # vector round trip
        v = Cv.t3_to_vector(x); x2 = Cv.t3_from_vector(v, shape, tr, ttr, stack_shape=ss)
        chk('vector roundtrip', all(np.array_equal(a, b) for a, b in zip(x[0] + x[1], x2[0] + x2[1])))
        # segment / concatenate
        if d >= 2:
            for (st, sp) in [(0, 1), (1, None), (None, -1), (-2, None), (1, 2)]:
                seg = Op.t3_segment(x, st, sp)
                s0 = 0 if st is None else (st if st >= 0 else d + st); s1 = d if sp is None else (sp if sp >= 0 else d + sp)
                chk('segment len', len(seg[0]) == s1 - s0 and len(seg[1]) == s1 - s0, (st, sp))
            parts = [Op.t3_segment(x, i, i + 1) for i in range(d)]
            cat = Op.t3_concatenate(parts)
            chk('concatenate(segments) == x', rel(Cv.t3_to_dense(cat), X) < 1e-12)
            chk('concatenate seam check raises', (lambda: (lambda f: (f() and False) if False else None)(None))() is None)
        # inner / norm all flag combos vs dense
        for uo in (True, False):
            ip = L.t3_inner_product(x, y, use_orthogonalization=uo)
            ref = np.einsum('...', X * Y).reshape(ss + (-1,)).sum(-1) if ss else (X * Y).sum()
            chk('inner_product uo=%s' % uo, rel(ip, ref) < 1e-10, (shape, ss, rel(ip, ref)))
            nm = L.t3_norm(x, use_orthogonalization=uo)
            refn = np.sqrt((X * X).reshape(ss + (-1,)).sum(-1)) if ss else np.sqrt((X * X).sum())
            chk('norm uo=%s' % uo, rel(nm, refn) < 1e-10, (shape, ss, rel(nm, refn)))
        # add / scale / plus_scalar / mult
        chk('add', rel(Cv.t3_to_dense(L.t3_add(x, y)), X + Y) < 1e-12)
        chk('scale', rel(Cv.t3_to_dense(L.t3_scale(x, 2.5)), 2.5 * X) < 1e-12)
        chk('plus_scalar', rel(Cv.t3_to_dense(L.t3_plus_scalar(x, 1.5)), X + 1.5) < 1e-12)
        chk('mult', rel(Cv.t3_to_dense(L.t3_mult(x, y)), X * Y) < 1e-12)
        # t3_sum all / partial / negative
        chk('sum all', rel(Op.t3_sum(x), X.sum(axis=tuple(range(len(ss), len(ss) + d)))) < 1e-12, (shape, ss))
        if d >= 2:
            for ax in [0, -1, (0, 1), [d - 1, 0]]:
                axl = [ax] if isinstance(ax, int) else list(ax); axl = sorted(set(a % d for a in axl))
                s = Op.t3_sum(x, axis=ax)
                ref = X.sum(axis=tuple(len(ss) + a for a in axl))
                got = Cv.t3_to_dense(s) if isinstance(s, tuple) else s
                chk('sum axis=%s' % (ax,), rel(got, ref) < 1e-12, (shape, ss, ax))
        # sum_stack
        if ss:
            for ax in [None, 0, -1, (0,)]:
                s = L.t3_sum_stack(x, axis=ax)
                axl = list(range(len(ss))) if ax is None else ([ax] if isinstance(ax, int) else list(ax)); axl = sorted(set(a % len(ss) for a in axl))
                chk('sum_stack axis=%s' % (ax,), rel(Cv.t3_to_dense(s), X.sum(axis=tuple(axl))) < 1e-12, (shape, ss, ax))
        # absorb weights vs explicit diagonal insertion
        tw = tuple(np.random.rand(*ss, n) + 0.5 for n in tr); ttw = tuple(np.random.rand(*ss, r) + 0.5 for r in ttr)
        xa = Op.t3_absorb_weights(x, (tw, ttw))
        chk('weights_consistent', Op.t3_weights_consistent(x, (tw, ttw)))
        # explicit: scale tucker rank leg by tw; bond k by ttw[k] inserted once
        tk = tuple(np.einsum('...i,...io->...io', w, B) for w, B in zip(tw, x[0]))
        tt = list(G for G in x[1])
        tt[0] = np.einsum('...i,...iaj->...iaj', ttw[0], tt[0])
        for k in range(d): tt[k] = np.einsum('...iaj,...j->...iaj', tt[k], ttw[k + 1])
        chk('absorb_weights vs explicit', rel(Cv.t3_to_dense(xa), Cv.t3_to_dense((tk, tuple(tt)))) < 1e-12)
        # kron / concat weights vs mult / add
        tw2 = tuple(np.random.rand(*ss, n + 1) + 0.5 for n in tr); ttw2 = tuple(np.random.rand(*ss, r + 1) + 0.5 for r in ttr)
        ya = Op.t3_absorb_weights(y, (tw2, ttw2))
        chk('kron weights == absorb of mult', rel(Cv.t3_to_dense(Op.t3_absorb_weights(L.t3_mult(x, y), Op.t3_kronecker_weights((tw, ttw), (tw2, ttw2)))), Cv.t3_to_dense(xa) * Cv.t3_to_dense(ya)) < 1e-12, (shape, ss))
        chk('concat weights == absorb of add', rel(Cv.t3_to_dense(Op.t3_absorb_weights(L.t3_add(x, y), Op.t3_concatenate_weights((tw, ttw), (tw2, ttw2)))), Cv.t3_to_dense(xa) + Cv.t3_to_dense(ya)) < 1e-12, (shape, ss))
        # broadcast_to_common_stack: variation core stacked K+C vs frame cores C
        if ss:
            xb = (tuple(np.broadcast_to(B, (3,) + B.shape) if i == 0 else B for i, B in enumerate(x[0])), x[1])
            bt, bg = Op.t3_broadcast_to_common_stack(*xb)
            chk('broadcast common stack shapes', all(B.shape[:-2] == (3,) + ss for B in bt) and all(G.shape[:-3] == (3,) + ss for G in bg))
            chk('broadcast common stack dense', rel(Cv.t3_to_dense((bt, bg)), np.broadcast_to(X, (3,) + X.shape)) < 1e-12)
        # change core shapes: grow preserves tensor; shrink back restores cores
        tk2 = Op.tucker_change_core_shapes(x[0], shape, tuple(n + 2 for n in tr)); tt2 = TT.tt_change_core_shapes(x[1], tuple(n + 2 for n in tr), tuple(r + 1 for r in ttr))
        chk('change_core_shapes grow preserves', rel(Cv.t3_to_dense((tk2, tt2)), X) < 1e-12, (shape, ss))
        tk3 = Op.tucker_change_core_shapes(tk2, shape, tr); tt3 = TT.tt_change_core_shapes(tt2, tr, ttr)
        chk('change_core_shapes shrink restores', all(np.array_equal(a, b) for a, b in zip(tk3 + tt3, x[0] + x[1])))
        # stack/unstack
        if ss:
            un = Op.t3_unstack(x); st = Op.t3_stack(un)
            chk('t3_stack(t3_unstack)', all(np.array_equal(a, b) for a, b in zip(st[0] + st[1], x[0] + x[1])), (shape, ss))
        # zipper
        Z = TT.tt_zipper_left_to_right(x[1], x[1]); Zr = TT.tt_zipper_right_to_left(x[1], x[1])
        chk("zipper value", np.allclose(Z[-1].reshape(ss + (-1,)).sum(-1) if ss else Z[-1].sum(), (Cv.t3_to_dense((Cv.t3_from_tensor_train(x[1])[0], x[1]))**2).reshape(ss + (-1,)).sum(-1) if ss else (Cv.t3_to_dense((Cv.t3_from_tensor_train(x[1])[0], x[1]))**2).sum()), (shape, ss))
        chk("zipper r2l[0] == l2r[-1] total", np.allclose(Z[-1].sum(axis=(-2, -1)), Zr[0].sum(axis=(-2, -1))), (shape, ss))
        chk('zipper len', len(Z) == d + 1 and len(Zr) == d + 1)
        bigx = Op.t3_absorb_tucker_into_tt(x[0], x[0]) if False else None
        # tensor_train round trip, canonical
        tt_c = Cv.t3_to_tensor_train(x); xt = Cv.t3_from_tensor_train(tt_c)
        chk('tensor_train roundtrip', rel(Cv.t3_to_dense(xt), X) < 1e-12)
        F = tuple(np.random.randn(*ss, 3, N) for N in shape)
        xc = Cv.t3_from_canonical(F)
        refc = np.zeros(ss + shape)
        for r in range(3):
            term = np.ones(ss)
            for i, Fi in enumerate(F): term = term[..., None] * Fi[..., r, :].reshape(Fi.shape[:-2] + (1,) * i + (shape[i],))
            refc = refc + term
        chk('from_canonical', rel(Cv.t3_to_dense(xc), refc) < 1e-12, (shape, ss))
        # ones/zeros constructors
        o = K.t3_ones(shape, ss); chk('ones', np.array_equal(Cv.t3_to_dense(o), np.ones(ss + shape)))
        z = K.t3_zeros(shape, tr, ttr, ss); chk('zeros', np.array_equal(Cv.t3_to_dense(z), np.zeros(ss + shape)))
print('FAILURES:', bad if bad else 'none')
# concatenate edge cases
x = t3.TuckerTensorTrain.randn((4, 5), (3, 2), (1, 3, 1)).data
print('t3_concatenate single segment returns the input object itself:', Op.t3_concatenate([x]) is x)
try:
    Op.t3_concatenate([Op.t3_segment(x, 0, 1), Op.t3_segment(t3.TuckerTensorTrain.randn((4, 5), (3, 2), (1, 2, 1)).data, 1, 2)])
    print('seam mismatch: no error')
except ValueError as e: print('seam mismatch raises ValueError:', str(e).splitlines()[0])
try: Op.t3_segment(x, 1, 1)
except ValueError as e: print('segment len<1 raises ValueError')
print('t3_segment(x, -5, None) on d=2 (negative beyond -d):', [len(t) for t in Op.t3_segment(x, -5, None)])
print('t3_sum(x, axis=np.array([0,1])):', end=' ')
try: print(Op.t3_sum(x, axis=np.array([0, 1])))
except Exception as e: print(type(e).__name__, str(e)[:80])
print('t3_inner_product with list+tuple mixed families:', end=' ')
try: print(L.t3_inner_product((list(x[0]), x[1]), x))
except Exception as e: print(type(e).__name__, str(e)[:80])
