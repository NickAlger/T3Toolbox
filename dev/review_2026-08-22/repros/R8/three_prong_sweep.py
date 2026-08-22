"""R8: three-prong check of every public UniformTuckerTensorTrain op at asymmetric shapes and
varying-rank stacks: (1) per-element dense vs ragged twin, (2) exact output masks vs an independently
derived expectation, (3) garbage-padded input == clean input (real parts)."""
import numpy as np, itertools, traceback
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.ranks as ranks
from t3toolbox.backend.common import prefix_mask
from t3toolbox.backend.stacking import apply_func_to_leaf_subtrees

np.random.seed(0)
TOL = 1e-9
FAILS = []

def relerr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

def corrupt(ux, scale=10.0):
    tkm, ttm = ux.masks.data
    d, N, n, r = ux.d, ux.N, ux.n, ux.r
    stack = ux.stack_shape
    shape_mask = prefix_mask(ux.shape, N).reshape((d,) + (1,) * len(stack) + (1, N))
    tk_real = tkm[..., :, None] & shape_mask
    tt_real = ttm[:-1][..., :, None, None] & tkm[..., None, :, None] & ttm[1:][..., None, None, :]
    g_tk = scale * np.random.randn(*ux.tucker_supercore.shape) * (~tk_real)
    g_tt = scale * np.random.randn(*ux.tt_supercore.shape) * (~tt_real)
    return ut3.UniformTuckerTensorTrain(ux.tucker_supercore + g_tk, ux.tt_supercore + g_tt, ux.shape, ux.masks)

def check(name, cond, detail=''):
    if not cond:
        FAILS.append((name, detail))
        print('FAIL', name, detail)

def is_prefix(m):
    m = np.asarray(m, bool)
    flat = m.reshape(-1, m.shape[-1])
    return all(np.array_equal(row, prefix_mask(int(row.sum()), m.shape[-1])) for row in flat)

# ---------------------------------------------------------------- configs (asymmetric, d in 1..4)
CONFIGS = [  # (shape, tucker_ranks, tt_ranks)
    ((5,),        (3,),        (1, 1)),
    ((4, 6),      (2, 3),      (1, 3, 1)),
    ((4, 5, 3),   (2, 3, 2),   (1, 2, 3, 1)),
    ((3, 6, 4, 5),(2, 3, 3, 2),(1, 2, 4, 2, 1)),
]
STACKS = [(), (1,), (2,), (2, 3)]

def ragged_tree(shape, tk, tt, stack, vary):
    """A stack-shaped tree of ragged T3s; with vary=True every element gets its own (smaller) ranks."""
    def one(i):
        if not vary or i == 0:
            return t3.TuckerTensorTrain.randn(shape, tk, tt)
        tk2 = tuple(max(1, a - (i % 2)) for a in tk)
        tt2 = (1,) + tuple(max(1, b - ((i + j) % 2)) for j, b in enumerate(tt[1:-1])) + (1,)
        return t3.TuckerTensorTrain.randn(shape, tk2, tt2)
    n_el = int(np.prod(stack)) if stack else 1
    leaves = [one(i) for i in range(n_el)]
    if not stack:
        return leaves[0]
    def build(level, off):
        if level == len(stack) - 1:
            return tuple(leaves[off + k] for k in range(stack[level]))
        step = int(np.prod(stack[level + 1:]))
        return tuple(build(level + 1, off + k * step) for k in range(stack[level]))
    return build(0, 0)

def to_uniform(tree, N=None, n=None, r=None):
    if isinstance(tree, t3.TuckerTensorTrain):
        return ut3.UniformTuckerTensorTrain.from_t3(tree, N=N, n=n, r=r)
    return ut3.UniformTuckerTensorTrain.stack(
        apply_func_to_leaf_subtrees(tree, lambda x: ut3.UniformTuckerTensorTrain.from_t3(x, N=N, n=n, r=r), None))

def dense_tree(tree):
    if isinstance(tree, t3.TuckerTensorTrain):
        return tree.to_dense()
    return np.stack([dense_tree(t) for t in tree])

def map_tree(f, tree):
    if isinstance(tree, t3.TuckerTensorTrain):
        return f(tree)
    return tuple(map_tree(f, t) for t in tree)

def leaves(tree):
    if isinstance(tree, t3.TuckerTensorTrain):
        return [tree]
    return [l for t in tree for l in leaves(t)]

def ranks_tree(tree, which):
    """(d,)+stack (or (d+1,)+stack) array of per-element ranks of a ragged tree."""
    def r(t):
        return np.asarray(t.tucker_ranks if which == 'tucker' else t.tt_ranks)
    if isinstance(tree, t3.TuckerTensorTrain):
        return r(tree)
    return np.moveaxis(np.stack([ranks_tree(t, which) for t in tree]), 0, -1) if False else _rk(tree, which)

def _rk(tree, which):
    if isinstance(tree, t3.TuckerTensorTrain):
        return np.asarray(tree.tucker_ranks if which == 'tucker' else tree.tt_ranks)
    sub = [_rk(t, which) for t in tree]           # each (L,)+inner_stack
    return np.stack(sub, axis=1)                    # (L, len(tree)) + inner_stack

def _flat(tree):
    if isinstance(tree, ut3.UniformTuckerTensorTrain):
        return [tree]
    return [l for t in tree for l in _flat(t)]

for (shape, tk, tt), stack, vary, force_pad in itertools.product(CONFIGS, STACKS, [False, True], [False, True]):
    if vary and not stack:
        continue
    tag = 'shape=%s tk=%s tt=%s stack=%s vary=%s pad=%s' % (shape, tk, tt, stack, vary, force_pad)
    try:
        tree = ragged_tree(shape, tk, tt, stack, vary)
        kw = dict(N=max(shape) + 2, n=max(tk) + 1, r=max(tt) + 1) if force_pad else (dict(N=max(shape), n=max(tk), r=max(tt)) if vary else {})
        ux = to_uniform(tree, **kw)
        uxg = corrupt(ux)
        D = dense_tree(tree)
        d = len(shape)

        # ---- structure / masks of the conversion itself
        check('from_t3 tucker_ranks ' + tag, np.array_equal(ux.tucker_ranks, _rk(tree, 'tucker')))
        check('from_t3 tt_ranks ' + tag, np.array_equal(ux.tt_ranks, _rk(tree, 'tt')))
        check('masks host numpy ' + tag, all(type(m) is np.ndarray for m in ux.masks.data))

        # ---- to_dense / to_t3 round trips
        check('to_dense ' + tag, relerr(ux.to_dense(), D) < TOL)
        check('to_dense garbage ' + tag, relerr(uxg.to_dense(), D) < TOL)
        back = ux.to_t3()
        check('to_t3 dense ' + tag, relerr(dense_tree(back), D) < TOL)
        backg = uxg.to_t3()
        check('to_t3 garbage dense ' + tag, relerr(dense_tree(backg), D) < TOL)
        check('to_t3 ranks ' + tag, all(np.array_equal(a.tucker_ranks, b.tucker_ranks) and np.array_equal(a.tt_ranks, b.tt_ranks)
                                      for a, b in zip(leaves(back), leaves(tree))))

        # ---- reverse
        check('reverse ' + tag, relerr(ux.reverse().to_dense(), np.moveaxis(D, list(range(-d, 0)), list(range(-1, -d - 1, -1)))) < TOL)
        check('reverse masks ' + tag, np.array_equal(ux.reverse().tt_ranks, ux.tt_ranks[::-1]) and np.array_equal(ux.reverse().tucker_ranks, ux.tucker_ranks[::-1]))
        check('reverse garbage ' + tag, relerr(uxg.reverse().to_dense(), ux.reverse().to_dense()) < TOL)

        # ---- scalar ops
        check('scale ' + tag, relerr((2.5 * ux).to_dense(), 2.5 * D) < TOL)
        check('neg ' + tag, relerr((-ux).to_dense(), -D) < TOL)
        check('scale garbage ' + tag, relerr((2.5 * uxg).to_dense(), 2.5 * D) < TOL)

        # ---- add / sub (+ masks: concatenation, possibly gappy, rank sums; then squash sets boundary to 1)
        tree2 = ragged_tree(shape, tk, tt, stack, vary)
        uy = to_uniform(tree2, **kw)
        uyg = corrupt(uy)
        D2 = dense_tree(tree2)
        s = ux + uy
        check('add ' + tag, relerr(s.to_dense(), D + D2) < TOL)
        check('sub ' + tag, relerr((ux - uy).to_dense(), D - D2) < TOL)
        check('add garbage ' + tag, relerr((uxg + uyg).to_dense(), D + D2) < TOL)
        exp_tk = ux.tucker_ranks + uy.tucker_ranks
        exp_tt = ux.tt_ranks + uy.tt_ranks
        exp_tt[0] = 1; exp_tt[-1] = 1
        check('add tucker ranks ' + tag, np.array_equal(s.tucker_ranks, exp_tk))
        check('add tt ranks ' + tag, np.array_equal(s.tt_ranks, exp_tt))
        # exact masks: concat of the two masks (interior), boundary -> [1,0,...]
        tkm_e = np.concatenate([ux.masks.tucker_edge_mask, uy.masks.tucker_edge_mask], axis=-1)
        ttm_e = np.concatenate([ux.masks.tt_edge_mask, uy.masks.tt_edge_mask], axis=-1)
        b = np.zeros_like(ttm_e[0]); b[..., 0] = True
        ttm_e[0] = b; ttm_e[-1] = b
        check('add exact tucker mask ' + tag, np.array_equal(s.masks.tucker_edge_mask, tkm_e))
        check('add exact tt mask ' + tag, np.array_equal(s.masks.tt_edge_mask, ttm_e))
        check('add to_t3 through gappy masks ' + tag, relerr(dense_tree(s.to_t3()), D + D2) < TOL)
        check('add then t3svd canonical prefix ' + tag, all(is_prefix(m) for m in s.t3svd()[0].masks.data))
        check('add then t3svd dense ' + tag, relerr(s.t3svd()[0].to_dense(), D + D2) < TOL)

        # ---- squash_tails
        check('squash garbage ' + tag, relerr(uxg.squash_tails().to_dense(), D) < TOL)

        # ---- sum_stack
        if stack:
            ss = ux.sum_stack()
            ref = D.reshape((-1,) + shape).sum(axis=0)
            check('sum_stack ' + tag, relerr(ss.to_dense(), ref) < TOL)
            check('sum_stack garbage ' + tag, relerr(uxg.sum_stack().to_dense(), ref) < TOL)
            S = int(np.prod(stack))
            check('sum_stack tucker ranks ' + tag, np.array_equal(ss.tucker_ranks, ux.tucker_ranks.reshape(d, -1).sum(axis=-1)))
            e_tt = ux.tt_ranks.reshape(d + 1, -1).sum(axis=-1); e_tt[0] = 1; e_tt[-1] = 1
            check('sum_stack tt ranks ' + tag, np.array_equal(ss.tt_ranks, e_tt))
            check('sum_stack stack_shape ' + tag, ss.stack_shape == ())

        # ---- inner / norm
        Dn = np.sqrt((D * D).reshape((-1,) + shape).sum(axis=tuple(range(1, d + 1)))).reshape(stack) if stack else np.linalg.norm(D)
        Dip = (D * D2).reshape((-1,) + shape).sum(axis=tuple(range(1, d + 1))).reshape(stack) if stack else np.sum(D * D2)
        for orth in (True, False):
            check('norm orth=%s %s' % (orth, tag), relerr(ux.norm(orth), Dn) < TOL)
            check('norm garbage orth=%s %s' % (orth, tag), relerr(uxg.norm(orth), Dn) < TOL)
            check('inner orth=%s %s' % (orth, tag), relerr(ux.inner(uy, orth), Dip) < TOL)
            check('inner garbage orth=%s %s' % (orth, tag), relerr(uxg.inner(uyg, orth), Dip) < TOL)
        check('sum ' + tag, relerr(ux.sum(), D.reshape((-1,) + shape).sum(axis=tuple(range(1, d + 1))).reshape(stack) if stack else D.sum()) < TOL)
        check('sum garbage ' + tag, relerr(uxg.sum(), ux.sum()) < TOL)

        # ---- sampling: entries / apply / probe (W stack + C stack), ragged & packed vectors
        W = (3,)
        ww = [np.random.randn(*W, Ni) for Ni in shape]
        idx = np.stack([np.random.randint(0, Ni, size=W) for Ni in shape])      # (d,)+W
        ein_in = 'abcd'[:d]
        stack_letters = 'xyz'[:len(stack)]
        # apply: shape W+stack
        ref_apply = np.einsum('%s%s,%s->w%s' % (stack_letters, ein_in, ','.join('w' + c for c in ein_in), stack_letters), D, *ww)
        check('apply ' + tag, relerr(ux.apply(ww), ref_apply) < TOL)
        check('apply garbage ' + tag, relerr(uxg.apply(ww), ref_apply) < TOL)
        check('apply packed ' + tag, relerr(ux.apply(np.stack([np.pad(w, ((0, 0),) * len(W) + ((0, ux.N - w.shape[-1]),)) for w in ww])), ref_apply) < TOL)
        # entries: shape W+stack
        ref_entries = np.stack([D[(Ellipsis,) + tuple(idx[:, k])] for k in range(W[0])])
        check('entries ' + tag, relerr(ux.entries(idx), ref_entries) < TOL)
        check('entries garbage ' + tag, relerr(uxg.entries(idx), ref_entries) < TOL)
        # probe: len=d, each W+stack+(Ni,)
        zz = ux.probe(ww)
        zzg = uxg.probe(ww)
        for i in range(d):
            others = [c for c in ein_in if c != ein_in[i]]
            ref = np.einsum('%s%s,%s->w%s%s' % (stack_letters, ein_in, ','.join('w' + c for c in others), stack_letters, ein_in[i]),
                            D, *[ww[j] for j in range(d) if j != i])
            check('probe mode %d %s' % (i, tag), zz[i].shape == ref.shape and relerr(zz[i], ref) < TOL, str((zz[i].shape, ref.shape)))
            check('probe garbage mode %d %s' % (i, tag), relerr(zzg[i], ref) < TOL)
        # packed probe mirrors packedness
        packed_ww = np.stack([np.pad(w, ((0, 0),) * len(W) + ((0, ux.N - w.shape[-1]),)) for w in ww])
        zp = ux.probe(packed_ww)
        check('probe packed shape ' + tag, zp.shape == (d,) + W + stack + (ux.N,), str(zp.shape))
        for i in range(d):
            check('probe packed val %d %s' % (i, tag), relerr(zp[i][..., :shape[i]], zz[i]) < TOL)

        # ---- orthogonalizations: dense preserved, garbage-robust, masks prefix + exact ranks vs ragged
        for nm, f, rf in [
            ('down',  lambda u: u.down_orthogonalize_tucker_cores(), lambda t: t.down_orthogonalize_tucker_cores()),
            ('up',    lambda u: u.up_orthogonalize_tt_cores(),       lambda t: t.up_orthogonalize_tt_cores()),
            ('left',  lambda u: u.left_orthogonalize_tt_cores(),     lambda t: t.left_orthogonalize_tt_cores()),
            ('right', lambda u: u.right_orthogonalize_tt_cores(),    lambda t: t.right_orthogonalize_tt_cores()),
        ]:
            o = f(ux)
            rt = map_tree(rf, tree)
            check('%s orth dense %s' % (nm, tag), relerr(o.to_dense(), D) < TOL)
            check('%s orth garbage %s' % (nm, tag), relerr(f(uxg).to_dense(), D) < TOL)
            check('%s orth masks prefix %s' % (nm, tag), all(is_prefix(m) for m in o.masks.data))
            check('%s orth tucker ranks vs ragged %s' % (nm, tag), np.array_equal(o.tucker_ranks, _rk(rt, 'tucker')), str((o.tucker_ranks.tolist(), _rk(rt, 'tucker').tolist())))
            check('%s orth tt ranks vs ragged %s' % (nm, tag), np.array_equal(o.tt_ranks, _rk(rt, 'tt')), str((o.tt_ranks.tolist(), _rk(rt, 'tt').tolist())))
            check('%s orth garbage masks equal %s' % (nm, tag), all(np.array_equal(a, b) for a, b in zip(f(uxg).masks.data, o.masks.data)))
        lo = ux.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
        ro = ux.down_orthogonalize_tucker_cores().right_orthogonalize_tt_cores()
        check('is_left_orthogonal ' + tag, bool(np.all(lo.is_left_orthogonal())))
        check('is_right_orthogonal ' + tag, bool(np.all(ro.is_right_orthogonal())))
        check('is_left_orthogonal garbage ' + tag, bool(np.all(corrupt(lo).is_left_orthogonal())))
        check('not left orth of raw ' + tag, not bool(np.all(ux.is_left_orthogonal())) or d == 1)

        # ---- t3svd (no cap, cap), rank_adjustment_sweep, minimal ranks
        us, sk, st = ux.t3svd()
        check('t3svd dense ' + tag, relerr(us.to_dense(), D) < TOL)
        check('t3svd garbage dense ' + tag, relerr(uxg.t3svd()[0].to_dense(), D) < TOL)
        check('t3svd left orth ' + tag, bool(np.all(us.is_left_orthogonal())))
        check('t3svd masks prefix ' + tag, all(is_prefix(m) for m in us.masks.data))
        rs = map_tree(lambda t: t.t3svd()[0], tree)
        check('t3svd tucker ranks vs ragged ' + tag, np.array_equal(us.tucker_ranks, _rk(rs, 'tucker')), str((us.tucker_ranks.tolist(), _rk(rs, 'tucker').tolist())))
        check('t3svd tt ranks vs ragged ' + tag, np.array_equal(us.tt_ranks, _rk(rs, 'tt')), str((us.tt_ranks.tolist(), _rk(rs, 'tt').tolist())))
        # singular values vs ragged (real slots)
        for i, leaf in enumerate(leaves(tree)):
            _, rsk, rst = leaf.t3svd()
            sel = (slice(None),) + tuple(np.unravel_index(i, stack)) if stack else (slice(None),)
            for j in range(d):
                got = sk[(j,) + sel[1:]] if stack else sk[j]
                m = us.masks.tucker_edge_mask[(j,) + sel[1:]] if stack else us.masks.tucker_edge_mask[j]
                check('t3svd tucker svals mode %d el %d %s' % (j, i, tag), relerr(got[m], rsk[j]) < 1e-7, str((got[m], rsk[j])))
            for j in range(d + 1):
                got = st[(j,) + sel[1:]] if stack else st[j]
                m = us.masks.tt_edge_mask[(j,) + sel[1:]] if stack else us.masks.tt_edge_mask[j]
                check('t3svd tt svals bond %d el %d %s' % (j, i, tag), relerr(got[m], rst[j]) < 1e-7, str((got[m], rst[j])))
        # truncation
        cap_tk = tuple(max(1, a - 1) for a in tk)
        cap_tt = (1,) + tuple(max(1, b - 1) for b in tt[1:-1]) + (1,)
        ut_, _, _ = ux.t3svd(max_tucker_ranks=cap_tk, max_tt_ranks=cap_tt)
        rt_ = map_tree(lambda t: t.t3svd(max_tucker_ranks=cap_tk, max_tt_ranks=cap_tt)[0], tree)
        check('t3svd capped dense vs ragged ' + tag, relerr(ut_.to_dense(), dense_tree(rt_)) < 1e-8)
        check('t3svd capped tucker ranks ' + tag, np.array_equal(ut_.tucker_ranks, _rk(rt_, 'tucker')), str((ut_.tucker_ranks.tolist(), _rk(rt_, 'tucker').tolist())))
        check('t3svd capped tt ranks ' + tag, np.array_equal(ut_.tt_ranks, _rk(rt_, 'tt')), str((ut_.tt_ranks.tolist(), _rk(rt_, 'tt').tolist())))
        check('t3svd capped garbage ' + tag, relerr(uxg.t3svd(max_tucker_ranks=cap_tk, max_tt_ranks=cap_tt)[0].to_dense(), ut_.to_dense()) < 1e-8)
        # rank_adjustment_sweep on the t3svd result
        ra = us.rank_adjustment_sweep('right_to_left')
        check('ras dense ' + tag, relerr(ra.to_dense(), D) < TOL)
        check('ras right orth ' + tag, bool(np.all(ra.is_right_orthogonal())))
        check('ras minimal ' + tag, bool(np.all(ra.has_minimal_ranks)))
        check('ras garbage ' + tag, relerr(corrupt(us).rank_adjustment_sweep('right_to_left').to_dense(), D) < TOL)
        rr = map_tree(lambda t: t.t3svd()[0].rank_adjustment_sweep('right_to_left'), tree)
        check('ras ranks vs ragged ' + tag, np.array_equal(ra.tucker_ranks, _rk(rr, 'tucker')) and np.array_equal(ra.tt_ranks, _rk(rr, 'tt')),
              str((ra.tucker_ranks.tolist(), _rk(rr, 'tucker').tolist(), ra.tt_ranks.tolist(), _rk(rr, 'tt').tolist())))
        # minimal_ranks property vs backend
        mn = ux.minimal_ranks
        mt, mtt = ranks.compute_minimal_ranks(shape, ux.tucker_ranks, ux.tt_ranks)
        check('minimal_ranks ' + tag, np.array_equal(mn[0], mt) and np.array_equal(mn[1], mtt))

        # ---- unstack / stack
        if stack:
            parts = ux.unstack()
            re = ut3.UniformTuckerTensorTrain.stack(parts)
            check('unstack/stack dense ' + tag, relerr(re.to_dense(), D) < TOL)
            check('unstack/stack masks ' + tag, all(np.array_equal(a, b) for a, b in zip(re.masks.data, ux.masks.data)))
            check('unstack leaf dense ' + tag, all(relerr(l.to_dense(), r.to_dense()) < TOL for l, r in zip(
                _flat(parts), leaves(tree))))
    except Exception as e:
        FAILS.append(('EXC ' + tag, repr(e)))
        print('EXC', tag, repr(e))
        traceback.print_exc()

print('\n==== %d failures' % len(FAILS))
from collections import Counter
print(Counter(f[0].split(' shape=')[0] for f in FAILS))
