"""R9-C: pytree registrations (flatten/unflatten equality, value-hashed aux), stacked has_minimal_ranks /
tangent_space_dimension vs ragged, chunk_size consistency, sum_stack/sum_tangents axis semantics, doc
claims (module docstring 'no save', reverse commutes with to_dense, stale 'forthcoming' notes)."""
import numpy as np, traceback, os, tempfile
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as ut3m
import t3toolbox.manifold as t3m
import t3toolbox.frame_variations_format as bvf
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ufv_operations as ufvo

def rep(msg, ok, val=None):
    print(('PASS ' if ok else 'FAIL ') + msg + ('' if val is None else '  [%s]' % (val,)))

np.random.seed(0)
# hetero C=(2,) stack of different ranks, K=(3,)
HET = [((4, 5, 6), (2, 2, 2), (1, 2, 2, 1)), ((4, 5, 6), (3, 3, 2), (1, 1, 2, 1))]
PAD = dict(N=6, nU=4, nD=4, rL=3, rR=3)
us, rs = [], []
for s in HET:
    x = t3.TuckerTensorTrain.randn(*s)
    rb, rv = bvf.t3_orthogonal_representations(x)
    us.append(ut3m.UT3Tangent(ubv.UT3Frame.from_t3frame(rb, **PAD), ubv.UT3Variations.from_t3variations(rv, **PAD)))
    rs.append(t3m.T3Tangent(rb, rv))
vC = ut3m.UT3Tangent.stack_frame(tuple(us))                       # C=(2,)
vKC = ut3m.UT3Tangent.stack_tangents((vC, 2.0 * vC, -vC))          # K=(3,), C=(2,)
rep('stack shapes', (vKC.stack_shape, vKC.frame_stack_shape, vKC.tangent_stack_shape) == ((3, 2), (2,), (3,)), vKC.stack_shape)

# ---- stacked has_minimal_ranks / minimal_ranks / tangent_space_dimension vs ragged per element ----------
try:
    hm = vC.frame.has_minimal_ranks
    mr = vC.frame.minimal_ranks
    rep('stacked has_minimal_ranks shape == C', np.asarray(hm).shape == (2,), hm)
    rep('stacked has_minimal_ranks per element == ragged', all(bool(hm[c]) == bool(rs[c].frame.has_minimal_ranks) for c in range(2)),
        [bool(rs[c].frame.has_minimal_ranks) for c in range(2)])
    rep('stacked minimal_ranks shapes (d,)+C,(d+1,)+C', (np.asarray(mr[0]).shape, np.asarray(mr[1]).shape) == ((3, 2), (4, 2)), (np.asarray(mr[0]).shape, np.asarray(mr[1]).shape))
    for c in range(2):
        exp = ranks.compute_minimal_ranks(HET[c][0], HET[c][1], HET[c][2])
        rep('  minimal_ranks[%d] == per-element' % c, tuple(np.asarray(mr[0])[:, c]) == tuple(exp[0]) and tuple(np.asarray(mr[1])[:, c]) == tuple(exp[1]), (np.asarray(mr[0])[:, c], exp))
    tsd = vKC.tangent_space_dimension
    rep('tangent_space_dimension shape == C (K shared)', np.asarray(tsd).shape == (2,), tsd)
    rep('tangent_space_dimension per element == ragged manifold_dim', all(int(tsd[c]) == int(ranks.compute_manifold_dim(*HET[c])) for c in range(2)), [int(ranks.compute_manifold_dim(*HET[c])) for c in range(2)])
    hn = vKC.has_numerically_minimal_ranks()
    rep('has_numerically_minimal_ranks shape == C', np.asarray(hn).shape == (2,), hn)
except Exception:
    traceback.print_exc()

# ---- pytree registrations -------------------------------------------------------------------------------
try:
    import jax
    for name, obj in [('UT3Frame', vKC.frame), ('UT3Variations', vKC.variations), ('UT3Tangent', vKC)]:
        leaves, tdef = jax.tree_util.tree_flatten(obj)
        back = jax.tree_util.tree_unflatten(tdef, leaves)
        # rebuilt-identical object -> same treedef (value-hashed aux)
        if name == 'UT3Frame':
            f = obj; twin = ubv.UT3Frame(*(s.copy() for s in f.supercores), tuple(f.shape), ubv.UT3FrameMasks(*(m.copy() for m in f.masks.data)))
        elif name == 'UT3Variations':
            v = obj; twin = ubv.UT3Variations(*(s.copy() for s in v.supercores), tuple(v.shape), ubv.UT3VariationsMasks(*(m.copy() for m in v.masks.data)))
        else:
            twin = ut3m.UT3Tangent(vKC.frame.copy(), vKC.variations.copy())
        _, tdef2 = jax.tree_util.tree_flatten(twin)
        rep('%s treedef of a rebuilt-identical object is EQUAL (value-hashed aux)' % name, tdef == tdef2)
        rep('%s treedef hash equal' % name, hash(tdef) == hash(tdef2))
        rep('%s unflatten(flatten) round trip equal on real content' % name,
            all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(leaves, jax.tree_util.tree_leaves(back))) and type(back) is type(obj))
    # UT3FrameWeights
    Wt = ubv.UT3FrameWeights.from_ut3weights(ut3.UT3Weights.from_ut3svd(vC.frame.to_ut3()))
    leaves, tdef = jax.tree_util.tree_flatten(Wt)
    twin = ubv.UT3FrameWeights(*(s.copy() for s in Wt.supercores), ubv.UT3VariationsMasks(*(m.copy() for m in Wt.masks.data)))
    _, tdef2 = jax.tree_util.tree_flatten(twin)
    rep('UT3FrameWeights treedef of rebuilt-identical equal', tdef == tdef2)
    back = jax.tree_util.tree_unflatten(tdef, leaves)
    rep('UT3FrameWeights round trip', type(back) is ubv.UT3FrameWeights and all(np.array_equal(a, b) for a, b in zip(back.supercores, Wt.supercores)))
    # the comment at uniform_frame_variations_format.py:1340 says UT3FrameMasks is identity-hashed:
    m1 = vC.frame.masks; m2 = ubv.UT3FrameMasks(*(m.copy() for m in m1.data))
    rep('UT3FrameMasks is VALUE hashed/eq (contradicts the :1340 comment)', m1 == m2 and hash(m1) == hash(m2) and m1 is not m2)
except ImportError:
    print('no jax')
except Exception:
    traceback.print_exc()

# ---- chunk_size: ragged tangent probe-derivatives transpose with chunk_size None/100/3 identical ---------
try:
    np.random.seed(5)
    shape, n, r = (5, 6, 4), (2, 3, 2), (1, 2, 2, 1)
    x = t3.TuckerTensorTrain.randn(shape, n, r)
    u = ut3.UniformTuckerTensorTrain.from_t3(x)
    vv = ut3m.UNIFORM_COREWISE.randn(ut3m.UNIFORM_MANIFOLD.frame(x and u))
    ww = [np.random.randn(9, Ni) for Ni in shape]; pp = [np.random.randn(9, Ni) for Ni in shape]
    zj = vv.probe_derivatives(ww, pp, 2)
    rr = [np.random.randn(*np.asarray(z).shape) for z in zj]
    outs = [ut3m.UT3Tangent.probe_derivatives_transpose(rr, ww, pp, vv.frame, 2, sum_over_probes=True, chunk_size=cs) for cs in (None, 100, 3, 4)]
    worst = max(float(np.abs(np.asarray(o.variations.supercores[i]) - np.asarray(outs[0].variations.supercores[i])).max()) for o in outs[1:] for i in range(2))
    rep('tangent probe_derivatives_transpose: chunk_size None/100/3/4 identical', worst < 1e-10, worst)
    # the UT3 corewise twin has no chunk_size parameter at all (backlog): confirm the signature
    import inspect
    sig = inspect.signature(ut3.UniformTuckerTensorTrain.probe_corewise_derivatives_transpose)
    rep('CHARACTERIZE backlog: UT3.probe_corewise_derivatives_transpose has NO chunk_size param', 'chunk_size' not in sig.parameters, str(sig))
    sigr = inspect.signature(t3.TuckerTensorTrain.probe_corewise_derivatives_transpose)
    print('    ragged twin signature:', sigr)
    # numerics: uniform corewise transpose == ragged with chunk_size default and with None (should all agree)
    zc = u.probe_derivatives(ww, pp, 2)
    rc = [np.random.randn(*np.asarray(z).shape) for z in zc]
    gu = u.probe_corewise_derivatives_transpose(rc, ww, pp, 2, sum_over_probes=True)
    kw = {}
    if 'chunk_size' in sigr.parameters: kw = dict(chunk_size=None)
    gr = x.probe_corewise_derivatives_transpose(rc, ww, pp, 2, sum_over_probes=True, **kw)
    gr100 = x.probe_corewise_derivatives_transpose(rc, ww, pp, 2, sum_over_probes=True)
    def cores_of(o):
        if hasattr(o, 'tucker_cores'): return list(o.tucker_cores) + list(o.tt_cores)
        return list(o[0]) + list(o[1])
    gtk, gtt = gu   # bare supercore pair (d,)+(n,N), (d,)+(r,n,r)
    rc_ = cores_of(gr); worst = 0.0
    for i in range(3):
        a = np.asarray(gtk[i])[:rc_[i].shape[0], :rc_[i].shape[1]]; worst = max(worst, np.abs(a - np.asarray(rc_[i])).max())
        b = np.asarray(gtt[i])[:rc_[3+i].shape[0], :rc_[3+i].shape[1], :rc_[3+i].shape[2]]; worst = max(worst, np.abs(b - np.asarray(rc_[3+i])).max())
    rep('UT3 corewise deriv transpose == ragged (chunk_size=None)', worst < 1e-9, worst)
    worst = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(cores_of(gr100), cores_of(gr)))
    rep('ragged corewise deriv transpose chunk 100 == None', worst < 1e-9, worst)
except Exception:
    traceback.print_exc()

# ---- sum_stack / sum_tangents axis semantics ----------------------------------------------------------------
try:
    s0 = vKC.variations.sum_stack(axis=0)
    rep('UT3Variations.sum_stack(axis=0) sums K (3,2)->(2,)', s0.stack_shape == (2,), s0.stack_shape)
    try:
        sm1 = vKC.variations.sum_stack(axis=-1)
        rep('CHARACTERIZE: sum_stack(axis=-1) result stack', False, sm1.stack_shape)
    except Exception as e:
        rep('CHARACTERIZE: sum_stack(axis=-1) raises', True, '%s: %s' % (type(e).__name__, str(e)[:100].replace('\n', ' ')))
    # ragged twin
    rvKC = t3m.T3Tangent.stack_tangents((rs[0], 2.0 * rs[0], -rs[0]))
    try:
        rm1 = rvKC.variations.sum_stack(axis=-1)
        rep('ragged T3Variations.sum_stack(axis=-1) works ->', True, rm1.stack_shape)
    except Exception as e:
        rep('ragged T3Variations.sum_stack(axis=-1) raises', True, type(e).__name__)
    st = vKC.sum_tangents()
    rep('sum_tangents() == 2*vC (3 terms: v + 2v - v)', bool(st.allclose(2.0 * vC).all()))
except Exception:
    traceback.print_exc()

# ---- doc claims -------------------------------------------------------------------------------------------------
import t3toolbox.uniform_frame_variations_format as mod
doc = mod.__doc__
rep('module docstring claims "no save/to_vector" asymmetry', 'no ``save``' in doc, doc.strip().splitlines()[-2:])
rep('...but UT3Frame.save AND UT3Variations.save/load exist', hasattr(ubv.UT3Frame, 'save') and hasattr(ubv.UT3Variations, 'save') and hasattr(ubv.UT3Variations, 'load'))
tmp = tempfile.mkdtemp(); fn = os.path.join(tmp, 'v.npz')
vKC.variations.save(fn); back = ubv.UT3Variations.load(fn)
rep('UT3Variations.save/load round trip', np.array_equal(back.tucker_variations, vKC.variations.tucker_variations) and back.masks == vKC.variations.masks)
rep('UT3Variations has no to_vector/from_vector/size/data_size (undocumented asymmetry vs T3Variations)',
    not any(hasattr(ubv.UT3Variations, a) for a in ('to_vector', 'from_vector', 'size', 'data_size')))
# reverse commutes with to_dense (docstring says "will ... once the doubled-rank conversion lands")
rv = vKC.reverse()
rep('UT3Tangent.reverse().to_dense() == to_dense() with mode axes reversed (docstring says "will ... once ... lands")',
    np.allclose(np.asarray(rv.to_dense()), np.asarray(vKC.to_dense()).transpose(0, 1, 4, 3, 2)))
rep('UT3Tangent.corewise_inner docstring says "(forthcoming) manifold geometry"', 'forthcoming' in ut3m.UT3Tangent.corewise_inner.__doc__)
rep('UT3Tangent.reverse docstring says "Will commute ... once ... lands"', 'once the doubled-rank conversion lands' in ut3m.UT3Tangent.reverse.__doc__)
rep('uniform_manifold module docstring says "Deferred to later 3b slices"', 'Deferred to later 3b' in ut3m.__doc__)
import t3toolbox.backend.utv_sampling as uts
rep('utv_sampling module docstring says transpose "lands in 3b-6c"', 'lands in 3b-6c' in uts.__doc__)
rep('frontend ut3_orthogonal_representations docstring documents a nonexistent xnp param + T3Base return',
    'xnp:' in ubv.ut3_orthogonal_representations.__doc__ and 'T3Base' in ubv.ut3_orthogonal_representations.__doc__)
import inspect
src = inspect.getsource(mod)
rep('uniform_frame_variations_format references dev/uniform_fix_plan.md (file is in dev/archive)', 'dev/uniform_fix_plan.md' in src and not os.path.exists('/home/nick/repos/T3Toolbox/dev/uniform_fix_plan.md'))
