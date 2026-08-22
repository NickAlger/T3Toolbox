"""H4 matrix: uniform frontend ops x edge structures x {numpy, jax eager, jit}; ragged-vs-uniform value check."""
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Tangent, MANIFOLD
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.uniform_tucker_tensor_train import UniformTuckerTensorTrain as UT3, UT3Weights, ut3_absorb_weights, ut3_weighted_norm, ut3_weighted_inner
from t3toolbox.uniform_frame_variations_format import UT3Frame, UT3Variations, ut3_orthogonal_representations, UT3FrameWeights, ufv_absorb_weights
from t3toolbox.uniform_manifold import UT3Tangent, UNIFORM_MANIFOLD as UM, UNIFORM_COREWISE as UC
from t3toolbox.backend import common
sys.path.insert(0, '.')
from matrix_ragged import STRUCTS, STACKS, W, mk_aux, compare, canon, to_jax_tree, flat

JIT_STRUCTS = {'d2_rank1', 'd3_mode1', 'd3_nonmin'}
JIT_STACKS = [(), (2,)]

def ucanon(r):
    if isinstance(r, (UT3, UT3Tangent, UT3Frame)):
        return np.asarray(r.to_dense())
    if isinstance(r, UT3Variations):
        return tuple(np.asarray(a) for a in r.supercores)
    if isinstance(r, (UT3Weights, UT3FrameWeights)):
        return tuple(np.asarray(a) for a in r.supercores)
    if isinstance(r, (tuple, list)):
        return tuple(ucanon(a) for a in r)
    return canon(r)

def ucompare(a, b, tol=1e-8):
    return compare(ucanon(a), ucanon(b), tol)

def supercore_leak(r):
    sc = getattr(r, 'supercores', None)
    if sc is None:
        sc = jax.tree_util.tree_leaves(r)
    for a in jax.tree_util.tree_leaves(sc):
        if isinstance(a, np.ndarray) and a.dtype.kind == 'f':
            return True
    return False

def ops():
    O = {}
    fv = lambda x: ut3_orthogonal_representations(x)
    O['to_dense'] = (lambda x, a: x.to_dense(), True)
    O['norm'] = (lambda x, a: x.norm(), True)
    O['inner'] = (lambda x, a: x.inner(x), True)
    O['add'] = (lambda x, a: x + x, True)
    O['mul'] = (lambda x, a: 2.0 * x, True)
    O['sum'] = (lambda x, a: x.sum(), True)
    O['reverse'] = (lambda x, a: x.reverse(), True)
    O['squash_tails'] = (lambda x, a: x.squash_tails(), True)
    O['to_t3'] = (lambda x, a: x.to_t3() if not x.stack_shape else x.to_t3()[0], True)
    O['entries'] = (lambda x, a: x.entries(a['idx']), True)
    O['apply'] = (lambda x, a: x.apply(a['ww']), True)
    O['probe'] = (lambda x, a: x.probe(a['ww']), True)
    O['probe_corewise_T'] = (lambda x, a: x.probe_corewise_transpose(a['zt'], a['ww']), True)
    O['apply_corewise_T'] = (lambda x, a: x.apply_corewise_transpose(a['c'], a['ww']), True)
    O['entries_corewise_T'] = (lambda x, a: x.entries_corewise_transpose(a['c'], a['idx']), True)
    O['probe_derivs'] = (lambda x, a: x.probe_derivatives(a['ww'], a['pp'], 2), True)
    O['apply_derivs'] = (lambda x, a: x.apply_derivatives(a['ww'], a['pp'], 2), True)
    O['entries_derivs'] = (lambda x, a: x.entries_derivatives(a['idx'], a['pp'], 2), True)
    O['probe_corewise_derivs_T'] = (lambda x, a: x.probe_corewise_derivatives_transpose(a['zt_jet'], a['ww'], a['pp'], 2), True)
    O['apply_corewise_derivs_T'] = (lambda x, a: x.apply_corewise_derivatives_transpose(a['c_jet'], a['ww'], a['pp'], 2), True)
    O['entries_corewise_derivs_T'] = (lambda x, a: x.entries_corewise_derivatives_transpose(a['c_jet'], a['idx'], a['pp'], 2), True)
    O['t3svd'] = (lambda x, a: x.t3svd()[0], True)
    O['t3svd_svals'] = (lambda x, a: x.t3svd()[1:], True)
    O['down_orth'] = (lambda x, a: x.down_orthogonalize_tucker_cores(), True)
    O['left_orth'] = (lambda x, a: x.left_orthogonalize_tt_cores(), True)
    O['right_orth'] = (lambda x, a: x.right_orthogonalize_tt_cores(), True)
    O['up_orth'] = (lambda x, a: x.up_orthogonalize_tt_cores(), True)
    O['is_left_orth'] = (lambda x, a: x.left_orthogonalize_tt_cores().is_left_orthogonal(), False)
    O['minimal_ranks'] = (lambda x, a: x.minimal_ranks, False)
    O['has_minimal_ranks'] = (lambda x, a: x.has_minimal_ranks, False)
    O['rank_adj_sweep'] = (lambda x, a: x.rank_adjustment_sweep(), True)
    O['sum_stack'] = (lambda x, a: x.sum_stack() if x.stack_shape else x, True)
    O['stack_unstack'] = (lambda x, a: UT3.stack(x.unstack()) if x.stack_shape else x, True)
    O['UT3Weights_from_ut3svd'] = (lambda x, a: UT3Weights.from_ut3svd(x), False)
    O['ut3_weighted_norm'] = (lambda x, a: ut3_weighted_norm(x.t3svd()[0], UT3Weights.from_ut3svd(x)), True)
    O['ut3_absorb_weights'] = (lambda x, a: ut3_absorb_weights(x.t3svd()[0], UT3Weights.from_ut3svd(x)), True)
    O['ut3_weighted_inner'] = (lambda x, a: ut3_weighted_inner(x.t3svd()[0], UT3Weights.from_ut3svd(x), x.t3svd()[0], UT3Weights.from_ut3svd(x)), True)
    O['UT3Weights_reciprocal_sqrt'] = (lambda x, a: UT3Weights.from_ut3svd(x).sqrt().reciprocal(), True)
    O['UT3Weights_kron'] = (lambda x, a: (lambda w: w.kronecker(w))(UT3Weights.from_ut3svd(x)), True)
    O['UT3Weights_concat'] = (lambda x, a: (lambda w: w.concatenate(w))(UT3Weights.from_ut3svd(x)), True)
    # frames/tangents
    O['orth_reps_point'] = (lambda x, a: UT3Tangent(*fv(x)).to_ut3(include_shift=True), True)
    O['orth_reps_is_orth'] = (lambda x, a: fv(x)[0].is_orthogonal(), False)
    O['frame_from_ut3'] = (lambda x, a: UT3Frame.from_ut3(x), True)
    O['frame_reverse'] = (lambda x, a: fv(x)[0].reverse(), True)
    O['frame_orthogonalize'] = (lambda x, a: UT3Frame.from_ut3(x).orthogonalize(), True)
    O['tangent_to_dense'] = (lambda x, a: UT3Tangent(*fv(x)).to_dense(), True)
    O['tangent_probe'] = (lambda x, a: UT3Tangent(*fv(x)).probe(a['ww']), True)
    O['tangent_apply'] = (lambda x, a: UT3Tangent(*fv(x)).apply(a['ww']), True)
    O['tangent_entries'] = (lambda x, a: UT3Tangent(*fv(x)).entries(a['idx']), True)
    O['tangent_probe_T'] = (lambda x, a: UT3Tangent.probe_transpose(a['zt'], a['ww'], fv(x)[0]), True)
    O['tangent_apply_T'] = (lambda x, a: UT3Tangent.apply_transpose(a['c'], a['ww'], fv(x)[0]), True)
    O['tangent_entries_T'] = (lambda x, a: UT3Tangent.entries_transpose(a['c'], a['idx'], fv(x)[0]), True)
    O['tangent_probe_derivs'] = (lambda x, a: UT3Tangent(*fv(x)).probe_derivatives(a['ww'], a['pp'], 2), True)
    O['tangent_apply_derivs'] = (lambda x, a: UT3Tangent(*fv(x)).apply_derivatives(a['ww'], a['pp'], 2), True)
    O['tangent_entries_derivs'] = (lambda x, a: UT3Tangent(*fv(x)).entries_derivatives(a['idx'], a['pp'], 2), True)
    O['tangent_probe_derivs_T'] = (lambda x, a: UT3Tangent.probe_derivatives_transpose(a['zt_jet'], a['ww'], a['pp'], fv(x)[0], 2), True)
    O['tangent_apply_derivs_T'] = (lambda x, a: UT3Tangent.apply_derivatives_transpose(a['c_jet'], a['ww'], a['pp'], fv(x)[0], 2), True)
    O['tangent_entries_derivs_T'] = (lambda x, a: UT3Tangent.entries_derivatives_transpose(a['c_jet'], a['idx'], a['pp'], fv(x)[0], 2), True)
    O['um_project'] = (lambda x, a: UM.project(UT3Tangent(*fv(x))), True)
    O['um_project_oblique'] = (lambda x, a: UM.project_oblique(UT3Tangent(*fv(x))), True)
    O['um_norm'] = (lambda x, a: UM.norm(UM.project(UT3Tangent(*fv(x)))), True)
    O['um_retract'] = (lambda x, a: UM.retract(UM.project(UT3Tangent(*fv(x)))), True)
    O['um_project_ambient'] = (lambda x, a: UM.project_ambient(fv(x)[0], x), True)
    O['um_transport'] = (lambda x, a: UM.transport(UM.project(UT3Tangent(*fv(x))), fv(x + x)[0]), True)
    O['uc_project'] = (lambda x, a: UC.project(UT3Tangent(*fv(x))), True)
    O['uc_retract'] = (lambda x, a: UC.retract(UT3Tangent(*fv(x))), True)
    O['tangent_corewise_norm'] = (lambda x, a: UT3Tangent(*fv(x)).corewise_norm(), True)
    O['tangent_zeros'] = (lambda x, a: UT3Tangent.zeros(fv(x)[0]), True)
    O['tangent_gauge_residual'] = (lambda x, a: UT3Tangent(*fv(x)).gauge_residual, True)
    O['tangent_reverse'] = (lambda x, a: UT3Tangent(*fv(x)).reverse().to_dense(), True)
    O['tangent_to_t3tangent'] = (lambda x, a: UT3Tangent(*fv(x)).to_t3tangent().to_dense() if not x.stack_shape else 0, True)
    O['stack_tangents'] = (lambda x, a: UT3Tangent.stack_tangents([UT3Tangent(*fv(x)), UT3Tangent(*fv(x))]).to_dense(), True)
    O['frame_weights'] = (lambda x, a: UT3FrameWeights.from_ut3weights(UT3Weights.from_ut3svd(x)), True)
    O['tangent_weighted_norm'] = (lambda x, a: (lambda xs: UT3Tangent(*fv(xs)).weighted_norm(UT3FrameWeights.from_ut3weights(UT3Weights.from_ut3svd(xs))))(x.t3svd()[0]), True)
    O['tangent_absorb_weights'] = (lambda x, a: (lambda xs: UT3Tangent(*fv(xs)).absorb_weights(UT3FrameWeights.from_ut3weights(UT3Weights.from_ut3svd(xs))).to_dense())(x.t3svd()[0]), True)
    O['ufv_absorb_weights'] = (lambda x, a: (lambda xs: ufv_absorb_weights(fv(xs)[1], UT3FrameWeights.from_ut3weights(UT3Weights.from_ut3svd(xs))))(x.t3svd()[0]), True)
    O['frame_weights_reciprocal'] = (lambda x, a: UT3FrameWeights.from_ut3weights(UT3Weights.from_ut3svd(x)).reciprocal(), True)
    return O

def ragged_twin(oname, x_np, a):
    """ragged reference for value comparison, where meaningful"""
    fvr = lambda x: t3_orthogonal_representations(x)
    R = {
        'to_dense': lambda: x_np.to_dense(), 'norm': lambda: x_np.norm(), 'inner': lambda: x_np.inner(x_np),
        'add': lambda: x_np + x_np, 'mul': lambda: x_np * 2.0, 'sum': lambda: x_np.sum(), 'reverse': lambda: x_np.reverse(),
        'entries': lambda: x_np.entries(a['idx']), 'apply': lambda: x_np.apply(a['ww']), 'probe': lambda: x_np.probe(a['ww']),
        'probe_derivs': lambda: x_np.probe_derivatives(a['ww'], a['pp'], 2), 'apply_derivs': lambda: x_np.apply_derivatives(a['ww'], a['pp'], 2),
        'entries_derivs': lambda: x_np.entries_derivatives(a['idx'], a['pp'], 2), 't3svd': lambda: x_np.t3svd()[0],
        'tangent_to_dense': lambda: T3Tangent(*fvr(x_np)).to_dense(), 'tangent_probe': lambda: T3Tangent(*fvr(x_np)).probe(a['ww']),
        'tangent_apply': lambda: T3Tangent(*fvr(x_np)).apply(a['ww']), 'tangent_entries': lambda: T3Tangent(*fvr(x_np)).entries(a['idx']),
        'tangent_probe_T': lambda: T3Tangent.probe_transpose(a['zt'], a['ww'], fvr(x_np)[0]).to_dense(),
        'tangent_apply_T': lambda: T3Tangent.apply_transpose(a['c'], a['ww'], fvr(x_np)[0]).to_dense(),
        'tangent_entries_T': lambda: T3Tangent.entries_transpose(a['c'], a['idx'], fvr(x_np)[0]).to_dense(),
        'tangent_probe_derivs': lambda: T3Tangent(*fvr(x_np)).probe_derivatives(a['ww'], a['pp'], 2),
        'tangent_probe_derivs_T': lambda: T3Tangent.probe_derivatives_transpose(a['zt_jet'], a['ww'], a['pp'], fvr(x_np)[0], 2).to_dense(),
        'tangent_apply_derivs_T': lambda: T3Tangent.apply_derivatives_transpose(a['c_jet'], a['ww'], a['pp'], fvr(x_np)[0], 2).to_dense(),
        'tangent_entries_derivs_T': lambda: T3Tangent.entries_derivatives_transpose(a['c_jet'], a['idx'], a['pp'], fvr(x_np)[0], 2).to_dense(),
        'um_project': lambda: MANIFOLD.project(T3Tangent(*fvr(x_np))).to_dense(),
        'um_norm': lambda: MANIFOLD.norm(MANIFOLD.project(T3Tangent(*fvr(x_np)))),
        'um_retract': lambda: MANIFOLD.retract(MANIFOLD.project(T3Tangent(*fvr(x_np)))).to_dense(),
        'um_project_ambient': lambda: MANIFOLD.project_ambient(fvr(x_np)[0], x_np).to_dense(),
        'um_transport': lambda: MANIFOLD.transport(MANIFOLD.project(T3Tangent(*fvr(x_np))), fvr(x_np + x_np)[0]).to_dense(),
        'uc_retract': lambda: MANIFOLD.retract(T3Tangent(*fvr(x_np))).to_dense() if False else None,
        'tangent_weighted_norm': lambda: (lambda xs: T3Tangent(*fvr(xs)).weighted_norm(tb.T3FrameWeights.from_t3weights(tb.T3Weights.from_t3svd(xs))))(x_np.t3svd()[0]),
        'ut3_weighted_norm': lambda: tb.tucker_tensor_train.t3_weighted_norm(x_np.t3svd()[0], tb.T3Weights.from_t3svd(x_np)),
    }
    f = R.get(oname)
    return None if f is None else f()

def run():
    rows = []
    for sname, struct in STRUCTS.items():
        if sname.startswith('d1'):
            continue   # uniform d=1 is a known ledger item
        for stack in STACKS:
            np.random.seed(0)
            x_np = T3.randn(*struct, stack_shape=stack)
            ux_np = UT3.from_t3(x_np)
            aux_np = mk_aux(struct, stack)
            ux_jx = ux_np.to_jax()
            aux_jx = to_jax_tree(aux_np)
            for oname, (fn, jittable) in ops().items():
                res = {}
                for bname, x, a in (('np', ux_np, aux_np), ('jax', ux_jx, aux_jx)):
                    try:
                        res[bname] = ('ok', fn(x, a))
                    except Exception as e:
                        res[bname] = ('err', '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:140] if str(e) else ''))
                if jittable and sname in JIT_STRUCTS and stack in JIT_STACKS:
                    try:
                        with tb.safety.unsafe():
                            res['jit'] = ('ok', jax.jit(lambda x, a: fn(x, a))(ux_jx, aux_jx))
                    except Exception as e:
                        res['jit'] = ('err', '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:140] if str(e) else ''))
                notes = []
                for b in res:
                    if res[b][0] == 'err':
                        notes.append('%s ERR %s' % (b, res[b][1]))
                if res['np'][0] == 'ok' and res['jax'][0] == 'ok':
                    m = ucompare(res['jax'][1], res['np'][1])
                    if m: notes.append('np/jax MISMATCH ' + m)
                    try:
                        if supercore_leak(res['jax'][1]): notes.append('jax->numpy SUPERCORE LEAK')
                    except Exception: pass
                if 'jit' in res and res['jit'][0] == 'ok' and res['np'][0] == 'ok':
                    m = ucompare(res['jit'][1], res['np'][1])
                    if m: notes.append('np/jit MISMATCH ' + m)
                if res['np'][0] == 'ok':
                    try:
                        ref = ragged_twin(oname, x_np, aux_np)
                        if ref is not None:
                            m = ucompare(res['np'][1], ref, tol=1e-7)
                            if m: notes.append('uniform/ragged MISMATCH ' + m)
                    except Exception as e:
                        notes.append('ragged twin ERR %s: %s' % (type(e).__name__, str(e).splitlines()[0][:100] if str(e) else ''))
                if notes:
                    rows.append(('UNIFORM', sname, stack, oname, '; '.join(notes)))
                    print('%-8s %-10s %-5s %-32s %s' % rows[-1], flush=True)
    print('TOTAL NOTES:', len(rows))

if __name__ == '__main__':
    run()
