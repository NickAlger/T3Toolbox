"""H4 matrix: ragged frontend ops x edge structures x {numpy, jax eager, jax jit}.

Compares invariants (dense tensors / scalars / coordinate tuples) across the three backends at
float64 (jax_enable_x64), reports errors, mismatches and jax-in -> numpy-out leaks.
"""
import sys, traceback, math
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import t3toolbox as tb
from t3toolbox import TuckerTensorTrain as T3, T3Frame, T3Variations, T3Tangent, MANIFOLD, COREWISE, T3Weights, T3FrameWeights
from t3toolbox.frame_variations_format import t3_orthogonal_representations
from t3toolbox.tucker_tensor_train import t3_absorb_weights, t3_weighted_norm, t3_weighted_inner
from t3toolbox.frame_variations_format import fv_absorb_weights
from t3toolbox.backend import common

STRUCTS = {
    'd1':        ((5,), (2,), (1, 1)),
    'd1_square': ((3,), (3,), (1, 1)),
    'd2':        ((4, 6), (2, 3), (1, 2, 1)),
    'd2_rank1':  ((4, 6), (1, 1), (1, 1, 1)),
    'd3_rank1':  ((4, 5, 6), (1, 1, 1), (1, 1, 1, 1)),
    'd3_mode1':  ((1, 5, 6), (1, 2, 3), (1, 2, 2, 1)),
    'd3_square': ((3, 4, 5), (3, 4, 5), (1, 2, 3, 1)),
    'd3_nonmin': ((4, 5, 6), (2, 2, 2), (1, 4, 4, 1)),
    'd4':        ((3, 4, 5, 2), (2, 3, 2, 2), (1, 2, 3, 2, 1)),
}
STACKS = [(), (1,), (2,)]
JIT_STRUCTS = {'d1', 'd2_rank1', 'd3_mode1', 'd3_nonmin'}
JIT_STACKS = [(), (2,)]
W = (3,)

def leaves(x):
    return jax.tree_util.tree_leaves(x)

def canon(r):
    """Gauge-invariant comparable form."""
    if isinstance(r, (T3, T3Tangent, T3Frame)):
        return np.asarray(r.to_dense())
    if isinstance(r, T3Variations):
        return tuple(np.asarray(a) for a in leaves(r.data))
    if isinstance(r, (tuple, list)):
        return tuple(canon(a) for a in r)
    if isinstance(r, (bool, int, float, np.integer, np.floating)):
        return r
    if common.is_ndarray(r):
        return np.asarray(r)
    if isinstance(r, T3Weights) or isinstance(r, T3FrameWeights):
        return tuple(np.asarray(a) for a in leaves(r.data))
    return r

def flat(c):
    if isinstance(c, tuple):
        out = []
        for a in c:
            out += flat(a)
        return out
    return [c]

def compare(a, b, tol=1e-8):
    fa, fb = flat(canon(a)), flat(canon(b))
    if len(fa) != len(fb):
        return 'len %d vs %d' % (len(fa), len(fb))
    for u, v in zip(fa, fb):
        if isinstance(u, (bool, np.bool_)) or isinstance(v, (bool, np.bool_)):
            if bool(u) != bool(v):
                return 'bool %r vs %r' % (u, v)
            continue
        u, v = np.asarray(u), np.asarray(v)
        if u.shape != v.shape:
            return 'shape %s vs %s' % (u.shape, v.shape)
        if u.dtype.kind in 'iub':
            if not np.array_equal(u, v):
                return 'int mismatch'
            continue
        den = max(np.linalg.norm(v.ravel()), 1.0)
        err = np.linalg.norm((u - v).ravel()) / den
        if not np.isfinite(err) or err > tol:
            return 'relerr %.2e' % err
    return None

def numpy_leak(r):
    """Any numpy float leaf in an op result that should be jax."""
    for a in leaves(r.data if hasattr(r, 'data') else r):
        if isinstance(a, np.ndarray) and a.dtype.kind == 'f':
            return True
    return False

def to_jax_tree(t):
    return jax.tree_util.tree_map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, t)

def mk_aux(struct, stack):
    shape, tk, tt = struct
    d = len(shape)
    rng = np.random.RandomState(1)
    ww = tuple(rng.randn(*(W + (N,))) for N in shape)
    pp = tuple(rng.randn(*(W + (N,))) for N in shape)
    zt = tuple(rng.randn(*(W + stack + (N,))) for N in shape)
    zt_jet = tuple(rng.randn(*((3,) + W + stack + (N,))) for N in shape)
    c = rng.randn(*(W + stack))
    c_jet = rng.randn(*((3,) + W + stack))
    idx = np.stack([rng.randint(0, N, size=W) for N in shape])
    vec = rng.randn(1)
    return dict(ww=ww, pp=pp, zt=zt, zt_jet=zt_jet, c=c, c_jet=c_jet, idx=idx)

# ---------------------------------------------------------------- op table: name -> (fn(x, aux), jittable)
def ops_t3():
    O = {}
    O['to_dense'] = (lambda x, a: x.to_dense(), True)
    O['norm'] = (lambda x, a: x.norm(), True)
    O['norm_noorth'] = (lambda x, a: x.norm(use_orthogonalization=False), True)
    O['inner_self'] = (lambda x, a: x.inner(x), True)
    O['inner_noorth'] = (lambda x, a: x.inner(x, use_orthogonalization=False), True)
    O['add'] = (lambda x, a: x + x, True)
    O['sub'] = (lambda x, a: x - x * 0.5, True)
    O['mul_scalar'] = (lambda x, a: 2.0 * x, True)
    O['hadamard'] = (lambda x, a: x * x, True)
    O['t3m'] = (lambda x, a: x.t3m(x), True)
    O['sum'] = (lambda x, a: x.sum(), True)
    O['sum_axis0'] = (lambda x, a: x.sum(axis=0), True)
    O['reverse'] = (lambda x, a: x.reverse(), True)
    O['squash_tails'] = (lambda x, a: x.squash_tails(), True)
    O['concatenate'] = (lambda x, a: x.concatenate(x), True)
    O['resize_up'] = (lambda x, a: x.resize(tuple(n + 1 for n in x.tucker_ranks), tuple([1] + [r + 1 for r in x.tt_ranks[1:-1]] + [1])), True)
    O['to_vector'] = (lambda x, a: x.to_vector(), True)
    O['from_vector_rt'] = (lambda x, a: T3.from_vector(x.to_vector(), x.structure if x.stack_shape == () else x.structure, stack_shape=x.stack_shape) if False else T3.from_vector(x.to_vector(), x.shape, x.tucker_ranks, x.tt_ranks, stack_shape=x.stack_shape), True)
    O['to_tensor_train'] = (lambda x, a: x.to_tensor_train(), True)
    O['from_tensor_train_rt'] = (lambda x, a: T3.from_tensor_train(x.to_tensor_train()), True)
    O['entries'] = (lambda x, a: x.entries(a['idx']), True)
    O['apply'] = (lambda x, a: x.apply(a['ww']), True)
    O['probe'] = (lambda x, a: x.probe(a['ww']), True)
    O['probe_ambient_T'] = (lambda x, a: x.probe_ambient_transpose(a['zt'], a['ww']), True)
    O['probe_corewise_T'] = (lambda x, a: x.probe_corewise_transpose(a['zt'], a['ww']), True)
    O['apply_ambient_T'] = (lambda x, a: x.apply_ambient_transpose(a['c'], a['ww']), True)
    O['apply_corewise_T'] = (lambda x, a: x.apply_corewise_transpose(a['c'], a['ww']), True)
    O['entries_ambient_T'] = (lambda x, a: x.entries_ambient_transpose(a['c'], a['idx'], x.shape), True)
    O['entries_corewise_T'] = (lambda x, a: x.entries_corewise_transpose(a['c'], a['idx']), True)
    O['probe_derivs'] = (lambda x, a: x.probe_derivatives(a['ww'], a['pp'], 2), True)
    O['apply_derivs'] = (lambda x, a: x.apply_derivatives(a['ww'], a['pp'], 2), True)
    O['entries_derivs'] = (lambda x, a: x.entries_derivatives(a['idx'], a['pp'], 2), True)
    O['probe_corewise_derivs_T'] = (lambda x, a: x.probe_corewise_derivatives_transpose(a['zt_jet'], a['ww'], a['pp'], 2), True)
    O['apply_corewise_derivs_T'] = (lambda x, a: x.apply_corewise_derivatives_transpose(a['c_jet'], a['ww'], a['pp'], 2), True)
    O['entries_corewise_derivs_T'] = (lambda x, a: x.entries_corewise_derivatives_transpose(a['c_jet'], a['idx'], a['pp'], 2), True)
    O['t3svd_maxranks'] = (lambda x, a: x.t3svd(max_tt_ranks=x.tt_ranks, max_tucker_ranks=x.tucker_ranks)[0], True)
    O['t3svd_rtol'] = (lambda x, a: x.t3svd(rtol=1e-10)[0], False)
    O['t3svd_svals'] = (lambda x, a: x.t3svd()[1:], False)
    O['down_orth'] = (lambda x, a: x.down_orthogonalize_tucker_cores(), True)
    O['up_orth'] = (lambda x, a: x.up_orthogonalize_tt_cores(), True)
    O['left_orth'] = (lambda x, a: x.left_orthogonalize_tt_cores(), True)
    O['right_orth'] = (lambda x, a: x.right_orthogonalize_tt_cores(), True)
    O['is_left_orth'] = (lambda x, a: x.left_orthogonalize_tt_cores().is_left_orthogonal(), False)
    O['is_right_orth'] = (lambda x, a: x.right_orthogonalize_tt_cores().is_right_orthogonal(), False)
    O['minimal_ranks'] = (lambda x, a: x.minimal_ranks, False)
    O['has_minimal_ranks'] = (lambda x, a: x.has_minimal_ranks, False)
    O['has_num_minimal'] = (lambda x, a: x.has_numerically_minimal_ranks(), False)
    O['rank_adj_sweep'] = (lambda x, a: x.rank_adjustment_sweep(), True)
    O['continuation_ranks'] = (lambda x, a: x.continuation_ranks()[0], False)
    O['down_svd_tucker_core'] = (lambda x, a: x.down_svd_tucker_core(0)[0], False)
    O['left_svd_tt_core'] = (lambda x, a: x.left_svd_tt_core(0)[0], False)
    O['stack_unstack'] = (lambda x, a: T3.stack(x.unstack()) if x.stack_shape else x, True)
    O['sum_stack'] = (lambda x, a: x.sum_stack() if x.stack_shape else x, True)
    O['segment'] = (lambda x, a: x.segment(0, x.d), True)
    O['T3Weights_from_t3svd'] = (lambda x, a: T3Weights.from_t3svd(x), False)
    O['t3_weighted_norm'] = (lambda x, a: t3_weighted_norm(x.t3svd()[0], T3Weights.from_t3svd(x)), False)
    O['t3_absorb_weights'] = (lambda x, a: t3_absorb_weights(x.t3svd()[0], T3Weights.from_t3svd(x)), False)
    O['t3_weighted_inner'] = (lambda x, a: t3_weighted_inner(x.t3svd()[0], T3Weights.from_t3svd(x), x.t3svd()[0], T3Weights.from_t3svd(x)), False)
    O['share_all'] = (lambda x, a: x.share([0] * x.d) if len(set(x.shape)) == 1 else x, False)
    O['has_shared'] = (lambda x, a: x.has_shared_tucker_factors([0] * x.d) if len(set(x.shape)) == 1 else True, False)
    return O

def ops_tangent():
    O = {}
    def fv(x):
        return t3_orthogonal_representations(x)
    O['orth_reps_frame'] = (lambda x, a: fv(x)[0], True)
    O['orth_reps_point'] = (lambda x, a: T3Tangent(*fv(x)).to_t3(), True)
    O['orth_reps_is_orth'] = (lambda x, a: fv(x)[0].is_orthogonal(), False)
    O['frame_from_t3'] = (lambda x, a: T3Frame.from_t3(x), True)
    O['frame_to_t3'] = (lambda x, a: T3Frame.from_t3(x).to_t3(), True)
    O['frame_reverse'] = (lambda x, a: fv(x)[0].reverse(), True)
    O['frame_orthogonalize'] = (lambda x, a: T3Frame.from_t3(x).orthogonalize(), True)
    O['frame_minimal_ranks'] = (lambda x, a: fv(x)[0].minimal_ranks, False)
    O['frame_has_minimal_ranks'] = (lambda x, a: fv(x)[0].has_minimal_ranks, False)
    O['tangent_to_dense'] = (lambda x, a: T3Tangent(*fv(x)).to_dense(), True)
    O['tangent_reverse'] = (lambda x, a: T3Tangent(*fv(x)).reverse().to_dense(), True)
    O['tangent_probe'] = (lambda x, a: T3Tangent(*fv(x)).probe(a['ww']), True)
    O['tangent_apply'] = (lambda x, a: T3Tangent(*fv(x)).apply(a['ww']), True)
    O['tangent_entries'] = (lambda x, a: T3Tangent(*fv(x)).entries(a['idx']), True)
    O['tangent_probe_T'] = (lambda x, a: T3Tangent.probe_transpose(a['zt'], a['ww'], fv(x)[0]), True)
    O['tangent_apply_T'] = (lambda x, a: T3Tangent.apply_transpose(a['c'], a['ww'], fv(x)[0]), True)
    O['tangent_entries_T'] = (lambda x, a: T3Tangent.entries_transpose(a['c'], a['idx'], fv(x)[0]), True)
    O['tangent_probe_derivs'] = (lambda x, a: T3Tangent(*fv(x)).probe_derivatives(a['ww'], a['pp'], 2), True)
    O['tangent_apply_derivs'] = (lambda x, a: T3Tangent(*fv(x)).apply_derivatives(a['ww'], a['pp'], 2), True)
    O['tangent_entries_derivs'] = (lambda x, a: T3Tangent(*fv(x)).entries_derivatives(a['idx'], a['pp'], 2), True)
    O['tangent_probe_derivs_T'] = (lambda x, a: T3Tangent.probe_derivatives_transpose(a['zt_jet'], a['ww'], a['pp'], fv(x)[0], 2), True)
    O['tangent_probe_derivs_T_nochunk'] = (lambda x, a: T3Tangent.probe_derivatives_transpose(a['zt_jet'], a['ww'], a['pp'], fv(x)[0], 2, chunk_size=None), True)
    O['tangent_apply_derivs_T'] = (lambda x, a: T3Tangent.apply_derivatives_transpose(a['c_jet'], a['ww'], a['pp'], fv(x)[0], 2), True)
    O['tangent_entries_derivs_T'] = (lambda x, a: T3Tangent.entries_derivatives_transpose(a['c_jet'], a['idx'], a['pp'], fv(x)[0], 2), True)
    O['manifold_project'] = (lambda x, a: MANIFOLD.project(T3Tangent(*fv(x))), True)
    O['manifold_project_oblique'] = (lambda x, a: MANIFOLD.project_oblique(T3Tangent(*fv(x))), True)
    O['manifold_inner'] = (lambda x, a: (lambda v: MANIFOLD.inner(v, v))(MANIFOLD.project(T3Tangent(*fv(x)))), True)
    O['manifold_norm'] = (lambda x, a: MANIFOLD.norm(MANIFOLD.project(T3Tangent(*fv(x)))), True)
    O['manifold_retract'] = (lambda x, a: MANIFOLD.retract(MANIFOLD.project(T3Tangent(*fv(x)))), True)
    O['manifold_project_ambient_t3'] = (lambda x, a: MANIFOLD.project_ambient(fv(x)[0], x), True)
    O['manifold_project_ambient_dense'] = (lambda x, a: MANIFOLD.project_ambient(fv(x)[0], x.to_dense()), True)
    O['manifold_project_ambient_t3svd'] = (lambda x, a: MANIFOLD.project_ambient(fv(x)[0], x.to_dense(), method='t3svd'), False)
    O['manifold_transport'] = (lambda x, a: MANIFOLD.transport(MANIFOLD.project(T3Tangent(*fv(x))), fv(x + x)[0]), True)
    O['corewise_project'] = (lambda x, a: COREWISE.project(T3Tangent(*fv(x))), True)
    O['corewise_inner'] = (lambda x, a: (lambda v: COREWISE.inner(v, v))(T3Tangent(*fv(x))), True)
    O['corewise_retract'] = (lambda x, a: COREWISE.retract(T3Tangent(*fv(x))), True)
    O['tangent_corewise_norm'] = (lambda x, a: T3Tangent(*fv(x)).corewise_norm(), True)
    O['tangent_to_vector_rt'] = (lambda x, a: (lambda v: T3Tangent.from_vector(v.to_vector(), v.frame))(T3Tangent(*fv(x))), True)
    O['tangent_zeros'] = (lambda x, a: T3Tangent.zeros(fv(x)[0]), True)
    O['tangent_unit'] = (lambda x, a: T3Tangent.unit(fv(x)[0], (False, 0, (0, 0))), True)
    O['tangent_sum_tangents'] = (lambda x, a: T3Tangent.sum_tangents([T3Tangent(*fv(x)), T3Tangent(*fv(x))]), True)
    O['tangent_gauge_residual'] = (lambda x, a: T3Tangent(*fv(x)).gauge_residual, True)
    O['tangent_is_gauged'] = (lambda x, a: MANIFOLD.project(T3Tangent(*fv(x))).is_gauged(), False)
    O['frame_weights'] = (lambda x, a: T3FrameWeights.from_t3weights(T3Weights.from_t3svd(x)), False)
    O['tangent_weighted_norm'] = (lambda x, a: (lambda xs: T3Tangent(*fv(xs)).weighted_norm(T3FrameWeights.from_t3weights(T3Weights.from_t3svd(xs))))(x.t3svd()[0]), False)
    O['tangent_absorb_weights'] = (lambda x, a: (lambda xs: T3Tangent(*fv(xs)).absorb_weights(T3FrameWeights.from_t3weights(T3Weights.from_t3svd(xs))).to_dense())(x.t3svd()[0]), False)
    O['manifold_dim'] = (lambda x, a: tb.manifold.manifold_dim(x.shape, x.tucker_ranks, x.tt_ranks), False)
    O['tangent_space_dim'] = (lambda x, a: T3Tangent(*fv(x)).tangent_space_dimension, False)
    O['manifold_randn_inner'] = (lambda x, a: (lambda v: MANIFOLD.norm(v) > 0)(MANIFOLD.randn(fv(x)[0])), False)
    O['stack_tangents'] = (lambda x, a: T3Tangent.stack_tangents([T3Tangent(*fv(x)), T3Tangent(*fv(x))]).to_dense(), True)
    O['unstack_tangents'] = (lambda x, a: T3Tangent.unstack_tangents(T3Tangent.stack_tangents([T3Tangent(*fv(x)), T3Tangent(*fv(x))]))[1].to_dense(), True)
    O['unstack_frame'] = (lambda x, a: T3Tangent.unstack_frame(T3Tangent(*fv(x)))[0].to_dense() if x.stack_shape else 0, True)
    return O

def run(ops, label):
    rows = []
    for sname, struct in STRUCTS.items():
        for stack in STACKS:
            np.random.seed(0)
            x_np = T3.randn(*struct, stack_shape=stack)
            aux_np = mk_aux(struct, stack)
            x_jx = x_np.to_jax()
            aux_jx = to_jax_tree(aux_np)
            for oname, (fn, jittable) in ops.items():
                res = {}
                for bname, x, a in (('np', x_np, aux_np), ('jax', x_jx, aux_jx)):
                    try:
                        res[bname] = ('ok', fn(x, a))
                    except Exception as e:
                        res[bname] = ('err', '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:140]))
                if jittable and sname in JIT_STRUCTS and stack in JIT_STACKS:
                    try:
                        jf = jax.jit(lambda x, a: fn(x, a))
                        with tb.safety.unsafe():
                            res['jit'] = ('ok', jf(x_jx, aux_jx))
                    except Exception as e:
                        res['jit'] = ('err', '%s: %s' % (type(e).__name__, str(e).splitlines()[0][:140]))
                # evaluate
                notes = []
                for b in res:
                    if res[b][0] == 'err':
                        notes.append('%s ERR %s' % (b, res[b][1]))
                if res['np'][0] == 'ok' and res['jax'][0] == 'ok':
                    m = compare(res['jax'][1], res['np'][1])
                    if m:
                        notes.append('np/jax MISMATCH ' + m)
                    try:
                        if numpy_leak(res['jax'][1]):
                            notes.append('jax->numpy LEAK')
                    except Exception:
                        pass
                if 'jit' in res and res['jit'][0] == 'ok' and res['np'][0] == 'ok':
                    m = compare(res['jit'][1], res['np'][1])
                    if m:
                        notes.append('np/jit MISMATCH ' + m)
                if notes:
                    rows.append((label, sname, stack, oname, '; '.join(notes)))
                    print('%-8s %-10s %-5s %-32s %s' % rows[-1], flush=True)
    return rows

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    rows = []
    if which in ('t3', 'both'):
        rows += run(ops_t3(), 'T3')
    if which in ('tangent', 'both'):
        rows += run(ops_tangent(), 'TANGENT')
    print('TOTAL NOTES:', len(rows))
