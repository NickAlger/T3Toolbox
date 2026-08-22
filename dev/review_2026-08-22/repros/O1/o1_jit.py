"""Phase 3: jit == eager (x64) for the sampling/tangent ops, ragged and uniform."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from o1_common import *
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import t3toolbox.safety as safety
ORDER = 2
def run(sname, struct):
    shape, tr, ttr = struct; d = len(shape)
    for C in [(), (2,)]:
        np.random.seed(1)
        x = t3.TuckerTensorTrain.randn(shape, tr, ttr, stack_shape=C); xj = x.to_jax()
        frame, _ = bvf.t3_orthogonal_representations(x); framej = frame.to_jax()
        ux = ut3.UniformTuckerTensorTrain.from_t3(x).t3svd()[0].rank_adjustment_sweep('right_to_left'); uxj = ux.to_jax()
        uframe, _ = ubv.ut3_orthogonal_representations(ux); uframej = uframe.to_jax()
        for W in [(3,)]:
            ww = rand_ww(shape, W, 7); pp = rand_ww(shape, W, 8); index = rand_index(shape, W, 9)
            wwj = [jnp.asarray(w) for w in ww]; ppj = [jnp.asarray(p) for p in pp]
            pairs = [('probe', lambda o, w: o.probe(w), ww, wwj), ('apply', lambda o, w: o.apply(w), ww, wwj),
                     ('entries', lambda o, w: o.entries(index), ww, wwj),
                     ('probe_derivatives', lambda o, w: o.probe_derivatives(w, ppj if isinstance(w[0], jnp.ndarray) else pp, ORDER), ww, wwj),
                     ('apply_derivatives', lambda o, w: o.apply_derivatives(w, ppj if isinstance(w[0], jnp.ndarray) else pp, ORDER), ww, wwj)]
            for rep, o_np, o_j in [('ragged', x, xj), ('uniform', ux, uxj)]:
                for name, fn, a_np, a_j in pairs:
                    def _chk():
                        eager = fn(o_np, a_np)
                        jitted = jax.jit(lambda o, w: fn(o, w))(o_j, a_j)
                        if isinstance(eager, (list, tuple)): return max(relerr(np.asarray(b), np.asarray(a)) for a, b in zip(eager, jitted))
                        return relerr(np.asarray(jitted), np.asarray(eager))
                    check('jit_' + name, sname, rep, C, W, (), None, _chk, tol=1e-10)
                # corewise transpose
                c = np.random.RandomState(3).randn(*(W + C))
                def _cwt():
                    e = o_np.apply_corewise_transpose(c, ww, sum_over_probes=True)
                    j = jax.jit(lambda o, cc, w: o.apply_corewise_transpose(cc, w, sum_over_probes=True))(o_j, jnp.asarray(c), wwj)
                    return max(relerr(np.asarray(bj), np.asarray(ae)) for fe, fj in zip(e[:2], j[:2]) for ae, bj in zip(fe, fj))
                check('jit_apply_corewise_transpose', sname, rep, C, W, (), None, _cwt, tol=1e-10)
            for K in KS:
                np.random.seed(3)
                v = t3m.MANIFOLD.randn(frame, stack_shape=K); vj = v.to_jax()
                uv = ut3m.UNIFORM_MANIFOLD.randn(uframe, stack_shape=K); uvj = uv.to_jax()
                for rep, t_np, t_j, geom, fr_np, fr_j in [('ragged', v, vj, t3m.MANIFOLD, frame, framej), ('uniform', uv, uvj, ut3m.UNIFORM_MANIFOLD, uframe, uframej)]:
                    check('jit_tv_probe', sname, rep, C, W, K, None, lambda: max(relerr(np.asarray(b), np.asarray(a)) for a, b in zip(t_np.probe(ww), jax.jit(lambda t, w: t.probe(w))(t_j, wwj))), tol=1e-10)
                    check('jit_tv_apply_derivatives', sname, rep, C, W, K, None, lambda: relerr(np.asarray(jax.jit(lambda t, w, p: t.apply_derivatives(w, p, ORDER))(t_j, wwj, ppj)), np.asarray(t_np.apply_derivatives(ww, pp, ORDER))), tol=1e-10)
                    zt = [np.random.RandomState(5 + i).randn(*(W + K + C + (N,))) for i, N in enumerate(shape)]
                    def _pt():
                        e = type(t_np).probe_transpose(zt, ww, fr_np, sum_over_probes=True).to_dense()
                        j = jax.jit(lambda z, w, f: type(t_np).probe_transpose(z, w, f, sum_over_probes=True).to_dense())([jnp.asarray(a) for a in zt], wwj, fr_j)
                        return relerr(np.asarray(j), np.asarray(e))
                    check('jit_tv_probe_transpose', sname, rep, C, W, K, None, _pt, tol=1e-10)
                    ztj_ = [np.random.RandomState(15 + i).randn(*((ORDER + 1,) + W + K + C + (N,))) for i, N in enumerate(shape)]
                    def _pdt():
                        e = type(t_np).probe_derivatives_transpose(ztj_, ww, pp, fr_np, ORDER, sum_over_probes=True).to_dense()
                        j = jax.jit(lambda z, w, p, f: type(t_np).probe_derivatives_transpose(z, w, p, f, ORDER, sum_over_probes=True).to_dense())([jnp.asarray(a) for a in ztj_], wwj, ppj, fr_j)
                        return relerr(np.asarray(j), np.asarray(e))
                    check('jit_tv_probe_derivatives_transpose', sname, rep, C, W, K, None, _pdt, tol=1e-10)
                    check('jit_retract', sname, rep, C, W, K, None, lambda: relerr(np.asarray(jax.jit(lambda t: geom.retract(t * 0.1).to_dense())(t_j)), np.asarray(geom.retract(t_np * 0.1).to_dense())), tol=1e-10)
                    check('jit_project', sname, rep, C, W, K, None, lambda: relerr(np.asarray(jax.jit(lambda t: geom.project(t).to_dense())(t_j)), np.asarray(geom.project(t_np).to_dense())), tol=1e-10)
                    check('jit_inner', sname, rep, C, W, K, None, lambda: relerr(np.asarray(jax.jit(lambda t: geom.inner(t, t))(t_j)), np.asarray(geom.inner(t_np, t_np))), tol=1e-10)
if __name__ == '__main__':
    for s in ['d3', 'd4']: run(s, STRUCTS[s]); print('done', s, flush=True)
    dump(os.path.join(os.path.dirname(__file__), 'results_jit.md'))
