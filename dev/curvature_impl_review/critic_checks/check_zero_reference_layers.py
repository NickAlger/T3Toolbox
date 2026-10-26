"""Critic pre-check (read-only): (a) the zero-tensor reference is completion-dependent (the frame at X = 0 is an
arbitrary orthonormal completion): compare ragged vs uniform, and ragged under a rotated (still exact-zero) start;
(b) on the O2 optimizer-sweep setup (ragged-vs-uniform newton_losses paired at 1e-7), does the forcing term
eta = min(0.5, (|g|/g0)^0.5) differ between layers once g0 is the zero reference?"""
import sys, os; sys.path.insert(0, __import__("os").getcwd()); sys.path.insert(0, "tests/oracle_sweeps")
import numpy as np
import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.optimizers as topt
import t3toolbox.manifold as t3m
import t3toolbox.uniform_manifold as ut3m
import sweep_optimizers as SO

def zero_ref(geom, kind, sample, y, x0, order=None, weight=None):
    problem, init, _ = topt._setup(geom, kind, sample, y, x0, order, weight, None, chunk_size=None)
    import t3toolbox.corewise as cw
    g = problem.local_model(cw.corewise_zeros_like(init)).gradient
    gs = problem.local_model(init).gradient
    return float(problem.geom.inner(g, g)) ** 0.5, float(problem.geom.inner(gs, gs)) ** 0.5

rows = []
for d in (3, 2):
    for kind in ('apply', 'probe', 'probe_derivatives'):
        order = 2 if kind.endswith('_derivatives') else None
        x, sharing, sample, weight, y = SO.setup(d, False, kind, order, 'none', 11)
        ux = ut3.UniformTuckerTensorTrain.from_t3(x)
        g0r, gsr = zero_ref(t3m.MANIFOLD, kind, sample, y, x, order)
        g0u, gsu = zero_ref(ut3m.UNIFORM_MANIFOLD, kind, sample, y, ux, order)
        eta = lambda g, g0: min(0.5, (g / g0) ** 0.5)
        print(f"d={d} {kind:18s} zero-ref ragged {g0r:.4e} uniform {g0u:.4e} (rel diff {abs(g0r - g0u) / g0r:.1e}) | "
              f"|g(x0)| {gsr:.3e} | eta0 ragged {eta(gsr, g0r):.4f} uniform {eta(gsu, g0u):.4f}")
