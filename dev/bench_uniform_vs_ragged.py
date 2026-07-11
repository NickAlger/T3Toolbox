# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
"""Wall-clock benchmark: ragged (eager) vs uniform (eager) vs uniform (jit) least-squares fitting.

The uniform layer replaces the ragged Python loop over the mode axis ``d`` with one stacked supercore +
``jax.lax.scan``, and (under jit) holds the masks fixed so the per-step kernel compiles ONCE
(``docs/uniform_backend_jit_recipe.md``). **Both wins are a GPU story.** On CPU expect little or no
speedup -- XLA-CPU plus jax 0.10's slower jit can even lose to plain numpy on small problems -- so the
point of this script is to be *re-run on a GPU server*, where the scan over ``d`` and the recompile-free
step should pull ahead as ``d`` / the ranks / ``W`` grow.

Run (CPU or GPU)::

    PYTHONPATH=$PWD <env-python> dev/bench_uniform_vs_ragged.py

Tune the problem via env vars (bigger ``d`` and ranks favor the uniform layer)::

    SHAPE=10,10,10,10,10,10,10,10 TR=4 TTR=6 W=800 BATCH=128 MAX_ITER=400 ... dev/bench_uniform_vs_ragged.py
"""
import os
import time
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.backend.optimizers as bopt
import t3toolbox.backend.fitting as bfit
import t3toolbox.backend.uniform_fitting as uf
from t3toolbox.backend import apply as bapply

SHAPE    = tuple(int(x) for x in os.environ.get("SHAPE", "8,8,8,8,8,8").split(","))   # d modes
TUCKER   = tuple(int(os.environ.get("TR",  "3")) for _ in SHAPE)                      # tucker ranks
TT       = (1,) + tuple(int(os.environ.get("TTR", "4")) for _ in SHAPE[:-1]) + (1,)   # tt ranks
W        = int(os.environ.get("W",        "400"))    # number of measurements (probe/apply samples)
BATCH    = int(os.environ.get("BATCH",    "64"))     # minibatch
MAX_ITER = int(os.environ.get("MAX_ITER", "300"))


def timed(label, fn, warmup):
    """Report wall-clock for ``fn`` (warm caches / jit-compile first if ``warmup``)."""
    if warmup:
        fn()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    print(f"  {label:26s} {dt:9.3f} s")
    return dt


def main():
    np.random.seed(0)
    x_true = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
    ww = [np.random.randn(W, n) for n in SHAPE]
    data = bapply.t3_apply(x_true.data, ww)
    x0 = t3.TuckerTensorTrain.randn(SHAPE, TUCKER, TT)
    ux0 = ut3.UniformTuckerTensorTrain.from_t3(x0)

    prob_rag = bopt.least_squares_problem(bopt.MANIFOLD, bfit.APPLY, ww, data)
    prob_uni = uf.uniform_least_squares_problem('manifold', 'apply', ux0, ww, data)

    def run(problem, x0_cores, **kw):
        return lambda: bopt.mc_sgd(problem, x0_cores, np.random.default_rng(0),
                                   batch=BATCH, max_iter=MAX_ITER, **kw)

    print(f"problem: d={len(SHAPE)} modes  shape={SHAPE}  tucker={TUCKER}  tt={TT}")
    print(f"         W={W}  batch={BATCH}  max_iter={MAX_ITER}  (mc_sgd, manifold, apply)\n")

    t_rag = timed("ragged  eager (numpy)", run(prob_rag, x0.data), warmup=False)
    t_uni = timed("uniform eager (numpy)", run(prob_uni, (ux0.data[0], ux0.data[1])), warmup=False)

    t_jit = None
    try:
        import jax
        import jax.numpy as jnp
        plat = jax.devices()[0].platform
        ww_j = [jnp.asarray(w) for w in ww]
        data_j = jnp.asarray(data)
        ux0_j = ux0.to_jax()                                  # supercores -> jax; masks stay host numpy
        prob_uni_j = uf.uniform_least_squares_problem('manifold', 'apply', ux0_j, ww_j, data_j)
        print(f"\n  jax {jax.__version__} on {plat.upper()}  (jit warms up = first-step compile)")
        t_jit = timed("uniform jit   (jax)  ", run(prob_uni_j, (ux0_j.data[0], ux0_j.data[1]), use_jit=True),
                      warmup=True)
    except ImportError:
        print("\n  (jax not installed -> skipping the uniform-jit variant)")

    print("\nspeedups (higher = uniform faster):")
    print(f"  uniform-eager vs ragged-eager: {t_rag / t_uni:6.2f}x")
    if t_jit is not None:
        print(f"  uniform-jit   vs ragged-eager: {t_rag / t_jit:6.2f}x")
        print(f"  uniform-jit   vs uniform-eager:{t_uni / t_jit:6.2f}x")
    print("\nNOTE: on CPU the jit variant often does NOT beat numpy (XLA-CPU + jax 0.10's slow jit); the "
          "uniform layer's win is a GPU story -- re-run this on a GPU server, and scale up SHAPE/TR/TTR/W.")


if __name__ == '__main__':
    main()
