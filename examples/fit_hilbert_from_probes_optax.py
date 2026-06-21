"""Fit a Tucker tensor train to PROBE measurements of a Hilbert tensor -- corewise + OPTAX.

The optax counterpart of ``fit_hilbert_from_probes_adam.py``: the *same* problem and corewise setup, but
the optimizer is an **optax** optimizer (``optax.adam`` with a cosine schedule) instead of our hand-written
Adam. It exists to work out, in practice, **how T3Toolbox objects interact with optax** -- a question real
users will hit, and one we want answered before we design the optimizer module.

The key fact: a ``TuckerTensorTrain``'s ``.data = (tucker_cores, tt_cores)`` is a **native JAX pytree** (a
nested tuple of arrays), and so is the corewise gradient (``model.gradient.variations.data``). optax is
built to operate on pytrees, so it hooks in with **no glue** -- no flat ``to_vector``/``from_vector``
bridge (that is for non-pytree optimizers like scipy), no per-leaf bookkeeping:

    optimizer = optax.adam(optax.cosine_decay_schedule(lr, n_steps))
    opt_state = optimizer.init(cores)                       # cores = (tucker_cores, tt_cores) pytree
    ...
    grad = model.gradient.variations.data                  # matching pytree of core gradients
    updates, opt_state = optimizer.update(grad, opt_state, cores)
    cores = optax.apply_updates(cores, updates)            # new (tucker_cores, tt_cores)

optax runs on JAX, so once the cores enter it they become JAX arrays and the whole fit runs on the JAX
backend (the library infers jax dispatch from the core arrays). Everything else -- the probe data,
minibatching, rank continuation, validation -- is identical to the hand-Adam example, so the two are a
clean A/B: with the same cosine-decayed Adam the results match, confirming the hook is faithful.

Requires ``optax`` (and a recent jax). Run from the repo root (in an env with optax):
    python examples/fit_hilbert_from_probes_optax.py
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.manifold as t3m
import t3toolbox.fitting as fitting


# --------------------------------------------------------------------------------------------------
# Problem configuration (matches fit_hilbert_from_probes_adam.py)
# --------------------------------------------------------------------------------------------------
SHAPE        = (12, 12, 12, 12)
N_TRAIN      = 200
N_VAL        = 100
NOISE_LEVEL  = 0.01
RANK_LEVELS  = (1, 2, 3, 4, 5, 6)
SEED         = 0

OPTAX_LR     = 2e-2                    # peak learning rate (cosine-decayed), same as the hand-Adam example
OPTAX_BATCH  = 32                      # probes per minibatch
OPTAX_MAXITER = 1200


# --------------------------------------------------------------------------------------------------
# Target tensor + probe sampling (identical to the hand-Adam example)
# --------------------------------------------------------------------------------------------------
def hilbert_tensor(shape):
    grids = np.meshgrid(*[np.arange(N) for N in shape], indexing="ij")
    return 1.0 / (1.0 + sum(grids))


def unit_probes(M, shape, rng):
    ww = [rng.standard_normal((M, N)) for N in shape]
    return [w / np.linalg.norm(w, axis=1, keepdims=True) for w in ww]


def dense_probe(A, ww):
    d = len(ww)
    mode_axes = list(range(d)); w_axis = d
    out = []
    for free in range(d):
        ops = [A, mode_axes]
        for j in range(d):
            if j != free:
                ops += [ww[j], [w_axis, mode_axes[j]]]
        ops += [[w_axis, mode_axes[free]]]
        out.append(np.einsum(*ops))
    return out


def rms_all(arrs):
    ss = sum(float(np.sum(np.asarray(a) ** 2)) for a in arrs)
    n = sum(np.asarray(a).size for a in arrs)
    return float(np.sqrt(ss / n))


def probe_relerr(pred, data):
    num = sum(float(np.sum((np.asarray(p) - np.asarray(dd)) ** 2)) for p, dd in zip(pred, data))
    den = sum(float(np.sum(np.asarray(dd) ** 2)) for dd in data)
    return float(np.sqrt(num / den))


# --------------------------------------------------------------------------------------------------
# Corewise optimization with an optax optimizer (no flat bridge -- optax eats the native pytree)
# --------------------------------------------------------------------------------------------------
def random_start(tucker_ranks, tt_ranks, ww, data):
    """A small random corewise start, rescaled so the initial probes match the data magnitude (corewise
    needs a nonzero start; see fit_hilbert_from_entries_lbfgs.py / docs/entries_completion_findings.md)."""
    X = t3.TuckerTensorTrain.randn(SHAPE, tucker_ranks, tt_ranks)
    scale = rms_all(data) / max(rms_all(X.probe(ww)), 1e-12)
    tucker_cores, tt_cores = X.data
    c = scale ** (1.0 / (len(tucker_cores) + len(tt_cores)))
    return t3.TuckerTensorTrain(tuple(c * C for C in tucker_cores), tuple(c * C for C in tt_cores))


def corewise_optax(X0, ww, data, rng, optimizer, batch=OPTAX_BATCH, max_iter=OPTAX_MAXITER):
    """Fit X to probe data with an **optax** optimizer, minibatching over probes. ``X.data =
    (tucker_cores, tt_cores)`` is a JAX pytree, so optax operates on it directly: ``optimizer.init`` /
    ``.update`` consume the cores and the corewise gradient (also a matching pytree) with no flat bridge.

    The whole per-step computation -- build the point, probe the minibatch, take the corewise gradient,
    optax-update -- is **``jax.jit``-compiled** as one function of ``(cores, opt_state, ww_B, data_B)``.
    The minibatch shapes are fixed within a rank level, so it compiles once and the 1200 steps reuse it;
    that is the difference between **~7.5 min (eager) and ~55 s (jit)** here -- an ~8x speedup. (It
    recompiles once per rank level, where the core shapes change.) The forward probe and the corewise
    model are already jit-safe."""
    cores = jax.tree_util.tree_map(jnp.asarray, X0.data)   # (tucker, tt) pytree, on device
    opt_state = optimizer.init(cores)
    ww_j = [jnp.asarray(w) for w in ww]                   # full train probes, on device
    data_j = [jnp.asarray(dd) for dd in data]
    n = ww_j[0].shape[0]

    @jax.jit
    def step(cores, opt_state, ww_B, data_B):
        X = t3.TuckerTensorTrain(*cores)                  # jax cores -> jax dispatch throughout
        pred_B = X.probe(ww_B)
        r_B = [pred_B[i] - data_B[i] for i in range(len(ww_B))]
        grad = fitting.probe_model(t3m.COREWISE, X, ww_B, r_B).gradient.variations.data
        updates, opt_state = optimizer.update(grad, opt_state, cores)
        return optax.apply_updates(cores, updates), opt_state

    for k in range(max_iter):
        sel = rng.choice(n, size=min(batch, n), replace=False)
        ww_B = [w[sel] for w in ww_j]
        data_B = [dd[sel] for dd in data_j]
        cores, opt_state = step(cores, opt_state, ww_B, data_B)
    return t3.TuckerTensorTrain(*cores), dict(iters=max_iter)


# --------------------------------------------------------------------------------------------------
# Rank helpers
# --------------------------------------------------------------------------------------------------
def level_ranks(level, shape):
    d = len(shape)
    return tuple(min(level, N) for N in shape), (1,) + (level,) * (d - 1) + (1,)


def oracle_relerr(A, tucker_ranks, tt_ranks):
    Xr, _, _ = t3.TuckerTensorTrain.t3svd_dense(A, max_tucker_ranks=tucker_ranks, max_tt_ranks=tt_ranks)
    return float(np.linalg.norm(Xr.to_dense() - A)) / float(np.linalg.norm(A))


# --------------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    rng_opt = np.random.default_rng(SEED + 1)
    d = len(SHAPE)

    print(__doc__.split("\n\n")[0])
    print(f"\nHilbert tensor: shape {SHAPE}  ({np.prod(SHAPE):,} entries),  order d={d}")
    print(f"optax {optax.__version__} on jax {jax.__version__}")

    A = hilbert_tensor(SHAPE)
    A_norm = float(np.linalg.norm(A))

    M = N_TRAIN + N_VAL
    ww_all = unit_probes(M, SHAPE, rng)
    y_clean = dense_probe(A, ww_all)
    y_rms = rms_all(y_clean)
    y_all = [yc + NOISE_LEVEL * y_rms * rng.standard_normal(yc.shape) for yc in y_clean]
    ww_tr = [w[:N_TRAIN] for w in ww_all]; ww_va = [w[N_TRAIN:] for w in ww_all]
    y_tr = [y[:N_TRAIN] for y in y_all]; y_va = [y[N_TRAIN:] for y in y_all]

    print(f"Measurements (probes): {N_TRAIN} train + {N_VAL} validation, {NOISE_LEVEL*100:.0f}% noise.")
    print(f"Fit on COREWISE by optax.adam (lr={OPTAX_LR} cosine-decayed, minibatch {OPTAX_BATCH} probes).\n")

    optimizer = optax.adam(optax.cosine_decay_schedule(OPTAX_LR, OPTAX_MAXITER))

    header = (f"{'level':>5} {'tucker / tt ranks':>24} {'DOF':>5} {'iters':>5} "
              f"{'train':>9} {'val':>9} {'true':>9} {'oracle':>9}")
    print(header)
    print("-" * len(header))

    records = []
    t_start = time.perf_counter()
    for r in RANK_LEVELS:
        tucker_ranks, tt_ranks = level_ranks(r, SHAPE)
        dof = t3m.manifold_dim((SHAPE, tucker_ranks, tt_ranks))
        X0 = random_start(tucker_ranks, tt_ranks, ww_tr, y_tr)
        X, stats = corewise_optax(X0, ww_tr, y_tr, rng_opt, optimizer)

        train_e = probe_relerr(X.probe(ww_tr), y_tr)
        val_e = probe_relerr(X.probe(ww_va), y_va)
        true_e = float(np.linalg.norm(np.asarray(X.to_dense()) - A)) / A_norm
        oracle_e = oracle_relerr(A, tucker_ranks, tt_ranks)
        records.append(dict(level=r, dof=dof, val=val_e, true=true_e))
        rank_str = f"{tucker_ranks} {tt_ranks}"
        print(f"{r:>5} {rank_str:>24} {dof:>5} {stats['iters']:>5} "
              f"{train_e:>9.3e} {val_e:>9.3e} {true_e:>9.3e} {oracle_e:>9.3e}")

    best = min(records, key=lambda rec: rec["val"])
    print("-" * len(header))
    print(f"\nNoise floor (relative): {NOISE_LEVEL:.1e}    (total fit time {time.perf_counter()-t_start:.1f}s)")
    print(f"Best ranks by validation error: level {best['level']}  "
          f"(val {best['val']:.3e}, true error {best['true']:.3e}, DOF {best['dof']}).")


if __name__ == "__main__":
    main()
