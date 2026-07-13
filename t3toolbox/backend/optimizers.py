# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Backend-first optimization algorithms for fixed-rank T3 least-squares fitting.

These operate on **raw cores / tangent tuples** via backend functions only -- no frontend imports
(no ``TuckerTensorTrain`` / ``T3Tangent`` / ``GaussNewtonModel`` / geometry classes). A raw-``.data``
user can call them directly; the frontend ``optimizers.py`` is a thin adapter that validates the input
once and assembles the ``Problem`` oracle from the same backend functions (design:
``dev/archive/optimizers_plan.md``).

Because the numerical safety preconditions live only in the frontend, this layer is **check-free** of
them (structural shape guards inside the backend functions remain, and are jit-safe). So ``jit`` needs no
``unsafe()`` wrapping; the ``use_jit`` machinery is a thin jax-only layer on top of the ordinary
numpy/jax dispatch (the per-step kernels jit via ``_maybe_jit``; the inner CG runs as a
``lax.while_loop`` -- eager-vs-jit agreement is covered in ``tests/backend/test_optimizers.py``).

The oracle:

* ``Problem``     -- (geometry ops, sampling kind, sample, data); builds per-point ``LocalModel``
  objects and retracts.
* ``LocalModel``  -- the GN model linearized at a point (``.gradient`` / ``.objective`` /
  ``.hvp`` / ``.gn_quadratic`` / ``.retract``), the backend twin of ``fitting.GaussNewtonModel``.
* ``GeometryOps`` -- (frame, project, retract) on raw data; ``MANIFOLD_OPS`` / ``COREWISE_OPS``
  singletons.

Tangent vectors are raw ``(tucker_var, tt_var)`` tuples; vector arithmetic is the ``corewise`` ops.
"""
import dataclasses as dc
import math
import time
import typing as typ

from t3toolbox.backend.common import *
from t3toolbox.backend import tv_operations as tops
from t3toolbox.backend.fv_conversions import t3_orthogonal_representations
import t3toolbox.corewise as cw

__all__ = [
    'GeometryOps',
    'COREWISE_OPS',
    'MANIFOLD_OPS',
    'Problem',
    'LocalModel',
    'NewtonInfo',
    'least_squares_problem',
    'flat_draw',
    'gradient_descent',
    'mc_sgd',
    'adam',
    'newton_cg',
]

Tangent = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]   # (tucker_variations, tt_variations)


# --------------------------------------------------------------------------------------------------
# Geometry ops on raw data (frame / project / retract) -- the backend twin of the frontend geometries
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class GeometryOps:
    frame:    typ.Callable    # x_cores=(U,G)            -> frame=(U,O,P,Q)        the linearization frame
    project: typ.Callable    # (frame, variations)       -> variations           gauge Π (identity for corewise)
    retract: typ.Callable    # (frame, variations)       -> x_cores=(U,G)        chart retraction
    inner:   typ.Callable    # (v1, v2)                 -> scalar               coordinate ⟨·,·⟩ (check-free twin of Geometry.inner)


def _corewise_frame(
        x_cores: Tangent,    # (tucker_cores, tt_cores)
) -> typ.Tuple:              # (U, O, P, Q) = (U, G, G, G); the raw cores ARE the frame
    tucker_cores, tt_cores = x_cores
    return (tucker_cores, tt_cores, tt_cores, tt_cores)


COREWISE_OPS = GeometryOps(
    frame=_corewise_frame,
    project=lambda frame, var: var,                                   # Euclidean cores: no gauge projection
    retract=lambda frame, var: cw.corewise_add((frame[0], frame[2]), var),   # additive: (U,P)=(U,G) += var
    inner=cw.corewise_dot,                                           # ragged coordinate dot (layer-agnostic tree dot)
)


def _manifold_frame(
        x_cores: Tangent,    # (tucker_cores, tt_cores)
) -> typ.Tuple:              # (U, O, P, Q) orthonormal frame (Algorithm 11)
    frame, _ = t3_orthogonal_representations(x_cores)
    return frame


MANIFOLD_OPS = GeometryOps(
    frame=_manifold_frame,
    project=lambda frame, var: tops.tv_orthogonal_gauge_projection(frame, var),   # Π  (gauge-fix the tangent)
    retract=lambda frame, var: tops.tv_retract(frame, var),                       # implicit truncated T3-SVD
    inner=cw.corewise_dot,                                                    # ragged coordinate dot
)


# --------------------------------------------------------------------------------------------------
# The problem oracle
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class LocalModel:
    """The Gauss-Newton model linearized at a point -- the backend twin of ``fitting.GaussNewtonModel``.
    Built by ``Problem.local_model``; the frame sweep is computed once and reused by every method below."""
    geom:     GeometryOps
    kind:     typ.Any        # backend fitting.SamplingKind (forward / transpose / sumsq / w_axes)
    sample:   typ.Any        # ww (probe/apply) or index (entries)
    frame:     typ.Tuple      # (U, O, P, Q)
    sweep:    typ.Tuple      # precomputed frame sweep (reused across gradient/jacobian/hvp)
    residual: typ.Any        # r = S(x) - data
    n_w:      int            # number of leading sample-stack axes

    @property
    def objective(self):                                 # c = ½‖r‖²
        return 0.5 * self.kind.sumsq(self.residual, self.n_w)

    @property
    def gradient(self) -> Tangent:                       # g = Π 𝒥ᵀ r
        return self.geom.project(self.frame, self.kind.transpose(self.residual, self.sample, self.frame, self.sweep))

    def jacobian(self, p: Tangent):                      # 𝒥 Π p
        return self.kind.forward(self.geom.project(self.frame, p), self.sample, self.frame, self.sweep)

    def gn_quadratic(self, p: Tangent):                  # ‖𝒥 Π p‖²  (one forward; the Cauchy denominator)
        return self.kind.sumsq(self.jacobian(p), self.n_w)

    def hvp(self, p: Tangent) -> Tangent:                # H p = Π 𝒥ᵀ 𝒥 Π p
        z = self.jacobian(p)
        return self.geom.project(self.frame, self.kind.transpose(z, self.sample, self.frame, self.sweep))

    def retract(self, p: Tangent) -> Tangent:            # chart step from this frame -> new x_cores
        return self.geom.retract(self.frame, p)


@dc.dataclass(frozen=True)
class Problem:
    """A fixed-rank least-squares fitting problem ``min_x ½‖S(x) - data‖²`` for a sampling op ``S`` on
    geometry ``geom``. **Layout-agnostic**: it holds the operator (``kind``), the geometry, and the FULL
    ``(sample, data)``. ``local_model`` / ``objective`` linearize / evaluate at a point on the full data,
    or on an explicitly-passed minibatch ``(sample, data)`` (e.g. from a ``draw``). The ``Problem`` itself
    owns **no** minibatch-layout logic -- where the sample stack ``W`` lives, how to slice it -- that is
    the ``kind``'s (``kind.take`` for the default flat draw) or the user's ``draw``."""
    geom:   GeometryOps
    kind:   typ.Any       # backend fitting.SamplingKind
    sample: typ.Any       # the FULL sample (ww / index / (ww,pp) / (index,pp))
    data:   typ.Any       # the FULL observed data S(x_true) (+ noise)

    def local_model(self, x_cores: Tangent, sample=None, data=None) -> LocalModel:
        """Linearize at ``x_cores`` on the full data (``sample=None``) or an explicit minibatch."""
        if sample is None:
            sample, data = self.sample, self.data
        frame = self.geom.frame(x_cores)
        sweep = self.kind.precompute(frame, sample)
        residual = cw.corewise_sub(self.kind.point_forward(x_cores, sample), data)
        return LocalModel(self.geom, self.kind, sample, frame, sweep, residual, self.kind.w_axes(sample))

    def objective(self, x_cores: Tangent, sample=None, data=None):
        """``½‖S(x)-data‖²`` on the full data (``sample=None``) or an explicit minibatch; no frame sweep
        (cheap -- for the line search / the full-batch stop signal)."""
        if sample is None:
            sample, data = self.sample, self.data
        residual = cw.corewise_sub(self.kind.point_forward(x_cores, sample), data)
        return 0.5 * self.kind.sumsq(residual, self.kind.w_axes(sample))


def least_squares_problem(
        geom:   GeometryOps,   # COREWISE_OPS / MANIFOLD_OPS
        kind:   typ.Any,       # backend fitting.{APPLY,ENTRIES,PROBE} or a derivative kind
        sample: typ.Any,       # ww / index / (ww,pp) / (index,pp)
        data:   typ.Any,       # observed values
) -> Problem:
    """Assemble a least-squares ``Problem`` from a geometry, a sampling kind, the sample, and the observed
    data. (The frontend adapter calls this with the same backend objects a raw-data user would.)"""
    return Problem(geom, kind, sample, data)


def flat_draw(
        problem: Problem,
        batch:   int,        # measurements per minibatch
) -> typ.Callable:           # draw(rng) -> (sample_B, data_B)
    """The **default** minibatch draw: a uniform random subset of ``batch`` measurements across the whole
    (flattened) sample stack ``W`` -- the robust default. Returns a ``draw(rng) -> (sample_B, data_B)``
    over ``problem``'s full data, via the kind's ``take``. A user may pass **any** ``draw`` instead (slice
    X / P / order, importance-sample, ...). Host numpy by default; pass jax data + a jax draw to keep the
    minibatch on device (the optimizer never compiles the draw -- only the per-step kernel)."""
    n = problem.kind.n_measurements(problem.sample)

    def draw(rng):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        return problem.kind.take(problem.sample, problem.data, idx)
    return draw


# --------------------------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------------------------
def _maybe_jit(fn, use_jit, x0, problem):
    """jit ``fn`` iff ``use_jit`` AND jax is available AND the inputs (x0, sample, data) are **all** jax
    (so the minibatch gather + kernel run on device). Otherwise return ``fn`` unchanged -- the silent
    eager fallback (numpy, eager-jax, or no jax). Compiles once and is reused across the fixed-shape steps."""
    if (use_jit and jax_available and tree_contains_jax(x0)
            and tree_contains_jax(problem.sample) and tree_contains_jax(problem.data)):
        import jax
        return jax.jit(fn)
    return fn


def gradient_descent(
        problem:  Problem,    # the fixed-rank least-squares problem
        x0:       Tangent,    # initial cores (U, G)
        n_iter:   int   = 100,
        gtol_rel: float = 1e-8,   # stop when ‖g‖ <= gtol_rel * ‖g_0‖
        c_armijo: float = 1e-4,   # Armijo sufficient-decrease constant
) -> typ.Tuple[Tangent, dict]:    # (x_cores, stats)
    """Steepest descent with a Cauchy initial step and an **Armijo backtracking line search**. The step
    length starts at the Cauchy value ``α = ‖g‖² / ‖𝒥g‖²`` (the 1D minimizer of the local GN quadratic
    along ``−g``) and backtracks (``α ← α/2``) until ``f(retract(−α g)) ≤ f − c·α‖g‖²`` -- so it descends
    on any geometry, including the additive corewise chart where a bare Cauchy step overshoots the
    high-degree objective. Exercises the whole backend-first stack (``gradient`` / ``gn_quadratic`` /
    ``objective`` / ``retract``). (Eager; the jit kernel + the `xwhile` line search come in G3.3/G3.4.)"""
    x = x0
    losses = []
    g0norm = None
    for it in range(n_iter):
        lm = problem.local_model(x)
        f = float(lm.objective)
        losses.append(f)
        g = lm.gradient
        gg = float(problem.geom.inner(g, g))                # ‖g‖²
        if g0norm is None:
            g0norm = gg ** 0.5 if gg > 0 else 1.0
        if gg ** 0.5 <= gtol_rel * g0norm:                  # converged
            break
        alpha = gg / max(float(lm.gn_quadratic(g)), 1e-30 * gg)   # Cauchy initial step
        x_trial = x
        for _bt in range(50):                               # Armijo backtracking
            x_trial = lm.retract(cw.corewise_scale(g, -alpha))
            if float(problem.objective(x_trial)) <= f - c_armijo * alpha * gg:
                break
            alpha *= 0.5
        x = x_trial
    return x, {'losses': losses, 'n_iter': len(losses)}


def mc_sgd(
        problem:     Problem,    # the fixed-rank least-squares problem
        x0:          Tangent,    # initial cores (U, G)
        rng,                     # np.random.Generator -- passed to the draw each step
        batch:       int,        # measurements per minibatch (for the default flat draw; ignored if draw given)
        draw:        typ.Optional[typ.Callable] = None,  # custom draw(rng)->(sample_B,data_B); None = flat default
        max_iter:    int   = 3000,
        check_every: int   = 25,   # iterations between full-batch loss checks (the absolute-iteration window)
        smooth_tau:  float = 2.0,  # EMA timescale of the loss check, in checks
        plateau_lag: int   = 4,    # stop when the smoothed loss rises over this many checks
        use_jit:     bool  = False,  # jit the per-step kernel (gradient + Cauchy step + retract) when jax
) -> typ.Tuple[Tangent, dict]:     # (x_cores, stats)
    """Manifold Cauchy SGD (T4S §5.3.2): minibatch + the tuning-free Cauchy step
    ``α = ‖g‖² / ‖𝒥g‖²`` -- no learning rate. Intended for the **manifold** geometry (its bounded
    retraction makes the raw Cauchy step stable; on the additive corewise chart use ``adam``). Stops on an
    exponentially-smoothed **full-batch** loss with an **absolute-iteration window** (``plateau_lag *
    check_every`` iterations) -- decoupled from batch size (the epoch-based window made small batches
    fragile; findings in a separate research repo, maintainer-local). ``use_jit`` jits the per-step kernel (the host loop
    draws minibatches; the full-batch stop check stays on the host)."""
    draw = draw if draw is not None else flat_draw(problem, batch)
    a_smooth = 1.0 - math.exp(-1.0 / smooth_tau)

    def step(cores, sample_B, data_B):                      # the jit-able per-step kernel
        lm = problem.local_model(cores, sample_B, data_B)
        g = lm.gradient
        gg = problem.geom.inner(g, g)
        xnp, _, _ = get_backend(False, tree_contains_jax(g))
        alpha = gg / xnp.maximum(lm.gn_quadratic(g), 1e-30)
        return lm.retract(cw.corewise_scale(g, -alpha))
    step = _maybe_jit(step, use_jit, x0, problem)

    x = x0
    s_hist = []
    n_iter = 0
    for k in range(max_iter):
        n_iter = k + 1
        x = step(x, *draw(rng))
        if n_iter % check_every == 0:                       # full-batch loss check (the stop signal, host)
            L = float(problem.objective(x))
            s = L if not s_hist else a_smooth * L + (1.0 - a_smooth) * s_hist[-1]
            s_hist.append(s)
            if len(s_hist) > plateau_lag and (s_hist[-1] - s_hist[-1 - plateau_lag]) > 0.0:
                break
    return x, {'losses': s_hist, 'n_iter': n_iter}


def adam(
        problem:  Problem,    # the fixed-rank least-squares problem
        x0:       Tangent,    # initial cores (U, G)
        rng,                  # np.random.Generator -- passed to the draw each step
        batch:    int,        # measurements per minibatch (for the default flat draw; ignored if draw given)
        draw:     typ.Optional[typ.Callable] = None,  # custom draw(rng)->(sample_B,data_B); None = flat default
        lr:       float = 1e-2,
        max_iter: int   = 2000,
        betas:    typ.Tuple[float, float] = (0.9, 0.999),
        eps:      float = 1e-8,
        cosine:   bool  = True,   # cosine-decay the learning rate over max_iter (helps it settle)
        use_jit:  bool  = False,  # jit the per-step kernel (gradient + Adam update + retract) when jax
) -> typ.Tuple[Tangent, dict]:    # (x_cores, stats)
    """Adam over the cores -- the dependency-free first-order method for the **corewise** geometry. The
    moments ``m`` / ``v`` are trees matching the cores (a ``corewise_map`` per step); elementwise, so this
    is exactly per-core Adam. (``lr`` is a real hyperparameter -- Adam is not tuning-free, unlike the Cauchy
    step.) On corewise the step is additive (``lm.retract`` = ``cores += step``). ``use_jit`` jits the
    per-step kernel (``lr_t``/``t`` flow in as traced args, so the schedule does not force a recompile)."""
    b1, b2 = betas

    def step(cores, m, v, sample_B, data_B, lr_t, t):      # the jit-able per-step kernel
        lm = problem.local_model(cores, sample_B, data_B)
        g = lm.gradient
        m = cw.corewise_map(lambda mi, gi: b1 * mi + (1.0 - b1) * gi, m, g)
        v = cw.corewise_map(lambda vi, gi: b2 * vi + (1.0 - b2) * gi * gi, v, g)
        bc1, bc2 = 1.0 - b1 ** t, 1.0 - b2 ** t            # bias corrections
        xnp, _, _ = get_backend(False, tree_contains_jax(g))
        update = cw.corewise_map(lambda mi, vi: lr_t * (mi / bc1) / (xnp.sqrt(vi / bc2) + eps), m, v)
        return lm.retract(cw.corewise_scale(update, -1.0)), m, v
    step = _maybe_jit(step, use_jit, x0, problem)

    draw = draw if draw is not None else flat_draw(problem, batch)
    cores = x0
    m = cw.corewise_zeros_like(cores)
    v = cw.corewise_zeros_like(cores)
    losses = []
    for k in range(max_iter):
        sample_B, data_B = draw(rng)
        lr_t = lr * (0.5 * (1.0 + math.cos(math.pi * k / max_iter)) if cosine else 1.0)
        cores, m, v = step(cores, m, v, sample_B, data_B, lr_t, k + 1)
        if (k + 1) % 50 == 0:
            losses.append(float(problem.objective(cores)))
    return cores, {'losses': losses, 'n_iter': max_iter}


def _cg_solve(hvp, rhs, tol, maxiter, use_jit, inner):
    """Solve ``H p = rhs`` by conjugate gradients (``H = hvp``, symmetric PSD), to residual ``‖r‖ ≤ tol``.
    ``inner`` is the geometry's coordinate ``⟨·,·⟩`` (``geom.inner`` -- masked for the uniform layer, so
    padding is never summed). The body is **backend-agnostic and branch-free** (an ``xnp.where`` curvature
    guard: a nonpositive ``dᵀHd`` -- a gauge direction of the singular corewise ``H`` -- takes a zero step
    and the ``ok`` flag stops CG, i.e. truncated CG), so the SAME body runs eager (numpy/jax) or jit
    (``lax.while_loop``) through :py:func:`common.xwhile`.

    Returns the solution ``p`` plus the final loop state needed for diagnostics: the iteration count
    ``i``, the residual² ``rs = ‖H p − rhs‖² = ‖H p + g‖²`` (``rhs = −g``), and the positive-curvature
    flag ``ok`` (``False`` == the loop stopped on a truncation). The caller derives converged / truncated /
    maxiter from ``(rs, i, ok)``."""
    xnp, _, _ = get_backend(False, tree_contains_jax(rhs))
    tol2 = tol * tol
    rs0 = inner(rhs, rhs)
    state0 = (cw.corewise_zeros_like(rhs), rhs, rhs, rs0, 0, rs0 > tol2)   # (p, r, d, rs, i, ok)

    def cond(s):
        p, r, d, rs, i, ok = s
        return (rs > tol2) & (i < maxiter) & ok

    def body(s):
        p, r, d, rs, i, ok = s
        Hd = hvp(d)
        dHd = inner(d, Hd)
        pos = dHd > 0.0
        alpha = xnp.where(pos, rs / xnp.where(pos, dHd, 1.0), 0.0)        # 0 step on nonpositive curvature
        p = cw.corewise_add(p, cw.corewise_scale(d, alpha))
        r = cw.corewise_sub(r, cw.corewise_scale(Hd, alpha))
        rs_new = inner(r, r)
        beta = xnp.where(pos, rs_new / rs, 0.0)
        d = cw.corewise_add(r, cw.corewise_scale(d, beta))
        return (p, r, d, rs_new, i + 1, pos)

    p, r, d, rs, i, ok = xwhile(cond, body, state0, use_jit)
    return p, i, rs, ok


@dc.dataclass(frozen=True)
class NewtonInfo:
    """Per-iteration diagnostics passed to a ``newton_cg`` ``callback`` -- everything one Newton step
    produces, so a custom callback (or the display in :py:mod:`t3toolbox.backend.optimizer_display`) can
    report anything without the loop anticipating it. ``x_cores`` (the point *before* the step) and ``lm``
    (its :py:class:`LocalModel` -- residual / sample / frame) are carried so a callback can compute
    per-block errors or evaluate a validation forward. The step-related fields are ``None`` on the final
    **converged** line (no CG / line search ran). The scalar subset (all but ``x_cores`` / ``lm``) is what
    lands in ``stats['history']``."""
    iteration:    int              # Newton iteration index (0-based)
    objective:    float            # weighted ½‖ω⊙r‖² at x (= lm.objective)
    gnorm:        float            # ‖g‖ (Riemannian gradient norm, geom.inner)
    g0norm:       float            # ‖g₀‖ (first iterate's gradient norm; the gtol_rel reference)
    converged:    bool             # gnorm <= gtol_rel*g0norm -- this is the final line (step fields None)
    x_cores:      typ.Any = None                       # the point BEFORE the step (for a val forward)
    lm:           typ.Any = None                       # the LocalModel at x (residual/sample/frame)
    forcing_eta:  typ.Optional[float] = None           # inexact-Newton forcing term η
    cg_tol:       typ.Optional[float] = None           # CG stop tolerance = η·‖g‖
    cg_iters:     typ.Optional[int]   = None           # CG iterations run
    cg_resid:     typ.Optional[float] = None           # achieved ‖H p + g‖ (CG residual)
    cg_converged: typ.Optional[bool]  = None           # CG hit its tolerance (not maxiter / not truncated)
    cg_truncated: typ.Optional[bool]  = None           # CG stopped on nonpositive curvature (gauge-singular H)
    ls_steps:     typ.Optional[int]   = None           # Armijo backtracks (0 = full step; α = 2^-ls_steps)
    alpha:        typ.Optional[float] = None           # accepted step length
    slope:        typ.Optional[float] = None           # gᵀp (the directional derivative along the step)
    pHp:          typ.Optional[float] = None           # pᵀHp = ‖𝒥p‖² (for the predicted reduction)
    delta_f:      typ.Optional[float] = None           # actual objective change f_new − f
    rho:          typ.Optional[float] = None           # actual / predicted reduction (GN-model trust)
    step_rel:     typ.Optional[float] = None           # ‖αp‖ / ‖x‖ (coordinate norms; relative step size)
    wall_time:    typ.Optional[float] = None           # seconds spent in this iteration (host clock)


_NEWTON_SCALAR_FIELDS = tuple(f.name for f in dc.fields(NewtonInfo) if f.name not in ('x_cores', 'lm'))


def _newton_scalar_record(info: NewtonInfo) -> dict:
    """The scalar subset of a :py:class:`NewtonInfo` (drops ``x_cores`` / ``lm``) -- one ``history`` row."""
    return {k: getattr(info, k) for k in _NEWTON_SCALAR_FIELDS}


def newton_cg(
        problem:    Problem,    # the fixed-rank least-squares problem
        x0:         Tangent,    # initial cores (U, G)
        max_newton: int   = 30,
        gtol_rel:   float = 1e-8,   # stop when ‖g‖ <= gtol_rel * ‖g_0‖
        cg_maxiter: int   = 200,
        c_armijo:   float = 1e-4,
        use_jit:    bool  = False,  # jit the inner CG (lax.while_loop) when the inputs are jax; else eager
        callback:   typ.Optional[typ.Callable] = None,  # callback(NewtonInfo) each iteration (host-side; e.g. a display)
) -> typ.Tuple[Tangent, dict]:      # (x_cores, stats)
    """Inexact Riemannian Newton-CG with an Armijo line search -- the manifold workhorse (the gauged ``H``
    is positive-definite there). Each Newton step builds the local GN model once, solves ``H p = −g`` by
    CG to an inexact forcing-term tolerance (the inner loop -- jit-able via :py:func:`_cg_solve`), then
    backtracks along ``retract(α p)``. The CG truncates on the gauge-singular corewise ``H``; the outer
    line search keeps it robust regardless. ``use_jit`` jits only the inner CG (the outer loop, line
    search, and convergence test stay on the host).

    ``callback``, if given, is called with a :py:class:`NewtonInfo` each iteration (including the final
    converged line) -- the hook for a live diagnostic display; it runs **host-side** (it reads the concrete
    residual), so it composes with ``use_jit`` (only the inner CG jits) but not with a hypothetical
    fully-jitted outer loop. Ready-made displays: :py:func:`t3toolbox.backend.optimizer_display.make_newton_display`.
    ``stats`` always carries ``'history'`` -- one :py:func:`_newton_scalar_record` per iteration."""
    x = x0
    g0norm = None
    losses, newton_iters, history = [], 0, []
    t_prev = time.perf_counter()
    for it in range(max_newton):
        lm = problem.local_model(x)
        f = float(lm.objective)
        losses.append(f)
        g = lm.gradient
        gnorm = float(problem.geom.inner(g, g)) ** 0.5
        if g0norm is None:
            g0norm = gnorm if gnorm > 0 else 1.0
        if gnorm <= gtol_rel * g0norm:                                   # converged -- final line, no step
            info = NewtonInfo(iteration=it, objective=f, gnorm=gnorm, g0norm=g0norm, converged=True,
                              x_cores=x, lm=lm)
            if callback is not None:
                callback(info)
            history.append(_newton_scalar_record(info))
            break
        newton_iters += 1
        eta = min(0.5, (gnorm / g0norm) ** 0.5)                          # inexact-Newton forcing term
        cg_tol = eta * gnorm
        neg_g = cw.corewise_scale(g, -1.0)
        p, cg_i, cg_rs, cg_ok = _cg_solve(lm.hvp, neg_g, tol=cg_tol, maxiter=cg_maxiter,
                                          use_jit=use_jit, inner=problem.geom.inner)
        cg_iters, cg_rs, cg_ok = int(cg_i), float(cg_rs), bool(cg_ok)
        cg_converged = cg_rs <= cg_tol * cg_tol
        cg_truncated = (not cg_converged) and (not cg_ok)
        slope = float(problem.geom.inner(g, p))
        if (not math.isfinite(slope)) or slope >= 0.0:                   # ensure a descent direction
            p, slope = neg_g, -gnorm * gnorm
        pHp = float(lm.gn_quadratic(p))                                  # pᵀHp = ‖𝒥p‖² (for ρ)
        alpha, ls_steps, f_new = 1.0, 40, f
        x_trial = x
        for bt in range(40):                                             # Armijo backtracking
            x_trial = lm.retract(cw.corewise_scale(p, alpha))
            f_new = float(problem.objective(x_trial))
            if f_new <= f + c_armijo * alpha * slope:
                ls_steps = bt
                break
            alpha *= 0.5
        delta_f = f_new - f
        predicted = alpha * slope + 0.5 * alpha * alpha * pHp            # GN-model change along αp
        rho = (delta_f / predicted) if predicted != 0.0 else float('nan')
        p_norm = float(problem.geom.inner(p, p)) ** 0.5
        x_norm = float(cw.corewise_norm(x))                             # coordinate norm (ragged-exact; uniform: D6)
        step_rel = (alpha * p_norm / x_norm) if x_norm > 0 else float('nan')
        t_now = time.perf_counter()
        info = NewtonInfo(iteration=it, objective=f, gnorm=gnorm, g0norm=g0norm, converged=False,
                          x_cores=x, lm=lm, forcing_eta=eta, cg_tol=cg_tol, cg_iters=cg_iters,
                          cg_resid=cg_rs ** 0.5, cg_converged=cg_converged, cg_truncated=cg_truncated,
                          ls_steps=ls_steps, alpha=alpha, slope=slope, pHp=pHp, delta_f=delta_f,
                          rho=rho, step_rel=step_rel, wall_time=t_now - t_prev)
        t_prev = t_now
        if callback is not None:
            callback(info)
        history.append(_newton_scalar_record(info))
        x = x_trial
    return x, {'losses': losses, 'newton': newton_iters, 'history': history}
