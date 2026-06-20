# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Backend-first optimization algorithms for fixed-rank T3 least-squares fitting.

These operate on **raw cores / tangent tuples** via backend functions only -- no frontend imports
(no ``TuckerTensorTrain`` / ``T3Tangent`` / ``GaussNewtonModel`` / geometry classes). A raw-``.data``
user can call them directly; the frontend ``optimizers.py`` is a thin adapter that validates the input
once and assembles the ``Problem`` oracle from the same backend functions (design:
``docs/optimizers_plan.md``).

Because the numerical safety preconditions live only in the frontend, this layer is **check-free** of
them (structural shape guards inside the backend functions remain, and are jit-safe). So ``jit`` needs no
``unsafe()`` wrapping; the ``use_jit`` machinery (deferred) is a thin jax-only layer on top of the
ordinary numpy/jax dispatch.

The oracle:
  * ``Problem``      -- (geometry ops, sampling kind, sample, data); builds ``LocalModel``s and retracts.
  * ``LocalModel``   -- the GN model linearized at a point (``.gradient`` / ``.objective`` / ``.hvp`` /
                        ``.gn_quadratic`` / ``.retract``), the backend twin of ``fitting.GaussNewtonModel``.
  * ``GeometryOps``  -- (base, project, retract) on raw data; ``MANIFOLD`` / ``COREWISE`` singletons.
Tangent vectors are raw ``(tucker_var, tt_var)`` tuples; vector arithmetic is the ``corewise`` ops.
"""
import dataclasses as dc
import typing as typ

from t3toolbox.backend.common import *
from t3toolbox.backend import probing
from t3toolbox.backend import fitting as bfit
from t3toolbox.backend import apply as bapply
from t3toolbox.backend import entries as bentries
import t3toolbox.corewise as cw

__all__ = [
    'GeometryOps',
    'COREWISE',
    'Problem',
    'LocalModel',
    'least_squares_problem',
    'gradient_descent',
]

Tangent = typ.Tuple[typ.Sequence[NDArray], typ.Sequence[NDArray]]   # (tucker_variations, tt_variations)


# --------------------------------------------------------------------------------------------------
# Geometry ops on raw data (base / project / retract) -- the backend twin of the frontend geometries
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class GeometryOps:
    base:    typ.Callable    # x_cores=(U,G)            -> base=(U,O,P,Q)        the linearization frame
    project: typ.Callable    # (base, variations)       -> variations           gauge Π (identity for corewise)
    retract: typ.Callable    # (base, variations)       -> x_cores=(U,G)        chart retraction


def _corewise_base(
        x_cores: Tangent,    # (tucker_cores, tt_cores)
) -> typ.Tuple:              # (U, O, P, Q) = (U, G, G, G); the raw cores ARE the frame
    tucker_cores, tt_cores = x_cores
    return (tucker_cores, tt_cores, tt_cores, tt_cores)


COREWISE = GeometryOps(
    base=_corewise_base,
    project=lambda base, var: var,                                   # Euclidean cores: no gauge projection
    retract=lambda base, var: cw.corewise_add((base[0], base[2]), var),   # additive: (U,P)=(U,G) += var
)
# MANIFOLD geometry ops (orthogonal_representations base, gauge projection, T3-SVD retraction) -- next slice.


# --------------------------------------------------------------------------------------------------
# The problem oracle
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True)
class LocalModel:
    """The Gauss-Newton model linearized at a point -- the backend twin of ``fitting.GaussNewtonModel``.
    Built by ``Problem.local_model``; the base sweep is computed once and reused by every method below."""
    geom:     GeometryOps
    kind:     typ.Any        # backend fitting.SamplingKind (forward / transpose / sumsq / w_axes)
    sample:   typ.Any        # ww (probe/apply) or index (entries)
    base:     typ.Tuple      # (U, O, P, Q)
    sweep:    typ.Tuple      # precomputed base sweep (reused across gradient/jacobian/hvp)
    residual: typ.Any        # r = S(x) - data
    n_w:      int            # number of leading sample-stack axes

    @property
    def objective(self):                                 # c = ½‖r‖²
        return 0.5 * self.kind.sumsq(self.residual, self.n_w)

    @property
    def gradient(self) -> Tangent:                       # g = Π 𝒥ᵀ r
        return self.geom.project(self.base, self.kind.transpose(self.residual, self.sample, self.base, self.sweep))

    def jacobian(self, p: Tangent):                      # 𝒥 Π p
        return self.kind.forward(self.geom.project(self.base, p), self.sample, self.base, self.sweep)

    def gn_quadratic(self, p: Tangent):                  # ‖𝒥 Π p‖²  (one forward; the Cauchy denominator)
        return self.kind.sumsq(self.jacobian(p), self.n_w)

    def hvp(self, p: Tangent) -> Tangent:                # H p = Π 𝒥ᵀ 𝒥 Π p
        z = self.jacobian(p)
        return self.geom.project(self.base, self.kind.transpose(z, self.sample, self.base, self.sweep))

    def retract(self, p: Tangent) -> Tangent:            # chart step from this base -> new x_cores
        return self.geom.retract(self.base, p)


@dc.dataclass(frozen=True)
class Problem:
    """A fixed-rank least-squares fitting problem: ``min_x ½‖S(x) - data‖²`` for a sampling op ``S``,
    on the geometry ``geom``. ``local_model(x_cores)`` linearizes it at a point."""
    geom:          GeometryOps
    kind:          typ.Any       # backend fitting.SamplingKind
    sample:        typ.Any       # ww or index
    data:          typ.Any       # observed S(x_true) (+ noise); same structure as point_forward's output
    point_forward: typ.Callable  # (x_cores, sample) -> S(x)   (the POINT sampling, for the residual)

    def local_model(self, x_cores: Tangent) -> LocalModel:
        base = self.geom.base(x_cores)
        sweep = self.kind.precompute(base, self.sample)
        residual = cw.corewise_sub(self.point_forward(x_cores, self.sample), self.data)
        return LocalModel(self.geom, self.kind, self.sample, base, sweep, residual, self.kind.w_axes(self.sample))

    def objective(self, x_cores: Tangent):           # ½‖S(x) - data‖²  -- cheap (no sweep/transpose), for line search
        residual = cw.corewise_sub(self.point_forward(x_cores, self.sample), self.data)
        return 0.5 * self.kind.sumsq(residual, self.kind.w_axes(self.sample))


# the POINT sampling S(x) per kind (for the residual r = S(x) - data); the kind's own forward is the TANGENT 𝒥
_POINT_FORWARD = {
    'apply':   lambda x_cores, ww:    bapply.tucker_tensor_train_apply(x_cores, ww),
    'entries': lambda x_cores, index: bentries.tucker_tensor_train_entries(x_cores, index),
    'probe':   lambda x_cores, ww:    probing.probe_t3(ww, x_cores),
}


def least_squares_problem(
        geom:   GeometryOps,   # COREWISE / MANIFOLD
        kind:   typ.Any,       # backend fitting.APPLY / ENTRIES / PROBE
        sample: typ.Any,       # ww or index
        data:   typ.Any,       # observed values
) -> Problem:
    """Assemble a least-squares ``Problem`` from a geometry, a sampling kind, the sample vectors, and the
    observed data. (The frontend adapter calls this with the same backend objects a raw-data user would.)"""
    return Problem(geom, kind, sample, data, _POINT_FORWARD[kind.name])


# --------------------------------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------------------------------
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
        gg = float(cw.corewise_dot(g, g))                   # ‖g‖²
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
