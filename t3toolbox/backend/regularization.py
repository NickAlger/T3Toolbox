# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
"""Backend regularization terms for the fitting objective ``min_x ½‖ω⊙(S(x)−y)‖² + ρ(x)``.

A ``Regularizer`` is an additive objective term ``ρ(X)`` folded into the local Gauss-Newton model
(:py:class:`t3toolbox.backend.optimizers.LocalModel`) and ``Problem.objective``, so it composes with
**every** optimizer, sampling kind, geometry, and representation with no changes to any of them. This
layer is **check-free** -- a raw-``.data`` user constructs and attaches a regularizer directly, exactly as
a frontend user does (design record + the razor check: ``dev/regularization_design.md`` §5a).

The interface is **geometry-agnostic**: each method receives a ``GeometryOps`` and leans only on its
primitives (``point_norm_sq`` / ``point_tangent`` / ``project`` / ``inner``), so the SAME regularizer
works on manifold or corewise. Extending to a new regularizer (e.g. the Grasedyck-Kramer inverse-unfolding
-singular-value weighting) is a new subclass with the same four methods -- ``Problem`` / ``LocalModel`` /
the optimizers are untouched.
"""
import dataclasses as dc
import typing as typ

import t3toolbox.corewise as cw

__all__ = [
    'Regularizer',
    'IdentityRegularizer',
]


class Regularizer:
    """Protocol for a quadratic regularizer ``ρ(X)`` added to the fitting objective.

    A concrete subclass supplies the four contributions the local GN model needs. ``geom`` is a
    :py:class:`t3toolbox.backend.optimizers.GeometryOps`; ``frame`` = ``(U,O,P,Q)``; ``p`` a tangent
    ``(tucker_var, tt_var)``; ``x_cores`` = ``(tucker_cores, tt_cores)``. Tangent-valued methods return a
    gauged/projected tangent."""
    def value(self, geom, x_cores):        # ρ(X)          -- for Problem.objective / line search (any point)
        raise NotImplementedError
    def gradient(self, geom, frame):       # g_R = Π∇ρ     -- a tangent at frame
        raise NotImplementedError
    def hessian(self, geom, frame, p):     # H_R p         -- a tangent at frame
        raise NotImplementedError
    def quadratic(self, geom, frame, p):   # ⟨p, H_R p⟩    -- scalar, for gn_quadratic
        raise NotImplementedError


@dc.dataclass(frozen=True)
class IdentityRegularizer(Regularizer):
    """Identity (Tikhonov) regularization ``ρ(X) = ½·strength·‖X‖²`` in the geometry's own tangent metric.

    On the **manifold** this is the Hilbert-Schmidt ridge ``½λ‖X‖²_HS`` -- ``H_R = λ·Π`` (``= λp`` on a
    gauged tangent), which is the **exact** Riemannian Hessian (``∇ρ = X`` is fully tangent, so the
    curvature term vanishes). On **corewise** it is core weight-decay ``½λ Σ‖core_i‖²``, which also makes
    the gauge-singular corewise Gauss-Newton Hessian strictly positive-definite -- a better-conditioned
    Newton system (though CG already converges on the singular ``H`` since the gradient ``g = 𝒥ᵀr`` lies in
    ``range(H)``). ``X_ref = 0``. The ``point_tangent`` = ``v_X`` construction (the attachment point as a
    single gauged tangent term) is in ``dev/regularization_design.md`` §4."""
    strength: float                        # λ >= 0  (0 disables the regularizer)

    def value(self, geom, x_cores):
        return 0.5 * self.strength * geom.point_norm_sq(x_cores)

    def gradient(self, geom, frame):       # λ·Π(X) = λ·v_X  (point_tangent already returns a gauged tangent)
        return cw.corewise_scale(geom.point_tangent(frame), self.strength)

    def hessian(self, geom, frame, p):     # λ·Π p
        return cw.corewise_scale(geom.project(frame, p), self.strength)

    def quadratic(self, geom, frame, p):   # ⟨p, λ·Π p⟩ = λ‖Π p‖²
        projected = geom.project(frame, p)
        return self.strength * geom.inner(projected, projected)


@dc.dataclass(frozen=True)
class _ScaledRegularizer(Regularizer):
    """``inner`` scaled by a constant ``factor`` -- ``factor·ρ``. Used by the **stochastic** optimizers,
    where the minibatch data gradient is a ``batch/n`` estimate of the full data gradient, so the
    (deterministic) regularizer is scaled by ``batch/n`` to keep ``λ``'s meaning consistent with the
    full-batch optimizers (``dev/regularization_design.md`` §8.1). Valid for any regularizer: scaling a
    function by ``factor`` scales its value, gradient, and Hessian all by ``factor``."""
    inner:  Regularizer
    factor: float

    def value(self, geom, x_cores):
        return self.factor * self.inner.value(geom, x_cores)

    def gradient(self, geom, frame):
        return cw.corewise_scale(self.inner.gradient(geom, frame), self.factor)

    def hessian(self, geom, frame, p):
        return cw.corewise_scale(self.inner.hessian(geom, frame, p), self.factor)

    def quadratic(self, geom, frame, p):
        return self.factor * self.inner.quadratic(geom, frame, p)
