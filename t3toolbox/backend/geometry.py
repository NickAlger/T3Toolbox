# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Backend geometries on raw data -- the *where you optimize* axis of a fit.

A geometry bundles the chart-level choices an optimizer needs and nothing else: the linearization
``frame``, the gauge projection ``project`` (``Pi``), the ``retract``, the coordinate ``inner``, and the
optional per-frame ``precompute`` companion. The frontend twins are
:py:class:`t3toolbox.manifold.ManifoldGeometry` / ``CorewiseGeometry`` and their uniform counterparts;
these are the check-free versions a raw-``.data`` user calls directly.

**Parameters are fields, not closures.** Each geometry is a frozen dataclass whose fields *are* its
defining data -- the mode ``shape``, the rank ``masks`` it is fixed at, and the sharing ``groups``. This
matters for more than tidiness: a geometry rides as jax ``aux_data``, so its ``__hash__``/``__eq__`` are
part of the jit compilation cache key. With the parameters sealed in closure cells (the previous
``GeometryOps`` record-of-lambdas) a rebuilt-but-identical geometry was always a *new* key, so every
rank-continuation level and every rebuilt model recompiled. Value-based hash/eq over the fields
(:py:class:`~t3toolbox.backend.common.ValueHashedFields`) makes the cache key reflect the geometry's
actual identity. Mathematically it is also the more faithful encoding: a uniform manifold *at a given
rank* is a different manifold from one at another rank, so the rank belongs in the object's data.

**The math stays reachable.** Per the backend rule, a method here only binds parameters and names a
role -- every line of actual math is also a standalone function, either in the operation modules
(:py:mod:`~t3toolbox.backend.tv_operations`, :py:mod:`~t3toolbox.backend.utv_operations`,
:py:mod:`~t3toolbox.backend.sharing`) or named in this module
(:py:func:`fv_base_point_tangent`, :py:func:`ufv_base_point_tangent`,
:py:func:`t3_left_orthogonal_norm_sq`, :py:func:`t3_alias_tied_tucker_factors`).

**Sharing is a field, not a wrapper.** ``groups=()`` is the ordinary geometry; a non-empty partition
restricts every projection to the TIED tangent subspace and keeps the retraction on the shared set
(SF-T3; ``docs/sharing.md``). One class, one code path. Build the canonical partition with
:py:func:`~t3toolbox.backend.sharing.validate_sharing`, or use the ``from_point`` /
``with_sharing`` constructors here.

**The exact Hessian's members** (``hessian='exact'`` only; a Gauss-Newton fit never calls them).
``curvature_precompute`` builds the per-frame curvature data once per local model -- on the manifold the
lift factors, ``grad f`` as a doubled-rank T3 and its inner-product sweep
(:py:class:`ManifoldCurvatureData`); ``lift`` / ``lift_transpose`` map coordinates to raw core variations
and back (:py:func:`~t3toolbox.backend.tv_operations.tv_corewise_lift`); ``tangential_curvature`` is the
``K_{grad f}`` term (:py:func:`~t3toolbox.backend.t3_linalg.t3_inner_product_corewise_hessian_from_sweep`).
The corewise geometries need neither a lift nor a tangential term. The lift's damping is a solver option
(``curvature_damping``), not a geometry field: ``None`` resolves to :py:data:`DEFAULT_CURVATURE_DAMPING`
here, and a value where there is no lift to damp raises.
"""
import dataclasses as dc
import typing as typ

import numpy as np

from t3toolbox.backend.common import *
from t3toolbox.backend import tv_operations as tops
from t3toolbox.backend import utv_operations as utv_ops
from t3toolbox.backend import ufv_conversions
from t3toolbox.backend import fv_operations
from t3toolbox.backend import ufv_operations
from t3toolbox.backend import ufv_masking
from t3toolbox.backend import sharing as sharing_module
from t3toolbox.backend import t3_linalg
from t3toolbox.backend import t3_orthogonalization as ragged_orth
from t3toolbox.backend import tt_operations
from t3toolbox.backend.fv_conversions import t3_orthogonal_representations, t3_corewise_frame
import t3toolbox.corewise as cw

__all__ = [
    'ManifoldGeometryOps',
    'CorewiseGeometryOps',
    'UniformManifoldGeometryOps',
    'UniformCorewiseGeometryOps',
    'ManifoldCurvatureData',
    'DEFAULT_CURVATURE_DAMPING',
    'canonical_groups',
    'fv_base_point_tangent',
    'ufv_base_point_tangent',
    't3_left_orthogonal_norm_sq',
    't3_alias_tied_tucker_factors',
]


canonical_groups = sharing_module.canonical_groups   # re-exported: the geometry's `groups` field is exactly this

DEFAULT_CURVATURE_DAMPING = 1e-6    # the exact Hessian's lift damping, RELATIVE to ||X||_HS (curvature_damping=None)


# --------------------------------------------------------------------------------------------------
# The math the geometries bind (named, so a raw-.data user can find and reuse it)
# --------------------------------------------------------------------------------------------------
def t3_left_orthogonal_norm_sq(
        x_cores:  typ.Tuple,   # (tucker_cores, tt_cores) -- a LEFT-orthogonal T3 (a frame's (U,P), or a retracted point)
) -> NDArray:                  # ‖X‖²_HS, shape = stack C (a 0-d scalar for C=()); PER-ELEMENT over the stack
    """``‖X‖²_HS = ‖last TT core‖²`` -- exact for a left-orthogonal T3 (the frame's ``(U,P)`` or a
    ``t3svd`` retraction output), so no dense tensor and no re-orthogonalization. The left-orthogonal
    precondition is check-free here (backend) and is the caller's responsibility: on a raw point the value
    is simply wrong (measured 3 vs 1400 on a ``randn`` point -- the 2026-08-22 review). Use
    :py:meth:`ManifoldGeometryOps.point_norm_sq` for an arbitrary point; verify a point with
    :py:func:`t3toolbox.backend.t3_orthogonalization.t3_orthogonality_residual`.
    (``docs/contributor/fitting_internals.md`` §"The base point as a tangent".)"""
    last = x_cores[1][-1]                                    # stack + (rL, n, rR)
    xnp, _, _ = get_backend(False, tree_contains_jax(x_cores))
    return xnp.sum(last * last, axis=(-3, -2, -1))           # per stack element (review H3-5)


def fv_base_point_tangent(
        frame:  typ.Tuple,    # (U, O, P, Q) -- an orthogonal frame
) -> typ.Tuple:               # v_X: the attachment point X as a gauged tangent (tucker_var, tt_var)
    """The attachment point ``X = (U, P)`` as a gauged tangent ``v_X`` -- **the DIRECT construction**: all
    variations zero except the last TT variation, set to the frame's last left core ``P_last``. This is
    already gauged (the last TT variation is the one slot with no gauge condition), so it needs neither an
    ambient projection nor a gauge projection: ``dense(v_X) = X`` and ``‖v_X‖_coord = ‖X‖_HS`` exactly.
    (It equals ``tv_project_t3_onto_tangent_space(frame, (U, P))`` -- verified -- but avoids that roundabout
    computation, whose environment contractions all collapse since the projected T3 IS the frame's own
    cores. See ``docs/contributor/fitting_internals.md`` §Regularization.)"""
    up, _down, left, _right = frame
    stack = up[0].shape[:-2]                                         # the frame stack C
    tucker_var, tt_var = fv_operations.fv_variations_zeros(
        fv_operations.fv_variation_shapes(frame), stack, tree_contains_jax(frame))
    return (tucker_var, tt_var[:-1] + (left[-1],))                   # last TT variation = P_last


def ufv_base_point_tangent(
        frame_data:  typ.Tuple,   # uniform frame .data = (up_sc, down_sc, left_sc, right_sc, shape, masks)
) -> typ.Tuple:                   # v_X as a bare variation supercore pair (tucker_var_sc, tt_var_sc)
    """The uniform twin of :py:func:`fv_base_point_tangent`: the attachment point as a gauged tangent,
    all variation slots zero except the last TT slice, set to the frame's last left core ``P_last``.

    The variation supercore shapes come from
    :py:func:`~t3toolbox.backend.ufv_operations.ufv_variation_shapes`, which is where the frame axis
    convention is written down. Do not re-derive them here: the tempting shortcut ("the Tucker
    variation has the ``up`` supercore's shape") holds only when ``nD == nU``, which is true of most
    unshared ranks and false as soon as a sharing group's rank differs from the down rank."""
    left_sc = frame_data[2]
    xnp, _, _ = get_backend(True, tree_contains_jax(frame_data))
    tucker_var_shape, tt_var_shape = ufv_operations.ufv_variation_shapes(frame_data)
    d = tucker_var_shape[0]
    tucker_var_sc = xnp.zeros(tucker_var_shape)                        # all zero
    tt_var_sc = xnp.concatenate([xnp.zeros((d - 1,) + tt_var_shape[1:]),
                                 left_sc[-1:]], axis=0)                # zero except last = P_last
    return (tucker_var_sc, tt_var_sc)


def t3_alias_tied_tucker_factors(
        tucker_cores:  typ.Sequence[NDArray],            # len=d; tied within each group by VALUE
        groups:        typ.Tuple[typ.Tuple[int, ...], ...],  # canonical partition (canonical_groups)
) -> typ.Tuple[NDArray, ...]:                            # len=d; ONE array object shared per group
    """Collapse each sharing group's Tucker factors to a **single array object**.

    The corewise additive retraction preserves tying by value (equal inputs plus equal updates give
    equal outputs), but leaves each group member holding its own equal-valued copy. Aliasing them to one
    object makes the tying structural rather than incidental -- so a downstream consumer that checks
    identity, or a `t3svd(sharing=...)`, sees one factor per group as the shared format intends."""
    out = list(tucker_cores)
    for group in sharing_module.nontrivial_groups(groups):
        for member in group[1:]:
            out[member] = out[group[0]]
    return tuple(out)


# --------------------------------------------------------------------------------------------------
# The exact Hessian's per-frame curvature data (hessian='exact')
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True, eq=False)   # eq=False -> identity hash/eq: array fields (the LiftFactors precedent)
class ManifoldCurvatureData:
    """The per-frame data of the exact Riemannian Hessian's curvature term on the manifold geometry, built
    by :py:meth:`ManifoldGeometryOps.curvature_precompute` once per local model (beside the sampling sweep).

    As a jax pytree every array is a leaf (the treedef's aux is the value-hashed static skeleton), so a new
    frame, a new gradient or a new damping changes leaf values only -- never the compilation key."""
    lift_factors:  fv_operations.LiftFactors   # the frame's damped lift factors (fv_lift_factors); eps a leaf
    z:             typ.Tuple                   # grad f as a doubled-rank T3: tv_to_t3(frame, misfit_gradient)
    z_sweep:       typ.Tuple                   # t3_precompute_inner_product_sweep(z, base_point)
    sigma_min:     NDArray                     # the smallest lift singular value (Tucker, bond; s_g if shared), shape=C

    @property
    def eps(self) -> NDArray:                  # the absolute lift damping used, shape=C (a diagnostic; newton_cg's lift_eps)
        return self.lift_factors.eps


def _lift_sigma_min(
        lift_factors:  fv_operations.LiftFactors,   # the lift's factors, stack C (tied at a shared point)
) -> NDArray:                                       # shape=C: the smallest singular value the lift inverts
    """The smallest singular value the lift inverts, per frame element: every bond spectrum ``sigma(T_i)``, the
    Tucker spectrum ``sigma(S_i)`` of every mode the lift does not tie, and -- at a shared point -- each
    nontrivial group's spectrum ``s_g`` in place of its modes' own (the tied lift inverts the stacked ``M_g``,
    not a group mode's ``S_i^T``, which can be rank-deficient on a regular shared point)."""
    spectra = list(lift_factors.bond_svals)
    tied = ()
    if lift_factors.shared_data is not None:
        tied = sharing_module.nontrivial_groups(lift_factors.shared_data.groups)
        spectra += list(lift_factors.shared_data.svd_s)
    tied_modes = set(ii for group in tied for ii in group)
    spectra += [s for ii, s in enumerate(lift_factors.tucker_svals) if ii not in tied_modes]
    xnp, _, _ = get_backend(False, tree_contains_jax(spectra))
    sigma_min = xnp.min(spectra[0], axis=-1)
    for s in spectra[1:]:
        sigma_min = xnp.minimum(sigma_min, xnp.min(s, axis=-1))
    return sigma_min


def _manifold_curvature_data(
        frame:              typ.Tuple,   # (U, O, P, Q) -- an orthogonal frame
        misfit_gradient:    typ.Tuple,   # gauged coordinates of grad f (the misfit's Riemannian gradient), stack C
        curvature_damping:  float,       # RELATIVE: eps = curvature_damping * ||X||_HS, >= 0
        shared_data:        typ.Optional[sharing_module.SharedFrameData] = None,   # shared: the frame's companion
) -> ManifoldCurvatureData:
    """:py:meth:`ManifoldGeometryOps.curvature_precompute`'s body, without its ``d = 1`` shortcut (at
    ``d = 1`` the normal space is trivial and the term this data feeds is exactly zero). With ``shared_data``
    the lift is the TIED one (:py:func:`~t3toolbox.backend.sharing.fv_tied_lift_factors`) and
    ``misfit_gradient`` is the shared (tied) gradient, so ``z`` is the shared ``grad f`` and the normal part
    ``A^T r - grad f`` is the shared manifold's."""
    if shared_data is None:
        lift_factors = fv_operations.fv_lift_factors(frame, curvature_damping)
    else:
        lift_factors = sharing_module.fv_tied_lift_factors(frame, curvature_damping, shared_data)
    z = tops.tv_to_t3(frame, misfit_gradient)
    z_sweep = t3_linalg.t3_precompute_inner_product_sweep(z, (frame[0], frame[2]))
    return ManifoldCurvatureData(lift_factors, z, z_sweep, _lift_sigma_min(lift_factors))


# --------------------------------------------------------------------------------------------------
# Ragged geometries
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True, eq=False)   # eq=False + ValueHashedFields: value hash/eq, see the module docstring
class ManifoldGeometryOps(ValueHashedFields):
    """The fixed-rank T3 manifold geometry on raw ``(tucker_cores, tt_cores)`` data -- the check-free twin
    of :py:data:`t3toolbox.manifold.MANIFOLD`.

    ``frame`` is the orthonormal frame (Algorithm 11), ``project`` the gauge projection ``Pi``, ``retract``
    the implicit truncated T3-SVD retraction, and ``inner`` the ragged coordinate dot (equal to
    Hilbert-Schmidt on an orthonormal, gauged frame). With a non-empty ``groups`` this is the SF-T3
    geometry: ``precompute`` derives the per-frame companion
    (:py:func:`~t3toolbox.backend.sharing.fv_shared_frame_data`, built once per local model),
    ``project`` composes the gauge projection with the tied post-pass, and ``retract`` builds the tied
    doubled-rank embedding and truncates with the grouped ``t3svd``. ``inner`` / ``point_norm_sq`` /
    ``point_tangent`` are unaffected by tying (the tied subspace is linear, so the restriction of the
    metric is itself, and the base-point tangent has zero Tucker variations, hence is trivially tied)."""

    groups:  typ.Tuple[typ.Tuple[int, ...], ...] = ()   # sharing partition; () = unshared

    def with_sharing(self, sharing, shape) -> 'ManifoldGeometryOps':
        """This geometry restricted to tied Tucker factors (``sharing=None`` gives it back unshared)."""
        return dc.replace(self, groups=canonical_groups(sharing, shape))

    def frame(
            self,
            x_cores:  typ.Tuple,   # (tucker_cores, tt_cores)
    ) -> typ.Tuple:                # (U, O, P, Q) orthonormal frame (Algorithm 11)
        """The linearization frame at ``x_cores``."""
        frame, _ = t3_orthogonal_representations(x_cores)
        return frame

    def stack_shape(
            self,
            x_cores:  typ.Tuple,   # (tucker_cores, tt_cores)
    ) -> typ.Tuple[int, ...]:      # C -- the frame/core stack (empty for a single tensor)
        """The point's frame stack ``C``. Which axes are the stack is a layout question, so it belongs
        to the geometry rather than to the caller (as :py:meth:`base_point` does)."""
        return tuple(x_cores[0][0].shape[:-2])

    def base_point(
            self,
            frame:  typ.Tuple,   # (U, O, P, Q)
    ) -> typ.Tuple:              # (tucker_cores, tt_cores) = (U, P) -- the tensor the frame is attached to
        """The point ``X = (U, P)`` the frame is attached to. What a frame *is* belongs to the geometry,
        so consumers (the regularizer, diagnostics) ask for the point rather than indexing the tuple."""
        return (frame[0], frame[2])

    def precompute(
            self,
            frame:  typ.Tuple,   # (U, O, P, Q)
    ) -> typ.Any:                # the per-frame geometry companion, or None when there is none
        """The per-frame companion ``project`` / ``retract`` reuse -- the SF-T3 shared-frame data when
        tied, ``None`` otherwise. Computed ONCE per local model (the ``sweep`` pattern) rather than once
        per matvec; see ``docs/contributor/precompute_and_caching.md``."""
        if not self.groups:
            return None
        return sharing_module.fv_shared_frame_data(frame, self.groups)

    def project(self, frame, variations, aux=None):
        """The gauge projection ``Pi`` (plus the tied post-pass when shared). ``aux`` is this frame's
        :py:meth:`precompute` companion; ``None`` recomputes it."""
        if not self.groups:
            return tops.tv_orthogonal_gauge_projection(frame, variations)
        return tops.tv_orthogonal_gauge_projection(
            frame, variations, shared_data=aux if aux is not None else self.precompute(frame))

    def retract(self, frame, variations, aux=None):
        """The manifold retraction: shift the doubled-rank embedding and truncate back to the frame
        ranks (the grouped ``t3svd`` when shared)."""
        if not self.groups:
            return tops.tv_retract(frame, variations)
        return tops.tv_retract(
            frame, variations, shared_data=aux if aux is not None else self.precompute(frame))

    def inner(self, a, b):
        """The coordinate ``<.,.>`` on tangents (Hilbert-Schmidt on an orthonormal, gauged frame),
        **per-element over the leading stacks** (``K+C``) -- shape = stack, a 0-d scalar unstacked,
        matching the uniform twin and the frontend ``MANIFOLD.inner`` (review H3-5)."""
        return cw.corewise_stack_dot(a, b, a[0][0].ndim - 2)     # Tucker variation core = stack+(nD, N)

    def point_norm_sq(self, x_cores):
        """``‖X‖²_HS`` -- the regularizer's objective term; **per-element over the stack ``C``**
        (shape = ``C``, a 0-d scalar unstacked). Exact for ANY point: left-orthogonalizes first
        (Tucker up-orth + one left TT sweep, no ``W`` factor -- negligible beside the misfit evaluation it
        always sits next to), then reads ``‖last TT core‖²``. The uniform twin does the same. The check-free
        fast path for a point known to be left-orthogonal is :py:func:`t3_left_orthogonal_norm_sq`."""
        tucker, tt = x_cores
        return t3_left_orthogonal_norm_sq(
            ragged_orth.t3_left_orthogonalize((tucker, tt_operations.tt_squash_tails(tt))))

    def point_tangent(self, frame):
        """The attachment point as a gauged tangent ``v_X`` -- the regularizer's gradient direction."""
        return fv_base_point_tangent(frame)

    def zero_point_frame(
            self,
            frame:  typ.Tuple,   # (U, O, P, Q) -- a frame of this geometry: the STRUCTURE to copy
    ) -> typ.Tuple:              # (U, O, P, Q) -- the canonical orthonormal frame at the zero tensor
        """The canonical frame at the zero tensor with ``frame``'s ranks -- where the default Newton
        reference ``‖g(0)‖`` is evaluated (``newton_cg(g0_reference='zero')``). One seeded Haar Tucker
        factor per sharing group: :py:func:`~t3toolbox.backend.fv_operations.fv_zero_point_frame`."""
        return fv_operations.fv_zero_point_frame(frame, self.groups)

    # ---- the exact Hessian (hessian='exact' only) -------------------------------------------------
    def curvature_precompute(
            self,
            frame:              typ.Tuple,                   # (U, O, P, Q)
            misfit_gradient:    typ.Tuple,                   # Pi J^T omega^2 r: the model's ONE misfit transpose
            aux:                typ.Any,                     # self.precompute(frame)
            curvature_damping:  typ.Optional[float],         # RELATIVE to ||X||_HS; None -> DEFAULT_CURVATURE_DAMPING
    ) -> typ.Optional[ManifoldCurvatureData]:                # None iff the curvature term vanishes here (d = 1)
        """The per-frame curvature data of the exact Riemannian Hessian
        ``H p = Pi J^T omega^2 J Pi p + Pi L^T (K_{A^T omega^2 r} - K_{grad f}) L Pi p``: the lift ``L``'s
        damped factors at ``eps = curvature_damping * ||X||_HS``
        (:py:func:`~t3toolbox.backend.fv_operations.fv_lift_factors`), the misfit gradient as a doubled-rank
        T3 ``z`` and its inner-product sweep at the base point (the tangential term ``K_{grad f}``), and the
        smallest lift singular value (a diagnostic).

        ``None`` at ``d = 1``: there every slot is at maximal rank, the representable set is open in ``R^N``
        (flat), and the curvature term is exactly zero, so the work is skipped -- a cost shortcut; the formula
        needs no special case and gives zero there numerically.

        **Shared** (``groups``): the shared set is parametrized as ``Phi(T theta_sh)``, ``T`` copying each
        group's factor, so its Hessian is the same formula through the TIED lift -- one factor direction per
        group, ``Udot_g = M_g^{+eps} [V_i1; ...]`` against ``aux``'s stacked-``S`` SVD
        (:py:func:`~t3toolbox.backend.sharing.fv_tied_lift_factors`), its adjoint summing the group's copies --
        with ``misfit_gradient`` the TIED gradient (what :py:meth:`project` with ``aux`` returns), so the
        normal part is the shared manifold's; ``hvp``'s outer projection ties the result. The smallest lift
        singular value then reads each group's spectrum ``s_g`` in place of its modes' own. A zero-padded
        restart's new ``s_g`` levels filter to exactly 0, so the gated tied channel gets no curvature."""
        cd = DEFAULT_CURVATURE_DAMPING if curvature_damping is None else float(curvature_damping)
        if not cd >= 0.0:
            raise ValueError('curvature_damping must be >= 0 (it is relative to ||X||_HS); got %r'
                             % (curvature_damping,))
        if len(frame[0]) == 1:
            return None
        if not self.groups:
            return _manifold_curvature_data(frame, misfit_gradient, cd)
        return _manifold_curvature_data(frame, misfit_gradient, cd,
                                        shared_data=aux if aux is not None else self.precompute(frame))

    def lift(self, frame, variations, curvature_data):
        """Gauged (or not) coordinates -> raw core variations of the base point ``(U, P)``, ``theta_hat = L v``
        (:py:func:`~t3toolbox.backend.tv_operations.tv_corewise_lift`; tied -- one factor direction per sharing
        group -- on a shared geometry); stack ``K + C``."""
        return tops.tv_corewise_lift(frame, variations, curvature_data.lift_factors)

    def lift_transpose(self, frame, core_variations, curvature_data):
        """Raw core variations -> UNGAUGED coordinates, ``L^T y`` (the exact adjoint of :py:meth:`lift`)."""
        return tops.tv_corewise_lift_transpose(frame, core_variations, curvature_data.lift_factors)

    def tangential_curvature(self, frame, core_directions, curvature_data):
        """``K_{grad f} theta_hat``: the core-coordinate Hessian of ``theta -> <grad f, Phi(theta)>`` along the
        directions, at the base point -- the term that turns the ambient curvature into its normal part
        (:py:func:`~t3toolbox.backend.t3_linalg.t3_inner_product_corewise_hessian_from_sweep`)."""
        return t3_linalg.t3_inner_product_corewise_hessian_from_sweep(
            curvature_data.z, self.base_point(frame), core_directions, curvature_data.z_sweep)


@dc.dataclass(frozen=True, eq=False)
class CorewiseGeometryOps(ValueHashedFields):
    """The core-parameter Euclidean geometry on raw ``(tucker_cores, tt_cores)`` data -- the check-free
    twin of :py:data:`t3toolbox.manifold.COREWISE`.

    The raw cores ARE the frame (the Section 6.3 substitution ``(P, Q, O) -> G``), there is no gauge, and
    the retraction is additive (``cores += variations``). With a non-empty ``groups`` the projection is
    the per-group arithmetic mean (the corewise coordinates are raw factor copies, so that IS the
    orthogonal projection onto the tied subspace) and the additive retraction preserves tying exactly, so
    ``retract`` only mean-ties its input first -- a bitwise no-op on already-tied input.

    Secondary to the manifold geometry: its Gauss-Newton Hessian is gauge-singular, which first-order and
    quasi-Newton methods tolerate but Newton must truncate around."""

    groups:  typ.Tuple[typ.Tuple[int, ...], ...] = ()   # sharing partition; () = unshared

    def with_sharing(self, sharing, shape) -> 'CorewiseGeometryOps':
        """This geometry restricted to tied Tucker factors (``sharing=None`` gives it back unshared)."""
        return dc.replace(self, groups=canonical_groups(sharing, shape))

    def frame(
            self,
            x_cores:  typ.Tuple,   # (tucker_cores, tt_cores)
    ) -> typ.Tuple:                # (U, O, P, Q) = (U, G, G, G) -- the raw cores ARE the frame
        """The (non-orthonormal) corewise frame: the Section 6.3 substitution ``(P, Q, O) -> G``."""
        return t3_corewise_frame(x_cores)

    def stack_shape(
            self,
            x_cores:  typ.Tuple,   # (tucker_cores, tt_cores)
    ) -> typ.Tuple[int, ...]:      # C -- the frame/core stack (empty for a single tensor)
        """The point's frame stack ``C``. Which axes are the stack is a layout question, so it belongs
        to the geometry rather than to the caller (as :py:meth:`base_point` does)."""
        return tuple(x_cores[0][0].shape[:-2])

    def base_point(self, frame):
        """The point ``X = (U, G)`` the frame is attached to (see :py:meth:`ManifoldGeometryOps.base_point`)."""
        return (frame[0], frame[2])

    def precompute(self, frame):
        """No per-frame companion on this geometry -- the tied mean needs only the static partition."""
        return None

    def project(self, frame, variations, aux=None):
        """The gauge projection: the identity (Euclidean core space), or the per-group mean when tied."""
        if not self.groups:
            return variations
        return sharing_module.fv_share_tucker_variations_corewise(variations, self.groups)

    def retract(self, frame, variations, aux=None):
        """The additive retraction ``(U, G) += variations`` (mean-tied first when shared, which keeps
        tied-in giving tied-out; the group's factors come back as one aliased array)."""
        new = cw.corewise_add(self.base_point(frame), self.project(frame, variations))
        if not self.groups:
            return new
        return (t3_alias_tied_tucker_factors(new[0], self.groups), new[1])

    def inner(self, a, b):
        """The Euclidean coordinate ``<.,.>`` -- **per-element over the leading stacks** (shape =
        stack, a 0-d scalar unstacked), matching the uniform twin (review H3-5)."""
        return cw.corewise_stack_dot(a, b, a[0][0].ndim - 2)     # Tucker core = stack+(n, N)

    def point_norm_sq(self, x_cores):
        """``Σ‖core_i‖²`` -- weight decay on the raw cores, **per-element over the stack ``C``**
        (shape = ``C``, a 0-d scalar unstacked)."""
        return cw.corewise_stack_dot(x_cores, x_cores, x_cores[0][0].ndim - 2)

    def point_tangent(self, frame):
        """The cores ``(U, G)`` as a tangent (the projection is the identity here; ``X_ref = 0``)."""
        return self.base_point(frame)

    def zero_point_frame(self, frame) -> None:
        """``None``: this geometry has no zero-point reference. The objective is multilinear in the cores,
        so the zero cores are a critical point of every corewise objective and the gradient there is exactly
        zero -- the default Newton reference on a corewise geometry is therefore the warm start."""
        return None

    # ---- the exact Hessian (hessian='exact' only) -------------------------------------------------
    def curvature_precompute(self, frame, misfit_gradient, aux, curvature_damping):
        """``()``: the corewise exact Hessian is ``J^T omega^2 J + K_{A^T omega^2 r}`` (the full Hessian of
        ``f o Phi``), with no lift and no tangential term, so there is no per-frame data -- and nothing for a
        ``curvature_damping`` to damp (a non-``None`` value raises, rather than being silently ignored)."""
        if curvature_damping is not None:
            raise ValueError("curvature_damping has no effect on a corewise geometry (there is no lift to damp)")
        return ()

    def lift(self, frame, variations, curvature_data):
        """The identity: the corewise coordinates ARE raw core variations."""
        return variations

    def lift_transpose(self, frame, core_variations, curvature_data):
        """The identity (see :py:meth:`lift`)."""
        return core_variations

    def tangential_curvature(self, frame, core_directions, curvature_data) -> None:
        """``None``: no tangential term (the corewise Hessian is the plain Hessian of ``f o Phi``)."""
        return None


# --------------------------------------------------------------------------------------------------
# Uniform geometries -- the same two, at a FIXED rank whose masks are part of the geometry's data
# --------------------------------------------------------------------------------------------------
@dc.dataclass(frozen=True, eq=False)
class UniformManifoldGeometryOps(ValueHashedFields):
    """The uniform **manifold** geometry at a fixed rank -- the raw-supercore twin of
    :py:data:`t3toolbox.uniform_manifold.UNIFORM_MANIFOLD`.

    The uniform layer's optimizer state is a bare ``(tucker_supercore, tt_supercore)`` pair, so the rank
    structure the operations need -- the mode ``shape``, the plain-UT3 ``masks``, and the variation
    ``var_masks`` -- lives here instead, loop-invariant and value-hashed
    (``docs/uniform_backend_jit_recipe.md``: hold the masks as state, trace only the supercores). Build
    it from the point you are about to optimize with :py:meth:`from_point`.

    The uniform layer requires a **minimal-rank** start (:py:func:`~t3toolbox.backend.uniform_fitting.uniform_minimal`):
    from a non-minimal frame the retraction truncates to the realizable rank, which no longer matches the
    fixed masks held here."""

    shape:      typ.Tuple[int, ...]   # the mode sizes
    masks:      typ.Tuple             # plain-UT3 rank masks (tucker_edge_mask, tt_edge_mask); HOST numpy
    var_masks:  typ.Tuple             # variation masks (up, down, frame_left[:-1], frame_right[1:]); HOST numpy
    groups:     typ.Tuple[typ.Tuple[int, ...], ...] = ()   # sharing partition; () = unshared

    @classmethod
    def from_point(
            cls,
            x0_data:  typ.Tuple,                              # UniformTuckerTensorTrain.data = (tk_sc, tt_sc, shape, masks)
            sharing:  typ.Optional[typ.Sequence] = None,      # len=d group labels; None = unshared
    ) -> 'UniformManifoldGeometryOps':
        """The geometry at ``x0``'s fixed rank. The variation masks come from ``x0``'s orthogonal
        representation (:py:func:`~t3toolbox.backend.ufv_conversions.ut3_orthogonal_representations`),
        which is where the SVD-prefix mask contract is established -- do not build them by hand
        (``docs/contributor/uniform_svd_prefix_orthogonalization.md``)."""
        _tk_sc, _tt_sc, shape, base_masks = x0_data
        _frame, variation_data = ufv_conversions.ut3_orthogonal_representations(x0_data)
        return cls(tuple(shape), readonly_mask_copies(base_masks), readonly_mask_copies(variation_data[3]),
                   canonical_groups(sharing, tuple(shape)))

    @classmethod
    def from_frame(
            cls,
            frame_data:   typ.Tuple,                          # uniform frame .data = (U, O, P, Q, shape, masks)
            plain_masks:  typ.Tuple,                          # the point's plain-UT3 (tucker, tt) masks; HOST bool
            sharing:      typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
    ) -> 'UniformManifoldGeometryOps':
        """The geometry at the fixed rank of a frame already in hand -- :py:meth:`from_point` without the
        orthogonalization. The variation masks are the frame's own, gauge-shifted
        (:py:func:`~t3toolbox.backend.ufv_masking.ufv_variation_masks`): the frame IS the orthogonal
        representation ``from_point`` would recompute, so the two agree whenever ``frame_data`` is the frame
        of a point with ``plain_masks`` (one sweep, a few SVDs and a TT orthogonalization saved per call).

        For a caller that already holds the frame -- a local model building its backend geometry once per
        frame -- rather than a point."""
        shape = tuple(frame_data[4])
        return cls(shape, readonly_mask_copies(plain_masks),
                   readonly_mask_copies(ufv_masking.ufv_variation_masks(frame_data[5])),
                   canonical_groups(sharing, shape))

    def with_sharing(self, sharing) -> 'UniformManifoldGeometryOps':
        """This geometry restricted to tied Tucker factors (``sharing=None`` gives it back unshared)."""
        return dc.replace(self, groups=canonical_groups(sharing, self.shape))

    @property
    def n_stack(self) -> int:
        """``|C|``, the frame stack rank (0 for a single tensor). The tucker edge mask is
        ``(d,) + C + (n,)``, so the stack is everything between."""
        return self.masks[0].ndim - 2

    def _variations(self, var_sc):
        """A bare variation supercore pair as full variation ``.data`` (the masks are ours to supply)."""
        return (var_sc[0], var_sc[1], self.shape, self.var_masks)

    def frame(
            self,
            x_sc:  typ.Tuple,   # bare (tucker_supercore, tt_supercore) at this geometry's rank
    ) -> typ.Tuple:             # uniform frame .data = (up, down, left, right, shape, masks)
        """The orthonormal frame at ``x_sc``, using this geometry's held shape and rank masks."""
        return ufv_conversions.ut3_orthogonal_representations(
            (x_sc[0], x_sc[1], self.shape, self.masks))[0]

    def stack_shape(
            self,
            x_sc:  typ.Tuple,      # bare (tucker_supercore, tt_supercore)
    ) -> typ.Tuple[int, ...]:      # C -- the frame/core stack (empty for a single tensor)
        """The point's frame stack ``C``. The uniform Tucker supercore is ``(d,) + C + (nU, N)``."""
        return tuple(x_sc[0].shape[1:-2])

    def base_point(self, frame_data):
        """The bare supercore pair ``(U, P)`` the frame is attached to."""
        return (frame_data[0], frame_data[2])

    def precompute(self, frame_data):
        """The per-frame SF-T3 companion when tied, ``None`` otherwise (see
        :py:meth:`ManifoldGeometryOps.precompute`)."""
        if not self.groups:
            return None
        return sharing_module.ufv_shared_frame_data(frame_data, self.groups)

    def project(self, frame_data, var_sc, aux=None):
        """The gauge projection ``Pi`` (plus the tied post-pass when shared); bare pair in and out."""
        variations = self._variations(var_sc)
        if not self.groups:
            gauged = utv_ops.utv_orthogonal_gauge_projection(frame_data, variations)
        else:
            gauged = utv_ops.utv_orthogonal_gauge_projection(
                frame_data, variations,
                shared_data=aux if aux is not None else self.precompute(frame_data))
        return (gauged[0], gauged[1])

    def retract(self, frame_data, var_sc, aux=None):
        """The manifold retraction (the tied embedding + grouped SVD when shared); bare pair out."""
        variations = self._variations(var_sc)
        if not self.groups:
            new_x = utv_ops.utv_retract(frame_data, variations)
        else:
            new_x = utv_ops.utv_retract(
                frame_data, variations,
                shared_data=aux if aux is not None else self.precompute(frame_data))
        return (new_x[0], new_x[1])

    def inner(self, a_sc, b_sc):
        """The MASKED coordinate ``<.,.>`` over bare variation pairs, so padding is never summed."""
        return utv_ops.utv_corewise_inner(self._variations(a_sc), self._variations(b_sc), self.n_stack)

    def point_norm_sq(self, x_sc):
        """``‖X‖² = ‖v_X‖²`` in the masked coordinate metric."""
        v_x = self.point_tangent(self.frame(x_sc))
        return self.inner(v_x, v_x)

    def point_tangent(self, frame_data):
        """The attachment point as a gauged tangent ``v_X`` (the direct construction)."""
        return ufv_base_point_tangent(frame_data)

    def zero_point_frame(
            self,
            frame_data:  typ.Tuple,   # uniform frame .data of this geometry: the STRUCTURE to copy
    ) -> typ.Tuple:                   # uniform frame .data at the zero tensor (the template's masks)
        """The canonical frame at the zero tensor (see :py:meth:`ManifoldGeometryOps.zero_point_frame`).
        Its real blocks are the ragged construction's, so the reference is layer-independent:
        :py:func:`~t3toolbox.backend.ufv_operations.ufv_zero_point_frame`. Host-side and eager."""
        return ufv_operations.ufv_zero_point_frame(frame_data, self.groups)


@dc.dataclass(frozen=True, eq=False)
class UniformCorewiseGeometryOps(ValueHashedFields):
    """The uniform **corewise** geometry at a fixed rank -- the raw-supercore twin of
    :py:data:`t3toolbox.uniform_manifold.UNIFORM_COREWISE`.

    As :py:class:`CorewiseGeometryOps`: the cores are the frame, no gauge, additive retraction; and as
    :py:class:`UniformManifoldGeometryOps`, the rank structure is held here because the optimizer state
    is a bare supercore pair. The variation masks are a pure derivation of the plain masks here (the
    ``(U, G, G, G)`` frame's gauge shift), so ``from_point`` needs no orthogonalization."""

    shape:   typ.Tuple[int, ...]   # the mode sizes
    masks:   typ.Tuple             # plain-UT3 rank masks (tucker_edge_mask, tt_edge_mask); HOST numpy
    groups:  typ.Tuple[typ.Tuple[int, ...], ...] = ()   # sharing partition; () = unshared

    @classmethod
    def from_point(
            cls,
            x0_data:  typ.Tuple,                          # UniformTuckerTensorTrain.data
            sharing:  typ.Optional[typ.Sequence] = None,  # len=d group labels; None = unshared
    ) -> 'UniformCorewiseGeometryOps':
        """The geometry at ``x0``'s fixed rank."""
        _tk_sc, _tt_sc, shape, base_masks = x0_data
        return cls(tuple(shape), readonly_mask_copies(base_masks), canonical_groups(sharing, tuple(shape)))

    def with_sharing(self, sharing) -> 'UniformCorewiseGeometryOps':
        """This geometry restricted to tied Tucker factors (``sharing=None`` gives it back unshared)."""
        return dc.replace(self, groups=canonical_groups(sharing, self.shape))

    @property
    def n_stack(self) -> int:
        """``|C|``, the frame stack rank (see :py:attr:`UniformManifoldGeometryOps.n_stack`)."""
        return self.masks[0].ndim - 2

    @property
    def var_masks(self) -> typ.Tuple:
        """The variation masks of the ``(U, G, G, G)`` frame -- the corewise frame's mask set put
        through the gauge shift. A pure derivation of :py:attr:`masks`, so it is a property rather
        than a stored field."""
        tucker_mask, tt_mask = self.masks
        return ufv_masking.ufv_variation_masks((tucker_mask, tucker_mask, tt_mask, tt_mask))

    def _variations(self, var_sc):
        return (var_sc[0], var_sc[1], self.shape, self.var_masks)

    def frame(
            self,
            x_sc:  typ.Tuple,   # bare (tucker_supercore, tt_supercore)
    ) -> typ.Tuple:             # uniform frame .data = (U, G, G, G, shape, masks)
        """The corewise frame: the cores themselves, with the doubled mask set."""
        return ufv_conversions.ut3_corewise_frame((x_sc[0], x_sc[1], self.shape, self.masks))

    def stack_shape(
            self,
            x_sc:  typ.Tuple,      # bare (tucker_supercore, tt_supercore)
    ) -> typ.Tuple[int, ...]:      # C -- the frame/core stack (empty for a single tensor)
        """The point's frame stack ``C``. The uniform Tucker supercore is ``(d,) + C + (nU, N)``."""
        return tuple(x_sc[0].shape[1:-2])

    def base_point(self, frame_data):
        """The bare supercore pair ``(U, G)`` the frame is attached to."""
        return (frame_data[0], frame_data[2])

    def precompute(self, frame_data):
        """No per-frame companion on this geometry."""
        return None

    def project(self, frame_data, var_sc, aux=None):
        """The identity (Euclidean core space), or the per-group mean when tied."""
        if not self.groups:
            return var_sc
        tied = sharing_module.ufv_share_tucker_variations_corewise(self._variations(var_sc), self.groups)
        return (tied[0], tied[1])

    def retract(self, frame_data, var_sc, aux=None):
        """The additive retraction (mean-tied first when shared, which keeps tied-in giving tied-out)."""
        new_x = utv_ops.utv_corewise_retract(frame_data, self._variations(self.project(frame_data, var_sc)))
        return (new_x[0], new_x[1])

    def inner(self, a_sc, b_sc):
        """The MASKED coordinate ``<.,.>`` (the Euclidean metric here)."""
        return utv_ops.utv_corewise_inner(self._variations(a_sc), self._variations(b_sc), self.n_stack)

    def point_norm_sq(self, x_sc):
        """``Σ‖core_i‖²`` in the masked coordinate metric -- weight decay."""
        return self.inner(x_sc, x_sc)

    def point_tangent(self, frame_data):
        """The cores ``(U, G)`` as a tangent (the projection is the identity here)."""
        return self.base_point(frame_data)

    def zero_point_frame(self, frame_data) -> None:
        """``None``: no zero-point reference on a corewise geometry (see
        :py:meth:`CorewiseGeometryOps.zero_point_frame`)."""
        return None


if jax_available:
    import jax

    # The curvature data is a jax pytree whose arrays are all LEAVES (the lift factors, grad f's doubled-rank
    # cores and its sweep, sigma_min); the aux is the value-hashed static skeleton of `partition_static`, which
    # keeps any static structure static (a uniform twin's shape ints and host masks) -- so a new frame, gradient
    # or damping changes leaf values only, never the jit cache key.
    def _flatten_curvature_data(c):
        return partition_static((c.lift_factors, c.z, c.z_sweep, c.sigma_min))

    def _unflatten_curvature_data(skeleton, children):
        return ManifoldCurvatureData(*rebuild_static(children, skeleton))

    jax.tree_util.register_pytree_node(ManifoldCurvatureData, _flatten_curvature_data, _unflatten_curvature_data)
