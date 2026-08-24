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
from t3toolbox.backend import t3_orthogonalization as ragged_orth
from t3toolbox.backend import tt_operations
from t3toolbox.backend.fv_conversions import t3_orthogonal_representations, t3_corewise_frame
import t3toolbox.corewise as cw

__all__ = [
    'ManifoldGeometryOps',
    'CorewiseGeometryOps',
    'UniformManifoldGeometryOps',
    'UniformCorewiseGeometryOps',
    'canonical_groups',
    'fv_base_point_tangent',
    'ufv_base_point_tangent',
    't3_left_orthogonal_norm_sq',
    't3_alias_tied_tucker_factors',
]


canonical_groups = sharing_module.canonical_groups   # re-exported: the geometry's `groups` field is exactly this


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
