# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The shared-Tucker-factors (SF-T3) geometry wrapper: ``shared(base, sharing)``.

Wraps :py:data:`~t3toolbox.manifold.MANIFOLD` or :py:data:`~t3toolbox.manifold.COREWISE` so every
projection lands on the TIED tangent subspace (one Tucker factor per sharing group) and the
retraction stays on the shared set. One principle, two formulas -- each geometry ties by
orthogonal projection onto ITS tied subspace in ITS metric on ITS coordinates: the manifold
geometry's gauged coordinates carry the frame's ``S`` factors, so its post-pass is the
Gram-weighted (SVD-solved) tilted projection; the corewise coordinates are raw factor copies, so
its post-pass is the per-group arithmetic mean. Optimizers consume a shared geometry exactly like
a base one (``t3t.newton_cg(shared_manifold(sharing), 'apply', ww, b, x0)``).
"""
from __future__ import annotations

import typing as typ
import numpy as np

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.frame_variations_format as bvf
import t3toolbox.manifold as t3m
import t3toolbox.corewise as cw
import t3toolbox.safety as safety
import t3toolbox.backend.sharing as backend_sharing
import t3toolbox.backend.tv_operations as tv_operations
from t3toolbox.backend.common import *

__all__ = [
    'SharedGeometry',
    'shared',
    'shared_manifold',
    'shared_corewise',
]


def _require_tied_factors(
        data,               # (tucker_cores, tt_cores) raw data (a point or a frame's (U, P))
        sharing,            # len=d; group labels
        who:        str,    # the operation name, for the message
) -> None:
    '''Safe-mode TIED precondition: ``who`` requires the Tucker factors tied within each group.'''
    if safety.checks_active(data):
        atol = safety.effective_rtol(data)
        residual = backend_sharing.t3_sharing_residual(data, sharing)
        safety.require(
            bool((residual <= atol).all()),
            '{} requires the Tucker factors to be tied within each sharing group. Enter the '
            'shared format first (TuckerTensorTrain.share, or t3_share_tucker_cores for a '
            'nearly-tied point), or run in unsafe mode (safety.unsafe()).'.format(who))


class SharedGeometry:
    '''The shared (SF-T3) geometry: a base geometry restricted to tied Tucker factors.

    Construct via :py:func:`shared` / :py:func:`shared_manifold` / :py:func:`shared_corewise`.
    The wrapper is stateless up to its static ``(base, sharing)`` identity (value-based
    ``__eq__``/``__hash__``, so it is a stable jit aux); the per-frame companion
    (:py:class:`~t3toolbox.backend.sharing.T3SharedFrameData`) is DERIVED from a frame on
    demand -- pass it explicitly (``shared_data=``) to amortize across calls at one frame, as
    the fitting models do via :py:meth:`precompute_aux`.

    On a MANIFOLD base the full surface is available (``frame``/``project``/``project_oblique``/
    ``inner``/``norm``/``retract``/``project_ambient``/``transport``/``randn``); on a COREWISE
    base the surface matches the base (no ambient projection / transport). Safe mode enforces
    the base geometry's preconditions plus tied factors at ``frame``/``retract``/``transport``
    entry. Full shared rank is deliberately NOT a precondition -- zero-padded continuation
    restarts sit on the lower-shared-rank stratum by construction, and the tied projection's
    clipped solve is well-defined (minimum-norm) there.

    Examples
    --------
    >>> import numpy as np
    >>> import t3toolbox.tucker_tensor_train as t3
    >>> import t3toolbox.manifold as t3m
    >>> import t3toolbox.shared_geometry as sg
    >>> np.random.seed(0)
    >>> x = t3.TuckerTensorTrain.randn((6, 6, 5), (3, 3, 2), (1, 2, 2, 1)).share((0, 0, 1))
    >>> geom = sg.shared_manifold((0, 0, 1))
    >>> frame = geom.frame(x)                       # safe-mode tied check + orthonormal frame
    >>> v = geom.randn(frame)                       # a standard Gaussian on the TIED tangent space
    >>> print(v.is_gauged())
    True
    >>> y = geom.retract(v)                         # grouped retraction: stays exactly tied
    >>> print(y.data[0][0] is y.data[0][1])
    True
    >>> y0 = geom.retract(t3m.T3Tangent.zeros(frame))
    >>> print(bool(np.allclose(y0.to_dense(), x.to_dense())))   # retract(0) == the base point
    True
    '''

    def __init__(
            self,
            base,                       # t3m.MANIFOLD or t3m.COREWISE (the ragged singletons)
            sharing:    typ.Sequence,   # len=d; one hashable group label per mode
    ):
        if base is not t3m.MANIFOLD and base is not t3m.COREWISE:
            raise ValueError(
                'SharedGeometry wraps the ragged geometry singletons (manifold.MANIFOLD / '
                'manifold.COREWISE); got %r. (The uniform mirror is not built yet.)' % (base,))
        self.base = base
        self.sharing = tuple(sharing)
        for ii, label in enumerate(self.sharing):
            try:
                hash(label)
            except TypeError:
                raise ValueError('sharing labels must be hashable; got %r at mode %d' % (label, ii))

    # ---- static identity (a stable jit-aux key: rebuilt-equal wrappers are the same geometry) ----
    @property
    def base_name(self) -> str:
        return 'manifold' if self.base is t3m.MANIFOLD else 'corewise'

    def __eq__(self, other) -> bool:
        return (type(other) is SharedGeometry and other.base is self.base
                and other.sharing == self.sharing)

    def __hash__(self) -> int:
        return hash((SharedGeometry, self.base_name, self.sharing))

    def __repr__(self) -> str:
        return 'SharedGeometry(%s, sharing=%r)' % (self.base_name.upper(), self.sharing)

    def groups(self, shape) -> typ.Tuple[typ.Tuple[int, ...], ...]:
        """The canonical partition for a given mode-size tuple (validates the spec)."""
        return backend_sharing.validate_sharing(self.sharing, shape)

    # ------------------------------------------------------------------ frame + companion
    def frame(self, x: t3.TuckerTensorTrain) -> bvf.T3Frame:
        """The base geometry's frame at ``x``. Safe mode requires ``x``'s factors tied."""
        self.groups(x.shape)                                     # structural validation
        _require_tied_factors(x.data, self.sharing, 'SharedGeometry.frame')
        return self.base.frame(x)

    def shared_frame_data(self, frame: bvf.T3Frame) -> backend_sharing.T3SharedFrameData:
        """The per-frame companion (centers + the stacked-``S`` SVD), derived from ``frame``.

        ``svd_s`` is the group spectrum ``s_g`` -- the continuation/aptness statistic, free at
        every frame. MANIFOLD base only (the corewise post-pass needs no companion)."""
        return backend_sharing.fv_shared_frame_data(frame.data, self.groups(frame.shape))

    def precompute_aux(self, frame: bvf.T3Frame):
        """The fitting models' once-per-frame hook: the companion (MANIFOLD base) or ``None``
        (COREWISE base -- the mean needs only the static partition)."""
        if self.base is t3m.MANIFOLD:
            return self.shared_frame_data(frame)
        return None

    # ------------------------------------------------------------------ tangent ops
    def project(self, v: t3m.T3Tangent, shared_data=None) -> t3m.T3Tangent:
        """The tied projection ``Pi_sh``: orthogonal projection onto the TIED tangent subspace
        in this geometry's own metric (manifold: gauge projection + the Gram/SVD tilted
        post-pass; corewise: the per-group mean). Inherits the base's safe-mode preconditions."""
        if self.base is t3m.COREWISE:
            new_variations = backend_sharing.fv_mean_tucker_variations(
                v.variations.data, self.groups(v.shape))
            return t3m.T3Tangent(v.frame, bvf.T3Variations(*new_variations))
        t3m._require_orthogonal_frame(v.frame, 'SharedGeometry.project')
        if shared_data is None:
            shared_data = self.shared_frame_data(v.frame)
        new_variations = tv_operations.tv_orthogonal_gauge_projection(
            v.frame.data, v.variations.data, shared_data=shared_data)
        return t3m.T3Tangent(v.frame, bvf.T3Variations(*new_variations))

    def project_oblique(self, v: t3m.T3Tangent) -> t3m.T3Tangent:
        """The base geometry's vector-preserving gauge fix (MANIFOLD base only). Note it does
        NOT tie: the represented vector is preserved, so the result is tied exactly when the
        vector already lies in the tied tangent space."""
        if self.base is t3m.COREWISE:
            raise AttributeError('project_oblique is a manifold-geometry operation')
        return self.base.project_oblique(v)

    def inner(self, t1: t3m.T3Tangent, t2: t3m.T3Tangent):
        """The base geometry's inner product -- the tied subspace is linear, so the restricted
        metric is the metric (never reweighted by sharing)."""
        return self.base.inner(t1, t2)

    def norm(self, t: t3m.T3Tangent):
        """The base geometry's norm (see :py:meth:`inner`)."""
        return self.base.norm(t)

    def randn(self, frame: bvf.T3Frame, stack_shape: typ.Tuple[int, ...] = ()) -> t3m.T3Tangent:
        """A standard Gaussian on the TIED tangent space: the base draw, tied-projected."""
        return self.project(self.base.randn(frame, stack_shape=stack_shape))

    def randn_like(self, tangent: t3m.T3Tangent) -> t3m.T3Tangent:
        """A tied random tangent at ``tangent``'s frame, with its tangent stack ``K``."""
        return self.randn(tangent.frame, stack_shape=tangent.tangent_stack_shape)

    # ------------------------------------------------------------------ retraction + transport
    def retract(self, p: t3m.T3Tangent, shared_data=None) -> t3.TuckerTensorTrain:
        """The shared retraction: stays on the shared set exactly (one factor array per group).

        MANIFOLD base: the TIED doubled-rank embedding (the group's common ambient direction is
        recovered from the tied coordinates by the companion's clipped solve) truncated by the
        grouped T3-SVD. COREWISE base: mean-tie then the additive retraction. Safe mode
        requires the base preconditions, tied frame factors, and (manifold) tied tangent
        coordinates -- an untied tangent would be silently tied-projected by the embedding's
        solve, which is almost never what the caller meant."""
        groups = self.groups(p.shape)
        if self.base is t3m.COREWISE:
            new_variations = backend_sharing.fv_mean_tucker_variations(p.variations.data, groups)
            x_data = (p.frame.up_tucker_cores, p.frame.left_tt_cores)
            new = cw.corewise_add(x_data, new_variations)
            new_tucker = list(new[0])
            for group in backend_sharing.nontrivial_groups(groups):
                for ii in group[1:]:
                    new_tucker[ii] = new_tucker[group[0]]    # ONE array per group (values equal)
            return t3.TuckerTensorTrain(tuple(new_tucker), new[1])
        t3m._require_orthogonal_frame(p.frame, 'SharedGeometry.retract')
        _require_tied_factors((p.frame.up_tucker_cores, p.frame.left_tt_cores), self.sharing,
                              'SharedGeometry.retract')
        if shared_data is None:
            shared_data = self.shared_frame_data(p.frame)
        if safety.checks_active(p.frame.data, p.variations.data):
            atol = safety.effective_rtol(p.frame.data, p.variations.data)
            tied = backend_sharing.fv_share_tucker_variations(p.variations.data, shared_data)
            num = sum(float(np.linalg.norm(np.asarray(A) - np.asarray(B)))
                      for A, B in zip(tied[0], p.variations.tucker_variations))
            den = sum(float(np.linalg.norm(np.asarray(B))) for B in p.variations.tucker_variations)
            safety.require(num <= atol * max(den, 1.0),
                           'SharedGeometry.retract requires TIED tangent coordinates (project '
                           'with this geometry first); an untied tangent would be silently '
                           'tied-projected. Run in unsafe mode (safety.unsafe()) to skip.')
        cores = tv_operations.tv_retract(p.frame.data, p.variations.data, shared_data=shared_data)
        return t3.TuckerTensorTrain(*cores)

    def project_ambient(self, frame: bvf.T3Frame, grad, method: str = 'contraction',
                        shared_data=None) -> t3m.T3Tangent:
        """Project an ambient gradient onto the TIED tangent space (MANIFOLD base only)."""
        if self.base is t3m.COREWISE:
            raise AttributeError('project_ambient is a manifold-geometry operation')
        t3m._require_orthogonal_frame(frame, 'SharedGeometry.project_ambient')
        if shared_data is None:
            shared_data = self.shared_frame_data(frame)
        if isinstance(grad, t3.TuckerTensorTrain):
            variations = tv_operations.tv_project_t3_onto_tangent_space(
                frame.data, grad.data, shared_data=shared_data)
            return t3m.T3Tangent(frame, bvf.T3Variations(*variations))
        if method == 'contraction':
            variations = tv_operations.tv_project_dense_onto_tangent_space(
                frame.data, grad, shared_data=shared_data)
            return t3m.T3Tangent(frame, bvf.T3Variations(*variations))
        elif method == 't3svd':
            d = len(frame.shape)
            stack_shape = tuple(grad.shape[:grad.ndim - d])
            xg, _, _ = t3.TuckerTensorTrain.t3svd_dense(grad, stack_shape=stack_shape)
            return self.project_ambient(frame, xg, shared_data=shared_data)
        raise ValueError("project_ambient: method must be 'contraction' or 't3svd', got %r"
                         % (method,))

    def transport(self, v: t3m.T3Tangent, new_frame: bvf.T3Frame,
                  shared_data=None) -> t3m.T3Tangent:
        """Projective transport onto the TIED tangent space at ``new_frame`` (MANIFOLD base
        only). Safe mode requires ``new_frame``'s factors tied."""
        if self.base is t3m.COREWISE:
            raise AttributeError('transport is a manifold-geometry operation')
        _require_tied_factors((new_frame.up_tucker_cores, new_frame.left_tt_cores), self.sharing,
                              'SharedGeometry.transport')
        return self.project_ambient(new_frame, v.to_t3(), shared_data=shared_data)


def shared(
        base,                       # t3m.MANIFOLD or t3m.COREWISE
        sharing:    typ.Sequence,   # len=d; one hashable group label per mode
) -> SharedGeometry:
    '''The shared (SF-T3) geometry over a base geometry: optimize with the Tucker factors tied
    within each sharing group. Equal wrappers compare equal (a stable jit aux).'''
    return SharedGeometry(base, sharing)


def shared_manifold(sharing: typ.Sequence) -> SharedGeometry:
    '''``shared(MANIFOLD, sharing)`` -- the shared fixed-rank Riemannian geometry.'''
    return SharedGeometry(t3m.MANIFOLD, sharing)


def shared_corewise(sharing: typ.Sequence) -> SharedGeometry:
    '''``shared(COREWISE, sharing)`` -- the shared core-parameter Euclidean geometry.'''
    return SharedGeometry(t3m.COREWISE, sharing)


if jax_available:
    import jax

    # Zero-leaf pytree with a value-based static identity: a SharedGeometry passes through
    # jit/vmap as an ordinary argument and reconstructs to an equal wrapper (same aux key).
    jax.tree_util.register_pytree_node(
        SharedGeometry,
        lambda g: ((), (g.base_name, g.sharing)),
        lambda aux, children: SharedGeometry(
            t3m.MANIFOLD if aux[0] == 'manifold' else t3m.COREWISE, aux[1]),
    )
