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
import t3toolbox.uniform_tucker_tensor_train as ut3
import t3toolbox.uniform_frame_variations_format as ubv
import t3toolbox.uniform_manifold as um
import t3toolbox.safety as safety
import t3toolbox.backend.sharing as backend_sharing
import t3toolbox.backend.tv_operations as tv_operations
import t3toolbox.backend.utv_operations as utv_operations
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
            'shared format first (TuckerTensorTrain.share, or t3_tie_tucker_factors for a '
            'nearly-tied point), or run in unsafe mode (safety.unsafe()).'.format(who))


def _require_tied_factors_uniform(
        data,               # UT3-shaped data: (tucker_sc, tt_sc, shape, (tucker_mask, tt_mask))
        sharing,            # len=d; group labels
        who:        str,    # the operation name, for the message
) -> None:
    '''Safe-mode TIED precondition on uniform data (the masked-content residual).'''
    if safety.checks_active(data[:2]):
        atol = safety.effective_rtol(data[:2])
        residual = backend_sharing.ut3_sharing_residual(data, sharing)
        safety.require(
            bool((residual <= atol).all()),
            '{} requires the (masked) Tucker factors to be tied within each sharing group. Enter '
            'the shared format first, or run in unsafe mode (safety.unsafe()).'.format(who))


class SharedGeometry:
    '''The shared (SF-T3) geometry: a base geometry restricted to tied Tucker factors.

    Construct via :py:func:`shared` / :py:func:`shared_manifold` / :py:func:`shared_corewise`.
    The wrapper is stateless up to its static ``(base, sharing)`` identity (value-based
    ``__eq__``/``__hash__``, so it is a stable jit aux); the per-frame companion
    (:py:class:`~t3toolbox.backend.sharing.SharedFrameData`) is DERIVED from a frame on
    demand -- pass it explicitly (``shared_data=``) to amortize across calls at one frame, as
    the fitting models do via :py:meth:`precompute`.

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
        if (base is not t3m.MANIFOLD and base is not t3m.COREWISE
                and base is not um.UNIFORM_MANIFOLD and base is not um.UNIFORM_COREWISE):
            raise ValueError(
                'SharedGeometry wraps a geometry singleton (manifold.MANIFOLD / manifold.COREWISE / '
                'uniform_manifold.UNIFORM_MANIFOLD / UNIFORM_COREWISE); got %r.' % (base,))
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
        return {id(t3m.MANIFOLD): 'manifold', id(t3m.COREWISE): 'corewise',
                id(um.UNIFORM_MANIFOLD): 'uniform_manifold',
                id(um.UNIFORM_COREWISE): 'uniform_corewise'}[id(self.base)]

    @property
    def is_uniform(self) -> bool:
        """True when the base is a uniform geometry singleton (points/tangents are uniform objects)."""
        return self.base is um.UNIFORM_MANIFOLD or self.base is um.UNIFORM_COREWISE

    @property
    def _is_manifold_kind(self) -> bool:
        return self.base is t3m.MANIFOLD or self.base is um.UNIFORM_MANIFOLD

    def __eq__(self, other) -> bool:
        # type(self), NOT the hardcoded class: a SUBCLASS is a different geometry, and this pair is a jit
        # cache key (a SharedGeometry rides in GaussNewtonModel's aux). Hardcoding the name made a
        # subclass with different math compare and hash EQUAL to the plain wrapper, so whichever was
        # compiled first served both -- a silent wrong answer for the un-subclassed object.
        return (type(other) is type(self) and other.base is self.base
                and other.sharing == self.sharing)

    def __hash__(self) -> int:
        return hash((type(self), self.base_name, self.sharing))

    def __repr__(self) -> str:
        return 'SharedGeometry(%s, sharing=%r)' % (self.base_name.upper(), self.sharing)

    def groups(self, shape) -> typ.Tuple[typ.Tuple[int, ...], ...]:
        """The canonical partition for a given mode-size tuple (validates the spec)."""
        return backend_sharing.validate_sharing(self.sharing, shape)

    # ------------------------------------------------------------------ frame + companion
    def frame(self, x):
        """The base geometry's frame at ``x`` (ragged ``TuckerTensorTrain`` -> ``T3Frame``;
        uniform ``UniformTuckerTensorTrain`` -> ``UT3Frame``).

        **An untied ``x`` is tied first, silently**, by the per-group mean
        (:py:func:`~t3toolbox.backend.sharing.t3_tie_tucker_factors` / its uniform twin) -- the frame of
        a shared geometry describes a point on the shared set, so entering the format is this method's
        job rather than the caller's. An already-tied point is a bitwise fixed point, so the ordinary
        path is unchanged. This is what makes an untied initial guess a non-event for the frontend optimizers (which build their frames through this wrapper),
        and it also absorbs the slow drift a long run of low-precision first-order steps can produce."""
        self.groups(x.shape)                                     # structural validation
        if self.is_uniform and not isinstance(x, ut3.UniformTuckerTensorTrain):
            raise TypeError(
                'this SharedGeometry wraps the UNIFORM base %s: frame() takes a '
                'UniformTuckerTensorTrain, got %s -- build the geometry from the ragged base '
                '(shared(MANIFOLD/COREWISE, ...)) for ragged points (review R9-9)'
                % (self.base_name, type(x).__name__))
        if not self.is_uniform and not isinstance(x, t3.TuckerTensorTrain):
            raise TypeError(
                'this SharedGeometry wraps the RAGGED base %s: frame() takes a TuckerTensorTrain, '
                'got %s -- use shared(UNIFORM_MANIFOLD/UNIFORM_COREWISE, ...) for uniform points '
                '(review R9-9)' % (self.base_name, type(x).__name__))
        if self.is_uniform:
            return self.base.frame(um._ut3_from_data(
                backend_sharing.ut3_tie_tucker_factors(x.data, self.sharing)))
        return self.base.frame(t3.TuckerTensorTrain(
            *backend_sharing.t3_tie_tucker_factors(x.data, self.sharing)))

    def shared_frame_data(self, frame) -> backend_sharing.SharedFrameData:
        """The per-frame companion (centers + the stacked-``S`` SVD), derived from ``frame``
        (a ``T3Frame`` or a ``UT3Frame``).

        ``svd_s`` is the group spectrum ``s_g`` -- the continuation/aptness statistic, free at
        every frame. Manifold bases only (the corewise post-pass needs no companion)."""
        if self.is_uniform and not isinstance(frame, ubv.UT3Frame):
            raise TypeError('this SharedGeometry wraps the UNIFORM base %s: expected a UT3Frame, got %s '
                            '(review R9-9)' % (self.base_name, type(frame).__name__))
        if not self.is_uniform and not isinstance(frame, bvf.T3Frame):
            raise TypeError('this SharedGeometry wraps the RAGGED base %s: expected a T3Frame, got %s '
                            '(review R9-9)' % (self.base_name, type(frame).__name__))
        if self.is_uniform:
            return backend_sharing.ufv_shared_frame_data(frame.data, self.groups(frame.shape))
        return backend_sharing.fv_shared_frame_data(frame.data, self.groups(frame.shape))

    def precompute(self, frame):
        """The fitting models' once-per-frame hook: the companion (manifold bases) or ``None``
        (corewise bases -- the mean needs only the static partition)."""
        if self._is_manifold_kind:
            return self.shared_frame_data(frame)
        return None

    # ------------------------------------------------------------------ tangent ops
    def project(self, v, shared_data=None):
        """The tied projection ``Pi_sh``: orthogonal projection onto the TIED tangent subspace
        in this geometry's own metric (manifold: gauge projection + the Gram/SVD tilted
        post-pass; corewise: the per-group mean). Inherits the base's safe-mode preconditions.
        Ragged bases take/return ``T3Tangent``; uniform bases take/return ``UT3Tangent``."""
        if self.base is t3m.COREWISE:
            new_variations = backend_sharing.fv_share_tucker_variations_corewise(
                v.variations.data, self.groups(v.shape))
            return t3m.T3Tangent(v.frame, bvf.T3Variations(*new_variations))
        if self.base is um.UNIFORM_COREWISE:
            new_variations = backend_sharing.ufv_share_tucker_variations_corewise(
                v.variations.data, self.groups(v.shape))
            return um.UT3Tangent(v.frame, um._ut3variations_from_data(new_variations))
        if self.base is um.UNIFORM_MANIFOLD:
            um._require_orthogonal_frame(v.frame, 'SharedGeometry.project')
            if shared_data is None:
                shared_data = self.shared_frame_data(v.frame)
            new_variations = utv_operations.utv_orthogonal_gauge_projection(
                v.frame.data, v.variations.data, shared_data=shared_data)
            return um.UT3Tangent(v.frame, um._ut3variations_from_data(new_variations))
        t3m._require_orthogonal_frame(v.frame, 'SharedGeometry.project')
        if shared_data is None:
            shared_data = self.shared_frame_data(v.frame)
        new_variations = tv_operations.tv_orthogonal_gauge_projection(
            v.frame.data, v.variations.data, shared_data=shared_data)
        return t3m.T3Tangent(v.frame, bvf.T3Variations(*new_variations))

    def project_oblique(self, v):
        """The base geometry's vector-preserving gauge fix (manifold bases only). Note it does
        NOT tie: the represented vector is preserved, so the result is tied exactly when the
        vector already lies in the tied tangent space."""
        if not self._is_manifold_kind:
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
    def retract(self, p, shared_data=None):
        """The shared retraction: stays on the shared set exactly (one factor array per group;
        on the uniform layer, one factor content at every group mode).

        Manifold bases: the TIED doubled-rank embedding (the group's common ambient direction is
        recovered from the tied coordinates by the companion's clipped solve) truncated by the
        grouped T3-SVD. Corewise bases: mean-tie then the additive retraction. Safe mode
        requires the base preconditions, tied frame factors, and (manifold) tied tangent
        coordinates -- an untied tangent would be silently tied-projected by the embedding's
        solve, which is almost never what the caller meant."""
        groups = self.groups(p.shape)
        if self.base is um.UNIFORM_COREWISE:
            tied = backend_sharing.ufv_share_tucker_variations_corewise(p.variations.data, groups)
            return self.base.retract(um.UT3Tangent(p.frame, um._ut3variations_from_data(tied)))
        if self.base is um.UNIFORM_MANIFOLD:
            um._require_orthogonal_frame(p.frame, 'SharedGeometry.retract')
            _require_tied_factors_uniform(
                (p.frame.data[0], p.frame.data[2], p.frame.shape,
                 (p.frame.data[5][0], p.frame.data[5][2])),
                self.sharing, 'SharedGeometry.retract')
            if shared_data is None:
                shared_data = self.shared_frame_data(p.frame)
            if safety.checks_active(p.frame.data[:4], p.variations.data[:2]):
                atol = safety.effective_rtol(p.frame.data[:4], p.variations.data[:2])
                residual = backend_sharing.ufv_tied_variations_residual(p.variations.data, shared_data)
                safety.require(
                    bool((residual <= atol).all()),
                    'SharedGeometry.retract requires TIED tangent coordinates (project '
                    'with this geometry first); an untied tangent would be silently '
                    'tied-projected. Run in unsafe mode (safety.unsafe()) to skip.')
            return um._ut3_from_data(utv_operations.utv_retract(
                p.frame.data, p.variations.data, shared_data=shared_data))
        if self.base is t3m.COREWISE:
            new_variations = backend_sharing.fv_share_tucker_variations_corewise(p.variations.data, groups)
            x_data = (p.frame.up_tucker_cores, p.frame.left_tt_cores)
            new = cw.corewise_add(x_data, new_variations)
            # Tie the SUM rather than aliasing one mode's copy of it. Because
            # ``mean_i(U_i + V_i) = mean_i(U_i) + mean_i(V_i)``, tying after the add IS the corewise
            # (Euclidean) projection of ``x + v`` onto the tied set -- so the retraction is TOTAL: an
            # untied tangent (already handled by the mean above) and an untied base point both land on
            # the shared set, with nothing silently discarded. On a tied base point every summed core is
            # already identical and ``t3_tie_tucker_factors`` is a bitwise fixed point, so the sanctioned
            # path is unchanged to the last ulp; it also gives ONE array per group, as before.
            return t3.TuckerTensorTrain(*backend_sharing.t3_tie_tucker_factors(new, self.sharing))
        t3m._require_orthogonal_frame(p.frame, 'SharedGeometry.retract')
        _require_tied_factors((p.frame.up_tucker_cores, p.frame.left_tt_cores), self.sharing,
                              'SharedGeometry.retract')
        if shared_data is None:
            shared_data = self.shared_frame_data(p.frame)
        if safety.checks_active(p.frame.data, p.variations.data):
            atol = safety.effective_rtol(p.frame.data, p.variations.data)
            residual = backend_sharing.fv_tied_variations_residual(p.variations.data, shared_data)
            safety.require(
                bool((residual <= atol).all()),
                'SharedGeometry.retract requires TIED tangent coordinates (project '
                'with this geometry first); an untied tangent would be silently '
                'tied-projected. Run in unsafe mode (safety.unsafe()) to skip.')
        cores = tv_operations.tv_retract(p.frame.data, p.variations.data, shared_data=shared_data)
        return t3.TuckerTensorTrain(*cores)

    def project_ambient(self, frame, grad, method: str = 'contraction',
                        shared_data=None):
        """Project an ambient gradient onto the TIED tangent space (manifold bases only). The
        uniform base accepts a ``UniformTuckerTensorTrain`` gradient only (see
        :py:meth:`~t3toolbox.uniform_manifold.UniformManifoldGeometry.project_ambient`)."""
        if not self._is_manifold_kind:
            raise AttributeError('project_ambient is a manifold-geometry operation')
        if self.base is um.UNIFORM_MANIFOLD:
            if shared_data is None:
                shared_data = self.shared_frame_data(frame)
            raw = self.base.project_ambient(frame, grad)             # gauge projection (ORTH-checked)
            tied = backend_sharing.ufv_share_tucker_variations(raw.variations.data, shared_data)
            return um.UT3Tangent(frame, um._ut3variations_from_data(tied))
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

    def transport(self, v, new_frame, shared_data=None):
        """Projective transport onto the TIED tangent space at ``new_frame`` (manifold bases
        only). Safe mode requires ``new_frame``'s factors tied."""
        if not self._is_manifold_kind:
            raise AttributeError('transport is a manifold-geometry operation')
        if self.base is um.UNIFORM_MANIFOLD:
            _require_tied_factors_uniform(
                (new_frame.data[0], new_frame.data[2], new_frame.shape,
                 (new_frame.data[5][0], new_frame.data[5][2])),
                self.sharing, 'SharedGeometry.transport')
            return self.project_ambient(new_frame, v.to_ut3(), shared_data=shared_data)
        _require_tied_factors((new_frame.up_tucker_cores, new_frame.left_tt_cores), self.sharing,
                              'SharedGeometry.transport')
        return self.project_ambient(new_frame, v.to_t3(), shared_data=shared_data)


def shared(
        base,                       # MANIFOLD / COREWISE / UNIFORM_MANIFOLD / UNIFORM_COREWISE
        sharing:    typ.Sequence,   # len=d; one hashable group label per mode
) -> SharedGeometry:
    '''The shared (SF-T3) geometry over a base geometry: optimize with the Tucker factors tied
    within each sharing group. Equal wrappers compare equal (a stable jit aux). Uniform bases
    take/return the uniform objects (``UniformTuckerTensorTrain`` / ``UT3Frame`` / ``UT3Tangent``)
    and run the packed compile-once fitting path.'''
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
    _BASES_BY_NAME = {'manifold': t3m.MANIFOLD, 'corewise': t3m.COREWISE,
                      'uniform_manifold': um.UNIFORM_MANIFOLD,
                      'uniform_corewise': um.UNIFORM_COREWISE}
    jax.tree_util.register_pytree_node(
        SharedGeometry,
        lambda g: ((), (g.base_name, g.sharing)),
        lambda aux, children: SharedGeometry(_BASES_BY_NAME[aux[0]], aux[1]),
    )
