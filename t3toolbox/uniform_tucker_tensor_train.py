# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""
Uniform Tucker tensor trains (UT3): a stacked-supercore + boolean-mask representation of (a stack of)
ragged :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` s, laid out for GPU/jit efficiency.

A UT3 pads the cores of a Tucker tensor train up to common sizes ``n`` (Tucker rank), ``r`` (TT rank),
``N`` (mode dimension), stacks the ``d`` cores onto a leading axis to form *supercores*, and records the
real extent of each padded edge with boolean masks. It is, by design, a *faster representation of the
ragged layer* -- see ``docs/uniform_equivalence_contract.md`` and the other ``docs/uniform_*.md`` notes.

NOTE: this module is being built incrementally (slice 1: foundation). Linear algebra, orthogonalization,
SVD, sampling, and the jax pytree registration land in later slices.
"""
import typing as typ
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence, Tuple

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.ranks as ranks
import t3toolbox.backend.ut3_conversions as ut3_conversions
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.ut3_orthogonalization as ut3_orthogonalization
import t3toolbox.backend.ut3_linalg as ut3_linalg
import t3toolbox.backend.ut3_sampling as ut3_sampling
import t3toolbox.backend.ut3_svd as ut3_svd
import t3toolbox.backend.ut3_constructors as ut3_constructors
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.common as common
from t3toolbox.backend.common import NDArray

if common.has_jax:
    import jax

__all__ = [
    'UT3Masks',
    'UniformTuckerTensorTrain',
]

@dataclass(frozen=True, eq=False)  # eq=False -> the mixin's VALUE-based __hash__/__eq__ stand (a bare
class UT3Masks(common.ValueHashedMasks):  # eq=True would fail on arrays). See ValueHashedMasks.
    """The static rank structure of a uniform Tucker tensor train: its two boolean edge masks.

    Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form). Held as a
    separate object so it can ride as jax ``aux_data``. Hash/eq are **value-based** (the
    :py:class:`~t3toolbox.backend.common.ValueHashedMasks` mixin), so a rebuilt-but-identical holder is the
    *same* jit cache key -- no per-iteration recompile in optimization loops; see
    ``docs/uniform_pytree_composition.md``. (The physical ``shape`` is a separate static int tuple on
    :py:class:`UniformTuckerTensorTrain` -- not a mask, and already value-hashable.)
    """
    tucker_edge_mask: NDArray  # HOST bool, static, shape=(d,)   + stack_shape + (n,)
    tt_edge_mask:     NDArray  # HOST bool, static, shape=(d+1,) + stack_shape + (r,)

    @property
    def data(self) -> Tuple[NDArray, NDArray]:
        """The two raw rank-mask arrays, ``(tucker_edge_mask, tt_edge_mask)``."""
        return self.tucker_edge_mask, self.tt_edge_mask


@dataclass(frozen=True)
class UniformTuckerTensorTrain:
    """A uniform Tucker tensor train: two supercores (the data) + a :py:class:`UT3Masks` holder (the
    static structure).

    - ``tucker_supercore``: shape ``(d,) + stack_shape + (n, N)``
    - ``tt_supercore``: shape ``(d,) + stack_shape + (r, n, r)``
    - ``masks``: the :py:class:`UT3Masks` (shape mask + the two rank masks)

    The mode index ``d`` leads (outside the stack) so sweeps compile to ``lax.scan`` over axis 0
    (``docs/uniform_supercore_layout.md``). Ranks may differ across the stack; the physical shape may
    not (``docs/uniform_ranks_and_varieties.md``).
    """
    tucker_supercore: NDArray         # shape=(d,)+stack_shape+(n,N)
    tt_supercore:     NDArray         # shape=(d,)+stack_shape+(r,n,r)
    shape:            Tuple[int, ...] # len=d; (N0,...,N(d-1)) real mode dims, shared across the stack
    masks:            UT3Masks        # static rank structure (the two edge masks)

    # ----------------------------------------------------------------- views
    @cached_property
    def supercores(self) -> Tuple[NDArray, NDArray]:
        """``(tucker_supercore, tt_supercore)``."""
        return self.tucker_supercore, self.tt_supercore

    @cached_property
    def data(self) -> Tuple[NDArray, NDArray, Tuple[int, ...], Tuple[NDArray, NDArray]]:
        """Raw-array view, mirroring the fields: ``(tucker_supercore, tt_supercore, shape, (2 rank masks))``.

        Backend ``ut3_*`` functions take this layout (supercore-only ops use ``.data[:2]``; the static
        ``shape`` is ``.data[2]``; mask-using ops unpack ``.data[3]``). The ``UT3Masks`` holder stays a
        frontend concern.
        """
        return self.tucker_supercore, self.tt_supercore, self.shape, self.masks.data

    # ------------------------------------------------- padded (uniform) structure
    @cached_property
    def d(self) -> int:
        """Number of modes."""
        return self.tucker_supercore.shape[0]

    @cached_property
    def n(self) -> int:
        """Padded Tucker rank (``n >= max`` of the real Tucker ranks)."""
        return self.tucker_supercore.shape[-2]

    @cached_property
    def N(self) -> int:
        """Padded mode dimension (``N >= max`` of the real shapes)."""
        return self.tucker_supercore.shape[-1]

    @cached_property
    def r(self) -> int:
        """Padded TT rank (``r >= max`` of the real TT ranks)."""
        return self.tt_supercore.shape[-1]

    @cached_property
    def stack_shape(self) -> Tuple[int, ...]:
        """Stack shape (``()`` if unstacked). Lives at axes ``1 .. len(stack_shape)`` (``d`` is axis 0)."""
        return self.tucker_supercore.shape[1:-2]

    @cached_property
    def uniform_structure(self) -> Tuple[int, int, int, int, Tuple[int, ...]]:
        """``(d, N, n, r, stack_shape)`` -- the padded structure."""
        return self.d, self.N, self.n, self.r, self.stack_shape

    # ------------------------------------------------- original (real) structure
    # (`shape` is a stored field above -- the real mode dims, shared across the stack.)

    @cached_property
    def tucker_ranks(self) -> NDArray:  # dtype=int, shape=(d,)+stack_shape
        """Real Tucker ranks (from ``tucker_edge_mask``; may vary across the stack)."""
        return self.masks.tucker_edge_mask.sum(axis=-1)

    @cached_property
    def tt_ranks(self) -> NDArray:  # dtype=int, shape=(d+1,)+stack_shape
        """Real TT ranks (from ``tt_edge_mask``; may vary across the stack)."""
        return self.masks.tt_edge_mask.sum(axis=-1)

    @cached_property
    def structure(self) -> Tuple[Tuple[int, ...], NDArray, NDArray, Tuple[int, ...]]:
        """``(shape, tucker_ranks, tt_ranks, stack_shape)`` -- the real structure."""
        return self.shape, self.tucker_ranks, self.tt_ranks, self.stack_shape

    # ----------------------------------------------------------------- validation
    def validate(self):
        """Check the structural invariants (shapes mutually consistent, rank masks boolean, ``shape`` a
        length-``d`` tuple of mode dims within the padded ``N``). Raises ValueError."""
        tkm, ttm = self.masks.data
        for m, name in ((tkm, 'tucker_edge_mask'), (ttm, 'tt_edge_mask')):
            if not common.is_boolean_ndarray(m):
                raise ValueError(
                    'UniformTuckerTensorTrain: %s must be a boolean array (got %s).'
                    % (name, getattr(m, 'dtype', type(m))))

        d, stack, n, N, r = self.d, self.stack_shape, self.n, self.N, self.r
        if len(self.shape) != d:
            raise ValueError(
                'UniformTuckerTensorTrain: shape=%s has length %d, expected d=%d.'
                % (self.shape, len(self.shape), d))
        if any(Ni < 0 or Ni > N for Ni in self.shape):
            raise ValueError(
                'UniformTuckerTensorTrain: every entry of shape=%s must be in [0, padded N=%d].'
                % (self.shape, N))

        expected = {
            'tt_supercore':     (d,) + stack + (r, n, r),
            'tucker_edge_mask': (d,) + stack + (n,),
            'tt_edge_mask':     (d + 1,) + stack + (r,),
        }
        actual = {
            'tt_supercore':     tuple(self.tt_supercore.shape),
            'tucker_edge_mask': tuple(tkm.shape),
            'tt_edge_mask':     tuple(ttm.shape),
        }
        for k in expected:
            if actual[k] != expected[k]:
                raise ValueError(
                    'Inconsistent UniformTuckerTensorTrain: %s.shape = %s, expected %s '
                    '(d=%d, stack_shape=%s, n=%d, N=%d, r=%d).'
                    % (k, actual[k], expected[k], d, stack, n, N, r))

    def __post_init__(self):
        self.validate()

    def __repr__(self) -> str:
        ss = ', stack_shape=%s' % (self.stack_shape,) if self.stack_shape else ''
        return ('UniformTuckerTensorTrain(shape=%s, N=%d, n=%d, r=%d%s)'
                % (self.shape, self.N, self.n, self.r, ss))

    # ----------------------------------------------------------------- operations
    def apply_masks(self) -> 'UniformTuckerTensorTrain':
        """Zero the padded ("garbage") regions of the supercores (the masks are unchanged)."""
        mtk, mtt = ut3_masking.apply_masks_to_cores(self.data)
        return UniformTuckerTensorTrain(mtk, mtt, self.shape, self.masks)

    def to_dense(self) -> NDArray:
        """Form the dense tensor, ``shape = stack_shape + (N0,...,N(d-1))``. (Inspection/tests only.)"""
        return ut3_conversions.ut3_to_dense(self.data)

    # ----------------------------------------------------------------- ragged <-> uniform conversions
    @staticmethod
    def from_t3(
            x:  t3.TuckerTensorTrain,
            N:  Optional[int] = None,   # padded mode dim   (default max(Ni)); pass to force a larger pad
            n:  Optional[int] = None,   # padded Tucker rank (default max(tucker_ranks))
            r:  Optional[int] = None,   # padded TT rank    (default max(tt_ranks))
            squash_tails: bool = True,
    ) -> 'UniformTuckerTensorTrain':
        """Pack a ragged :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` into a uniform one."""
        return _from_data(ut3_conversions.t3_to_ut3(x.data, N=N, n=n, r=r, squash_tails=squash_tails))

    def to_t3(self):  # -> TuckerTensorTrain (unstacked) or a nested tree (shaped like stack_shape) of them
        """Convert back to ragged form.

        Unstacked: one :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain`. Stacked: a nested tree
        of them (a varying-rank stack has no single stacked ``TuckerTensorTrain``;
        ``docs/uniform_ranks_and_varieties.md``).
        """
        def _wrap(res):
            if common.is_ndarray(res[0][0]):   # res = (tucker_cores, tt_cores) leaf
                return t3.TuckerTensorTrain(*res)
            return tuple(_wrap(r) for r in res)

        return _wrap(ut3_conversions.ut3_to_t3(self.data))

    def reverse(self) -> 'UniformTuckerTensorTrain':
        """Reverse the mode order."""
        return _from_data(ut3_operations.ut3_reverse(self.data))

    def squash_tails(self) -> 'UniformTuckerTensorTrain':
        """Sum the leading/trailing TT bonds down to rank 1 (preserves the represented tensor)."""
        return _from_data(ut3_operations.ut3_squash_tails(self.data))

    # ----------------------------------------------------------------- linear algebra
    def __mul__(self, s) -> 'UniformTuckerTensorTrain':
        """Scale by a scalar."""
        return _from_data(ut3_linalg.ut3_scale(self.data, s))

    __rmul__ = __mul__

    def __neg__(self) -> 'UniformTuckerTensorTrain':
        return self * (-1.0)

    def __add__(self, other: 'UniformTuckerTensorTrain') -> 'UniformTuckerTensorTrain':
        """Add two uniform Tucker tensor trains (direct sum, then squash). Requires matching shape / d /
        stack_shape (structural ValueError); padded ``n``/``r`` need not match."""
        if self.shape != other.shape:
            raise ValueError('Cannot add UniformTuckerTensorTrains with different shapes: %s vs %s.'
                             % (self.shape, other.shape))
        if self.stack_shape != other.stack_shape:
            raise ValueError('Cannot add UniformTuckerTensorTrains with different stack_shapes: %s vs %s.'
                             % (self.stack_shape, other.stack_shape))
        return _from_data(ut3_operations.ut3_squash_tails(ut3_linalg.ut3_add(self.data, other.data)))

    def __sub__(self, other: 'UniformTuckerTensorTrain') -> 'UniformTuckerTensorTrain':
        return self + (-other)

    def sum_stack(self) -> 'UniformTuckerTensorTrain':
        """Sum the represented tensors over the entire stack -> one unstacked uniform T3 (genuine tensor
        sum, not corewise)."""
        return _from_data(ut3_operations.ut3_squash_tails(ut3_linalg.ut3_sum_stack(self.data)))

    def inner(self, other: 'UniformTuckerTensorTrain', use_orthogonalization: bool = True):
        """Hilbert-Schmidt inner product with another uniform Tucker tensor train (shape=stack_shape)."""
        if self.shape != other.shape:
            raise ValueError('Cannot inner-product UniformTuckerTensorTrains with different shapes: %s vs %s.'
                             % (self.shape, other.shape))
        xd, yd = self.data, other.data
        if use_orthogonalization:
            xd = ut3_orthogonalization.left_orthogonalize_tt_cores(
                ut3_orthogonalization.down_orthogonalize_tucker_cores(xd))
            yd = ut3_orthogonalization.left_orthogonalize_tt_cores(
                ut3_orthogonalization.down_orthogonalize_tucker_cores(yd))
        return ut3_linalg.ut3_inner_product(xd, yd)

    def norm(self, use_orthogonalization: bool = True):
        """Hilbert-Schmidt (Frobenius) norm of the represented tensor (shape=stack_shape)."""
        if use_orthogonalization:
            xd = ut3_orthogonalization.left_orthogonalize_tt_cores(
                ut3_orthogonalization.down_orthogonalize_tucker_cores(self.data))
            return ut3_linalg.ut3_norm_orthogonalized(xd)
        xnp, _, _ = common.get_backend(True, self.contains_jax)
        return xnp.sqrt(xnp.abs(ut3_linalg.ut3_inner_product(self.data, self.data)))

    # ----------------------------------------------------------------- sampling / evaluation
    def entries(self, index) -> NDArray:
        """Entry/entries of the represented tensor. ``index``: int array, ``shape=(d,)+idx_stack``."""
        return ut3_sampling.ut3_entries(self.data, index)

    def apply(self, vecs) -> NDArray:
        """Contract with vectors in all modes. ``vecs``: len-d, ith ``elm_shape=vec_stack+(Ni,)``."""
        return ut3_sampling.ut3_apply(self.data, vecs)

    def probe(self, ww):
        """Probe: contract all-but-one mode, for each mode. ``ww``: len-d, ith ``elm_shape=W+(Ni,)``."""
        return ut3_sampling.ut3_probe(ww, self.data)

    def sum(self, axis=None) -> NDArray:
        """Sum the represented tensor over all physical modes (shape=stack_shape). Partial sums (``axis``
        given) are deferred -- see docs/uniform_port_plan.md."""
        if axis is not None:
            raise NotImplementedError(
                'Partial sum (axis given) is deferred for UniformTuckerTensorTrain; only the full sum '
                '(axis=None) is implemented. See docs/uniform_port_plan.md.')
        return ut3_sampling.ut3_full_sum(self.data)

    # ----------------------------------------------------------------- orthogonalization
    # Thin wrappers over the .data-level backend (ut3_orthogonalization): the Tucker-core ops are
    # batched-SVD rewrites; the TT left/right ops share the polymorphic orthogonalization.py sweep. All
    # re-masking and mask/rank recomputation lives in the backend.

    def down_orthogonalize_tucker_cores(self) -> 'UniformTuckerTensorTrain':
        """Orthogonalize the Tucker cores, pushing the remainder up into the TT cores."""
        return _from_data(ut3_orthogonalization.down_orthogonalize_tucker_cores(self.data))

    def up_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Up-orthogonalize the TT cores, pushing the remainder down into the Tucker cores."""
        return _from_data(ut3_orthogonalization.up_orthogonalize_tt_cores(self.data))

    def left_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Left-orthogonalize the TT cores."""
        return _from_data(ut3_orthogonalization.left_orthogonalize_tt_cores(self.data))

    def right_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Right-orthogonalize the TT cores."""
        return _from_data(ut3_orthogonalization.right_orthogonalize_tt_cores(self.data))

    def is_left_orthogonal(self, atol: float = 1e-9) -> bool:
        """True if in left-orthogonal form (Tucker supercores down-orthogonal AND TT supercores
        left-orthogonal). A :py:meth:`t3svd` result is left-orthogonal. Non-enforcing checker; see
        :py:func:`~t3toolbox.backend.ut3_orthogonalization.ut3_orthogonality_residual`."""
        return bool(ut3_orthogonalization.ut3_orthogonality_residual(self.data, 'left') <= atol)

    def is_right_orthogonal(self, atol: float = 1e-9) -> bool:
        """True if in right-orthogonal form (Tucker supercores down-orthogonal AND TT supercores
        right-orthogonal). Non-enforcing checker; verify before ``t3svd(..., assume_orthogonal=True)``."""
        return bool(ut3_orthogonalization.ut3_orthogonality_residual(self.data, 'right') <= atol)

    @property
    def minimal_ranks(self) -> Tuple[NDArray, NDArray]:
        """Structural minimal ranks ``(min_tucker_ranks, min_tt_ranks)`` for this UT3's shape/ranks."""
        use_jax = self.contains_jax
        return ranks.compute_minimal_ranks(self.shape, self.tucker_ranks, self.tt_ranks, use_jax=use_jax)

    @property
    def has_minimal_ranks(self) -> bool:
        """True if this UT3's ranks are structurally minimal (every stack element)."""
        mn = self.minimal_ranks
        return bool(np.all(np.asarray(self.tucker_ranks) == np.asarray(mn[0]))
                    and np.all(np.asarray(self.tt_ranks) == np.asarray(mn[1])))

    # ----------------------------------------------------------------- T3-SVD
    def t3svd(self, max_tt_ranks=None, max_tucker_ranks=None, assume_orthogonal=False):
        """Mask-truncated T3-SVD -- the basic algorithm, matching ragged :py:meth:`TuckerTensorTrain.t3svd`
        on real parts. Always **left-orthogonal**; under truncation **not** necessarily minimal (use
        :py:meth:`rank_adjustment_sweep` to minimize). ``assume_orthogonal=True`` skips the
        orthogonalization, asserting the input is already right-orthogonal (verify with
        :py:meth:`is_right_orthogonal` -- not checked). Uniform truncates by **max rank only** -- unlike
        ragged ``t3svd`` there is no ``rtol``/``atol`` (a tolerance would make the output shape
        data-dependent, which the uniform layer forbids; see ``docs/uniform_ranks_and_varieties.md``).
        Per-stack-element ``max_*_ranks`` arrays are allowed. Returns ``(new UT3, Tucker svals, TT svals)``."""
        new_data, ss_tucker, ss_tt = ut3_svd.ut3svd(
            self.data, max_tucker_ranks=max_tucker_ranks, max_tt_ranks=max_tt_ranks,
            assume_orthogonal=assume_orthogonal)
        return _from_data(new_data), ss_tucker, ss_tt

    def rank_adjustment_sweep(self, direction: str = 'right_to_left') -> 'UniformTuckerTensorTrain':
        """A single directional sweep that drops structurally-redundant ranks (the separate
        rank-minimization step; :py:meth:`t3svd` does not minimize). ``'right_to_left'`` returns a
        right-orthogonal UT3, ``'left_to_right'`` a left-orthogonal one; it reaches minimal ranks **only
        if the input is orthogonal in the opposite direction** (a :py:meth:`t3svd` result is
        left-orthogonal, so ``'right_to_left'`` minimizes it).

        That precondition is **required**, not just optimal, and is **not enforced**: because the uniform
        layer commits to fixed (minimal) output shapes, sweeping the wrong direction for the input's gauge
        is **lossy** -- it discards real content (unlike the ragged version, which only under-minimizes).
        Verify the gauge with :py:meth:`is_left_orthogonal` / :py:meth:`is_right_orthogonal` first.
        See :py:func:`~t3toolbox.backend.ut3_svd.ut3_rank_adjustment_sweep`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.tucker_tensor_train as t3
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = t3.TuckerTensorTrain.randn((10, 10, 10), (9, 9, 9), (1, 9, 9, 1))
        >>> x2, _, _ = ut3.UniformTuckerTensorTrain.from_t3(x).t3svd(max_tucker_ranks=[9, 1, 9], max_tt_ranks=[1, 9, 2, 1])
        >>> print(x2.is_left_orthogonal(), x2.has_minimal_ranks)   # t3svd output: left-orth, non-minimal
        True False
        >>> good = x2.rank_adjustment_sweep('right_to_left')       # CORRECT (x2 is left-orthogonal)
        >>> print(good.has_minimal_ranks, np.allclose(good.to_dense(), x2.to_dense()))
        True True
        >>> bad = x2.rank_adjustment_sweep('left_to_right')        # WRONG direction -> corrupts the tensor
        >>> print(np.allclose(bad.to_dense(), x2.to_dense()))
        False
        """
        return _from_data(ut3_svd.ut3_rank_adjustment_sweep(self.data, direction))

    # ----------------------------------------------------------------- stacking
    def unstack(self):
        """Unstack into an array-like tree (shaped like ``stack_shape``) of unstacked UT3s."""
        return stacking.apply_func_to_leaf_subtrees(
            ut3_operations.ut3_unstack(self.data),
            lambda leaf: UniformTuckerTensorTrain(leaf[0], leaf[1], leaf[2], UT3Masks(*leaf[3])),
            ut3_operations.ut3_leaf_structure(self.d),
        )

    @staticmethod
    def stack(uxx) -> 'UniformTuckerTensorTrain':
        """Stack an array-like tree of UT3s into one stacked UT3."""
        data_tree = stacking.apply_func_to_leaf_subtrees(uxx, lambda u: u.data, None)
        tk, tt, shape, masks = ut3_operations.ut3_stack(data_tree)
        return UniformTuckerTensorTrain(tk, tt, shape, UT3Masks(*masks))

    # ----------------------------------------------------------------- dtype / copy
    @cached_property
    def contains_jax(self) -> bool:
        return common.tree_contains_jax(self.supercores)

    def to_jax(self) -> 'UniformTuckerTensorTrain':
        # Convert the SUPERCORES (data) to jax; the masks stay numpy (host structure -- a jax mask is a
        # tracer under jit and breaks the layer). See docs/uniform_pytree_composition.md.
        return UniformTuckerTensorTrain(
            common.to_jax(self.tucker_supercore), common.to_jax(self.tt_supercore), self.shape, self.masks)

    def to_numpy(self) -> 'UniformTuckerTensorTrain':
        # Supercores -> numpy; the masks are already numpy (host structure), so reuse the holder.
        return UniformTuckerTensorTrain(
            common.to_numpy(self.tucker_supercore), common.to_numpy(self.tt_supercore), self.shape, self.masks)

    def copy(self) -> 'UniformTuckerTensorTrain':
        return UniformTuckerTensorTrain(self.tucker_supercore, self.tt_supercore, self.shape, self.masks)

    # ----------------------------------------------------------------- constructors
    # Pure constructors keep a `use_jax` flag for the SUPERCORES (no array input to infer from); the
    # masks are always numpy (host) structure (docs/uniform_pytree_composition.md). The ranks may vary
    # per stack element (the variety) -- a backend feature a ragged round-trip cannot express.

    @staticmethod
    def zeros(
            shape:        Sequence[int],                          # (N0,...,N(d-1))
            tucker_ranks: typ.Union[int, Sequence[int], NDArray, None] = None,  # int|len-d|(d,)+stack; None->1
            tt_ranks:     typ.Union[int, Sequence[int], NDArray, None] = None,  # int|len-(d+1)|(d+1,)+stack; None->1
            stack_shape:  Sequence[int] = (),
            use_jax:      bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Uniform Tucker tensor train of zeros (padded regions masked to zero).

        ``tucker_ranks``/``tt_ranks`` accept a scalar, a per-mode sequence, or a full ``(d,)+stack`` /
        ``(d+1,)+stack`` array (the variety: ranks varying per stack element). ``None`` -> all ranks 1.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> z = ut3.UniformTuckerTensorTrain.zeros((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        >>> print(z.shape, z.stack_shape)
        (5, 6, 7) (2,)
        >>> print(float(np.linalg.norm(z.to_dense())))
        0.0
        """
        return _from_data(ut3_constructors.ut3_zeros(
            shape, tucker_ranks, tt_ranks, tuple(stack_shape), use_jax=use_jax))

    @staticmethod
    def ones(
            shape:       Sequence[int],          # (N0,...,N(d-1))
            stack_shape: Sequence[int] = (),
            use_jax:     bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Rank-1 uniform Tucker tensor train representing a tensor full of ones (every real entry == 1).

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> x = ut3.UniformTuckerTensorTrain.ones((5, 6, 7), stack_shape=(2,))
        >>> print(float(np.linalg.norm(x.to_dense() - np.ones((2, 5, 6, 7)))))
        0.0
        >>> print(np.asarray(x.tucker_ranks).max(), np.asarray(x.tt_ranks).max())
        1 1
        """
        return _from_data(ut3_constructors.ut3_ones(shape, tuple(stack_shape), use_jax=use_jax))

    @staticmethod
    def randn(
            shape:        Sequence[int],                          # (N0,...,N(d-1))
            tucker_ranks: typ.Union[int, Sequence[int], NDArray], # int|len-d|(d,)+stack
            tt_ranks:     typ.Union[int, Sequence[int], NDArray], # int|len-(d+1)|(d+1,)+stack
            stack_shape:  Sequence[int] = (),
            use_jax:      bool = False,
    ) -> 'UniformTuckerTensorTrain':
        """Uniform Tucker tensor train with random N(0,1) supercores (padded regions masked to zero).

        ``tucker_ranks``/``tt_ranks`` accept a scalar, a per-mode sequence, or a full ``(d,)+stack`` /
        ``(d+1,)+stack`` array -- the latter setting **per-stack-element ranks** (the variety) while
        keeping one padded supercore shape, which a ragged round-trip cannot express.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1), stack_shape=(2,))
        >>> print(x.shape, x.stack_shape)
        (5, 6, 7) (2,)
        >>> print(np.reshape(x.tucker_ranks, (3, 2))[:, 0].tolist())   # uniform across the stack here
        [3, 4, 2]
        >>> print(bool(np.any(x.tucker_supercore != 0.0)))            # random, not zeros
        True

        Per-stack-element ranks (the variety): a full ``(d,)+stack`` array gives each stack element its
        own ranks under one padded shape.

        >>> tucker_ranks = np.array([[2, 4], [3, 5], [2, 3]])         # (d=3, stack=2)
        >>> tt_ranks = np.array([[1, 1], [2, 4], [2, 3], [1, 1]])     # (d+1=4, stack=2)
        >>> xv = ut3.UniformTuckerTensorTrain.randn((6, 7, 8), tucker_ranks, tt_ranks, stack_shape=(2,))
        >>> print(np.asarray(xv.tucker_ranks).tolist())              # ranks genuinely differ per element
        [[2, 4], [3, 5], [2, 3]]
        >>> print(xv.n, xv.r)                                        # one padded shape: n=max, r=max
        5 4
        """
        return _from_data(ut3_constructors.ut3_randn(
            shape, tucker_ranks, tt_ranks, tuple(stack_shape), use_jax=use_jax))

    # Note: there are deliberately NO ``from_canonical`` / ``from_tensor_train`` / ``to_tensor_train``
    # methods. They would take *ragged* CP/TT data and round-trip through ``TuckerTensorTrain``, which is
    # ambiguous (ragged vs uniform input). Be explicit instead: build a ``TuckerTensorTrain`` (which has
    # those methods) and convert with :py:meth:`from_t3` / :py:meth:`to_t3`.

    # ----------------------------------------------------------------- save / load
    def save(
            self,
            file,  # path or open file object to write the .npz to
    ) -> None:
        """Save to a ``.npz`` file (2 supercores + 2 rank masks + the shape ints). See :py:meth:`load`.

        Examples
        --------
        >>> import numpy as np
        >>> import t3toolbox.uniform_tucker_tensor_train as ut3
        >>> np.random.seed(0)
        >>> x = ut3.UniformTuckerTensorTrain.randn((5, 6, 7), (3, 4, 2), (1, 3, 2, 1))
        >>> fname = 'ut3_file.npz'
        >>> x.save(fname)
        >>> x2 = ut3.UniformTuckerTensorTrain.load(fname)
        >>> print(float(np.linalg.norm(x2.to_dense() - x.to_dense())))
        0.0
        >>> x2.shape                                       # the static shape survives the round-trip
        (5, 6, 7)
        >>> print([str(m.dtype) for m in x2.masks.data])   # rank masks come back numpy (host) bool
        ['bool', 'bool']
        """
        ut3_constructors.ut3_save(file, self.data)

    @staticmethod
    def load(
            file,                   # path or open file object to read the .npz from
            use_jax: bool = False,  # supercore type; the masks always come back numpy (host) bool
    ) -> 'UniformTuckerTensorTrain':
        """Load from a ``.npz`` file written by :py:meth:`save`.

        The supercores follow ``use_jax``; the masks stay **numpy (host) bool** (a jax mask is a tracer
        under jit and breaks the layer; ``docs/uniform_pytree_composition.md``). See :py:meth:`save` for
        an example.
        """
        return _from_data(ut3_constructors.ut3_load(file, use_jax=use_jax))


def _from_data(
        data: typ.Tuple[NDArray, NDArray, typ.Tuple[int, ...], typ.Tuple[NDArray, NDArray]],
) -> 'UniformTuckerTensorTrain':
    """Wrap a backend ``.data`` tuple ``(tucker_supercore, tt_supercore, shape, (2 rank masks))`` into a
    ``UniformTuckerTensorTrain``. Every frontend operation is a thin wrapper: call the matching
    ``ut3_*`` backend function on ``self.data``, then re-wrap with this. The masks holder is the only
    frontend-side construction (the OO-frame exception to the backend/frontend razor)."""
    tk, tt, shape, masks = data
    return UniformTuckerTensorTrain(tk, tt, shape, UT3Masks(*masks))


# (ragged <-> uniform conversions are methods on UniformTuckerTensorTrain: `.from_t3` / `.to_t3`.)


if common.has_jax:
    # UniformTuckerTensorTrain as a jax pytree: the two supercores are the (traced) children; the static
    # aux_data is ``(shape, UT3Masks)``. Both are STRUCTURE (the real mode dims + which rank slots are
    # real), not data, so they belong in aux, not the traced leaves. BOTH are value-keyed: ``shape`` is a
    # value-hashable int tuple, and ``UT3Masks`` hashes/compares by mask CONTENT (the ValueHashedMasks
    # mixin). So the jit cache key reflects the rank STRUCTURE -- a rebuilt-but-identical object is the
    # same key (no per-iteration recompile when frames are re-orthogonalized in an optimization loop); a
    # genuinely different structure recompiles (correct). Because uniform output ranks are STATICALLY
    # determined (no rtol; shrink-to-structural-minimum), a jitted op's output masks stay compile-time
    # constants -- safe as aux. See docs/uniform_pytree_composition.md.
    jax.tree_util.register_pytree_node(
        UniformTuckerTensorTrain,
        lambda x: ((x.tucker_supercore, x.tt_supercore), (x.shape, x.masks)),
        lambda aux, children: UniformTuckerTensorTrain(children[0], children[1], aux[0], aux[1]),
    )
