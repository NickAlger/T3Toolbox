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
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import t3toolbox.tucker_tensor_train as t3
import t3toolbox.backend.ut3_conversions as ut3_conversions
import t3toolbox.backend.ut3_masking as ut3_masking
import t3toolbox.backend.ut3_operations as ut3_operations
import t3toolbox.backend.ut3_orthogonalization as ut3_orthogonalization
import t3toolbox.backend.ut3_linalg as ut3_linalg
import t3toolbox.backend.orthogonalization as orth
import t3toolbox.backend.t3_operations as ragged_operations
import t3toolbox.backend.stacking as stacking
import t3toolbox.backend.common as common
from t3toolbox.backend.common import NDArray

__all__ = [
    'UT3Masks',
    'UniformTuckerTensorTrain',
    't3_to_ut3',
    'ut3_to_t3',
]

# A uniform-T3 leaf in nested .data layout, for the tree machinery in stacking.py.
_UT3_LEAF_STRUCTURE = (None, None, (None, None, None))


@dataclass(frozen=True, eq=False)  # eq=False -> identity __hash__/__eq__, so this array-holding object
class UT3Masks:                    # can be jax aux_data (value hash/eq is impossible). See
    """The static structure of a uniform Tucker tensor train: its three boolean edge masks.

    Slot ``j`` of an edge is real iff its mask is ``True`` there (the prefix/canonical form). Held as a
    separate, identity-hashable object so it can ride as jax ``aux_data`` -- the ``T3Basis``<->``T3Tangent``
    pattern; see ``docs/uniform_pytree_composition.md``.
    """
    shape_mask:       NDArray  # dtype=bool, shape=(d, N)                 (no stack: shape is shared across the stack)
    tucker_edge_mask: NDArray  # dtype=bool, shape=(d,)   + stack_shape + (n,)
    tt_edge_mask:     NDArray  # dtype=bool, shape=(d+1,) + stack_shape + (r,)

    @property
    def data(self) -> Tuple[NDArray, NDArray, NDArray]:
        """The three raw mask arrays, ``(shape_mask, tucker_edge_mask, tt_edge_mask)``."""
        return self.shape_mask, self.tucker_edge_mask, self.tt_edge_mask


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
    tucker_supercore: NDArray   # shape=(d,)+stack_shape+(n,N)
    tt_supercore:     NDArray   # shape=(d,)+stack_shape+(r,n,r)
    masks:            UT3Masks  # static structure (shape mask + rank masks)

    # ----------------------------------------------------------------- views
    @cached_property
    def supercores(self) -> Tuple[NDArray, NDArray]:
        """``(tucker_supercore, tt_supercore)``."""
        return self.tucker_supercore, self.tt_supercore

    @cached_property
    def data(self) -> Tuple[NDArray, NDArray, Tuple[NDArray, NDArray, NDArray]]:
        """Raw-array view, mirroring the fields: ``(tucker_supercore, tt_supercore, (3 masks))``.

        Backend ``ut3_*`` functions take this nested layout (supercore-only ops use ``.data[:2]``;
        mask-using ops unpack ``.data[2]``). The ``UT3Masks`` holder stays a frontend concern.
        """
        return self.tucker_supercore, self.tt_supercore, self.masks.data

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
    @cached_property
    def shape(self) -> Tuple[int, ...]:  # len=d
        """Real shape ``(N0,...,N(d-1))`` (from ``shape_mask``; shared across the stack)."""
        return tuple(int(x) for x in self.masks.shape_mask.sum(axis=-1))

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
        """Check the structural invariants (shapes mutually consistent, masks boolean). Raises ValueError."""
        sm, tkm, ttm = self.masks.data
        for m, name in ((sm, 'shape_mask'), (tkm, 'tucker_edge_mask'), (ttm, 'tt_edge_mask')):
            if not common.is_boolean_ndarray(m):
                raise ValueError(
                    'UniformTuckerTensorTrain: %s must be a boolean array (got %s).'
                    % (name, getattr(m, 'dtype', type(m))))

        d, stack, n, N, r = self.d, self.stack_shape, self.n, self.N, self.r
        expected = {
            'tt_supercore':     (d,) + stack + (r, n, r),
            'shape_mask':       (d, N),
            'tucker_edge_mask': (d,) + stack + (n,),
            'tt_edge_mask':     (d + 1,) + stack + (r,),
        }
        actual = {
            'tt_supercore':     tuple(self.tt_supercore.shape),
            'shape_mask':       tuple(sm.shape),
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
        return UniformTuckerTensorTrain(mtk, mtt, self.masks)

    def to_dense(self) -> NDArray:
        """Form the dense tensor, ``shape = stack_shape + (N0,...,N(d-1))``.

        Masks the supercores, chain-contracts (shared with the ragged path), then static-prefix-slices
        the padded physical axes down to the real shape. For checking work / tests -- never form a large
        one in practice.
        """
        masked_tucker, masked_tt = ut3_masking.apply_masks_to_cores(self.data)
        T = ragged_operations.t3_to_dense_chain(masked_tucker, masked_tt)   # stack + (N,)*d (padded)
        sl = (Ellipsis,) + tuple(slice(0, Ni) for Ni in self.shape)
        return T[sl]

    def reverse(self) -> 'UniformTuckerTensorTrain':
        """Reverse the mode order."""
        sm, tkm, ttm = self.masks.data
        return UniformTuckerTensorTrain(
            self.tucker_supercore[::-1],
            ut3_operations.reverse_utt(self.tt_supercore),
            UT3Masks(sm[::-1], tkm[::-1], ttm[::-1]),
        )

    def squash_tails(self) -> 'UniformTuckerTensorTrain':
        """Sum the leading/trailing TT bonds down to rank 1 (preserves the represented tensor)."""
        use_jax = self.contains_jax
        xnp, _, _ = common.get_backend(True, use_jax)

        new_tt_supercore = ut3_operations.uniform_squash_tt_tails(self.tt_supercore)

        sm, tkm, ttm = self.masks.data
        rank1 = xnp.broadcast_to(xnp.arange(self.r) < 1, self.stack_shape + (self.r,))  # [True, False, ...]
        new_ttm = xnp.concatenate([rank1[None], ttm[1:-1], rank1[None]], axis=0)
        return UniformTuckerTensorTrain(self.tucker_supercore, new_tt_supercore, UT3Masks(sm, tkm, new_ttm))

    # ----------------------------------------------------------------- linear algebra
    def __mul__(self, s) -> 'UniformTuckerTensorTrain':
        """Scale by a scalar (scales the last Tucker supercore slice; masks/ranks unchanged)."""
        xnp, _, _ = common.get_backend(True, self.contains_jax)
        scaled = xnp.concatenate([self.tucker_supercore[:-1], s * self.tucker_supercore[-1:]], axis=0)
        return UniformTuckerTensorTrain(scaled, self.tt_supercore, self.masks)

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
        tk, tt, masks = ut3_linalg.ut3_add(self.data, other.data)
        return UniformTuckerTensorTrain(tk, tt, UT3Masks(*masks)).squash_tails()

    def __sub__(self, other: 'UniformTuckerTensorTrain') -> 'UniformTuckerTensorTrain':
        return self + (-other)

    def sum_stack(self) -> 'UniformTuckerTensorTrain':
        """Sum the represented tensors over the entire stack -> one unstacked uniform T3 (genuine tensor
        sum, not corewise)."""
        tk, tt, masks = ut3_linalg.ut3_sum_stack(self.data)
        return UniformTuckerTensorTrain(tk, tt, UT3Masks(*masks)).squash_tails()

    def inner(self, other: 'UniformTuckerTensorTrain', use_orthogonalization: bool = True):
        """Hilbert-Schmidt inner product with another uniform Tucker tensor train (shape=stack_shape)."""
        if self.shape != other.shape:
            raise ValueError('Cannot inner-product UniformTuckerTensorTrains with different shapes: %s vs %s.'
                             % (self.shape, other.shape))
        x, y = self, other
        if use_orthogonalization:
            x = x.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
            y = y.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
        return ut3_linalg.ut3_inner_product(x.data, y.data)

    def norm(self, use_orthogonalization: bool = True):
        """Hilbert-Schmidt (Frobenius) norm of the represented tensor (shape=stack_shape)."""
        if use_orthogonalization:
            x = self.down_orthogonalize_tucker_cores().left_orthogonalize_tt_cores()
            return ut3_linalg.ut3_norm_orthogonalized(x.data)
        xnp, _, _ = common.get_backend(True, self.contains_jax)
        return xnp.sqrt(xnp.abs(ut3_linalg.ut3_inner_product(self.data, self.data)))

    # ----------------------------------------------------------------- orthogonalization
    # Core-local Tucker orthogonalizations are uniform-specific BATCHED-SVD rewrites; the TT left/right
    # sweeps SHARE the polymorphic orthogonalization.py. Each re-masks on entry; the SVD remainder
    # R = ss.Vt has ss=0 in the padded slots, so no garbage propagates. Ranks naturally drop to the
    # structural minimum the SVD produces (we shrink to it and set the masks to match -- minimal for free).

    def down_orthogonalize_tucker_cores(self) -> 'UniformTuckerTensorTrain':
        """Orthogonalize the Tucker cores (rows orthonormal over the mode index), pushing the remainder
        up into the TT cores. Tucker rank -> min(shape, tucker_rank)."""
        xnp, _, _ = common.get_backend(False, self.contains_jax)
        mtk, mtt = ut3_masking.apply_masks_to_cores(self.data)
        new_tk, new_tt = ut3_orthogonalization.down_orthogonalize_tucker_cores(mtk, mtt)
        sm, tkm, ttm = self.masks.data
        shape_arr = sm.sum(axis=-1).reshape((self.d,) + (1,) * len(self.stack_shape))  # (d,)+(1,)*stack
        new_tucker_ranks = xnp.minimum(self.tucker_ranks, shape_arr)
        new_tkm = xnp.arange(new_tk.shape[-2]) < new_tucker_ranks[..., None]
        return UniformTuckerTensorTrain(new_tk, new_tt, UT3Masks(sm, new_tkm, ttm))

    def up_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Up-orthogonalize the TT cores (mode index orthonormal over the bonds), pushing the remainder
        down into the Tucker cores. Tucker rank -> min(tucker_rank, rL*rR)."""
        xnp, _, _ = common.get_backend(False, self.contains_jax)
        mtk, mtt = ut3_masking.apply_masks_to_cores(self.data)
        new_tk, new_tt = ut3_orthogonalization.up_orthogonalize_tt_cores(mtk, mtt)
        sm, tkm, ttm = self.masks.data
        tt_ranks = self.tt_ranks
        new_tucker_ranks = xnp.minimum(self.tucker_ranks, tt_ranks[:-1] * tt_ranks[1:])
        new_tkm = xnp.arange(new_tt.shape[-2]) < new_tucker_ranks[..., None]
        return UniformTuckerTensorTrain(new_tk, new_tt, UT3Masks(sm, new_tkm, ttm))

    def left_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Left-orthogonalize the TT cores (shared polymorphic sweep). TT bond ranks follow the L->R
        recurrence ``r[i+1] = min(r[i]*n[i], r[i+1])``."""
        xnp, _, _ = common.get_backend(False, self.contains_jax)
        _, mtt = ut3_masking.apply_masks_to_cores(self.data)
        new_tt = orth.left_orthogonalize_tt_cores(mtt)
        sm, tkm, ttm = self.masks.data
        new_tt_ranks = _left_orthogonalized_tt_ranks(self.tt_ranks, self.tucker_ranks, xnp)
        new_ttm = xnp.arange(self.r) < new_tt_ranks[..., None]
        return UniformTuckerTensorTrain(self.tucker_supercore, new_tt, UT3Masks(sm, tkm, new_ttm))

    def right_orthogonalize_tt_cores(self) -> 'UniformTuckerTensorTrain':
        """Right-orthogonalize the TT cores (shared polymorphic sweep). TT bond ranks follow the R->L
        recurrence ``r[i] = min(n[i]*r[i+1], r[i])``."""
        xnp, _, _ = common.get_backend(False, self.contains_jax)
        _, mtt = ut3_masking.apply_masks_to_cores(self.data)
        new_tt = orth.right_orthogonalize_tt_cores(mtt)
        sm, tkm, ttm = self.masks.data
        new_tt_ranks = _right_orthogonalized_tt_ranks(self.tt_ranks, self.tucker_ranks, xnp)
        new_ttm = xnp.arange(self.r) < new_tt_ranks[..., None]
        return UniformTuckerTensorTrain(self.tucker_supercore, new_tt, UT3Masks(sm, tkm, new_ttm))

    # ----------------------------------------------------------------- stacking
    def unstack(self):
        """Unstack into an array-like tree (shaped like ``stack_shape``) of unstacked UT3s."""
        return stacking.apply_func_to_leaf_subtrees(
            ut3_operations.ut3_unstack(self.data),
            lambda leaf: UniformTuckerTensorTrain(leaf[0], leaf[1], UT3Masks(*leaf[2])),
            _UT3_LEAF_STRUCTURE,
        )

    @staticmethod
    def stack(uxx) -> 'UniformTuckerTensorTrain':
        """Stack an array-like tree of UT3s into one stacked UT3."""
        data_tree = stacking.apply_func_to_leaf_subtrees(uxx, lambda u: u.data, None)
        tk, tt, masks = ut3_operations.ut3_stack(data_tree)
        return UniformTuckerTensorTrain(tk, tt, UT3Masks(*masks))

    # ----------------------------------------------------------------- dtype / copy
    @cached_property
    def contains_jax(self) -> bool:
        return common.tree_contains_jax(self.supercores)

    def to_jax(self) -> 'UniformTuckerTensorTrain':
        return UniformTuckerTensorTrain(
            common.to_jax(self.tucker_supercore), common.to_jax(self.tt_supercore),
            UT3Masks(*[common.to_jax(m) for m in self.masks.data]))

    def to_numpy(self) -> 'UniformTuckerTensorTrain':
        return UniformTuckerTensorTrain(
            common.to_numpy(self.tucker_supercore), common.to_numpy(self.tt_supercore),
            UT3Masks(*[common.to_numpy(m) for m in self.masks.data]))

    def copy(self) -> 'UniformTuckerTensorTrain':
        return UniformTuckerTensorTrain(self.tucker_supercore, self.tt_supercore, self.masks)


# ------------------------------------------------------------ orthogonalization rank recurrences
# After left/right TT orthogonalization the real bond ranks follow a sweep recurrence (the boundary
# bonds r[0], r[d] are untouched). Same logic as ranks.compute_minimal_ranks' passes. Vectorized over
# the stack; the d-loop is short (and unrolls cleanly under jit).

def _left_orthogonalized_tt_ranks(
        tt_ranks:     NDArray,  # dtype=int, shape=(d+1,)+stack_shape
        tucker_ranks: NDArray,  # dtype=int, shape=(d,)+stack_shape
        xnp,
) -> NDArray:                   # dtype=int, shape=(d+1,)+stack_shape
    d = tucker_ranks.shape[0]
    new = [tt_ranks[0]]
    for i in range(d - 1):
        new.append(xnp.minimum(new[i] * tucker_ranks[i], tt_ranks[i + 1]))
    new.append(tt_ranks[d])
    return xnp.stack(new)


def _right_orthogonalized_tt_ranks(
        tt_ranks:     NDArray,  # dtype=int, shape=(d+1,)+stack_shape
        tucker_ranks: NDArray,  # dtype=int, shape=(d,)+stack_shape
        xnp,
) -> NDArray:                   # dtype=int, shape=(d+1,)+stack_shape
    d = tucker_ranks.shape[0]
    new = [None] * (d + 1)
    new[d] = tt_ranks[d]
    for i in range(d - 1, 0, -1):
        new[i] = xnp.minimum(tucker_ranks[i] * new[i + 1], tt_ranks[i])
    new[0] = tt_ranks[0]
    return xnp.stack(new)


# ===================================================================== conversions

def t3_to_ut3(
        x: t3.TuckerTensorTrain,
        N: int = None,              # padded mode dim   (default max(Ni)); pass to force a larger pad
        n: int = None,              # padded Tucker rank (default max(tucker_ranks))
        r: int = None,              # padded TT rank    (default max(tt_ranks))
        squash_tails: bool = True,
) -> UniformTuckerTensorTrain:
    """Convert a :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain` to a uniform one."""
    tk, tt, masks = ut3_conversions.t3_to_ut3(x.data, N=N, n=n, r=r, squash_tails=squash_tails)
    return UniformTuckerTensorTrain(tk, tt, UT3Masks(*masks))


def ut3_to_t3(
        ux: UniformTuckerTensorTrain,
):  # -> TuckerTensorTrain (unstacked) or a nested tree (shaped like stack_shape) of them
    """Convert a uniform Tucker tensor train back to ragged form.

    Unstacked: one :py:class:`~t3toolbox.tucker_tensor_train.TuckerTensorTrain`. Stacked: a nested tree
    of them (a varying-rank stack has no single stacked ``TuckerTensorTrain``;
    ``docs/uniform_ranks_and_varieties.md``).
    """
    result = ut3_conversions.ut3_to_t3(ux.data)

    def _wrap(res):
        if common.is_ndarray(res[0][0]):   # res = (tucker_cores, tt_cores) leaf
            return t3.TuckerTensorTrain(*res)
        return tuple(_wrap(r) for r in res)

    return _wrap(result)
