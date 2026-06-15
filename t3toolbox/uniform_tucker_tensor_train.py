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
