# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""The transformation laws of the sampling types: a test helper (not a test file), on raw backend arrays.

A function declared of a type must commute with the laws of that type: move its arguments by their laws, and its
result moves by the law of its result. Two laws (``DATA_METRIC_PLAN.md`` 9.3 in the graded-messages notes;
``docs/glossary.md``, Symbols, for the letters):

- **The jet law** ``s_j -> ell_j s_j`` (:py:class:`JetLaw`): slot ``k`` of a jet (``P``: ``z``, ``y``, ``r``,
  ``Jp``, the input jets on their own sizes) times ``ell^k = prod_j ell_j**k_j``; of an element of ``P*`` (``c``,
  ``M r``, the weight ``omega``) times ``ell^-k``; of an element of ``P*`` in the engine's turned form times
  ``ell^-(n-k)``, ``n`` the jet shape. The parameters ``X``, ``X*`` and the inputs' modes do not move.
- **The input law** ``w_i -> Delta_i w_i`` (:py:class:`InputLaw`, ``Delta_i`` positive, of shape ``(N_i,)``), with
  every represented value fixed: the inputs times ``Delta_i``; a Tucker core (``X`` on the mode axis) times
  ``Delta_i^-1``, its dual (``X*``) times ``Delta_i``; a probe measurement (``P (x) V_i*``) times ``Delta_i^-1``, an
  element of ``P* (x) V_i`` (``c``, ``M r``, and a weight with a full edge) times ``Delta_i``; an entries measurement
  divided by ``prod_i Delta_i[n_i]``, an entries ``c`` multiplied by it; a frame's down store ``D_i`` times
  ``Delta_i^-1`` and a tangent's ``V`` blocks (``X*``) times ``Delta_i``, the TT stores and blocks fixed; the ambient
  transposes' output (``E*``, CP factors) compared as the dense tensor they sum to, times ``Delta_0 (x) ... (x)
  Delta_(d-1)``.

Both are homogeneity tests: necessary, not sufficient (a wrong constant, sign or binomial factor passes them); the
dense oracles stay the primary tests. They hold for the metric-free maps only (forwards, tangents, transposes, ``M``),
not for an orthonormal frame, a projection or a retraction. :py:func:`law_error` measures how far a function is from a
law; :py:func:`assert_law` asserts it is within rounding.

The laws of the 42 transposes and second-order transposes (``c`` in ``Y``) are tested in ``test_transposes_typed.py``
(M4); the packed movers there pad ``Delta`` with ones, the padding the masks zero.
"""
import numpy as np


def _leaves(value):
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def _map_arrays(function, value):
    """``function`` on each array of an object of the type of ``z`` (each mode of a tuple or a list, or the array)."""
    if isinstance(value, (tuple, list)):
        return tuple(function(i, np.asarray(v)) for i, v in enumerate(value))
    return function(None, np.asarray(value))


class JetLaw:
    """``s_j -> ell_j s_j``, one ``ell_j`` per jet variable of the jet shape ``order``."""

    def __init__(self, order, ell):
        self.order, self.ell = tuple(order), tuple(float(e) for e in ell)
        self.m = len(self.order)
        if len(self.ell) != self.m:
            raise ValueError('JetLaw: one ell per jet variable: %d for %s' % (len(self.ell), self.order))
        if not any(n >= 1 for n in self.order):
            raise ValueError('JetLaw: the law is vacuous unless some order is at least 1, got %s' % (self.order,))
        if any(e == 1.0 or e <= 0.0 for e in self.ell) or len(set(self.ell)) != self.m:
            raise ValueError('JetLaw: each ell must be positive, not 1, and distinct per variable, got %s' % (self.ell,))

    def _factors(self, sizes, power, turned=False):
        """``prod_j ell_j**(power * k_j)`` (``k_j -> n_j - k_j`` turned) on the slots of ``sizes``."""
        out = np.ones(sizes)
        for idx in np.ndindex(*sizes):
            k = tuple((n - a) if turned else a for a, n in zip(idx, self.order))
            out[idx] = np.prod([e ** (power * kj) for e, kj in zip(self.ell, k)])
        return out

    def _scale(self, value, power, lead, turned=False):
        def one(_, array):
            sizes = array.shape[lead:lead + self.m]
            factors = self._factors(sizes, power, turned)
            return array * factors.reshape((1,) * lead + sizes + (1,) * (array.ndim - lead - self.m))
        return _map_arrays(one, value)

    def P(self, value, lead=0):
        """A jet (``z``, ``y``, ``r``, ``Jp``, an input jet on its own sizes): slot ``k`` times ``ell^k``. ``lead`` is
        the number of axes before the jet axes (1 for a probe array whose first axis is the mode)."""
        return self._scale(value, 1, lead)

    def P_star(self, value, lead=0):
        """An element of ``P*`` in slot coordinates (``c``, ``M r``, the weight): slot ``k`` times ``ell^-k``."""
        return self._scale(value, -1, lead)

    def P_star_turned(self, value, lead=0):
        """An element of ``P*`` in the engine's turned form (``w_tilde``): slot ``k`` times ``ell^-(n-k)``."""
        return self._scale(value, -1, lead, turned=True)

    def inputs(self, ww):
        """The input jets, each on its own jet sizes: slot ``k`` times ``ell^k``."""
        return self.P(ww)


class InputLaw:
    """``w_i -> Delta_i w_i``, ``Delta_i`` positive of shape ``(N_i,)``, every represented value fixed."""

    def __init__(self, deltas):
        self.deltas = tuple(np.asarray(delta, dtype=float) for delta in deltas)
        if any(delta.ndim != 1 or (delta <= 0).any() for delta in self.deltas):
            raise ValueError('InputLaw: one positive vector per mode')

    def _edge(self, value, power):
        return _map_arrays(lambda i, array: array * self.deltas[i] ** power, value)

    def inputs(self, ww):
        """The input jets: ``Delta_i w_i`` on the mode edge."""
        return self._edge(ww, 1)

    def V_star(self, value):
        """A probe measurement (``z``, ``y``, ``r``, ``Jp``: ``P (x) V_i*``): ``Delta_i^-1`` on the edge."""
        return self._edge(value, -1)

    def V(self, value):
        """An element of ``P* (x) V_i`` (``c``, ``M r``) or a probe weight with a full edge: ``Delta_i`` on the edge."""
        return self._edge(value, 1)

    def tucker(self, x):
        """A point ``(B, G)``: the Tucker cores ``B_i Delta_i^-1`` (``X`` on the mode axis), the TT cores as given."""
        return tuple(B / delta for B, delta in zip(x[0], self.deltas)), tuple(x[1])

    def tucker_dual(self, blocks):
        """Tucker-core blocks of ``X*`` (a gradient's): ``g_B Delta_i``."""
        return tuple(g * delta for g, delta in zip(blocks, self.deltas))

    def frame(self, frame):
        """A frame ``(D, U, L, R)``: the down store ``D_i Delta_i^-1`` (``X`` on the mode axis), the TT stores fixed."""
        return (tuple(D / delta for D, delta in zip(frame[0], self.deltas)),) + tuple(tuple(s) for s in frame[1:4])

    def variations_dual(self, blocks):
        """A tangent's blocks of ``X*`` ``(V, H)`` (a transpose's output at a frame): ``V_i Delta_i``, ``H`` fixed."""
        return tuple(V * delta for V, delta in zip(blocks[0], self.deltas)), tuple(blocks[1])

    def packed_mode_dual(self, block):
        """A packed block of ``X*`` on the mode axis, ``(d,) + ... + (N,)`` (a uniform Tucker or ``V`` block): times
        ``Delta_i`` padded with ones to ``N`` (the padding is zero by the masks, so its factor does not matter)."""
        block = np.asarray(block)
        N = block.shape[-1]
        pad = np.ones((len(self.deltas), N))
        for i, delta in enumerate(self.deltas):
            pad[i, :len(delta)] = delta
        return block * pad.reshape((len(self.deltas),) + (1,) * (block.ndim - 2) + (N,))

    def entries_dual(self, c, index):
        """An entries ``c`` (``Y``, dual to the entries measurement): times ``prod_i Delta_i[n_i]``."""
        index = np.asarray(index)
        factor = np.prod([delta[index[i]] for i, delta in enumerate(self.deltas)], axis=0)        # W
        c = np.asarray(c)
        return c * factor.reshape(factor.shape + (1,) * (c.ndim - factor.ndim))

    def ambient(self, tensor):
        """A dense tensor of ``E*`` (an ambient transpose's CP factors, summed), its last ``d`` axes the modes: times
        ``Delta_0 (x) ... (x) Delta_(d-1)``."""
        tensor = np.asarray(tensor)
        d = len(self.deltas)
        for i, delta in enumerate(self.deltas):
            tensor = tensor * delta.reshape((len(delta),) + (1,) * (d - 1 - i))
        return tensor

    def entries(self, z, index):
        """An entries measurement at the integer ``index`` (``(d,) + W``): divided by ``prod_i Delta_i[n_i]``."""
        index = np.asarray(index)
        factor = np.prod([delta[index[i]] for i, delta in enumerate(self.deltas)], axis=0)        # W
        z = np.asarray(z)
        return z / factor.reshape(factor.shape + (1,) * (z.ndim - factor.ndim))


def law_error(function, args, moved_args, move_out):
    """The largest relative error, leaf by leaf, between ``move_out(function(*args))`` and ``function(*moved_args)``."""
    expected, got = _leaves(move_out(function(*args))), _leaves(function(*moved_args))
    if len(expected) != len(got):
        raise AssertionError('law_error: %d leaves against %d' % (len(expected), len(got)))
    error = 0.0
    for a, b in zip(expected, got):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.shape != b.shape:
            raise AssertionError('law_error: shapes %s and %s' % (a.shape, b.shape))
        error = max(error, float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)))
    return error


def assert_law(test, function, args, moved_args, move_out, rtol=1e-12):
    """Assert, in the ``unittest.TestCase`` ``test``, that ``function`` follows the law to ``rtol``."""
    test.assertLessEqual(law_error(function, args, moved_args, move_out), rtol)
