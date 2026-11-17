# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Test-only: the 41 literal ``jet_contract`` strings of the sampling terms, and a reference for them
that uses neither ``contract`` nor ``jet_contract``.

A variable has the shape ``T + W + [K] + C + (size,)``:

    T   the order axis of a full jet: (order+1,), or () at order 0
    A   the order axis of an affine variable (an input, a leaf): (2,), or () at order 0
    W   the sample stack, K the stack of directions (on the one operand that carries it),
    C   the stack of base points (on every core and every variable, not on an input)

``STRINGS`` holds the strings by family, each as it stands at a call site. Every string is written
for ONE site's arrays, which is what a step sees on both layouts: the site axis of the uniform layout
is not a group of these strings.

The reference (``reference``) shares no mechanism with the interpreter: every operand is expanded to
all the stacks (``full``), each stack flattened to ONE axis, the order padded with zeros, and the
truncated products are explicit loops over pairs of orders with a fixed lowercase ``np.einsum``.

Imported by ``test_jet_contract.py``, ``test_contractions_interpreter.py``,
``test_contractions_sharding.py`` and ``test_dispatch.py``. Numpy only.
"""
import itertools

import numpy as np

# family -> its strings. The operands are in the order (xi: w, D) (z: eta, D) (mu: mu, G, xi)
# (nu: G, xi, nu) (eta: mu, G, nu) (V: eta, w) (H: mu, xi, nu).
STRINGS = {
    'xi':  ['AWo,Cio->AWCi',                'AWo,KCio->AWKCi',                  # an input
            'TWCo,Cio->TWCi',               'TWKCo,Cio->TWKCi',                 'TWCo,KCio->TWKCi'],    # a seed
    'z':   ['TWCi,Cio->TWCo',               'TWKCi,Cio->TWKCo',                 'TWCi,KCio->TWKCo'],
    'mu':  ['TWCa,Caib,AWCi->TWCb',         'TWKCa,Caib,AWCi->TWKCb',           'TWCa,KCaib,AWCi->TWKCb',
            'TWCa,Caib,AWKCi->TWKCb',                                           # an affine leaf
            'TWCa,Caib,TWCi->TWCb',         'TWKCa,Caib,TWCi->TWKCb',           'TWCa,KCaib,TWCi->TWKCb',
            'TWCa,Caib,TWKCi->TWKCb'],                                          # a full leaf (a tilde one)
    'nu':  ['Caib,AWCi,TWCb->TWCa',         'Caib,AWCi,TWKCb->TWKCa',           'KCaib,AWCi,TWCb->TWKCa',
            'Caib,AWKCi,TWCb->TWKCa',
            'Caib,TWCi,TWCb->TWCa',         'Caib,TWCi,TWKCb->TWKCa',           'KCaib,TWCi,TWCb->TWKCa',
            'Caib,TWKCi,TWCb->TWKCa'],
    'eta': ['TWCa,Caib,TWCb->TWCi',         'TWKCa,Caib,TWCb->TWKCi',           'TWCa,KCaib,TWCb->TWKCi',
            'TWCa,Caib,TWKCb->TWKCi'],
    'V':   ['TWCi,AWo->Cio',                'TWKCi,AWo->KCio',                  # an input
            'TWCi,TWCo->Cio',               'TWKCi,TWCo->KCio',                 'TWCi,TWKCo->KCio'],    # a seed
    'H':   ['TWCa,AWCi,TWCb->Caib',         'TWKCa,AWCi,TWCb->KCaib',           'TWCa,AWKCi,TWCb->KCaib',
            'TWCa,AWCi,TWKCb->KCaib',
            'TWCa,TWCi,TWCb->Caib',         'TWKCa,TWCi,TWCb->KCaib',           'TWCa,TWKCi,TWCb->KCaib',
            'TWCa,TWCi,TWKCb->KCaib'],
}
ALL_STRINGS = [(family, string) for family, strings in STRINGS.items() for string in strings]

SIZE = dict(a=2, i=3, b=4, o=5)     # the single axes; all different, so a transposed axis is a shape error


def input_terms(string):
    return string.split('->')[0].split(',')


def output_term(string):
    return string.split('->')[1]


def n_order_operands(string):
    """How many operands have an order group."""
    return sum('T' in t or 'A' in t for t in input_terms(string))


def is_full_product(string):
    """Two full jets and T in the output: the one case where the two forms over the orders differ."""
    return 'T' in output_term(string) and sum('T' in t for t in input_terms(string)) == 2


def follows_the_form(string):
    """A full jet is taken one coefficient at a time: a product of two full jets, or a readout of three. The
    cases where ``over_orders`` is about something ('scan' scans, 'loop' loops with slices)."""
    n_full = sum('T' in t for t in input_terms(string))
    return is_full_product(string) or (not ('T' in output_term(string) or 'A' in output_term(string)) and n_full >= 3)


def is_product(string):
    """Two operands with an order group and T in the output: a truncated product."""
    return 'T' in output_term(string) and n_order_operands(string) == 2


def n_contractions(string, order):
    """How many ``contract`` calls ``jet_contract`` makes for this string (structural; both forms)."""
    if order == 0 or n_order_operands(string) == 1:
        return 1                                    # no order axis, or it rides
    terms = input_terms(string)
    n_full, n_affine = sum('T' in t for t in terms), sum('A' in t for t in terms)
    if 'T' in output_term(string):
        return 2 if n_affine else order + 1         # one per coefficient of the partner
    if n_full >= 2:                                 # a readout: two full jets are contracted along the order
        looped = [2] * n_affine + [order + 1] * (n_full - 2)
        return sum(1 for taken in itertools.product(*(range(n) for n in looped)) if sum(taken) <= order)
    return 2 if n_affine else 1


def spec_of(term):
    """'TWKCa' -> ('TWKC', 'a', affine?), with A read as the order."""
    groups = ''.join('T' if ch == 'A' else ch for ch in term if ch.isupper())
    return groups, ''.join(ch for ch in term if ch.islower()), 'A' in term


def make(rng, shape_of, groups, core, order_len=None):
    """A random operand. groups: letters among T W K C present on it, in that order; order_len: the length
    of its order axis when it has one (2 for an affine variable)."""
    shape = ()
    for g in groups:
        shape += (order_len,) if g == 'T' else shape_of[g]
    return rng.standard_normal(shape + tuple(SIZE[c] for c in core))


def build(string, order, shape_of, rng):
    """Random operands for one string at one order.

    Returns ``(arrays, operands)``: the arrays ``jet_contract`` is called with, and the same with their
    groups and core letters, ``[(array, groups, core), ...]``, as ``reference`` takes them."""
    arrays, operands = [], []
    for term in input_terms(string):
        groups, core, affine = spec_of(term)
        groups_here = groups if order else groups.replace('T', '')
        x = make(rng, shape_of, groups_here, core, 2 if affine else order + 1)
        arrays.append(x)
        operands.append((x, groups_here, core))
    return arrays, operands


def shape_of_stacks(string, W, K, C):
    """The stacks of one call: K is empty where the string has no direction."""
    return dict(W=W, K=K if 'K' in output_term(string) else (), C=C)


def full(x, shape_of, groups, n_slots):
    """x on every stack: (t, w, k, c) + core, each stack flattened to one axis, the order padded with zeros."""
    shape = x.shape
    out_axes, pos = [], 0
    for g in 'TWKC':
        n_axes = (1 if g in groups else 0) if g == 'T' else (len(shape_of[g]) if g in groups else 0)
        dims = shape[pos:pos + n_axes]
        out_axes.append(int(np.prod(dims)) if n_axes else 1)
        pos += n_axes
    x = x.reshape(tuple(out_axes) + shape[pos:])
    if x.shape[0] < n_slots:
        x = np.concatenate([x, np.zeros((n_slots - x.shape[0],) + x.shape[1:])], axis=0)
    target = [n_slots] + [int(np.prod(shape_of[g])) for g in 'WKC']
    return np.broadcast_to(x, tuple(target) + x.shape[4:])


def _reference_full(family, operands, shape_of, n_slots):
    """The term by explicit loops, on every stack. operands: [(array, groups, core letters), ...]."""
    fulls = [full(np.asarray(x, dtype=np.float64), shape_of, groups, n_slots) for x, groups, core in operands]
    stack = 'wkc'
    if family in ('xi', 'z'):                                           # the order rides on the first operand
        sub = {'xi': 'twkco,wkcio->twkci', 'z': 'twkci,wkcio->twkco'}[family]
        return np.einsum(sub, fulls[0], fulls[1][0])
    if family in ('mu', 'nu', 'eta'):
        sub = {'mu': '%sa,%saib,%si->%sb', 'nu': '%saib,%si,%sb->%sa', 'eta': '%sa,%saib,%sb->%si'}[family] % ((stack,) * 4)
        jets = {'mu': (0, 2), 'nu': (1, 2), 'eta': (0, 2)}[family]
        core_pos = ({0, 1, 2} - set(jets)).pop()
        out = None
        for i, r in itertools.product(range(n_slots), repeat=2):
            if i + r >= n_slots:
                continue
            ops = [None, None, None]
            ops[jets[0]], ops[jets[1]], ops[core_pos] = fulls[jets[0]][i], fulls[jets[1]][r], fulls[core_pos][0]
            val = np.einsum(sub, *ops)
            if out is None:
                out = np.zeros((n_slots,) + val.shape)
            out[i + r] += val
        return out                                                      # (t, w, k, c, size)
    if family == 'V':
        out = 0.0
        for i in range(n_slots):
            out = out + np.einsum('wkci,wkco->kcio', fulls[0][i], fulls[1][n_slots - 1 - i])
        return out
    assert family == 'H'
    out = 0.0
    for i, j in itertools.product(range(n_slots), repeat=2):
        if i + j < n_slots:
            out = out + np.einsum('wkca,wkci,wkcb->kcaib', fulls[0][i], fulls[1][j], fulls[2][n_slots - 1 - i - j])
    return out


def reference(family, string, operands, order, shape_of):
    """The value ``jet_contract(string, ...)`` must return, in the shape it must have. Float64."""
    n_slots = order + 1
    ref = _reference_full(family, operands, shape_of, n_slots)
    out_groups, out_core, out_affine = spec_of(output_term(string))
    if family in ('xi', 'z', 'mu', 'nu', 'eta'):                        # (t, w, k, c) + core: keep the output's slots
        ref = ref[:2 if out_affine else n_slots] if order else ref[0]
    shape = ()
    for g in out_groups:
        if g == 'T':
            shape += ((2 if out_affine else n_slots),) if order else ()
        else:
            shape += shape_of[g]
    return ref.reshape(shape + tuple(SIZE[c] for c in out_core))


def relerr(x, ref):
    return float(np.linalg.norm(np.asarray(x) - ref) / max(np.linalg.norm(ref), 1e-300))
