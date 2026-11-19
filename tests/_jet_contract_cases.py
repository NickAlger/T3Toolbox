# Authors: Nick Alger and Blake Christierson
# Copyright: MIT License (2026)
# Github: https://github.com/NickAlger/T3Toolbox
# Documentation: https://nickalger.github.io/T3Toolbox/index.html
"""Test-only: the 29 literal ``jet_contract`` strings of the sampling terms, the operand shapes the library
gives them, and a reference that uses neither ``contract`` nor ``jet_contract``.

A variable has the shape ``J + W + [K] + C + (size,)``:

    J   the jet axes: m of them, one per jet variable, with sizes of the operand's own (a full jet has
        order + 1 on each; a leaf or an input has its own, two for a line); () at order ()
    W   the sample stack, K the stack of directions (on the one operand that carries it),
    C   the stack of base points (on every core and every variable, not on an input)

``STRINGS`` holds the strings by family, each as it stands at a call site. Every string is written for ONE
site's arrays, which is what a step sees on both layouts: the site axis of the uniform layout is not a group
of these strings. A string no longer shows which operand is a leaf; ``LEAF`` says it, and
``library_shapes`` gives the jet sizes the library passes.

The reference (``reference``) shares no mechanism with ``jet_contract``: every stack group of an operand is
flattened to ONE axis, and the truncated product is explicit loops over the multi-indices of every jet
operand, with one fixed lowercase-and-group-letter ``np.einsum`` per combination, accumulated at the sum of
the multi-indices where it lies in the box (or where it equals the corner, for a readout).

Imported by ``test_jet_contract.py``, ``test_contractions_interpreter.py``,
``test_contractions_sharding.py`` and ``test_dispatch.py``. Numpy only.
"""
import itertools

import numpy as np

# family -> its strings. The operands are in the order (xi: w or a seed, D) (z: eta, D) (mu: mu, G, xi)
# (nu: G, xi, nu) (eta: mu, G, nu) (V: eta, w or a seed) (H: mu, xi, nu).
STRINGS = {
    'xi':  ['JWo,Cio->JWCi',                'JWo,KCio->JWKCi',                  # an input
            'JWCo,Cio->JWCi',               'JWKCo,Cio->JWKCi',                 'JWCo,KCio->JWKCi'],    # a seed
    'z':   ['JWCi,Cio->JWCo',               'JWKCi,Cio->JWKCo',                 'JWCi,KCio->JWKCo'],
    'mu':  ['JWCa,Caib,JWCi->JWCb',         'JWKCa,Caib,JWCi->JWKCb',           'JWCa,KCaib,JWCi->JWKCb',
            'JWCa,Caib,JWKCi->JWKCb'],                                          # the leaf xi: two slots, or full
    'nu':  ['Caib,JWCi,JWCb->JWCa',         'Caib,JWCi,JWKCb->JWKCa',           'KCaib,JWCi,JWCb->JWKCa',
            'Caib,JWKCi,JWCb->JWKCa'],
    'eta': ['JWCa,Caib,JWCb->JWCi',         'JWKCa,Caib,JWCb->JWKCi',           'JWCa,KCaib,JWCb->JWKCi',
            'JWCa,Caib,JWKCb->JWKCi'],
    'V':   ['JWCi,JWo->Cio',                'JWKCi,JWo->KCio',                  # an input
            'JWCi,JWCo->Cio',               'JWKCi,JWCo->KCio',                 'JWCi,JWKCo->KCio'],    # a seed
    'H':   ['JWCa,JWCi,JWCb->Caib',         'JWKCa,JWCi,JWCb->KCaib',           'JWCa,JWKCi,JWCb->KCaib',
            'JWCa,JWCi,JWKCb->KCaib'],
}
ALL_STRINGS = [(family, string) for family, strings in STRINGS.items() for string in strings]

# the operand that is a leaf (mu, nu, H: the leaf xi, of two slots for a line input or full for a tilde
# leaf) or a sample vector (xi, V: an input of its own sizes where its term has no C; a full seed where it has)
LEAF = {'xi': 0, 'mu': 2, 'nu': 1, 'V': 1, 'H': 1}

SIZE = dict(a=2, i=3, b=4, o=5, c=6, j=7)  # the single axes; all different, so a transposed axis is a shape error


def input_terms(string):
    return string.split('->')[0].split(',')


def output_term(string):
    return string.split('->')[1]


def jet_operands(string):
    """The positions of the operands whose term has J."""
    return [p for p, t in enumerate(input_terms(string)) if 'J' in t]


def is_product(string):
    """Two or more jet operands and J in the output: a truncated product."""
    return 'J' in output_term(string) and len(jet_operands(string)) >= 2


def is_readout(string):
    """No J in the output: the corner coefficient."""
    return 'J' not in output_term(string)


def box_of(order):
    return tuple(n + 1 for n in order)


def library_shapes(family, string, order):
    """The jet sizes the library passes for this string at the jet shape ``order``, with a line input:
    ``[(label, {position: jet sizes}, the output's jet shape), ...]``. A leaf of a line has two slots per jet
    variable (clipped to the box, as the engine clips it); a tilde leaf or a seed is full; an input rides to a
    box of its own (its jet sizes clipped to the full box)."""
    box = box_of(order)
    full = {p: box for p in jet_operands(string)}
    line = tuple(min(2, f) for f in box)
    leaf = LEAF.get(family)
    if leaf is None or (family in ('xi', 'V') and 'C' in input_terms(string)[leaf]):
        return [('full', full, order)]
    if family == 'xi':
        return [('line input', {0: line}, tuple(f - 1 for f in line))]
    if family == 'V':
        return [('line input', {**full, leaf: line}, order)]
    return [('leaf of a line', {**full, leaf: line}, order), ('full leaf', full, order)]


def make(rng, shape_of, term, jet_sizes):
    """A random operand for one term: jet_sizes for J (absent at order ()), shape_of for W, K, C."""
    shape = ()
    for ch in term:
        if ch == 'J':
            shape += jet_sizes
        elif ch.isupper():
            shape += shape_of[ch]
        else:
            shape += (SIZE[ch],)
    return rng.standard_normal(shape)


def build(string, sizes, shape_of, rng):
    """Random operands for one string. sizes: {position: jet sizes} for every operand with J (any jet sizes; empty
    tuples, or no entry, at order ())."""
    return [make(rng, shape_of, term, tuple(sizes.get(p, ()))) for p, term in enumerate(input_terms(string))]


def shape_of_stacks(string, W, K, C):
    """The stacks of one call: K is empty where the string has no direction."""
    return dict(W=W, K=K if 'K' in output_term(string) else (), C=C)


def _flattened(x, term, m, shape_of):
    """x with each stack group flattened to one axis (the jet axes kept, unflattened, in front), and the
    einsum letters of that flattened array: each stack group by its own letter, the jet group dropped."""
    shape, new_shape, letters, pos = x.shape, (), '', 0
    for ch in term:
        if ch == 'J':
            new_shape, pos = new_shape + shape[pos:pos + m], pos + m
        elif ch.isupper():
            n = len(shape_of[ch])
            new_shape, pos, letters = new_shape + (int(np.prod(shape[pos:pos + n])),), pos + n, letters + ch
        else:
            new_shape, pos, letters = new_shape + (shape[pos],), pos + 1, letters + ch
    return x.reshape(new_shape), letters


def reference(string, arrays, order, shape_of):
    """The value ``jet_contract(string, *arrays, order=order, ...)`` must return, in the shape it must have.
    Float64. By definition: every combination of the jet operands' multi-indices (all of their slots, beyond the
    box too) is contracted with one fixed einsum, and added at the sum of the indices where it lies in the box
    (J in the output) or where it equals the corner (no J in the output)."""
    m = len(order)
    terms, out = input_terms(string), output_term(string)
    flats = [_flattened(np.asarray(x, dtype=np.float64), t, m, shape_of) for x, t in zip(arrays, terms)]
    out_letters = ''.join(ch for ch in out if ch != 'J')
    subscripts = ','.join(letters for _, letters in flats) + '->' + out_letters
    jets = jet_operands(string)
    keep = 'J' in out
    result = None
    first = None
    for ks in itertools.product(*(list(np.ndindex(*arrays[p].shape[:m])) for p in jets)):
        total = tuple(sum(k[j] for k in ks) for j in range(m))
        ops = [flat for flat, _ in flats]
        for p, k in zip(jets, ks):
            ops[p] = ops[p][k]
        if first is None:
            first = np.zeros_like(np.einsum(subscripts, *ops))
        if (keep and any(t > n for t, n in zip(total, order))) or (not keep and total != tuple(order)):
            continue
        value = np.einsum(subscripts, *ops)
        if result is None:
            result = np.zeros(box_of(order) + value.shape) if keep else np.zeros(value.shape)
        if keep:
            result[total] += value
        else:
            result += value
    if result is None:                      # nothing reaches the box or the corner
        result = np.zeros(box_of(order) + first.shape) if keep else first
    shape = ()
    for ch in out:
        if ch == 'J':
            shape += box_of(order)
        elif ch.isupper():
            shape += shape_of[ch]
        else:
            shape += (SIZE[ch],)
    return result.reshape(shape)


def n_contractions(string, sizes, order, form):
    """How many ``contract`` calls ``jet_contract`` makes on numpy (where a scan is a Python loop: one call for the
    first coefficient and one per step). A regression pin of the rule as built: it restates the choice of the
    carrier, the pair and the scanned partner, the skip of a combination whose pair ranges are empty, and the one
    call of the zeros, so it is not independent evidence. The independent pin of the counts is P1's table in
    ``test_one_variable_with_the_library_shapes_makes_the_contractions_of_today``. Structural."""
    m = len(order)
    if m == 0:
        return 1
    box, jets = box_of(order), jet_operands(string)
    clipped = {p: tuple(min(s, f) for s, f in zip(sizes[p], box)) for p in jets}
    n_slots = {p: int(np.prod(clipped[p])) for p in jets}
    ranked = sorted(jets, key=lambda p: (-n_slots[p], p))
    keep = 'J' in output_term(string)
    if keep and len(jets) == 1:
        return 1                                            # a ride
    if not keep and len(jets) == 1:
        return 1                                            # a lone readout: one slice (or zeros)
    lead = ranked[:1] if keep else ranked[:2]               # the carrier, or the pair
    taken = [p for p in jets if p not in lead]
    scanned = None
    if form == 'scan' and m == 1:
        scanned = next((p for p in taken if clipped[p][0] == box[0]), None)
    looped = [p for p in taken if p != scanned]
    count = 0
    for ks in itertools.product(*(list(np.ndindex(*clipped[p])) for p in looped)):
        reach = tuple(sum(k[j] for k in ks) for j in range(m))
        if any(r > n for r, n in zip(reach, order)):
            continue
        if not keep and scanned is None:                    # the pair must meet the rest of the corner
            a, b = sorted(lead)
            corner = tuple(n - r for n, r in zip(order, reach))
            if any(max(0, c - sizes[b][j] + 1) > min(c, sizes[a][j] - 1) for j, c in enumerate(corner)):
                continue
        count += box[0] if scanned is not None else 1
    return max(count, 1)                                    # a corner nothing reaches: one call for the zeros


def relerr(x, ref):
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64) - ref) / max(np.linalg.norm(ref), 1e-300))
