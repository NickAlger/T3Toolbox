"""Target decision 5.4: the terms generic over every stack and over the order, with a literal string at each place.

A variable has the shape  [S] + T + W + [K] + C + (size,):
    S   the site group: (d,) when a map-type operation is done at all sites at once on the uniform layout, else ()
    T   the optional order axis of a full jet: (order+1,), or () at order 0
    A   the optional order axis of an affine variable (the input w and the leaves xi, xi_dot, xi_U_dot): (2,) or ()
    W   the sample stack: any number of axes
    K   the stack of directions: any number of axes, on the one operand of a term that carries the direction
    C   the stack of base points: any number of axes, on every core and every variable (not on the input w)

DECIDED (the maintainer, 2026-11-16, 16:24 and 16:26).  Each place a term is used has its own literal
contraction string, with K written on the operand that carries it, "so the code is more self-documenting".
The strings stand at the call sites in the step functions, through `jet_contract`, and every call passes both
`len_T` and `len_C`, so that an operand of the wrong rank raises:

    def mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H, len_T, len_C, over_orders):
        t1 = jet_contract('TWKCa,Caib,AWCi->TWKCb', mu_dot, R, xi,     len_T=len_T, len_C=len_C, over_orders=over_orders)
        t2 = jet_contract('TWCa,KCaib,AWCi->TWKCb', mu,     H, xi,     len_T=len_T, len_C=len_C, over_orders=over_orders)
        t3 = jet_contract('TWCa,Caib,AWKCi->TWKCb', mu,     U, xi_dot, len_T=len_T, len_C=len_C, over_orders=over_orders)
        return t1 + t2 + t3

`jet_contract` is the library's `contract` with one more rule: the order groups T and A multiply as truncated
polynomials (decision 2.9).
    one operand has an order group, and so has the output      the axis rides: one contraction
    two operands have one, and the output has T                the truncated product: two contractions for an
                                                               affine partner; 'loop' or 'scan' for a full one
    the output has no order group                              the top coefficient of the product, which is what
                                                               the block readouts are
At order 0 both groups are empty and `jet_contract` is exactly one `contract` call.

The seven terms are seven families of strings, written out below (STRINGS) as they would stand at their call
sites.  The site group S is on the strings of the map-type operations (xi, z, eta and the readouts) and not on
those of the sweeps (mu, nu), which go site by site.

Checked against a reference that uses neither `contract` nor `jet_contract`: every operand is expanded to all
the stacks, each stack flattened to one axis, and the truncated products are explicit loops over np.einsum.

Not here: the scan written through `xscan` (scripts/review/order_scan_check.py does that for W and C; here
'scan' is the same carried shift as a Python loop); the plan's contraction order (5.5); masks; the readouts with
the samples kept (3.7).  The first version of this script used one string per term with the direction's place
solved from the ranks (log entry 90); that form was not taken.

Usage:  python3 -I term_stacks_check.py /path/to/T3Toolbox
"""
import collections
import itertools
import sys

import numpy as np

contract = get_backend = tree_contains_jax = None       # bound by load_library, as in order_scan_check.py


def load_library(repo):
    global contract, get_backend, tree_contains_jax
    sys.path.insert(0, repo)
    from t3toolbox.backend import common
    from t3toolbox.backend import contractions
    contract, get_backend, tree_contains_jax = contractions.contract, common.get_backend, common.tree_contains_jax


N_CONTRACTIONS = [0]                                    # counted by the checks below


# ---- the order axis of one operand -----------------------------------------------------------------------------
def _slot(x, r, axis):                      # coefficient r; the order axis is gone
    return x[(slice(None),) * axis + (r,)]


def _slots(x, start, stop, axis):           # coefficients start .. stop-1; the order axis is kept
    return x[(slice(None),) * axis + (slice(start, stop),)]


def _turned(x, axis):                       # the order axis reversed
    return x[(slice(None),) * axis + (slice(None, None, -1),)]


def _moved_up(x, r, n_slots, axis):         # x in slots r .. of n_slots, zeros elsewhere, in x's dtype
    xnp = get_backend(False, tree_contains_jax((x,)))[0]
    have = x.shape[axis]

    def zeros(n):
        return xnp.zeros(x.shape[:axis] + (n,) + x.shape[axis + 1:], x.dtype)
    parts = ([zeros(r)] if r else []) + [x] + ([zeros(n_slots - r - have)] if n_slots - r - have else [])
    return parts[0] if len(parts) == 1 else xnp.concatenate(parts, axis=axis)


def _without_order(term):                   # an operand's term once one coefficient has been taken
    return term.replace('T', '').replace('A', '')


# ---- contract, with the order groups multiplying as truncated polynomials ------------------------------------
def jet_contract(
        subscripts,             # a grouped string.  T: the order axis of a full jet.  A: of an affine variable
        *operands,
        len_T,                  # 1 if order else 0.  The axis count of T, and of A
        len_C,                  # the axis count of the stack of base points
        over_orders='scan',     # 'scan' | 'loop': how a product of two FULL jets is carried out (decision 2.9)
        **group_lens,           # further len_<G>: len_S where the string has the site group; len_W, len_K for strictness
):
    terms, out = subscripts.replace(' ', '').split('->')
    terms = terms.split(',')
    len_S = group_lens.get('len_S', 0)

    def one_contraction(terms_, operands_):
        sub = ','.join(terms_) + '->' + out
        lens = {key: n for key, n in group_lens.items() if key[4] in sub}
        lens.update({'len_' + g: n for g, n in (('T', len_T), ('A', len_T), ('C', len_C)) if g in sub})
        N_CONTRACTIONS[0] += 1
        return contract(sub, *operands_, **lens)

    jets = [p for p, term in enumerate(terms) if 'T' in term or 'A' in term]
    out_has_order = 'T' in out or 'A' in out
    if len_T == 0 or (len(jets) == 1 and out_has_order):        # no order axis, or it rides: one contraction
        return one_contraction(terms, operands)

    for p in jets:                                              # the order axis leads, after the site group
        assert terms[p].lstrip('S')[0] in 'TA', 'the order group must come first, after S: %r' % terms[p]
    axis = {p: (len_S if terms[p][0] == 'S' else 0) for p in jets}
    full_jets = [p for p in jets if 'T' in terms[p]]
    if not full_jets:
        raise ValueError('%r: a product over the orders needs an operand with T' % subscripts)
    n_slots = operands[full_jets[0]].shape[axis[full_jets[0]]]  # order + 1
    ops, tms = list(operands), list(terms)

    if 'T' in out:                                              # ---- the truncated product of two jets
        if len(jets) != 2:
            raise ValueError('%r: a product with T in the output has two operands with an order group' % subscripts)
        carrier = full_jets[0]                                  # carries the order axis through the contraction
        (partner,) = [p for p in jets if p != carrier]          # is taken one coefficient at a time
        a, b = operands[carrier], operands[partner]
        n_b = b.shape[axis[partner]]                            # 2 for an affine partner
        tms[partner] = _without_order(terms[partner])
        axis_out = len_S if out[0] == 'S' else 0
        result = None
        if n_b > 2 and over_orders == 'scan':                   # the carried jet moves up one slot at each step
            for r in range(n_b):
                ops[carrier], ops[partner] = a, _slot(b, r, axis[partner])
                part = one_contraction(tms, ops)
                result = part if result is None else result + part
                a = _moved_up(_slots(a, 0, n_slots - 1, axis[carrier]), 1, n_slots, axis[carrier])
            return result
        for r in range(n_b):                                    # an affine partner, or 'loop': slices
            ops[carrier], ops[partner] = _slots(a, 0, n_slots - r, axis[carrier]), _slot(b, r, axis[partner])
            part = _moved_up(one_contraction(tms, ops), r, n_slots, axis_out)
            result = part if result is None else result + part
        return result

    if out_has_order:
        raise ValueError('%r: a product of jets is a full jet; the output cannot have A' % subscripts)

    # ---- the top coefficient of the product: the orders of the operands sum to the top
    pair = full_jets[:2] if len(full_jets) >= 2 else None       # two full jets are contracted along the order axis
    alone = None if pair else full_jets[0]                      # a single full jet gives one slot
    looped = [p for p in jets if p not in (pair or [alone])]    # the rest: one coefficient at a time
    for p in looped:
        tms[p] = _without_order(terms[p])
    if alone is not None:
        tms[alone] = _without_order(terms[alone])
    result = None
    for taken in itertools.product(*(range(operands[p].shape[axis[p]]) for p in looped)):
        left = n_slots - 1 - sum(taken)                         # what the remaining operands must sum to
        if left < 0:
            continue
        for p, r in zip(looped, taken):
            ops[p] = _slot(operands[p], r, axis[p])
        if pair:
            ops[pair[0]] = _slots(operands[pair[0]], 0, left + 1, axis[pair[0]])
            ops[pair[1]] = _turned(_slots(operands[pair[1]], 0, left + 1, axis[pair[1]]), axis[pair[1]])
        else:
            ops[alone] = _slot(operands[alone], left, axis[alone])
        part = one_contraction(tms, ops)
        result = part if result is None else result + part
    return result


# ---- two step functions, as they would stand in the engine -----------------------------------------------------
def mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H, len_T, len_C, over_orders='scan'):
    t1 = jet_contract('TWKCa,Caib,AWCi->TWKCb', mu_dot, R, xi,     len_T=len_T, len_C=len_C, over_orders=over_orders)
    t2 = jet_contract('TWCa,KCaib,AWCi->TWKCb', mu,     H, xi,     len_T=len_T, len_C=len_C, over_orders=over_orders)
    t3 = jet_contract('TWCa,Caib,AWKCi->TWKCb', mu,     U, xi_dot, len_T=len_T, len_C=len_C, over_orders=over_orders)
    return t1 + t2 + t3


def V_tilde_step(eta_tilde, eta, w, w_tilde, len_S, len_T, len_C):
    t1 = jet_contract('STWCi,SAWo->SCio',  eta_tilde, w,       len_S=len_S, len_T=len_T, len_C=len_C)
    t2 = jet_contract('STWCi,STWCo->SCio', eta,       w_tilde, len_S=len_S, len_T=len_T, len_C=len_C)
    return t1 + t2


# ---- the seven families of strings, each as it would stand at a call site --------------------------------------
# family -> the strings; the operands are in the order (xi: w, D) (z: eta, D) (mu: mu, G, xi) (nu: G, xi, nu)
# (eta: mu, G, nu) (V: eta, w) (H: mu, xi, nu).
STRINGS = {
    'xi':  ['SAWo,SCio->SAWCi',                 'SAWo,SKCio->SAWKCi',               # an input
            'STWCo,SCio->STWCi',                'STWKCo,SCio->STWKCi',              'STWCo,SKCio->STWKCi'],     # a seed
    'z':   ['STWCi,SCio->STWCo',                'STWKCi,SCio->STWKCo',              'STWCi,SKCio->STWKCo'],
    'mu':  ['TWCa,Caib,AWCi->TWCb',             'TWKCa,Caib,AWCi->TWKCb',           'TWCa,KCaib,AWCi->TWKCb',
            'TWCa,Caib,AWKCi->TWKCb',                                               # an affine leaf
            'TWCa,Caib,TWCi->TWCb',             'TWKCa,Caib,TWCi->TWKCb',           'TWCa,KCaib,TWCi->TWKCb',
            'TWCa,Caib,TWKCi->TWKCb'],                                              # a full leaf (a tilde one)
    'nu':  ['Caib,AWCi,TWCb->TWCa',             'Caib,AWCi,TWKCb->TWKCa',           'KCaib,AWCi,TWCb->TWKCa',
            'Caib,AWKCi,TWCb->TWKCa',
            'Caib,TWCi,TWCb->TWCa',             'Caib,TWCi,TWKCb->TWKCa',           'KCaib,TWCi,TWCb->TWKCa',
            'Caib,TWKCi,TWCb->TWKCa'],
    'eta': ['STWCa,SCaib,STWCb->STWCi',         'STWKCa,SCaib,STWCb->STWKCi',       'STWCa,SKCaib,STWCb->STWKCi',
            'STWCa,SCaib,STWKCb->STWKCi'],
    'V':   ['STWCi,SAWo->SCio',                 'STWKCi,SAWo->SKCio',               # an input
            'STWCi,STWCo->SCio',                'STWKCi,STWCo->SKCio',              'STWCi,STWKCo->SKCio'],     # a seed
    'H':   ['STWCa,SAWCi,STWCb->SCaib',         'STWKCa,SAWCi,STWCb->SKCaib',       'STWCa,SAWKCi,STWCb->SKCaib',
            'STWCa,SAWCi,STWKCb->SKCaib',
            'STWCa,STWCi,STWCb->SCaib',         'STWKCa,STWCi,STWCb->SKCaib',       'STWCa,STWKCi,STWCb->SKCaib',
            'STWCa,STWCi,STWKCb->SKCaib'],
}


# ---- the reference: no `contract`, no helper above ----------------------------------------------------------
SIZE = dict(a=2, i=3, b=4, o=5)


def make(rng, shape_of, groups, core, order_len=None):
    """A random operand.  groups: letters among S T W K C present on it, in that order; order_len: length of its
    order axis when it has one (2 for an affine variable)."""
    shape = ()
    for g in groups:
        shape += (order_len,) if g == 'T' else shape_of[g]
    return rng.standard_normal(shape + tuple(SIZE[c] for c in core))


def full(x, shape_of, groups, n_core, n_slots):
    """x on every stack: (s, t, w, k, c) + core, each stack flattened to one axis, the order padded with zeros."""
    shape = x.shape
    out_axes, pos = [], 0
    for g in 'STWKC':
        n_axes = (1 if g in groups else 0) if g == 'T' else (len(shape_of[g]) if g in groups else 0)
        dims = shape[pos:pos + n_axes]
        out_axes.append(int(np.prod(dims)) if n_axes else 1)
        pos += n_axes
    x = x.reshape(tuple(out_axes) + shape[pos:])
    if x.shape[1] < n_slots:
        pad = np.zeros(x.shape[:1] + (n_slots - x.shape[1],) + x.shape[2:])
        x = np.concatenate([x, pad], axis=1)
    target = [int(np.prod(shape_of[g])) if g != 'T' else n_slots for g in 'STWKC']
    return np.broadcast_to(x, tuple(target) + x.shape[5:])


def reference(kind, operands, shape_of, n_slots):
    """The term by explicit loops.  operands: list of (array, groups, core letters)."""
    fulls = [full(x, shape_of, groups, len(core), n_slots) for x, groups, core in operands]
    stack = 'swkc'
    if kind in ('xi', 'z'):                                             # the order rides on the first operand
        (w_or_eta, core_arr) = fulls
        sub = {'xi': 'stwkco,swkcio->stwkci', 'z': 'stwkci,swkcio->stwkco'}[kind]
        return np.einsum(sub, w_or_eta, core_arr[:, 0])
    if kind in ('mu', 'nu', 'eta'):
        sub = {'mu': '%sa,%saib,%si->%sb', 'nu': '%saib,%si,%sb->%sa', 'eta': '%sa,%saib,%sb->%si'}[kind] % ((stack,) * 4)
        jets = {'mu': (0, 2), 'nu': (1, 2), 'eta': (0, 2)}[kind]
        core_pos = ({0, 1, 2} - set(jets)).pop()
        out = None
        for i, r in itertools.product(range(n_slots), repeat=2):
            if i + r >= n_slots:
                continue
            ops = [None, None, None]
            ops[jets[0]], ops[jets[1]], ops[core_pos] = fulls[jets[0]][:, i], fulls[jets[1]][:, r], fulls[core_pos][:, 0]
            val = np.einsum(sub, *ops)
            if out is None:
                out = np.zeros((n_slots,) + val.shape)
            out[i + r] += val
        return np.moveaxis(out, 0, 1)                                   # (s, t, w, k, c, size)
    if kind == 'V':
        out = 0.0
        for i in range(n_slots):
            out = out + np.einsum('swkci,swkco->skcio', fulls[0][:, i], fulls[1][:, n_slots - 1 - i])
        return out
    out = 0.0                                                           # 'H'
    for i, j in itertools.product(range(n_slots), repeat=2):
        if i + j < n_slots:
            out = out + np.einsum('swkca,swkci,swkcb->skcaib', fulls[0][:, i], fulls[1][:, j],
                                  fulls[2][:, n_slots - 1 - i - j])
    return out


def unflatten(ref, shape_of, out_groups, n_order_out):
    """The reference's (s, [t], [w], k, c) + core back to the shape the terms return."""
    shape = ()
    core = ref.shape[len('s' + ('t' if 'T' in out_groups or n_order_out else '') + ('w' if 'W' in out_groups else '')
                         + 'kc'):]
    for g in out_groups:
        shape += (n_order_out,) if g == 'T' else shape_of[g]
    return ref.reshape(shape + core)


def relerr(x, ref):
    return float(np.linalg.norm(np.asarray(x) - ref) / max(np.linalg.norm(ref), 1e-300))


def spec_of(term):                          # 'STWKCa' -> ('STWKC', 'a', affine?) with A read as the order
    groups = ''.join('T' if ch == 'A' else ch for ch in term if ch.isupper())
    return groups, ''.join(ch for ch in term if ch.islower()), 'A' in term


def check_all(rng):
    print('1. Every string of every family against the explicit reference, on every combination of stacks.')
    worst = collections.defaultdict(float)
    n_checked = 0
    stacks = list(itertools.product([(), (2,)], [(), (3,), (2, 2)], [(2,), (2, 2)], [(), (2,)]))   # S, W, K, C
    for order in (0, 1, 3):
        n_slots = order + 1
        for S, W, K, C in stacks:
            for kind, strings in STRINGS.items():
                for string in strings:
                    if S and 'S' not in string:
                        continue                                        # the sweeps go site by site
                    terms, out = string.split('->')
                    specs = [spec_of(t) for t in terms.split(',')]
                    has_K = 'K' in out
                    shape_of = dict(S=S, W=W, K=K if has_K else (), C=C)
                    arrays, operands = [], []
                    for groups, core, affine in specs:
                        groups_here = groups if order else groups.replace('T', '')
                        x = make(rng, shape_of, groups_here, core, 2 if affine else n_slots)
                        arrays.append(x)
                        operands.append((x, groups_here, core))
                    ref = reference(kind, operands, shape_of, n_slots)
                    out_groups, _, out_affine = spec_of(out)
                    n_out = (2 if out_affine else n_slots) if 'T' in out_groups else 0
                    if kind in ('xi', 'z', 'mu', 'nu', 'eta'):
                        ref = ref[:, 0] if not order else ref[:, :n_out]
                    ref = unflatten(ref, shape_of, out_groups if order else out_groups.replace('T', ''),
                                    n_out if order else 0)
                    lens = dict(len_T=1 if order else 0, len_C=len(C))
                    if 'S' in string:
                        lens['len_S'] = len(S)
                    full_product = order and 'T' in out and sum('T' in t for t in terms.split(',')) == 2
                    for form in (('loop', 'scan') if full_product else ('scan',)):
                        got = jet_contract(string, *arrays, over_orders=form, **lens)
                        assert got.shape == ref.shape, (string, got.shape, ref.shape)
                        worst[kind] = max(worst[kind], relerr(got, ref))
                        n_checked += 1
    for kind in STRINGS:
        print('   %-4s %2d strings, largest relative error %.1e' % (kind, len(STRINGS[kind]), worst[kind]))
    print('   %d calls checked: orders 0, 1, 3; S in (), (2,); W in (), (3,), (2, 2); K in (2,), (2, 2); C in (), (2,)'
          % n_checked)
    assert max(worst.values()) < 1e-12


def check_steps_and_strictness(rng):
    print('2. The two step functions, order 0, and an operand of the wrong rank.')
    shape_of = dict(S=(), W=(2, 3), K=(4,), C=(2,))
    for order in (0, 3):
        n_slots, T = order + 1, ('T' if order else '')
        mu, mu_dot = make(rng, shape_of, T + 'WC', 'a', n_slots), make(rng, shape_of, T + 'WKC', 'a', n_slots)
        xi, xi_dot = make(rng, shape_of, T + 'WC', 'i', 2), make(rng, shape_of, T + 'WKC', 'i', 2)
        R, U, H = make(rng, shape_of, 'C', 'aib'), make(rng, shape_of, 'C', 'aib'), make(rng, shape_of, 'KC', 'aib')
        N_CONTRACTIONS[0] = 0
        got = mu_dot_step(mu_dot, mu, xi, xi_dot, R, U, H, len_T=1 if order else 0, len_C=1)
        n_made = N_CONTRACTIONS[0]
        ref = sum(reference('mu', [(a, ga, 'a'), (G, gG, 'aib'), (x, gx, 'i')], shape_of, n_slots)
                  for a, ga, G, gG, x, gx in ((mu_dot, T + 'WKC', R, 'C', xi, T + 'WC'),
                                              (mu, T + 'WC', H, 'KC', xi, T + 'WC'),
                                              (mu, T + 'WC', U, 'C', xi_dot, T + 'WKC')))
        ref = unflatten(ref if order else ref[:, 0], shape_of, T + 'WKC', n_slots if order else 0)
        print('   mu_dot_step at order %d: %d contractions for its 3 terms, result %s, relative error %.1e'
              % (order, n_made, got.shape, relerr(got, ref)))
        assert relerr(got, ref) < 1e-12 and n_made == (3 if not order else 6)
    S_of = dict(S=(2,), W=(3,), K=(), C=(2,))
    eta_tilde, eta = make(rng, S_of, 'STWC', 'i', 4), make(rng, S_of, 'STWC', 'i', 4)
    w, w_tilde = make(rng, S_of, 'STW', 'o', 2), make(rng, S_of, 'STWC', 'o', 4)
    got = V_tilde_step(eta_tilde, eta, w, w_tilde, len_S=1, len_T=1, len_C=1)
    ref = (reference('V', [(eta_tilde, 'STWC', 'i'), (w, 'STW', 'o')], S_of, 4)
           + reference('V', [(eta, 'STWC', 'i'), (w_tilde, 'STWC', 'o')], S_of, 4))
    ref = unflatten(ref, S_of, 'SC', 0)
    print('   V_tilde_step at order 3, all sites at once: result %s, relative error %.1e' % (got.shape, relerr(got, ref)))
    assert relerr(got, ref) < 1e-12

    shape_of = dict(S=(), W=(2, 3), K=(4,), C=(2,))
    mu = make(rng, shape_of, 'TWC', 'a', 4)
    xi = make(rng, shape_of, 'TWC', 'i', 2)
    H = make(rng, shape_of, 'KC', 'aib')
    outcomes = []
    for label, bad in (('in front of W', mu[:, None]), ('between W and C', mu[:, :, :, None])):
        try:
            jet_contract('TWCa,KCaib,AWCi->TWKCb', bad, H, xi, len_T=1, len_C=1)
            outcomes.append('%s: ACCEPTED' % label)
        except ValueError:
            outcomes.append('%s: raises' % label)
    print('   a stray axis on mu, with len_T and len_C passed.  ' + ';  '.join(outcomes))
    assert all(o.endswith('raises') for o in outcomes)
    seed = make(rng, shape_of, 'TWKC', 'o', 4)
    D = make(rng, shape_of, 'C', 'io')
    outcomes = []
    for label, lens in (('len_T and len_C', dict(len_T=1, len_C=1, len_S=0)),
                        ('with len_W as well', dict(len_T=1, len_C=1, len_S=0, len_W=2)),
                        ('with len_W and len_K', dict(len_T=1, len_C=1, len_S=0, len_W=2, len_K=1))):
        try:
            jet_contract('STWKCo,SCio->STWKCi', seed[:, :, None], D, **lens)
            outcomes.append('%s: ACCEPTED' % label)
        except ValueError:
            outcomes.append('%s: raises' % label)
    print('   a stray axis on a seed that carries K: W and K stand together on the only operand that has them, so')
    print('   the stray axis is read as one more axis of W or of K unless both lengths are given.')
    print('   ' + ';  '.join(outcomes))
    assert [o.split(': ')[1] for o in outcomes] == ['ACCEPTED', 'ACCEPTED', 'raises']


def check_jax(rng):
    print('3. jax.')
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print('   skipped (jax is not installed)')
        return
    shape_of = dict(S=(), W=(2, 2), K=(3,), C=(2,))
    f32 = np.float32
    mu = make(rng, shape_of, 'TWC', 'a', 4).astype(f32)
    H = make(rng, shape_of, 'KC', 'aib').astype(f32)
    nu = make(rng, shape_of, 'TWC', 'b', 4).astype(f32)
    nu_dot = make(rng, shape_of, 'TWKC', 'b', 4).astype(f32)
    xi = make(rng, shape_of, 'TWC', 'i', 2).astype(f32)
    worst = 0.0
    for string, ops, lens in (('TWCa,KCaib,AWCi->TWKCb', (mu, H, xi), {}),
                              ('STWCa,SKCaib,STWCb->STWKCi', (mu, H, nu), dict(len_S=0)),
                              ('STWCa,SAWCi,STWKCb->SKCaib', (mu, xi, nu_dot), dict(len_S=0))):
        on_numpy = jet_contract(string, *ops, len_T=1, len_C=1, **lens)
        jitted = jax.jit(lambda *a, string=string, lens=lens: jet_contract(string, *a, len_T=1, len_C=1, **lens))
        on_jax = jitted(*(jnp.asarray(o) for o in ops))
        assert on_jax.dtype == jnp.float32 and on_numpy.dtype == f32, (string, on_jax.dtype, on_numpy.dtype)
        worst = max(worst, relerr(np.asarray(on_jax), on_numpy))
    print('   a product with an affine leaf, a product of two full jets and a block readout, under jit, float32 in')
    print('   and out, against the same calls on numpy: agree to %s'
          % ('rounding (below 1e-5)' if worst < 1e-5 else 'NO: %.1e' % worst))
    assert worst < 1e-5


if __name__ == '__main__':
    load_library(sys.argv[1])
    rng = np.random.default_rng(0)
    check_all(rng)
    check_steps_and_strictness(rng)
    check_jax(rng)
