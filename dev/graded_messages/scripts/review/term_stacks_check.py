"""Target decision 5.4: the seven terms, written once, generic over every stack and over the order.

The question.  A variable has the shape  [S] + T + W + [K] + C + (size,):
    S   the site group: (d,) when a map-type operation is done at all sites at once on the uniform layout, else ()
    T   the optional order axis: (order+1,), or () at order 0.   A: the same for an affine variable, (2,) or ()
    W   the sample stack: any number of axes
    K   the stack of directions: any number of axes, on whatever depends on the direction
    C   the stack of base points: any number of axes, on every core and every variable (not on the input w)
In one term the direction sits on ONE operand, and which one differs from term to term:
    mu_dot_step = mu_term(mu_dot, R, xi) + mu_term(mu, H, xi) + mu_term(mu, U, xi_dot)
                          K on the carry       K on the core       K on the leaf
The library today writes a literal contraction string for every such place (backend/probing.py, _sigma_step).

What this prototype shows.  ONE contraction string per term serves every placement:

    mu_term:   'S T W X C a,  S Y C a i b,  S W Z C i  ->  S T W X Y Z C b'

X, Y, Z are the direction stack as carried by the first, second and third operand.  Their lengths are not
given: each is solved from the rank of its own operand, the other groups being known from four static integers
of the call (StackLens: how many axes S, T, W and C have).  So a call site is the formula and nothing else:
`mu_term(mu, H, xi, lens)`.  A term function checks that at most one operand carries a direction stack.

Over the orders (decision 2.9) a term is the truncated product of its operands' jets.  One operand carries the
order axis through the contraction, as the riding group T; the other is taken one coefficient at a time.  The
helper `_product_of_jets` does that for an affine partner (two contractions) and for a full one ('loop' or
'scan'), and at order 0 it is one contraction with no order axis anywhere.  The block readouts V_term and H_term
contract the order axis itself (the top coefficient of the product) and sum the samples.

Checked against a reference that uses neither `contract` nor these helpers: every operand is expanded to all the
stacks, each stack flattened to one axis, and the truncated products are explicit loops over np.einsum.

Not here: the scan written through `xscan` (scripts/review/order_scan_check.py does that for W and C; here
'scan' is the same carried shift as a Python loop); the plan's contraction order (5.5); masks.

Usage:  python3 -I term_stacks_check.py /path/to/T3Toolbox
"""
import collections
import itertools
import math
import sys

import numpy as np

contract = get_backend = tree_contains_jax = None       # bound by load_library, as in order_scan_check.py


def load_library(repo):
    global contract, get_backend, tree_contains_jax
    sys.path.insert(0, repo)
    from t3toolbox.backend import common
    from t3toolbox.backend import contractions
    contract, get_backend, tree_contains_jax = contractions.contract, common.get_backend, common.tree_contains_jax


# The four static integers of a call: how many axes each group has.  Hashable, so it can key a memoized factory.
StackLens = collections.namedtuple('StackLens', ['S', 'T', 'W', 'C'])   # S and T are 0 or 1

N_CONTRACTIONS = [0]                                                    # counted by the checks below


# ---- one contraction ---------------------------------------------------------------------------------------
def _grouped(
        subscripts,     # a grouped string.  S, T, A, W, C: lengths from `lens` (A as T).  X, Y, Z: solved here
        lens,           # StackLens
        *operands,
):
    """One `contract` call.  The length of X, Y or Z is the rank its operand has left over once the known groups
    and the single axes are counted; at most one operand may have any left over."""
    known = {'S': lens.S, 'T': lens.T, 'A': lens.T, 'W': lens.W, 'C': lens.C}
    group_lens = {'len_' + g: n for g, n in known.items() if g in subscripts}
    carrying = 0
    for term, operand in zip(subscripts.replace(' ', '').split('->')[0].split(','), operands):
        free = [g for g in term if g.isupper() and g not in known]
        if not free:
            continue
        (g,) = free                                                     # one direction group per operand
        left_over = operand.ndim - sum(known.get(ch, 1) for ch in term if ch != g)
        if left_over < 0:
            raise ValueError('term %r: the operand has %d axes too few' % (term, -left_over))
        group_lens['len_' + g] = left_over
        carrying += left_over > 0
    if carrying > 1:
        raise ValueError('%r: more than one operand carries a direction stack' % subscripts)
    N_CONTRACTIONS[0] += 1
    return contract(subscripts, *operands, **group_lens)


# ---- the order axis: it follows the site group -------------------------------------------------------------
def _xnp(*arrays):
    return get_backend(False, tree_contains_jax(arrays))[0]


def _slot(x, r, lens):                      # coefficient r
    return x[(slice(None),) * lens.S + (r,)]


def _slots(x, start, stop, lens):           # coefficients start .. stop-1, the order axis kept
    return x[(slice(None),) * lens.S + (slice(start, stop),)]


def _turned(x, lens):                       # the order axis reversed
    return x[(slice(None),) * lens.S + (slice(None, None, -1),)]


def _moved_up(x, r, n_slots, lens):         # x in slots r .. r+len-1 of n_slots, zeros elsewhere (x's dtype)
    xnp = _xnp(x)
    have = x.shape[lens.S]

    def zeros(n):
        return xnp.zeros(x.shape[:lens.S] + (n,) + x.shape[lens.S + 1:], x.dtype)
    parts = ([zeros(r)] if r else []) + [x] + ([zeros(n_slots - r - have)] if n_slots - r - have else [])
    return parts[0] if len(parts) == 1 else xnp.concatenate(parts, axis=lens.S)


def _product_of_jets(
        subscripts,     # T on the carrier, no order group on the partner
        lens,
        operands,       # tuple, in the order of the subscripts
        carrier,        # position of the operand that carries the order axis through the contraction
        partner,        # position of the operand that is taken one coefficient at a time
        over_orders,    # 'loop' | 'scan': how a product of two FULL jets is carried out (decision 2.9)
):
    """c[t] = sum over i + r = t of the contraction with carrier[i] and partner[r].  At order 0: one contraction."""
    if lens.T == 0:
        return _grouped(subscripts, lens, *operands)
    ops = list(operands)
    a, b = ops[carrier], ops[partner]
    n_slots = a.shape[lens.S]                                           # order + 1
    n_b = b.shape[lens.S]                                               # 2 for an affine partner
    if n_b > 2 and over_orders == 'scan':                               # the carried jet moves up one slot a step
        out = None
        for r in range(n_b):
            ops[carrier], ops[partner] = a, _slot(b, r, lens)
            part = _grouped(subscripts, lens, *ops)
            out = part if out is None else out + part
            a = _moved_up(_slots(a, 0, n_slots - 1, lens), 1, n_slots, lens)
        return out
    out = None                                                          # affine partner, or 'loop': slices
    for r in range(n_b):
        ops[carrier], ops[partner] = _slots(a, 0, n_slots - r, lens), _slot(b, r, lens)
        part = _moved_up(_grouped(subscripts, lens, *ops), r, n_slots, lens)
        out = part if out is None else out + part
    return out


# ---- the seven terms -----------------------------------------------------------------------------------------
def _is_input(w, lens):
    """An input w is A + W + (N,): two coefficients and no C.  A seed w_tilde is T + W + [K] + C + (N,): a full
    dual jet.  With C empty and no K their ranks are equal, and the length of the order axis tells them apart;
    at order 1 both have two coefficients and every formula below gives the same result for either."""
    return w.ndim == lens.S + lens.T + lens.W + 1 and (lens.T == 0 or w.shape[lens.S] == 2)


def xi_term(D, w, lens):
    """A leaf core applied to an input or to a seed.  The order axis rides: there is no product over orders."""
    if _is_input(w, lens):
        return _grouped('S A W o,       S X C i o  ->  S A W X C i', lens, w, D)
    return _grouped('S T W Z C o,   S X C i o  ->  S T W X Z C i', lens, w, D)


def z_term(eta, D, lens):
    """The trunk above a Tucker edge through the leaf core, the mode leg left open.  The order axis rides."""
    return _grouped('S T W X C i,   S Y C i o  ->  S T W X Y C o', lens, eta, D)


def mu_term(mu, G, xi, lens, over_orders='scan'):
    return _product_of_jets('S T W X C a,  S Y C a i b,  S W Z C i  ->  S T W X Y Z C b',
                            lens, (mu, G, xi), 0, 2, over_orders)


def nu_term(G, xi, nu, lens, over_orders='scan'):
    return _product_of_jets('S Y C a i b,  S W Z C i,  S T W X C b  ->  S T W X Y Z C a',
                            lens, (G, xi, nu), 2, 1, over_orders)


def eta_term(mu, G, nu, lens, over_orders='scan'):
    return _product_of_jets('S T W X C a,  S Y C a i b,  S W Z C b  ->  S T W X Y Z C i',
                            lens, (mu, G, nu), 0, 2, over_orders)


def V_term(eta, w, lens):
    """A block shaped like a leaf core: the top coefficient of the product, summed over the samples."""
    if lens.T == 0:
        if _is_input(w, lens):
            return _grouped('S W X C i,   S W o      ->  S X C i o', lens, eta, w)
        return _grouped('S W X C i,   S W Z C o  ->  S X Z C i o', lens, eta, w)
    top = eta.shape[lens.S] - 1
    if _is_input(w, lens):                                              # an input: two coefficients, two slots
        return (_grouped('S W X C i,   S W o  ->  S X C i o', lens, _slot(eta, top, lens), _slot(w, 0, lens))
                + _grouped('S W X C i,   S W o  ->  S X C i o', lens, _slot(eta, top - 1, lens), _slot(w, 1, lens)))
    return _grouped('S T W X C i,   S T W Z C o  ->  S X Z C i o', lens, eta, _turned(w, lens))


def H_term(mu, xi, nu, lens):
    """A block shaped like a TT core: the top coefficient of the product of three jets, summed over the samples.
    The order axis of mu and of nu is contracted; xi is taken one coefficient at a time."""
    if lens.T == 0:
        return _grouped('S W X C a,   S W Y C i,   S W Z C b  ->  S X Y Z C a i b', lens, mu, xi, nu)
    n_slots = mu.shape[lens.S]
    out = None
    for j in range(xi.shape[lens.S]):                                   # 2 for an affine xi
        part = _grouped('S T W X C a,   S W Y C i,   S T W Z C b  ->  S X Y Z C a i b', lens,
                        _slots(mu, 0, n_slots - j, lens), _slot(xi, j, lens),
                        _turned(_slots(nu, 0, n_slots - j, lens), lens))
        out = part if out is None else out + part
    return out


# ---- the library's way, for comparison: a literal string for each place the direction can sit ---------------
MU_TERM_LITERAL = {None: 'TWCa,Caib,WCi->TWCb',
                   0:    'TWKCa,Caib,WCi->TWKCb',
                   1:    'TWCa,KCaib,WCi->TWKCb',
                   2:    'TWCa,Caib,WKCi->TWKCb'}


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


# The operands of each term: (name, core letters, has the samples, may carry the direction).
TERMS = {
    'xi':  ([('D', 'io', False, True)], 'input-or-seed'),
    'z':   ([('eta', 'i', True, True), ('D', 'io', False, True)], None),
    'mu':  ([('mu', 'a', True, True), ('G', 'aib', False, True), ('xi', 'i', True, True)], 2),
    'nu':  ([('G', 'aib', False, True), ('xi', 'i', True, True), ('nu', 'b', True, True)], 1),
    'eta': ([('mu', 'a', True, True), ('G', 'aib', False, True), ('nu', 'b', True, True)], 2),
    'V':   ([('eta', 'i', True, True)], 'input-or-seed'),
    'H':   ([('mu', 'a', True, True), ('xi', 'i', True, True), ('nu', 'b', True, True)], 1),
}
CALL = {'xi': lambda ops, lens, form: xi_term(ops[1], ops[0], lens),          # ops are in the reference's order
        'z': lambda ops, lens, form: z_term(ops[0], ops[1], lens),
        'mu': lambda ops, lens, form: mu_term(*ops, lens, over_orders=form),
        'nu': lambda ops, lens, form: nu_term(*ops, lens, over_orders=form),
        'eta': lambda ops, lens, form: eta_term(*ops, lens, over_orders=form),
        'V': lambda ops, lens, form: V_term(ops[0], ops[1], lens),
        'H': lambda ops, lens, form: H_term(*ops, lens)}


def cases(order):
    """Every term, every place the direction can sit, every kind of partner.  Yields
    (label, kind, [(name, groups, core, order_len)], out_groups, n_order_out)."""
    T = 'T' if order else ''
    n_slots = order + 1

    def groups(samples, k, c=True):
        return 'S' + (T if samples else '') + ('W' if samples else '') + ('K' if k else '') + ('C' if c else '')
    for kind in TERMS:
        base, extra = TERMS[kind]
        if kind in ('xi', 'V'):                                         # the second operand is an input or a seed
            for second in ('input', 'seed'):
                names = ([('w', 'o')] + [(n, c) for n, c, _, _ in base]) if kind == 'xi' else \
                        ([(n, c) for n, c, _, _ in base] + [('w', 'o')])
                w_pos = 0 if kind == 'xi' else 1
                for k_on in [None] + [p for p in range(2) if not (second == 'input' and p == w_pos)]:
                    ops = []
                    for p, (name, core) in enumerate(names):
                        if p == w_pos:
                            ops.append((name, groups(True, k_on == p, c=(second == 'seed')), core,
                                        (2 if second == 'input' else n_slots)))
                        else:
                            ops.append((name, groups(kind == 'V', k_on == p), core, n_slots))
                    out_groups = 'S' + (T + 'W' if kind == 'xi' else '') + ('K' if k_on is not None else '') + 'C'
                    n_out = (2 if second == 'input' else n_slots) if kind == 'xi' else 0
                    yield ('%s_term, %s, direction on %s' % (kind, second, 'none' if k_on is None else names[k_on][0]),
                           kind, ops, out_groups, n_out)
            continue
        partners = ('affine', 'full') if (extra is not None and base[extra][0] == 'xi') else ('full',)
        for partner in partners:
            for k_on in [None] + list(range(len(base))):
                ops = []
                for p, (name, core, samples, _) in enumerate(base):
                    n_ord = 2 if (name == 'xi' and partner == 'affine') else n_slots
                    ops.append((name, groups(samples, k_on == p), core, n_ord))
                out_groups = ('S' + ('' if kind == 'H' else T + 'W') + ('K' if k_on is not None else '') + 'C')
                yield ('%s_term, %s partner, direction on %s' % (kind, partner, 'none' if k_on is None else base[k_on][0]),
                       kind, ops, out_groups, 0 if kind == 'H' else n_slots)


def check_all(rng):
    print('1. Every term against the explicit reference: all places of the direction, both kinds of partner,')
    print('   both forms over the orders, on every combination of stacks.')
    worst = collections.defaultdict(float)
    n_checked = 0
    stacks = list(itertools.product([(), (2,)], [(), (3,), (2, 2)], [(2,), (2, 2)], [(), (2,)]))   # S, W, K, C
    for order in (0, 1, 3):
        n_slots = order + 1
        for S, W, K, C in stacks:
            shape_of = dict(S=S, W=W, K=K, C=C)
            lens = StackLens(len(S), 1 if order else 0, len(W), len(C))
            for label, kind, spec, out_groups, n_out in cases(order):
                arrays = [make(rng, shape_of, g, core, n_ord) for _, g, core, n_ord in spec]
                ref = reference(kind, [(x, g, core) for x, (_, g, core, _) in zip(arrays, spec)],
                                dict(shape_of, K=K if 'K' in out_groups else ()), n_slots)
                if kind in ('xi', 'z', 'mu', 'nu', 'eta'):
                    if not order:
                        ref = ref[:, 0]                                 # no order axis at order 0
                    elif n_out == 2:
                        ref = ref[:, :2]                                # an affine result has two coefficients
                want_shape_of = dict(shape_of, K=K if 'K' in out_groups else ())
                ref = unflatten(ref, want_shape_of, out_groups, n_out if order else 0)
                for form in (('loop', 'scan') if kind in ('mu', 'nu', 'eta') and order else ('-',)):
                    got = CALL[kind](arrays, lens, form)
                    assert got.shape == ref.shape, (label, got.shape, ref.shape)
                    worst[kind] = max(worst[kind], relerr(got, ref))
                    n_checked += 1
    for kind in TERMS:
        print('   %-9s largest relative error %.1e' % (kind + '_term', worst[kind]))
    print('   %d calls checked: orders 0, 1, 3; S in (), (2,); W in (), (3,), (2, 2); K in (2,), (2, 2); C in (), (2,)'
          % n_checked)
    assert max(worst.values()) < 1e-12


def check_order0_and_guard(rng):
    print('2. Order 0, the guard, and the literal strings.')
    shape_of = dict(S=(), W=(2, 3), K=(4,), C=(2,))
    lens = StackLens(0, 0, 2, 1)
    mu = make(rng, shape_of, 'WC', 'a')
    G = make(rng, shape_of, 'C', 'aib')
    H = make(rng, shape_of, 'KC', 'aib')
    xi = make(rng, shape_of, 'WC', 'i')
    xi_dot = make(rng, shape_of, 'WKC', 'i')
    mu_dot = make(rng, shape_of, 'WKC', 'a')
    N_CONTRACTIONS[0] = 0
    step = mu_term(mu_dot, G, xi, lens) + mu_term(mu, H, xi, lens) + mu_term(mu, G, xi_dot, lens)
    print('   a dotted step at order 0: %d contractions for its 3 terms; result W + K + C + (rR,) = %s; no order axis'
          % (N_CONTRACTIONS[0], step.shape))
    assert N_CONTRACTIONS[0] == 3 and step.shape == (2, 3, 4, 2, SIZE['b'])
    try:
        mu_term(mu_dot, H, xi, lens)
        raised = False
    except ValueError as err:
        raised = 'more than one operand' in str(err)
    print('   the direction on two operands of one term raises: %s' % raised)
    assert raised
    T_lens = StackLens(0, 1, 2, 1)
    shape_T = dict(shape_of)
    mu_T, mu_dot_T = make(rng, shape_T, 'TWC', 'a', 4), make(rng, shape_T, 'TWKC', 'a', 4)
    worst = 0.0
    for k_on, ops in ((None, (mu_T, G, xi)), (0, (mu_dot_T, G, xi)), (1, (mu_T, H, xi)), (2, (mu_T, G, xi_dot))):
        literal = contract(MU_TERM_LITERAL[k_on], *ops, len_T=1, len_C=1)
        one_string = _grouped('S T W X C a,  S Y C a i b,  S W Z C i  ->  S T W X Y Z C b', T_lens, *ops)
        worst = max(worst, relerr(one_string, np.asarray(literal)))
    print('   the one string against the four literal strings of the library\'s style (order axis riding): %.1e' % worst)
    assert worst < 1e-14


def check_jax(rng):
    print('3. jax.')
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print('   skipped (jax is not installed)')
        return
    shape_of = dict(S=(), W=(2, 2), K=(3,), C=(2,))
    lens = StackLens(0, 1, 2, 1)
    n_slots = 4
    mu = make(rng, shape_of, 'TWC', 'a', n_slots).astype(np.float32)
    H = make(rng, shape_of, 'KC', 'aib').astype(np.float32)
    nu = make(rng, shape_of, 'TWC', 'b', n_slots).astype(np.float32)
    xi = make(rng, shape_of, 'TWC', 'i', 2).astype(np.float32)
    worst = 0.0
    for name, fn, ops in (('mu_term', mu_term, (mu, H, xi)), ('eta_term', eta_term, (mu, H, nu)),
                          ('H_term', H_term, (mu, xi, nu[..., None, :, :] * np.ones((3, 1, 1), np.float32)))):
        on_numpy = fn(*ops, lens)
        jitted = jax.jit(lambda *a, fn=fn: fn(*a, lens))
        on_jax = jitted(*(jnp.asarray(o) for o in ops))
        assert on_jax.dtype == jnp.float32 and on_numpy.dtype == np.float32, (name, on_jax.dtype, on_numpy.dtype)
        worst = max(worst, relerr(np.asarray(on_jax), on_numpy))
    print('   mu_term, eta_term and H_term under jit, the direction on the core or on nu, float32 in and out,')
    print('   against the same functions on numpy: agree to %s' % ('rounding (below 1e-5)' if worst < 1e-5 else 'NO: %.1e' % worst))
    assert worst < 1e-5


if __name__ == '__main__':
    load_library(sys.argv[1])
    rng = np.random.default_rng(0)
    check_all(rng)
    check_order0_and_guard(rng)
    check_jax(rng)
