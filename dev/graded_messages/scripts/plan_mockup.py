"""STATUS (2026-11-14): this mockup shows the interface as it was BEFORE the review of that date and has not been
rewritten.  The decided interface is in ../DESIGN_AFTER_REVIEW.md: `sampling_variables` with keyword arguments in
place of `evaluate(want, have)`, `SamplePlan` with nested scans in place of `Plan`, new names (`_dot`, `_tilde`,
`gamma`, `JtJp`), no `centers` argument, no projection inside, weighted least squares.  The explicit functions with
the decided names are in review/jet_steps_check.py.  Known defect: `run` still holds a reference cycle (finding D14).

MOCKUP: how a user would compute the probing objective, gradient and a Gauss-Newton Hessian-vector product
with an execution plan.  Part 1 is what the user writes.  Part 2 is a small working stand-in for what the
library would provide, so that the mockup runs and can be checked against T3Toolbox.

The per-sample intermediates at a point form a dependency graph of named components.  There is one entry
point, `evaluate`: a call says which components it WANTS and which it already HAS.  Everything else is computed
inside each block of samples, dropped at its last use, and never outlives the block.  Keeping a component for later, returning the
per-sample residual, and fusing two operations into one pass are all the same thing: wanting a component.

    how to execute (blocks, association order, executor)   ->  the Plan
    what to compute and what to hand back                  ->  want
    what is already known                                  ->  have

A sketch of a proposed interface, not library code: numpy only, order 0, one sample axis W, host loops.
The plan fields that concern jets, scans and devices are listed in `Plan` but only the ones marked LIVE are
honoured here.  Nothing is timed.

Usage:  python3 plan_mockup.py /path/to/T3Toolbox [--dot graph.dot]
"""
import itertools
import math
import sys
import tracemalloc
from dataclasses import dataclass

import numpy as np

if len(sys.argv) > 1:
    sys.path.append(sys.argv[1])


# ====================================================================================================
# PART 1.  WHAT THE USER WRITES
# ====================================================================================================
def user_example():
    rng = np.random.default_rng(0)
    np.random.seed(0)

    # A point, some samples, some data
    shape, tucker_ranks, tt_ranks, W = (60, 50, 70, 40), (6, 7, 6, 5), (1, 8, 9, 8, 1), 4096
    x = random_point(shape, tucker_ranks, tt_ranks)                 # the point: (Tucker cores, TT cores)
    w = [rng.standard_normal((W, N)) for N in shape]                # sample vectors, one array per mode
    y = [rng.standard_normal((W, N)) for N in shape]                # data, one array per mode

    frame, centers = orthogonal_frame(x)                            # (D, U, L, R), and the point's center cores
    p = random_tangent(frame, rng)                                  # a direction for the Hessian product

    # ---- Make a plan: HOW to execute.  Three ways. --------------------------------------------------
    plan = Plan()                                                   # 1. the defaults
    plan = Plan(samples=512, assembly=64)                           # 2. explicit
    plan = fill(frame, centers, w, budget_bytes=8 * 2**20, keep=INPUTS)   # 3. a policy: size the block to a
                                                                    #    budget, given what will be kept

    # ---- Say what you want; hand in what you have. --------------------------------------------------
    f = evaluate(frame, centers, w, plan, want=('f',), y=y)['f']    # the objective only, e.g. at a trial point

    out = evaluate(frame, centers, w, plan, want=('f', 'g') + INPUTS, y=y)   # objective and gradient in one pass,
    f, g = out['f'], out['g']                                       # keeping xi as a by-product
    have = {name: out[name] for name in INPUTS}

    Hp = evaluate(frame, centers, w, plan, want=('Hp',), p=p, have=have)['Hp']   # Gauss-Newton Hessian applied to p

    r = evaluate(frame, centers, w, plan, want=('r',), y=y, have=have)['r']      # per-sample residuals, on request

    # ---- The same things under their usual names (thin wrappers over evaluate). ---------------------
    f    = objective(frame, centers, w, y, plan)
    f, g = objective_and_gradient(frame, centers, w, y, plan, have=have)
    Hp   = gn_product(frame, centers, w, p, plan, have=have)

    return x, w, y, frame, centers, p, plan, (f, g, Hp)


# ====================================================================================================
# PART 2.  WHAT THE LIBRARY WOULD PROVIDE  (a small working stand-in)
# ====================================================================================================

# ----------------------------------------------------------------------------------------------------
# The plan: an immutable value that says HOW to execute.  Every field has a default.  It never changes
# what a call returns, and it does not say what to keep.
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Plan:
    samples:  int = None            # LIVE. Outer block: samples carried together through a whole operation.
                                    #       None = all of them at once.
    assembly: int = None            # LIVE. Inner block for the gradient assemblies. None = the outer block.
    path:     str = 'carry_first'   # LIVE. Association inside a sweep step:
                                    #       'carry_first' = (mu . G) . xi      'input_first' = mu . (G . xi)
    orders:   str = 'sequential'    # not in this mockup (order 0). Jet products: 'sequential' | 'vectorized'
    sites:    str = 'sequential'    # not in this mockup. Per-site maps: 'sequential' | 'vectorized'
    executor: str = 'host_loop'     # not in this mockup. Where the sample loop runs:
                                    #       'host_loop' | 'scan' | 'shard_map' | ...


# ----------------------------------------------------------------------------------------------------
# Component names.  A keep set is any tuple of per-sample names; these are only convenient presets.
# ----------------------------------------------------------------------------------------------------
SUMMED = ('f', 'g', 'Hp')                              # summed over samples: core-sized results
POINT = ('xi', 'mu', 'nu', 'eta')                      # change with the point: valid for every call at this point
POINT_DATA = ('r', 'xi_z', 'mu_z', 'nu_z', 'eta_z')    # change with the point and the data
#   every other per-sample component ('z', 'xi_t', ..., 'Jp', 'Jp.xi_z', ...) changes with the direction p
#   or is cheap to rebuild; any of them may still be wanted.

NOTHING = ()
INPUTS = ('xi',)
FRAME = POINT


# ----------------------------------------------------------------------------------------------------
# The one entry point
# ----------------------------------------------------------------------------------------------------
def evaluate(frame, centers, w, plan, want, have=None, y=None, p=None):
    """Compute the components named in `want`, given the per-sample components in `have`.

    Returns {name: value}.  Summed components ('f', 'g', 'Hp') come back core-sized; per-sample components
    come back for all samples.  Anything not wanted is computed inside a block only if something wanted reads it,
    and is dropped at its last read.
    `y` is needed by anything downstream of the residual, `p` by anything downstream of the direction."""
    if p is not None:
        p = project(frame, p)
    summed = [name for name in want if name in SUMMED]
    mapped = [name for name in want if name not in SUMMED]

    def kernel(s):                                              # a pure function of one block of samples
        out = run(block_graph(frame, centers, s['w'], s.get('y'), p, plan), s['have'], summed + mapped)
        return tuple(out[name] for name in summed) or None, {name: out[name] for name in mapped} or None

    per_sample = {'w': w, 'have': dict(have or {})}
    if y is not None:
        per_sample['y'] = y
    total, out = over_samples(kernel, per_sample, plan.samples)
    result = dict(out or {})
    for name, value in zip(summed, total or ()):
        result[name] = value if name == 'f' else project(frame, value)     # g and Hp are gauged tangents (V, H)
    return result


# The usual names, as thin wrappers.
def objective(frame, centers, w, y, plan, have=None):
    return evaluate(frame, centers, w, plan, want=('f',), have=have, y=y)['f']


def objective_and_gradient(frame, centers, w, y, plan, have=None):
    out = evaluate(frame, centers, w, plan, want=('f', 'g'), have=have, y=y)
    return out['f'], out['g']


def gn_product(frame, centers, w, p, plan, have=None):
    return evaluate(frame, centers, w, plan, want=('Hp',), have=have, p=p)['Hp']


# ----------------------------------------------------------------------------------------------------
# The dependency graph for one block, declared as data:  name -> (the components it reads, its rule).
# The names and the reads are the same for every block; only the arrays the rules close over differ.
# ----------------------------------------------------------------------------------------------------
def block_graph(frame, centers, w, y, p, plan):
    D, U, L, R = frame
    Vc, _ = centers                                    # the point's Tucker center cores
    ones, zero = np.ones((w[0].shape[0], 1)), np.zeros((w[0].shape[0], 1))
    graph = {}

    def node(name, reads, rule):
        graph[name] = (reads, rule)

    # plain components, the probe output, the residual, the objective
    node('xi',  (),            lambda: sitewise(UP, [(D, w)]))
    node('mu',  ('xi',),       lambda xi: sweep(RIGHT, ones, [(CARRY, L, xi)], plan))
    node('nu',  ('xi',),       lambda xi: sweep(LEFT, ones, [(R, xi, CARRY)], plan, reverse=True))
    node('eta', ('mu', 'nu'),  lambda mu, nu: sitewise(DOWN, [(mu, U, nu[1:])]))
    node('z',   ('eta',),      lambda eta: sitewise(OUT, [(eta, Vc)]))     # read off the frame's components:
    if y is not None:                                                      # no sweep through the point's own cores
        node('r', ('z',),      lambda z: [z_i - y_i for z_i, y_i in zip(z, y)])
        node('f', ('r',),      lambda r: 0.5 * sum(float(np.sum(r_i ** 2)) for r_i in r))
        residual_grade(node, seed='r', prefix='', out='g', frame=frame, w=w, zero=zero, plan=plan)
    if p is not None:                                  # the tangent grade along p = (V, H), then J^T of J p
        V, H = p
        node('xi_t',  (), lambda: sitewise(UP, [(V, w)]))
        node('mu_t',  ('xi', 'mu', 'xi_t'),
             lambda xi, mu, xi_t: sweep(RIGHT, zero, [(CARRY, R, xi), (mu, H, xi), (mu, U, xi_t)], plan))
        node('nu_t',  ('xi', 'nu', 'xi_t'),
             lambda xi, nu, xi_t: sweep(LEFT, zero, [(L, xi, CARRY), (H, xi, nu[1:]), (U, xi_t, nu[1:])], plan,
                                        reverse=True))
        node('eta_t', ('mu', 'nu', 'mu_t', 'nu_t'),
             lambda mu, nu, mu_t, nu_t: sitewise(DOWN, [(mu_t, R, nu[1:]), (mu, H, nu[1:]), (mu, L, nu_t[1:])]))
        node('Jp',    ('eta', 'eta_t'), lambda eta, eta_t: sitewise(OUT, [(eta_t, D), (eta, V)]))
        residual_grade(node, seed='Jp', prefix='Jp.', out='Hp', frame=frame, w=w, zero=zero, plan=plan)
    return graph


def residual_grade(node, seed, prefix, out, frame, w, zero, plan):
    """The residual grade seeded by the per-sample component `seed`, and its summed assembly `out` = J^T seed."""
    D, U, L, R = frame
    xz, mz, nz, ez = prefix + 'xi_z', prefix + 'mu_z', prefix + 'nu_z', prefix + 'eta_z'
    node(xz, (seed,),               lambda s: sitewise(UP, [(D, s)]))
    node(mz, ('xi', 'mu', xz),      lambda xi, mu, xi_z: sweep(RIGHT, zero, [(CARRY, L, xi), (mu, L, xi_z)], plan))
    node(nz, ('xi', 'nu', xz),      lambda xi, nu, xi_z: sweep(LEFT, zero, [(R, xi, CARRY), (R, xi_z, nu[1:])], plan,
                                                               reverse=True))
    node(ez, ('mu', 'nu', mz, nz),  lambda mu, nu, mu_z, nu_z: sitewise(DOWN, [(mu_z, U, nu[1:]), (mu, U, nu_z[1:])]))
    node(out, ('xi', 'mu', 'nu', 'eta', seed, xz, mz, nz, ez),
         lambda xi, mu, nu, eta, s, xi_z, mu_z, nu_z, eta_z: (
             sitewise(GRAD_B, [(eta_z, w), (eta, s)], sum_w=True, block=plan.assembly),
             sitewise(GRAD_G, [(mu_z, xi, nu[1:]), (mu, xi_z, nu[1:]), (mu, xi, nu_z[1:])], sum_w=True,
                      block=plan.assembly)))


def run(graph, have, want):
    """Evaluate the components in `want`, given those in `have`, for one block.

    Because the graph is declared, two things are known before anything is computed: exactly which components
    are needed (walk back from `want`, stopping at `have`), and when each is read for the last time.  Each
    component is dropped at its last read, so nothing outlives its use, let alone the block."""
    order, seen = [], set()

    def visit(name):
        if name in have or name in seen:
            return
        if name not in graph:
            raise ValueError('component %r is not available: it needs y (data) or p (a direction)' % name)
        seen.add(name)
        for dep in graph[name][0]:
            visit(dep)
        order.append(name)

    for name in want:
        visit(name)
    readers = {}
    for name in order:
        for dep in graph[name][0]:
            readers[dep] = readers.get(dep, 0) + 1
    values = dict(have)
    out = {}
    for name in order:
        reads, rule = graph[name]
        values[name] = rule(*(values[dep] for dep in reads))
        for dep in reads:
            readers[dep] -= 1
            if readers[dep] == 0 and dep not in want:
                del values[dep]
        if name in want and readers.get(name, 0) == 0:
            out[name] = values.pop(name)
    for name in want:
        if name not in out:
            out[name] = values[name]
    return out


# ----------------------------------------------------------------------------------------------------
# A policy is a plain function that returns a Plan.  This one is "fill a memory budget".
# What to keep is the caller's decision; the block size is its consequence.
# ----------------------------------------------------------------------------------------------------
def fill(frame, centers, w, budget_bytes, keep=NOTHING, fraction=0.9, probe_blocks=(256, 512)):
    """Size the outer block so that the predicted peak of a Hessian product, plus the components in `keep`
    for all samples, stays within `fraction` of `budget_bytes`.

    Peak memory is affine in the block size, so two small probe runs give the line. Nothing is timed.
    Each probe runs two blocks, so the cost of accumulating across blocks is in the measurement.
    (On jax the two numbers would come from the compiled program's memory analysis, without running it.)"""
    W = w[0].shape[0]
    p = zero_tangent(frame)

    def probe(b):
        ws = [a[:2 * b] for a in w]
        plan = Plan(samples=b)
        have = evaluate(frame, centers, ws, plan, want=tuple(keep))
        tracemalloc.start()
        evaluate(frame, centers, ws, plan, want=('Hp',), p=p, have=have)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak, tree_bytes(have) / (2 * b)

    b1, b2 = probe_blocks
    (m1, _), (m2, kept_per_sample) = probe(b1), probe(b2)
    per_sample = (m2 - m1) / (b2 - b1)
    fixed = m1 - per_sample * b1
    room = fraction * budget_bytes - kept_per_sample * W - fixed
    if room < per_sample:
        raise ValueError('the budget does not cover keeping %r for all samples; keep less' % (tuple(keep),))
    block = min(W, 2 ** int(math.log2(room / per_sample)))          # quantized, so plans repeat across runs
    return Plan(samples=block)


# ----------------------------------------------------------------------------------------------------
# The sample loop.  One implementation; a scan, a shard map or a streaming loop would replace this body.
# ----------------------------------------------------------------------------------------------------
def over_samples(kernel, per_sample, block):
    """Run `kernel` on blocks of samples. The kernel returns (reduced, mapped): `reduced` parts are added
    across blocks, `mapped` parts are written into per-sample outputs allocated once. Either may be None."""
    W = tree_leaves(per_sample)[0].shape[0]
    block = W if block is None else min(block, W)
    total, out = None, None
    for lo in range(0, W, block):
        reduced, mapped = kernel(tree_map(lambda a: a[lo:lo + block], per_sample))
        if reduced is not None:
            total = reduced if total is None else tree_add(total, reduced)
        if mapped is not None:
            if out is None:
                out = tree_map(lambda a: np.empty((W,) + a.shape[1:], a.dtype), mapped)
            tree_write(out, lo, mapped)
        del reduced, mapped                                         # nothing from this block outlives it
    return total, out



# ----------------------------------------------------------------------------------------------------
# Rules and the two drivers (as in the worked example), now reading the plan
# ----------------------------------------------------------------------------------------------------
UP, RIGHT, LEFT, DOWN, OUT = 'io,o->i', 'a,aib,i->b', 'aib,i,b->a', 'a,aib,b->i', 'i,io->o'
GRAD_G, GRAD_B = 'a,i,b->aib', 'i,o->io'
CARRY = object()

FIRST_PAIR = {                                   # which two operands of a sweep step are contracted first
    'carry_first': {RIGHT: (0, 1), LEFT: (0, 2)},
    'input_first': {RIGHT: (1, 2), LEFT: (0, 1)},
}


def _has_w(op, legs):
    return op.ndim == len(legs) + 1


def _term(rule, ops, sum_w=False, first_pair=None):
    ins, out = rule.split('->')
    ins = ins.split(',')
    subs = ','.join(('w' + s) if _has_w(op, s) else s for s, op in zip(ins, ops)) + '->' + ('' if sum_w else 'w') + out
    path = ['einsum_path', first_pair, (0, 1)] if first_pair else True
    return np.einsum(subs, *ops, optimize=path)


def sitewise(rule, terms, sum_w=False, block=None):
    """One rule at every site. With sum_w, the sample axis is summed, in blocks of `block` samples."""
    legs = rule.split('->')[0].split(',')
    d = min(len(op) for op in terms[0])
    out = []
    for i in range(d):
        ops_i = [[op[i] for op in t] for t in terms]
        if not (sum_w and block):
            out.append(sum(_term(rule, ops, sum_w) for ops in ops_i))
            continue
        W = next(op.shape[0] for op, s in zip(ops_i[0], legs) if _has_w(op, s))
        total = 0
        for lo in range(0, W, block):
            total = total + sum(_term(rule, [op[lo:lo + block] if _has_w(op, s) else op for op, s in zip(ops, legs)], True)
                                for ops in ops_i)
        out.append(total)
    return out


def sweep(rule, start, terms, plan, reverse=False):
    """Thread a carry along the chain. Returns the message on every bond 0..d."""
    first_pair = FIRST_PAIR[plan.path][rule]
    d = min(len(op) for op in terms[0] if op is not CARRY)
    out = [None] * (d + 1)
    out[d if reverse else 0] = carry = start
    for i in (reversed(range(d)) if reverse else range(d)):
        carry = sum(_term(rule, [carry if op is CARRY else op[i] for op in t], first_pair=first_pair) for t in terms)
        out[i if reverse else i + 1] = carry
    return out


# ----------------------------------------------------------------------------------------------------
# Small helpers: nested containers of arrays, and the existing library functions the example leans on
# ----------------------------------------------------------------------------------------------------
def tree_map(f, t):
    if isinstance(t, dict):
        return {k: tree_map(f, v) for k, v in t.items()}
    if isinstance(t, (list, tuple)):
        return type(t)(tree_map(f, v) for v in t)
    return f(t)


def tree_leaves(t):
    if isinstance(t, dict):
        return [a for v in t.values() for a in tree_leaves(v)]
    if isinstance(t, (list, tuple)):
        return [a for v in t for a in tree_leaves(v)]
    return [t]


def tree_add(a, b):
    if isinstance(a, (list, tuple)):
        return type(a)(tree_add(x, y) for x, y in zip(a, b))
    return a + b


def tree_write(out, lo, piece):
    if isinstance(out, dict):
        for k in out:
            tree_write(out[k], lo, piece[k])
    elif isinstance(out, (list, tuple)):
        for o, q in zip(out, piece):
            tree_write(o, lo, q)
    else:
        out[lo:lo + piece.shape[0]] = piece


def tree_bytes(t):
    return sum(a.nbytes for a in tree_leaves(t))


def random_point(shape, tucker_ranks, tt_ranks):
    import t3toolbox.backend.t3_constructors as t3_constructors
    return t3_constructors.t3_corewise_randn(shape, tucker_ranks, tt_ranks)


def orthogonal_frame(x):
    """The orthogonal frame (D, U, L, R) of the point x, and the point's center cores (Tucker, TT)."""
    import t3toolbox.backend.fv_conversions as fv_conversions
    return fv_conversions.t3_orthogonal_representations(x)


def project(frame, v):
    import t3toolbox.backend.tv_operations as tv_operations
    return tv_operations.tv_orthogonal_gauge_projection(frame, v)


def zero_tangent(frame):
    D, U, L, R = frame
    return ([np.zeros((u.shape[1], dd.shape[1])) for u, dd in zip(U, D)],
            [np.zeros((l.shape[0], l.shape[1], rr.shape[2])) for l, rr in zip(L, R)])


def random_tangent(frame, rng):
    return project(frame, tree_map(lambda a: rng.standard_normal(a.shape), zero_tangent(frame)))


# ====================================================================================================
# PART 3.  CHECK: every plan and every keep set gives the library's answer; only the memory changes
# ====================================================================================================
def peak_of(call):
    tracemalloc.start()
    out = call()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak, out


def main():
    import t3toolbox.tucker_tensor_train as t3
    import t3toolbox.frame_variations_format as fvf
    import t3toolbox.manifold as t3m
    import t3toolbox.fitting as fitting
    import t3toolbox.backend.probing as probing

    x, w, y, frame, centers, p, plan_from_fill, _ = user_example()
    W = w[0].shape[0]
    mb = lambda b: b / 2**20

    r_ref = [z - yy for z, yy in zip(probing.t3_probe(w, x), y)]
    model = fitting.probe_model(t3m.MANIFOLD, t3.TuckerTensorTrain(*x), w, r_ref)
    f_ref = float(model.objective_value)
    g_ref = model.gradient.data[1].data
    Hp_ref = model.gn_hessian(t3m.T3Tangent(model.frame, fvf.T3Variations(*p))).data[1].data
    flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
    rel = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))

    # ---- 1. plans and keep sets against the library -------------------------------------------------
    cases = [('Plan()', 'FRAME', Plan(), FRAME),
             ('Plan()', 'NOTHING', Plan(), NOTHING),
             ('Plan(samples=512)', 'INPUTS', Plan(samples=512), INPUTS),
             ("Plan(samples=512, assembly=64, path='input_first')", 'NOTHING',
              Plan(samples=512, assembly=64, path='input_first'), NOTHING),
             ('fill(..., budget_bytes=8 MiB, keep=INPUTS)', 'INPUTS', plan_from_fill, INPUTS)]
    print('W = %d samples; the data w and y are %.1f MB each' % (W, mb(tree_bytes(w))))
    print('the fill policy chose:', plan_from_fill)
    print()
    print('1. every plan and keep set against the library')
    print('%-52s %-8s %8s %11s   %s' % ('plan', 'kept', 'kept MB', 'Hp peak MB', 'difference from the library (f, g, Hp, r)'))
    for plan_name, keep_name, plan, keep in cases:
        out = evaluate(frame, centers, w, plan, want=('f', 'g', 'r') + keep, y=y)
        have = {name: out[name] for name in keep}
        peak, Hp = peak_of(lambda: evaluate(frame, centers, w, plan, want=('Hp',), p=p, have=have)['Hp'])
        print('%-52s %-8s %8.2f %11.2f   %.0e  %.0e  %.0e  %.0e' % (
            plan_name, keep_name, mb(tree_bytes(have)), mb(peak), abs(out['f'] - f_ref) / f_ref,
            rel(out['g'], g_ref), rel(Hp, Hp_ref), rel([out['r']], [r_ref])))

    # ---- 2. any subset of valid components may be handed in -----------------------------------------
    n, small = 256, Plan(samples=64)
    ws, ys = [a[:n] for a in w], [a[:n] for a in y]
    names = POINT + POINT_DATA
    everything = evaluate(frame, centers, ws, small, want=names, y=ys)
    g_small = evaluate(frame, centers, ws, small, want=('g',), y=ys)['g']
    Hp_small = evaluate(frame, centers, ws, small, want=('Hp',), p=p)['Hp']
    subsets = lambda group: (s for k in range(len(group) + 1) for s in itertools.combinations(group, k))
    worst_g = max(rel(evaluate(frame, centers, ws, small, want=('g',), y=ys,
                               have={name: everything[name] for name in s})['g'], g_small) for s in subsets(names))
    worst_Hp = max(rel(evaluate(frame, centers, ws, small, want=('Hp',), p=p,
                                have={name: everything[name] for name in s})['Hp'], Hp_small) for s in subsets(POINT))
    print()
    print('2. any subset of the keepable components may be handed in')
    print('   g  under all %d subsets of %d components: worst difference %.0e' % (2 ** len(names), len(names), worst_g))
    print('   Hp under all %d subsets of %d components:  worst difference %.0e' % (2 ** len(POINT), len(POINT), worst_Hp))

    # ---- 3. what wanting a per-sample component costs ------------------------------------------------
    plan = Plan(samples=512)
    have = evaluate(frame, centers, w, plan, want=INPUTS)
    print()
    print('3. peak MB during the call, including what it returns, under Plan(samples=512) with xi handed in')
    print('   the residual r is %.1f MB, the size of the data' % mb(tree_bytes(r_ref)))
    for want in (('f',), ('f', 'r'), ('f', 'g'), ('f', 'g', 'r'), ('f', 'g') + POINT_DATA[1:]):
        peak, _ = peak_of(lambda: evaluate(frame, centers, w, plan, want=want, y=y, have=have))
        print('   want=%-46s %6.2f' % (want, mb(peak)))

    # ---- 4. keeping against blocking -----------------------------------------------------------------
    print()
    print('4. one Gauss-Newton product: kept MB + peak MB during the call = total (the data is not counted)')
    print('   %-10s %24s %24s %24s' % ('kept', 'one block', 'blocks of 512', 'blocks of 128'))
    for keep_name, keep in (('FRAME', FRAME), ('INPUTS', INPUTS), ('NOTHING', NOTHING)):
        cells = []
        for b in (None, 512, 128):
            plan = Plan(samples=b)
            have = evaluate(frame, centers, w, plan, want=keep)
            peak, _ = peak_of(lambda: evaluate(frame, centers, w, plan, want=('Hp',), p=p, have=have))
            cells.append('%5.2f + %5.2f = %5.2f' % (mb(tree_bytes(have)), mb(peak), mb(tree_bytes(have) + peak)))
        print('   %-10s %24s %24s %24s' % ((keep_name,) + tuple(cells)))

    # ---- 5. sizes ------------------------------------------------------------------------------------
    print()
    print('5. bytes per sample of each keepable component (float64), for this problem')
    full = evaluate(frame, centers, ws, small, want=names, y=ys)
    print('   ' + '   '.join('%s %d' % (name, tree_bytes(full[name]) // n) for name in names))
    print('   the data, w and y together: %d' % ((tree_bytes(ws) + tree_bytes(ys)) // n))

    # ---- 6. the dependency graph, as the code runs it ------------------------------------------------
    graph = block_graph(frame, centers, ws, ys, project(frame, p), small)
    edges = [(dep, name) for name, (reads, _) in graph.items() for dep in reads]
    print()
    print('6. the dependency graph, as declared  (component <- the components it reads;')
    print('   every rule also reads frame cores; xi, xi_t, g and Hp read w; r reads y; the tangent grade reads p)')
    for name, (reads, _) in graph.items():
        print('   %-9s <- %s' % (name, ', '.join(reads) or '(inputs only)'))
    if '--dot' in sys.argv:
        path = sys.argv[sys.argv.index('--dot') + 1]
        with open(path, 'w') as fh:
            fh.write(dot_source(edges))
        print('   wrote', path)


def dot_source(edges):
    """The declared graph, with the inputs added by hand, as Graphviz DOT."""
    groups = [('plain', 'Plain: changes with the point', ['xi', 'mu', 'nu', 'eta', 'z']),
              ('resid', 'Residual grade, seed r: changes with the point and the data', ['r', 'f', 'xi_z', 'mu_z', 'nu_z', 'eta_z', 'g']),
              ('tangent', 'Tangent grade: changes with the direction p', ['xi_t', 'mu_t', 'nu_t', 'eta_t', 'Jp']),
              ('gn', 'Residual grade, seed J p: changes with the direction p', ['Jp.xi_z', 'Jp.mu_z', 'Jp.nu_z', 'Jp.eta_z', 'Hp'])]
    inputs = [('w', 'xi'), ('w', 'xi_t'), ('w', 'g'), ('w', 'Hp'), ('y', 'r'), ('p', 'xi_t'), ('p', 'mu_t'),
              ('p', 'nu_t'), ('p', 'eta_t'), ('p', 'Jp')]
    lines = ['digraph components {', '  rankdir=LR; node [shape=box, fontname="Helvetica", fontsize=11];',
             '  w [style=dashed]; y [style=dashed]; p [style=dashed];']
    for key, label, names in groups:
        lines.append('  subgraph cluster_%s { label="%s"; fontname="Helvetica"; fontsize=11;' % (key, label))
        lines += ['    "%s"%s;' % (name, ' [shape=ellipse]' if name in SUMMED else '') for name in names]
        lines.append('  }')
    lines += ['  "%s" -> "%s"%s;' % (a, b, ' [style=dashed]' if (a, b) in inputs else '') for a, b in inputs + list(edges)]
    return '\n'.join(lines + ['}']) + '\n'


if __name__ == '__main__':
    main()
