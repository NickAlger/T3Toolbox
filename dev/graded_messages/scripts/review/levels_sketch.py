"""SKETCH: inner and outer blocking controlled the same way.

The sample stack W has several axes. A plan is an ordered tuple of LEVELS, outermost first. Each level says
    keep   components to compute once for the whole block the level receives, and hand down
    scan   axes of the sample stack (indices into the ORIGINAL stack) to loop over, in nesting order
Axes that no level scans are vectorized. Inside the innermost level the ordinary evaluator runs on what is left.
Keeping across calls is the same thing one step further out: the caller's `have`.

Usage: python3 levels_sketch.py /path/to/T3Toolbox /path/to/handoff/scripts
"""
import importlib.util, pathlib, sys
from dataclasses import dataclass
import numpy as np

sys.path.append(sys.argv[1])
_spec = importlib.util.spec_from_file_location('plan_mockup', pathlib.Path(sys.argv[2]) / 'plan_mockup.py')
pm = importlib.util.module_from_spec(_spec); sys.modules['plan_mockup'] = pm; _spec.loader.exec_module(pm)
SUMMED = pm.SUMMED


@dataclass(frozen=True)
class Level:
    keep: tuple = ()     # computed for the whole block this level receives, before its scan
    scan: tuple = ()     # sample-stack axes this level loops over


def at(tree, pos, i):                                   # one index along sample-stack position `pos`
    return pm.tree_map(lambda a: a[(slice(None),) * pos + (i,)], tree)


def evaluate(frame, centers, inputs, levels, want, have, stack, p=None, axes=None):
    """inputs: {'w': [...], 'y': [...]} and have: {name: ...}; every array leads with the sample stack `stack`.
    Returns {name: value}: summed components reduced over the whole stack, per-sample ones with the stack."""
    axes = list(range(len(stack))) if axes is None else axes            # original ids of the remaining axes
    if not levels:
        return kernel(frame, centers, inputs, have, want, stack, p)
    level, rest = levels[0], levels[1:]
    if level.keep:                                                       # materialize for this whole block
        have = {**have, **kernel(frame, centers, inputs, have, level.keep, stack, p)}
    if not level.scan:
        return evaluate(frame, centers, inputs, rest, want, have, stack, p, axes)
    pos = axes.index(level.scan[0])
    inner = ((Level(scan=level.scan[1:]),) if len(level.scan) > 1 else ()) + rest
    sub_stack, sub_axes = stack[:pos] + stack[pos + 1:], axes[:pos] + axes[pos + 1:]
    outs = [evaluate(frame, centers, at(inputs, pos, i), inner, want, at(have, pos, i), sub_stack, p, sub_axes)
            for i in range(stack[pos])]                                  # the scan
    result = {}
    for name in want:
        if name in SUMMED:
            total = outs[0][name]
            for o in outs[1:]:
                total = pm.tree_add(total, o[name]) if not np.isscalar(total) else total + o[name]
            result[name] = total
        else:                                                            # stack per-sample parts back along the axis
            result[name] = stack_trees([o[name] for o in outs], pos)
    return result


def stack_trees(trees, pos):
    first = trees[0]
    if isinstance(first, (list, tuple)):
        return type(first)(stack_trees([t[j] for t in trees], pos) for j in range(len(first)))
    return np.stack(trees, axis=pos)


def kernel(frame, centers, inputs, have, want, stack, p):
    """The ordinary evaluator on one block, with whatever sample axes are left vectorized (flattened here)."""
    k = len(stack)
    flat = lambda t: pm.tree_map(lambda a: a.reshape((-1,) + a.shape[k:]), t)
    graph = pm.block_graph(frame, centers, flat(inputs['w']), flat(inputs['y']), p, pm.Plan())
    out = pm.run(graph, flat(have), list(want))
    return {n: v if n in SUMMED else pm.tree_map(lambda a: a.reshape(tuple(stack) + a.shape[1:]), v) for n, v in out.items()}


def main():
    rng = np.random.default_rng(0); np.random.seed(0)
    shape, stack = (20, 18, 22, 16), (4, 8, 16)                          # 512 samples as a three-axis stack
    x = pm.random_point(shape, (5, 6, 5, 4), (1, 6, 7, 5, 1))
    frame, centers = pm.orthogonal_frame(x)
    p = pm.random_tangent(frame, rng)
    inputs = {'w': [rng.standard_normal(stack + (N,)) for N in shape], 'y': [rng.standard_normal(stack + (N,)) for N in shape]}
    W = int(np.prod(stack))
    flatten = lambda t: [a.reshape((W,) + a.shape[3:]) for a in t]
    ref = pm.evaluate(frame, centers, flatten(inputs['w']), pm.Plan(), want=('f', 'g', 'r'), y=flatten(inputs['y']))
    Hp_ref = pm.evaluate(frame, centers, flatten(inputs['w']), pm.Plan(), want=('Hp',), p=p)['Hp']
    flat = lambda v: np.concatenate([np.ravel(a) for part in v for a in part])
    rel = lambda a, b: np.linalg.norm(flat(a) - flat(b)) / np.linalg.norm(flat(b))
    pp = pm.project(frame, p)
    FRAME, RESID = ('xi', 'mu', 'nu', 'eta'), ('r', 'xi_z', 'mu_z', 'nu_z', 'eta_z')
    TANGENT = ('Jp', 'Jp.xi_z', 'Jp.mu_z', 'Jp.nu_z', 'Jp.eta_z')
    cases = [
        ('no levels: everything vectorized', ()),
        ('outer only: scan axis 0', (Level(scan=(0,)),)),
        ('inner only, as the library chunks today: keep every per-sample component, scan axis 1 for the assembly',
         (Level(keep=FRAME + RESID + TANGENT, scan=(1,)),)),
        ('nested: scan axis 0; then keep the frame components and scan axis 1', (Level(scan=(0,)), Level(keep=FRAME, scan=(1,)))),
        ('three levels: scan 0; keep xi, scan 1; keep mu and nu, scan 2',
         (Level(scan=(0,)), Level(keep=('xi',), scan=(1,)), Level(keep=('mu', 'nu'), scan=(2,)))),
        ('one level, two axes, not in stack order: scan (2, 0)', (Level(scan=(2, 0)),)),
        ('inner axis before outer axis: scan 1; then keep the frame components and scan 0', (Level(scan=(1,)), Level(keep=FRAME, scan=(0,)))),
    ]
    print('sample stack %s = %d samples; differences from the flat, unblocked evaluation' % (stack, W))
    for name, levels in cases:
        out = evaluate(frame, centers, inputs, levels, ('f', 'g', 'r'), {}, stack, pp)
        Hp = evaluate(frame, centers, inputs, levels, ('Hp',), {}, stack, pp)['Hp']
        g, Hp = pm.project(frame, out['g']), pm.project(frame, Hp)
        r = [a.reshape((W,) + a.shape[3:]) for a in out['r']]
        print('  %-104s f %.0e  g %.0e  Hp %.0e  r %.0e' % (name, abs(out['f'] - ref['f']) / ref['f'], rel(g, ref['g']),
                                                              rel(Hp, Hp_ref), rel([r], [ref['r']])))


if __name__ == '__main__':
    main()
