"""The dependency graph of every sampling case, generated from the graded program and not written by hand.

`graded_messages_prototype.py` runs ONE program (seven site rules) for entries, apply and probe, their Jacobians,
transposes and exact-Hessian terms, with or without jets, and reproduces the library (see its own check).
This script runs the same program with every product-rule term traced, so that for each case it can print
which components exist, what each one reads, and which cores it goes through.

A case is three choices:
    readout    where the output is taken: the right boundary (apply, entries) or every mode leg (probe)
    frame      the manifold frame (D, U, L, R), or the corewise frame (B, G, G, G) at the base point
    operation  the grade that is wanted: plain (value), t (J p), z (J^T r), tz (the exact curvature term)

Usage:  python3 component_graphs.py /path/to/T3Toolbox
"""
import importlib.util
import itertools
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('proto', HERE / 'graded_messages_prototype.py')
proto = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proto)
UP, RIGHT, LEFT, DOWN, OUT, GRAD_G, GRAD_B, CLOSE = (proto.UP, proto.RIGHT, proto.LEFT, proto.DOWN, proto.OUT,
                                                     proto.GRAD_G, proto.GRAD_B, proto.CLOSE)


def gmul_traced(rule, ops, names, P, out_name, reads, cores, core_names, want=None):
    """proto.gmul, recording for each output grade the (operand, grade) pairs and the cores of its terms."""
    ins, out = rule.split('->')
    ins = ins.split(',')
    is_msg = [isinstance(o, dict) for o in ops]
    msgs = [k for k in range(len(ops)) if is_msg[k]]
    subs = ','.join(('w' + s) if is_msg[k] else s for k, s in enumerate(ins)) + '->w' + out
    result = {}
    for gs in itertools.product(*[list(o.keys()) if m else o.grades() for o, m in zip(ops, is_msg)]):
        if sum(len(g) for g in gs) != len(set(''.join(gs))):
            continue
        total = ''.join(sorted(''.join(gs)))
        if want is not None and total not in want:
            continue
        arrays, used = [], []
        for k, (o, g) in enumerate(zip(ops, gs)):
            if is_msg[k]:
                arrays.append(o[g])
            elif 't' in g:
                arrays.append(o.dot)
                used.append(core_names[len(ins[k])]['dot'])
            else:
                port = [ins[k2] for k2 in msgs if 't' in gs[k2]]
                port = port[0] if port else [c for c in out if c in ins[k]][0]
                arrays.append(o.toward.get(port))
                used.append(core_names[len(ins[k])].get(port))
        if any(a is None for a in arrays):
            continue
        hit = False
        for orders in itertools.product(range(P), repeat=len(msgs)):
            t = sum(orders)
            if t >= P:
                continue
            sl = [arrays[k][orders[msgs.index(k)]] if is_msg[k] else arrays[k] for k in range(len(ops))]
            if any(not np.any(x) for x in sl):
                continue
            val = np.einsum(subs, *sl)
            if total not in result:
                result[total] = np.zeros((P,) + val.shape, val.dtype)
            result[total][t] += val
            hit = True
        if hit:
            reads.setdefault((out_name, total), set()).update((names[k], gs[k]) for k in msgs)
            cores.setdefault((out_name, total), set()).update(used)
    return result


def run_traced(B, G, w, left, right, P, core_names, grad=('z', 'tz')):
    """proto.run with every rule traced. Returns (outputs, reads, cores)."""
    reads, cores = {}, {}
    mul = lambda rule, ops, names, out_name, want=None: gmul_traced(rule, ops, names, P, out_name, reads, cores,
                                                                    core_names, want)
    d = len(G)
    X = [mul(UP, [B[i], w[i]], (None, 'w'), 'xi') for i in range(d)]
    M = [left]
    for i in range(d):
        M.append(mul(RIGHT, [M[i], G[i], X[i]], ('mu', None, 'xi'), 'mu'))
    N = [None] * d + [right]
    for i in reversed(range(d)):
        N[i] = mul(LEFT, [G[i], X[i], N[i + 1]], (None, 'xi', 'nu'), 'nu')
    E = [mul(DOWN, [M[i], G[i], N[i + 1]], ('mu', None, 'nu'), 'eta') for i in range(d)]
    Z = [mul(OUT, [E[i], B[i]], ('eta', None), 'z') for i in range(d)]
    dG = [mul(GRAD_G, [M[i], X[i], N[i + 1]], ('mu', 'xi', 'nu'), 'dG', grad) for i in range(d)]
    dB = [mul(GRAD_B, [E[i], w[i]], ('eta', 'w'), 'dB', grad) for i in range(d)]
    Phi = mul(CLOSE, [M[d], N[d]], ('mu', 'b_R'), 'phi')        # the closure reads the right boundary vector
    return dict(X=X, M=M, N=N, E=E, Z=Z, dG=dG, dB=dB, Phi=Phi), reads, cores


SHORTCUT = {}
MANIFOLD = {3: {'a': 'R', 'b': 'L', 'i': 'U', 'dot': 'H'}, 2: {'i': 'D', 'o': 'D', 'dot': 'V'}}
COREWISE = {3: {'a': 'G', 'b': 'G', 'i': 'G', 'dot': 'Gdot'}, 2: {'i': 'B', 'o': 'B', 'dot': 'Bdot'}}


def label(node):
    name, grade = node
    if name == 'w':
        return {'': 'w', 'z': 'r'}[grade]            # a leaf at grade z is the probe residual
    if name == 'b_R':
        return 'b_R'
    return name + ('_' + grade if grade else '')


def needed_graph(reads, cores, wanted, seeds):
    """Walk back from the wanted components. Returns [(component, reads, cores)] in dependency order."""
    order, seen = [], set()

    def visit(node):
        if node in seen or node[0] in ('w', 'b_R'):
            return
        seen.add(node)
        deps = sorted(n for n in reads.get(node, ()) if n != node)
        for n in deps:
            visit(n)
        order.append((node, deps))

    for node in wanted:
        if node in reads:
            visit(node)
    rows = []
    for node, deps in order:
        names = [label(n) for n in deps]
        if node in seeds:
            names.append(seeds[node])
        rows.append((label(node), names, sorted(c for c in cores.get(node, ()) if c)))
    return rows


def cases(repo, P):
    """Yield (kind, operation, frame name, rows) for every case, at jet length P."""
    rng = np.random.default_rng(0)
    rn = rng.standard_normal
    Ns, nD, nU = (6, 5, 7, 4), (3, 4, 2, 3), (2, 3, 3, 2)
    rL, rR = (1, 3, 2, 4, 1), (1, 2, 4, 3, 1)
    d, W = len(Ns), 5
    D = [rn((nD[i], Ns[i])) for i in range(d)]
    U = [rn((rL[i], nU[i], rR[i + 1])) for i in range(d)]
    L = [rn((rL[i], nD[i], rL[i + 1])) for i in range(d)]
    R = [rn((rR[i], nD[i], rR[i + 1])) for i in range(d)]
    V = [rn((nU[i], Ns[i])) for i in range(d)]
    H = [rn((rL[i], nD[i], rR[i + 1])) for i in range(d)]
    frame, variation = (D, U, L, R), (V, H)
    Bc, Gc = D, L
    Bd, Gd = [rn(b.shape) for b in Bc], [rn(g.shape) for g in Gc]
    ww, pp = [rn((W, N)) for N in Ns], [rn((W, N)) for N in Ns]
    index = np.stack([rng.integers(0, N, size=W) for N in Ns])
    onehot = [np.eye(N)[index[i]] for i, N in enumerate(Ns)]
    bnd = proto.ones_boundary(P, W)

    for kind, vecs in (('apply', ww), ('entries', onehot), ('probe', ww)):
        leaf = lambda i: proto.jet(P, vecs[i], pp[i])
        plain = [{'': leaf(i)} for i in range(d)]
        out_name = 'z' if kind == 'probe' else 'phi'
        if kind == 'probe':                                    # the residual sits on the leaves
            resid = [{'': leaf(i), 'z': proto.dual_jet(P, rn((P, W, Ns[i])))} for i in range(d)]
            seeded_w, seeded_right, seeds = resid, bnd, {}
        else:                                                  # a scalar residual seeds the right boundary
            seeded_w = plain
            seeded_right = {'': bnd[''], 'z': proto.dual_jet(P, rn((P, W)))[..., None]}
            seeds = {('nu', 'z'): 'c'}
        grad = lambda g: [('dB', g), ('dG', g)]
        runs = [('value', 'corewise', proto.corewise((Bc, Gc)), plain, bnd, COREWISE, [(out_name, '')], {}),
                ('J p', 'manifold', proto.manifold(frame, variation), plain, bnd, MANIFOLD, [(out_name, 't')], {}),
                ('J^T r', 'manifold', proto.manifold(frame), seeded_w, seeded_right, MANIFOLD, grad('z'), seeds),
                ('exact K p', 'corewise', proto.corewise((Bc, Gc), (Bd, Gd)), seeded_w, seeded_right, COREWISE,
                 grad('tz'), seeds)]
        for op, frame_name, (B, G), w, right, core_names, wanted, sd in runs:
            out, reads, cores = run_traced(B, G, w, bnd, right, P, core_names)
            ref = proto.run(B, G, w, bnd, right, P)            # the traced copy is the prototype's program
            for key in ('Phi',):
                for g in ref[key]:
                    assert np.array_equal(ref[key][g], out[key][g])
            for key in ('Z', 'dG', 'dB'):
                for a, b in zip(ref[key], out[key]):
                    for g in a:
                        assert np.array_equal(a[g], b[g])
            if kind == 'apply' and op == 'J^T r' and P == 1:   # a shortcut: with a scalar residual, nu_z = c * nu
                plain_nu = proto.run(B, G, plain, bnd, bnd, P)['N']
                c = right['z'][0]
                SHORTCUT['nu_z'] = max(np.abs(out['N'][i]['z'][0] - c * plain_nu[i][''][0]).max()
                                       / np.abs(out['N'][i]['z'][0]).max() for i in range(d))
            yield kind, op, frame_name, needed_graph(reads, cores, wanted, sd)


def main(repo):
    sys.path.append(repo)
    at = {P: list(cases(repo, P)) for P in (1, 3)}
    same_jets = all([(r[0], sorted(r[1])) for r in a[3]] == [(r[0], sorted(r[1])) for r in b[3]]
                    for a, b in zip(at[1], at[3]))
    table = {(k, op): rows for k, op, _, rows in at[1]}
    same_entries = all(table[('apply', op)] == table[('entries', op)] for op in ('value', 'J p', 'J^T r', 'exact K p'))
    print('The dependency graph of every case, traced from the graded program (order 0).')
    print('entries has the same graph as apply in every case: %s' % same_entries)
    print('with jets (orders 0..2) every case has the same components and reads as at order 0: %s' % same_jets)
    print('apply and entries: the adjoint sweep nu_z equals the scalar residual times the plain nu: largest relative difference %.0e'
          % SHORTCUT['nu_z'])
    for kind in ('probe', 'apply'):
        for k, op, frame_name, rows in at[1]:
            if k != kind:
                continue
            print()
            print('%s, %s, %s frame' % (('apply and entries' if kind == 'apply' else kind), op, frame_name))
            for name, deps, cs in rows:
                print('   %-8s <- %-44s through %s' % (name, ', '.join(deps), ', '.join(cs) or '-'))

    # the grid: which component each case needs
    letters, grades = ('xi', 'mu', 'nu', 'eta'), ('', 't', 'z', 'tz')
    ops = ('value', 'J p', 'J^T r', 'exact K p')
    print()
    print('Which components each case needs  (P = probe, A = apply and entries, . = not needed)')
    print('   %-8s %s' % ('', '   '.join('%-9s' % op for op in ops)))
    for grade in grades:
        for letter in letters:
            name = letter + ('_' + grade if grade else '')
            cells = []
            for op in ops:
                marks = ''.join(m for m, kind in (('P', 'probe'), ('A', 'apply'))
                                if any(r[0] == name for r in table[(kind, op)]))
                cells.append('%-9s' % (marks or '.'))
            print('   %-8s %s' % (name, '   '.join(cells)))


if __name__ == '__main__':
    main(sys.argv[1])
