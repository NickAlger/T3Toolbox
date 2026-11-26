"""Tests whose own text is unchanged but which read a test helper the build changed or deleted.

    python helper_users.py OLDROOT NEWROOT

A *helper* is any non-test unit under tests/ (a function, a module or class table, a setUp, a class attribute).
A touched helper is one whose text changed or which went (unitdiff.py's units). It reaches:
  - in its own file: a module-level helper every unit that names it; a class member the units of its class (and of
    the subclasses in the file);
  - in a helper module (tests/**/_*.py, tests/oracle_sweeps/*.py): the units of every test file that imports that
    module and names the helper;
transitively (a helper that reads a touched helper is touched). Printed: every test function whose text is identical
in both roots and which a touched helper reaches, with the helpers.
"""
import ast
import collections
import io
import os
import re
import sys
import tokenize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unitdiff  # noqa: E402


def names_in(text):
    """The NAME tokens of a unit's code (strings and comments excluded)."""
    out = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NAME:
                out.add(tok.string)
    except (tokenize.TokenError, IndentationError):
        out |= set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text))
    return out


def imported_modules(src):
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out |= {a.name.split('.')[-1] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out.add(n.module.split('.')[-1])
            out |= {a.name for a in n.names}
    return out


def subclasses(src):
    """{class: {itself and every class of the file deriving from it, transitively}}."""
    bases = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.ClassDef):
            bases[n.name] = {b.id for b in n.bases if isinstance(b, ast.Name)}
    out = {c: {c} for c in bases}
    changed = True
    while changed:
        changed = False
        for c, bs in bases.items():
            for b in bs:
                if b in out and c not in out[b]:
                    out[b].add(c)
                    changed = True
    for c in list(out):
        for c2 in list(out[c]):
            out[c] |= out.get(c2, set())
    return out


def leaf(key):
    return key.split('.')[-1].replace(' <header>', '')


def is_test(key):
    return leaf(key).startswith('test') and not key.endswith('<header>')


def main(old_root, new_root):
    rels = sorted({r for r in unitdiff.py_files(old_root) + unitdiff.py_files(new_root) if r.startswith('tests/')})
    U = {}
    src_new = {}
    for rel in rels:
        po, pn = os.path.join(old_root, rel), os.path.join(new_root, rel)
        uo = unitdiff.units(open(po).read())[0] if os.path.exists(po) else {}
        un = unitdiff.units(open(pn).read())[0] if os.path.exists(pn) else {}
        U[rel] = (uo, un)
        src_new[rel] = open(pn).read() if os.path.exists(pn) else ''
    helper_mod = {rel: os.path.basename(rel)[:-3] for rel in rels
                  if os.path.basename(rel).startswith('_') or '/oracle_sweeps/' in rel}
    SUBS = {rel: subclasses(src) if src else {} for rel, src in src_new.items()}
    IMPORTS = {rel: imported_modules(src or 'pass') for rel, src in src_new.items()}
    touched = {}                                     # (rel, key) -> how
    for rel, (uo, un) in U.items():
        for k in uo:
            if is_test(k) or k.startswith(('import:', 'stmt:')) or leaf(k) == '<docstring>':
                continue
            if k not in un:
                touched[(rel, k)] = 'deleted'
            elif uo[k][2] != un[k][2]:
                touched[(rel, k)] = 'changed'

    def reaches(rel, key, text):
        """The touched helpers that the unit (rel, key) reads."""
        names = names_in(text)
        cls = key.split('.')[0] if '.' in key else None
        subs = SUBS[rel]
        hit = set()
        for (hrel, hk), how in touched.items():
            if (hrel, hk) == (rel, key):
                continue
            hl = leaf(hk)
            if hrel == rel:
                hcls = hk.split('.')[0] if '.' in hk else None
                if hcls is None and hl in names:
                    hit.add((hrel, hk))
                elif hcls is not None and cls is not None and cls in subs.get(hcls, {hcls}) and \
                        (hl in names or hl in ('setUp', 'setUpClass', 'tearDown') or hk.endswith('<header>')):
                    hit.add((hrel, hk))
            elif hrel in helper_mod and helper_mod[hrel] in IMPORTS[rel] and \
                    hl in names and '.' not in hk:
                hit.add((hrel, hk))
        return hit

    changed = True
    while changed:                                   # helpers reading touched helpers are touched
        changed = False
        for rel, (uo, un) in U.items():
            for k, (_, _, text) in un.items():
                if is_test(k) or (rel, k) in touched or k.startswith(('import:', 'stmt:')) or \
                        leaf(k) == '<docstring>':
                    continue
                if reaches(rel, k, text):
                    touched[(rel, k)] = 'reads a touched helper'
                    changed = True
    print('touched helpers:')
    for (rel, k), how in sorted(touched.items()):
        print('  %s: %s (%s)' % (rel, k, how))
    print('\nidentical tests reached by a touched helper:')
    for rel, (uo, un) in sorted(U.items()):
        for k, (_, _, text) in un.items():
            if not is_test(k) or k not in uo or uo[k][2] != text:
                continue
            hit = reaches(rel, k, text)
            if hit:
                print('  %s::%s  <- %s' % (rel, k.replace('.', '::'),
                                           sorted('%s:%s' % (os.path.basename(r), h) for r, h in hit)))


if __name__ == '__main__':
    main(*sys.argv[1:3])
