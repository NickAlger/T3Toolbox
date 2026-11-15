"""R1 freeness check (plan §3.4, critic r14): every image of a NAME rename is unbound and unread in its scope.

For every NAME hit of the dry run (``dev/store_rename_apply.py``), its image ``N`` must be FREE in the hit's scope at
``v2026.4.0`` (the working tree's ``t3toolbox/``, ``tests/``, ``examples/`` equal the tag):

* the scope of a code hit is its outermost enclosing ``def`` (methods: the method), docstrings excluded; of a doctest
  hit, the docstring's examples (one doctest namespace); of a module-level hit, the module's code outside defs;
* ``N`` is free there iff every NAME occurrence of ``N`` in the scope is itself renamed away by the same dry run (so a
  simultaneous swap ``UM_t <-> DM_t`` or a chain ``U -> D``, ``D -> Psi`` passes as a pair), except occurrences inside
  a nested ``def`` that binds ``N`` itself (a parameter, as ``core_dot``'s ``gB``; or a local or generator
  variable, as ``core_dot``'s ``b``) while no renamed image lies inside it: reported and excepted.

Usage::

    python dev/store_rename_slices/r1_freeness.py > dev/store_rename_slices/R1_freeness.md

Reports every image checked (per tier), every collision with its context, and every excepted nested-def parameter.
"""
import ast
import collections
import importlib.util
import os
import sys
import typing as typ

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location('store_rename_apply', os.path.join(REPO, 'dev', 'store_rename_apply.py'))
sra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sra)

LETTER_TIERS = ('C', 'C-B', 'C-D', 'C-1.2a-B', 'C-1.2a-D', 'C-VH', 'C-sig', 'C-site', 'A3x', 'B3')


def scopes_of(src: str) -> typ.Tuple[typ.List[typ.Tuple[int, int, str]], typ.List[typ.Tuple[int, int]],
                                       typ.List[typ.Tuple[int, int, typ.Set[str]]]]:
    """(outermost function spans, docstring spans, nested defs with their parameter names)."""
    tree = ast.parse(src)
    outer, docs, nested = [], [], []

    def visit(node, depth):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                first = min([ch.lineno] + [d.lineno for d in getattr(ch, 'decorator_list', [])])
                if depth == 0 and not isinstance(ch, ast.Lambda):
                    outer.append((first, ch.end_lineno, ch.name))
                elif depth > 0 and not isinstance(ch, ast.Lambda):
                    a = ch.args
                    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
                    params |= {x.arg for x in (a.vararg, a.kwarg) if x}
                    bound = set(params) | {n.id for n in ast.walk(ch)
                                           if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                    nested.append((first, ch.end_lineno, bound))
                visit(ch, depth + (0 if isinstance(ch, ast.Lambda) else 1))
            else:
                visit(ch, depth)
    visit(tree, 0)
    for n in [tree] + list(ast.walk(tree)):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            s = n.body[0]
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                docs.append((s.value.lineno, s.value.end_lineno))
    return outer, docs, nested


def main() -> None:
    entries = sra.Entries()
    p1 = sra.Pass1(entries, sra.load_fences(), sra.load_letter_sites())
    p1.record_names = []
    hits = []
    srcs = {}
    for path in sra.python_paths():
        src = open(os.path.join(REPO, path), encoding='utf-8').read()
        srcs[path] = src
        _, h = p1.python_file(path, src)
        hits += h
    renamed = {(h.file, h.line, h.col) for h in hits}
    names = collections.defaultdict(list)                 # file -> [(line, col, tok, ctx)]
    for path, line, col, tok, ctx in p1.record_names:
        names[path].append((line, col, tok, ctx))
    sc = {p: scopes_of(s) for p, s in srcs.items()}

    def scope_key(path, line, ctx):
        outer, docs, _ = sc[path]
        if ctx.startswith('DOCTEST'):
            d = [s for s in docs if s[0] <= line <= s[1]]
            return ('doctest', min(d, key=lambda s: s[1] - s[0]))
        o = [s for s in outer if s[0] <= line <= s[1]]
        return ('def', (o[0][0], o[0][1], o[0][2])) if o else ('module', (0, 0, '<module>'))

    def in_scope(path, key, line, ctx):
        kind, span = key
        if kind == 'doctest':
            return ctx.startswith('DOCTEST') and span[0] <= line <= span[1]
        if ctx.startswith('DOCTEST'):
            return False
        outer = sc[path][0]
        if kind == 'def':
            return span[0] <= line <= span[1]
        return not any(a <= line <= b for a, b, _ in outer)

    checked = collections.Counter()
    images_by_scope = collections.defaultdict(lambda: collections.defaultdict(list))   # (file, key) -> N -> hits
    for h in hits:
        if not h.ctx.endswith('NAME'):
            continue
        images_by_scope[(h.file, scope_key(h.file, h.line, h.ctx))][h.new].append(h)
    collisions, excepted = [], []
    for (path, key), imgs in sorted(images_by_scope.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        nested = sc[path][2]
        for N, hs in sorted(imgs.items()):
            for t in {h.tier for h in hs}:
                checked[t] += 1
            occ = [(l, c, tok, ctx) for (l, c, tok, ctx) in names[path]
                   if tok == N and in_scope(path, key, l, ctx) and (path, l, c) not in renamed]
            bad = []
            for l, c, tok, ctx in occ:
                nd = [n for n in nested if n[0] <= l <= n[1] and N in n[2]
                      and not any(n[0] <= h.line <= n[1] for h in hs)]
                if nd:
                    excepted.append((path, key, N, l, hs[0]))
                else:
                    bad.append((l, c))
            if bad:
                collisions.append((path, key, N, bad, hs))
    P = print
    P('# R1 freeness check (plan §3.4)')
    P('')
    P('Generated by `python dev/store_rename_slices/r1_freeness.py`. Scope = outermost enclosing def (docstrings '
      'excluded), or the docstring for doctest hits. An image is free iff every occurrence of it in the scope is '
      'itself renamed away (swaps and chains pass as pairs); occurrences inside a nested def that binds it itself '
      '(parameter or local) with no renamed image inside are excepted (listed).')
    P('')
    P('Images checked (one per (scope, image)), by tier: ' + ', '.join('%s %d' % kv for kv in sorted(checked.items())))
    P('')
    letter = [c for c in collisions if any(h.tier in LETTER_TIERS for h in c[4])]
    other = [c for c in collisions if c not in letter]
    for title, group in (('Collisions: letter-tier images (tier C, B/D rows, §1.2a, σ̂, Psi, A3 exception, B3)', letter),
                         ('Collisions: other tiers (A, B1; informational)', other)):
        P('## %s (%d)' % (title, len(group)))
        P('')
        for path, key, N, bad, hs in group:
            src_lines = srcs[path].splitlines()
            P('- `%s` scope %s `%s`: image `%s` (from %s) already occurs unrenamed at %s'
              % (path, key[0], key[1][2] if key[0] != 'doctest' else 'lines %d-%d' % key[1], N,
                 ', '.join(sorted({'`%s` %s' % (h.old, h.tier) for h in hs})),
                 ', '.join('%d:%d' % b for b in bad[:6])))
            for l, c in bad[:3]:
                P('  - `%d: %s`' % (l, src_lines[l - 1].strip()[:140]))
        P('')
    P('## Excepted: bound inside a nested def (%d)' % len(excepted))
    P('')
    for path, key, N, l, h in excepted:
        P('- `%s:%d` image `%s` (from `%s`, %s) is bound by the nested def there' % (path, l, N, h.old, h.tier))


if __name__ == '__main__':
    main()
