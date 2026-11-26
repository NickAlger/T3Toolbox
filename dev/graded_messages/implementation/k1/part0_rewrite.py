#!/usr/bin/env python3
"""Part 0's scripted rewrite of the tests' constructions (K1_DESIGN_NOTE.md section 2.2, part 0).

    python3 part0_rewrite.py [--apply] FILE...

For each `LeastSquaresBatch(` construction whose scope has exactly one old oracle pair `(kind, order)` in reach (the
scope, the class's setUp/body, the module: diff_audit.py's rule), the call becomes
`LeastSquaresBatch(kind, order, sample, y, weight=..., minibatch_scale=..., drawn_indices=...)` (the third and later
positional arguments turned into those keywords); every `LeastSquaresOracle(...)` becomes `LeastSquaresOracle()`. Every
other construction (no pair, or two, or a subclass of the oracle) is printed as MANUAL and left alone. Without
`--apply` it only reports. The edits are textual, on the exact source segment of each call (comments and alignment
outside the call untouched); the report lists each one, old and new, for the review.
"""
import ast
import sys

sys.path.insert(0, __import__('os').path.dirname(__file__))
import diff_audit as da

KEYS = ('weight', 'minibatch_scale', 'drawn_indices')


def main():
    apply_ = '--apply' in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith('--')]
    total = dict(batch=0, oracle=0, manual=0)
    for path in files:
        text = open(path).read()
        tree = ast.parse(text)
        scopes = da.scopes_of(path if path.startswith('tests/') else 'tests/' + path, text)
        # map each call node to its scope by line: rebuild a walker that records nodes
        edits = []
        classes = da._subclasses(tree)

        class W(ast.NodeVisitor):
            def __init__(self):
                self.stack = ['<module>']
                self.cls = [None]

            def visit_ClassDef(self, node):
                self.cls.append(node.name)
                self.stack.append('%s.<body>' % node.name)
                self.generic_visit(node)
                self.stack.pop()
                self.cls.pop()

            def visit_FunctionDef(self, node):
                self.stack.append('%s%s' % ((self.cls[-1] + '.') if self.cls[-1] else '', node.name))
                self.cls.append(self.cls[-1])
                self.generic_visit(node)
                self.cls.pop()
                self.stack.pop()

            def visit_Call(self, node):
                name = da._callee(node)
                scope = scopes.get(self.stack[-1])
                if name == da.ORACLE:
                    edits.append((node, 'oracle', None))
                elif name in classes:
                    edits.append((node, 'manual', 'a subclass of the oracle'))
                elif name == da.BATCH:
                    reach = da._fallbacks(scopes, scope) if scope else []
                    pairs = sorted(set(da._old_pairs(scopes, reach)))
                    if len(pairs) == 1:
                        edits.append((node, 'batch', pairs[0]))
                    else:
                        edits.append((node, 'manual', 'pairs in reach: %s' % (pairs,)))
                self.generic_visit(node)
        W().visit(tree)
        lines = text.splitlines(True)
        offsets = [0]
        for l in lines:
            offsets.append(offsets[-1] + len(l))

        def pos(lineno, col):
            return offsets[lineno - 1] + len(lines[lineno - 1].encode()[:col].decode())

        repl = []
        for node, what, info in edits:
            a, b = pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset)
            old = text[a:b]
            if what == 'manual':
                print('MANUAL %s:%d %s  [%s]' % (path, node.lineno, ' '.join(old.split())[:140], info))
                total['manual'] += 1
                continue
            func = text[pos(node.func.lineno, node.func.col_offset):pos(node.func.end_lineno, node.func.end_col_offset)]
            if what == 'oracle':
                new = func + '()'
                total['oracle'] += 1
            else:
                if any(isinstance(x, ast.Starred) for x in node.args) or any(k.arg is None for k in node.keywords):
                    print('MANUAL %s:%d %s  [starred]' % (path, node.lineno, ' '.join(old.split())[:140]))
                    total['manual'] += 1
                    continue
                args = [text[pos(x.lineno, x.col_offset):pos(x.end_lineno, x.end_col_offset)] for x in node.args]
                kws = [text[pos(k.lineno, k.col_offset):pos(k.end_lineno, k.end_col_offset)] for k in node.keywords]
                head = list(info) + args[:2]
                tail = ['%s=%s' % (KEYS[i], s) for i, s in enumerate(args[2:])]
                new = func + '(' + ', '.join(head + tail + kws) + ')'
                total['batch'] += 1
            repl.append((a, b, old, new))
            print('EDIT   %s:%d %s  ->  %s' % (path, node.lineno, ' '.join(old.split())[:110], ' '.join(new.split())[:150]))
        if apply_ and repl:
            for a, b, old, new in sorted(repl, reverse=True):
                text = text[:a] + new + text[b:]
            open(path, 'w').write(text)
    print('TOTAL', total)


if __name__ == '__main__':
    main()
