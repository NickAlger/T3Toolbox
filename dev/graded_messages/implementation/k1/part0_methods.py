#!/usr/bin/env python3
"""Part 0's scripted move of the oracle's batch methods to the batch, in the tests (K1_DESIGN_NOTE.md section 2.2).

    python3 part0_methods.py [--apply] FILE...

Rewrites, on the exact source segment of each call (nothing outside the call touched), and prints every edit:
    <o>.take(<b>, <i>)               -> <b>.take(<i>)            (<o> any expression whose text names an oracle)
    <o>.n_measurements(<b>)          -> <b>.n_measurements
    <o>.check_batch(<b>, <rest>)     -> <b>.check(<rest>)
    pack_batch(<k>, <o>, <b>, <N>)   -> pack_batch(<b>, <N>)     (four positional arguments: the old signature)
    flat_draw(<o>, <b>, <n>)         -> flat_draw(<b>, <n>)
    prepare_batch(<o>, <b>, ...)     -> prepare_batch(<b>, ...)  (first argument naming an oracle)
A call argument that is itself a call is parenthesized only when it is not a name, an attribute or a call.
"""
import ast
import sys


def main():
    apply_ = '--apply' in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = 0
    for path in files:
        text = open(path).read()
        lines = text.splitlines(True)
        offs = [0]
        for l in lines:
            offs.append(offs[-1] + len(l))
        pos = lambda ln, col: offs[ln - 1] + len(lines[ln - 1].encode()[:col].decode())
        seg = lambda node: text[pos(node.lineno, node.col_offset):pos(node.end_lineno, node.end_col_offset)]
        repl = []
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
            base = seg(f.value) if isinstance(f, ast.Attribute) else ''
            args = [seg(a) for a in node.args]
            kws = [seg(k) for k in node.keywords]
            wrap = lambda a, s: s if isinstance(a, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)) else '(%s)' % s
            new = None
            if name == 'take' and 'oracle' in base.lower() + (' full' if base == 'full' else '') and len(args) == 2:
                new = '%s.take(%s)' % (wrap(node.args[0], args[0]), args[1])
            elif name == 'take' and base == 'full' and len(args) == 2:
                new = '%s.take(%s)' % (wrap(node.args[0], args[0]), args[1])
            elif name == 'n_measurements' and 'oracle' in base.lower() and len(args) == 1:
                new = '%s.n_measurements' % wrap(node.args[0], args[0])
            elif name == 'check_batch' and 'oracle' in base.lower() and len(args) >= 1:
                new = '%s.check(%s)' % (wrap(node.args[0], args[0]), ', '.join(args[1:] + kws))
            elif name == 'pack_batch' and len(args) == 4:
                new = '%s(%s)' % (seg(f), ', '.join(args[2:] + kws))
            elif name == 'flat_draw' and len(args) == 3:
                new = '%s(%s)' % (seg(f), ', '.join(args[1:] + kws))
            elif name == 'prepare_batch' and args and 'oracle' in args[0].lower():
                new = '%s(%s)' % (seg(f), ', '.join(args[1:] + kws))
            if new is not None:
                a, b = pos(node.lineno, node.col_offset), pos(node.end_lineno, node.end_col_offset)
                repl.append((a, b, text[a:b], new, node.lineno))
        # innermost first is not needed: none of these calls nest one another in the tests (checked: overlaps refused)
        repl.sort()
        for (a1, b1, *_), (a2, b2, *_) in zip(repl, repl[1:]):
            assert b1 <= a2, ('overlapping edits', path)
        for a, b, old, new, ln in repl:
            print('EDIT %s:%d  %s  ->  %s' % (path, ln, ' '.join(old.split())[:120], ' '.join(new.split())[:120]))
            n += 1
        if apply_:
            for a, b, old, new, ln in sorted(repl, reverse=True):
                text = text[:a] + new + text[b:]
            open(path, 'w').write(text)
    print('TOTAL', n)


if __name__ == '__main__':
    main()
