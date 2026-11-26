"""Reviewer 4: the text of K1 part 1's scripted rewrite, read against the base (standard library only).

    python3 part1_textcheck.py [--base 78a34d77] [--root REPO] reference
    python3 part1_textcheck.py [--base ...] [--root ...] files PATH ...
    python3 part1_textcheck.py [--base ...] [--root ...] prose

reference: backend/sampling_definitions.py, unit by unit (a unit: a top-level def or class with its decorators,
    comments and docstring; the module docstring; the statements between). The INTENDED substitutions are applied to
    the base text first:
      - the signature line `weight: ... # of the type of z, each axis 1 ...` -> the two lines of note 4.1;
      - the signature line `y: ... # the data, of the type of z` -> note 4.1's;
      - `_the_seed(S, sample, y, weight, w_tilde, ...)`, `_dual_jet(S, X, weight)`, `_weight_squared(S, weight, i, ...)`
        -> the two weights in weight's place;
    and what still differs is printed hunk by hunk: those are the lines to read (the docstrings of prose_edits.md 6,
    the private helpers of 6.7, anything else is a finding). Also: per function, the set of `#` columns of its
    signature before and after (a re-aligned or broken column shows), and every changed line over 120 characters.
files: for each file, every changed hunk classified: REFLOW (the same words, other line breaks: prose re-flowed, a
    finding unless an inserted sentence forced it), COMMENTS (the comments added and removed, by tokenize), CODE; and
    lines over 120 characters that are new.
prose: every "After" block of prose_edits.md section 6 searched, whitespace-normalized, in the new reference (a
    `<to be pasted: ...>` line matches any output). FOUND or MISSING per block.
"""
import argparse
import ast
import collections
import difflib
import io
import os
import re
import subprocess
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..'))
REF = 't3toolbox/backend/sampling_definitions.py'
PROSE = os.path.join(HERE, '..', 'prose_edits.md')

OLD_W = ("        weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z's size; "
         "None = ones")
NEW_W = ("        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries "
         "();\n                                                             #   None = ones\n"
         "        full_weight:      typ.Optional[typ.Any] = None,      # z's shape, each W axis 1 or W's size; None = ones")
OLD_Y = "        y:                typ.Optional[typ.Any] = None,      # the data, of the type of z"
NEW_Y = "        y:                typ.Optional[typ.Any] = None,      # the data, z's shape exactly: probe len=d, F+W+(Ni,)"
SUBS = [
    (re.compile(r'_the_seed\(S, sample, y, weight, w_tilde, '), '_the_seed(S, sample, y, block_weight, full_weight, w_tilde, '),
    (re.compile(r'(_dual_jet\(S, .*), weight\)\)$', re.M), r'\1, block_weight, full_weight))'),
    (re.compile(r'_weight_squared\(S, weight, i, '), '_weight_squared(S, block_weight, full_weight, i, '),
]


def show(base, path, root):
    if base == 'WORKTREE':
        p = os.path.join(root, path)
        return open(p).read() if os.path.exists(p) else None
    try:
        return subprocess.run(['git', '-C', root, 'show', '%s:%s' % (base, path)], check=True, capture_output=True,
                              text=True).stdout
    except subprocess.CalledProcessError:
        return None


def units(text):
    """name -> text, top-level units in order; statements between defs are grouped under the next def's name."""
    tree = ast.parse(text)
    lines = text.splitlines()
    out = collections.OrderedDict()
    prev_end = 0
    for k, node in enumerate(tree.body):
        start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])]) - 1
        name = getattr(node, 'name', None)
        if k == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            name = '<module docstring>'
        if name is None:
            name = '<stmt %d>' % k
        if start > prev_end:
            out.setdefault('<between, before %s>' % name, '\n'.join(lines[prev_end:start]))
        out[name] = '\n'.join(lines[start:node.end_lineno])
        prev_end = node.end_lineno
    if prev_end < len(lines):
        out['<end>'] = '\n'.join(lines[prev_end:])
    return out


def intended(text):
    text = text.replace(OLD_W, NEW_W).replace(OLD_Y, NEW_Y)
    for rx, rep in SUBS:
        text = rx.sub(rep, text)
    return text


def sig_columns(unit):
    """The set of '#' columns of a def's signature lines (def line to the line ending the header)."""
    cols = []
    for line in unit.splitlines():
        if line.lstrip().startswith(('"""', "'''")):
            break
        j = line.find('  # ')
        if j >= 0:
            cols.append(j + 2)
        if line.rstrip().endswith(':') and ('->' in line or line.lstrip().startswith(')')):
            break
    return sorted(set(cols))


def do_reference(base, root):
    old, new = show(base, REF, root), show('WORKTREE', REF, root)
    uo, un = units(old), units(new)
    print('reference: %d units at %s, %d now' % (len(uo), base, len(un)))
    for name in un:
        if name not in uo:
            print('ADDED   %s' % name)
    for name in uo:
        if name not in un:
            print('DELETED %s' % name)
    residual = 0
    for name, text in uo.items():
        if name not in un:
            continue
        a, b = intended(text).splitlines(), un[name].splitlines()
        if text != un[name] and isinstance(name, str):
            ca, cb = sig_columns(text), sig_columns(un[name])
            if ca != cb:
                print('COLUMNS %s: signature comment columns %s -> %s' % (name, ca, cb))
        for line in un[name].splitlines():
            if len(line) > 120 and line not in text.splitlines():
                print('LONG    %s (%d): %s' % (name, len(line), line.strip()[:100]))
        if a == b:
            continue
        residual += 1
        print('--- RESIDUAL %s (after the intended substitutions)' % name)
        for line in difflib.unified_diff(a, b, lineterm='', n=1):
            if not line.startswith(('---', '+++')):
                print('    ' + line)
    print('units with a residual difference: %d' % residual)


def comments(text):
    out = collections.Counter()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.string.strip()] += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def do_files(base, root, paths):
    for path in paths:
        old, new = show(base, path, root) or '', show('WORKTREE', path, root) or ''
        a, b = old.splitlines(), new.splitlines()
        counts = collections.Counter()
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            A, B = a[i1:i2], b[j1:j2]
            words = lambda L: ' '.join(' '.join(L).split())
            if A and B and words(A) == words(B):
                counts['REFLOW'] += 1
                print('REFLOW   %s:%d-%d (same words, other breaks; %d -> %d lines)' % (path, j1 + 1, j2, len(A), len(B)))
                continue
            ca, cb = comments('\n'.join(A)), comments('\n'.join(B))
            if ca != cb:
                counts['COMMENTS'] += 1
                for c in (ca - cb):
                    print('COMMENT- %s:%d %s' % (path, i1 + 1, c[:110]))
                for c in (cb - ca):
                    print('COMMENT+ %s:%d %s' % (path, j1 + 1, c[:110]))
            else:
                counts['CODE'] += 1
            for k, line in enumerate(B):
                if len(line) > 120:
                    print('LONG     %s:%d (%d) %s' % (path, j1 + k + 1, len(line), line.strip()[:90]))
        print('== %s: %s' % (path, dict(counts)))


def prose_blocks():
    text = open(PROSE).read()
    s6 = text[text.index('## 6. Part 1'):text.index('## 7.')]
    lines = s6.splitlines()
    blocks, cur, label, before, start_label = [], [], '', False, ''
    for k, line in enumerate(lines):
        if line.startswith('### '):
            label = line[4:]
        if line.startswith('    ') or (cur and line == ''):
            if not cur:
                start_label = label
            cur.append(line)
            continue
        if cur:
            blocks.append((start_label, before, cur))
            cur = []
        if line.strip():
            before = 'before' in line.lower() and 'after' not in line.lower()
    if cur:
        blocks.append((start_label, before, cur))
    return [(lab, [l[4:] for l in blk if l.strip()]) for lab, bef, blk in blocks if not bef]


def norm(s):
    return ' '.join(s.split())


def do_prose(root):
    new = norm(show('WORKTREE', REF, root))
    found = missing = 0
    for label, blk in prose_blocks():
        parts = []
        for line in blk:
            m = re.match(r'\s*<to be pasted: (.*)>\s*$', line)
            # '...' in the draft is an elision (unchanged text, or a message pasted from a run): any text
            parts.append(r'\S.*?' if m else r'.*?'.join(re.escape(norm(p)) for p in norm(line).split('...')))
        rx = re.compile(r'\s*'.join(parts))
        if rx.search(new):
            found += 1
            print('FOUND   %s: %s' % (label, norm(blk[0])[:80]))
        else:
            missing += 1
            # the longest prefix of the block that is found, to say where it departs
            k = len(parts)
            while k > 0 and not re.search(r'\s*'.join(parts[:k]), new):
                k -= 1
            print('MISSING %s: %s\n        found through line %d of %d; departs at: %s'
                  % (label, norm(blk[0])[:80], k, len(blk), blk[k] if k < len(blk) else ''))
    print('prose blocks: %d found, %d missing' % (found, missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='78a34d77')
    ap.add_argument('--root', default=REPO)
    ap.add_argument('mode', choices=('reference', 'files', 'prose'))
    ap.add_argument('paths', nargs='*')
    args = ap.parse_args()
    if args.mode == 'reference':
        do_reference(args.base, args.root)
    elif args.mode == 'files':
        do_files(args.base, args.root, args.paths)
    else:
        do_prose(args.root)


if __name__ == '__main__':
    main()
