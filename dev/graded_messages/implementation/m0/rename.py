"""M0: the jet group `J -> P` and the size `J_in_i -> F_in_i`, as a token-level rename with an edit log.

    python rename.py ROOT forward LOG.tsv     # rename t3toolbox/ and tests/ under ROOT in place
    python rename.py ROOT inverse LOG.tsv     # the rules reversed (for the round trip)
    python rename.py ROOT inverse LOG.tsv --dry   # count only: on an untouched tree it must find nothing

Not a test (no `test_` name): a one-off of slice M0, kept as its record (M0_HANDOVER.md). Python's `tokenize` tells
code, strings and comments apart, so the group is told from the Jacobian `J` by context, never by the bare letter:

  S1  a string literal that is a grouped subscripts string (with `->`): every `J` is the group
  S2  the literals 'J', 'JW', 'JWKC' (the grammar's set, the test machinery's group reads)
  N1  identifiers: a `_`-segment that spells a group suffix (`mu_JWCa`, `jet_JX`), and the table NAMES
  N2  the `GroupLengths` field: a keyword `J=` and an attribute `.J` (found by AST)
  W1  size words in strings and comments (`J_in_i`, `J_in`, ...) and the NAMES words
  W2  shape grammar: the packed `(d,)+J+` (-> `F_in`) and a `J` followed by `+ W`, `+ X`, `+ ...`, `+ stack_shape`
  W3  the four files wholly about the group: every bare `J`
  W4  a word of group letters only, with `J` and `W` among them (`JWCa`, `WJCa`, `SJWCi` in prose and messages)
  F   per-file: `test_sampling_definitions.py`'s jet ring `J` -> `ring`; `ut3_operations.py`'s `J_i` -> `F_in_i`
  E   a short explicit list of sites, by line, reviewed by eye

The inverse applies the same rules with old and new swapped. Run on the untouched tree it must find nothing (no
target token exists yet: the collision check); run on the renamed tree it must give back the untouched tree.
"""
import ast
import io
import pathlib
import re
import sys
import tokenize

GROUP_FILES = {'t3toolbox/backend/contractions.py', 'tests/_jet_contract_cases.py', 'tests/test_jet_contract.py',
               'tests/test_contractions_interpreter.py'}
# Pre-existing bare `P` in the group files, which the inverse of W3 must not touch (none is the group): a projector in
# comments, and a local `P = np.polynomial.polynomial`. By line; the rename moves no line.
INVERSE_W3_SKIP = {('tests/test_contractions_interpreter.py', 241), ('tests/test_contractions_interpreter.py', 242),
                   ('tests/test_contractions_interpreter.py', 365), ('tests/test_contractions_interpreter.py', 366),
                   ('tests/test_jet_contract.py', 217), ('tests/test_jet_contract.py', 225)}

NAMES = {                                   # whole words, in code, strings and comments
    'len_J': 'len_P', 'lead_J': 'lead_P', 'factors_J': 'factors_P',
    'J_in_i': 'F_in_i', 'J_in_j': 'F_in_j', 'J_in': 'F_in', 'J_IN': 'F_IN',
    'test_every_J_literal_is_a_jet_contract_literal': 'test_every_P_literal_is_a_jet_contract_literal',
    'test_counts_are_nonnegative_ints_and_J_any_count': 'test_counts_are_nonnegative_ints_and_P_any_count',
}
FILE_NAMES = {                              # per file: (old, new, token kinds)
    'tests/test_sampling_definitions.py': [('J', 'ring', ('NAME',))],
    't3toolbox/backend/ut3_operations.py': [('J_i', 'F_in_i', ('STRING', 'COMMENT'))],
}
EXPLICIT = [                                # (file, line, old, new): the n-th whole-word match on the line, n = 1
    ('t3toolbox/backend/sampling_steps.py', 28, '``J``', '``P``'),
    ('t3toolbox/backend/sampling_steps.py', 39, '``J``', '``P``'),
    ('t3toolbox/backend/sampling_variables.py', 219, 'with ``J``', 'with ``P``'),
    ('t3toolbox/backend/sampling_variables.py', 1188, 'J = tuple', 'jet_sizes = tuple'),
    ('t3toolbox/backend/sampling_variables.py', 1189, 'return J +', 'return jet_sizes +'),
    ('t3toolbox/backend/ut3_operations.py', 132, ', J the largest', ', F_in the largest'),
    ('t3toolbox/backend/ut3_operations.py', 159, 'J = tuple', 'F_in = tuple'),
    ('t3toolbox/backend/ut3_operations.py', 171, 'zip(J, v', 'zip(F_in, v'),
    ('tests/test_sampling_variables.py', 328, '``J`` the', '``P`` the'),
    ('tests/test_least_squares_oracle.py', 77, 'standard_normal(J + (n, N)) for J, N', 'standard_normal(F_in + (n, N)) for F_in, N'),
    ('tests/backend/test_uniform_fitting.py', 718, 'standard_normal((J, W, n)) for J, n', 'standard_normal((F_in, W, n)) for F_in, n'),
    ('tests/test_sampling_functions.py', 491, 'standard_normal(J + W + (N,)) for J, N', 'standard_normal(F_in + W + (N,)) for F_in, N'),
]

SUBS = re.compile(r'^[A-Za-z]*(,[A-Za-z]*)*->[A-Za-z]*$')
BARE = r'(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])'


def rules(direction):
    """The rules of one direction: J -> P (forward) or P -> J (inverse)."""
    a, b = ('J', 'P') if direction == 'forward' else ('P', 'J')
    names = NAMES if direction == 'forward' else {v: k for k, v in NAMES.items()}
    file_names = {f: [(o, n, k) if direction == 'forward' else (n, o, k) for o, n, k in v]
                  for f, v in FILE_NAMES.items()}
    explicit = [(f, l, o, n) if direction == 'forward' else (f, l, n, o) for f, l, o, n in EXPLICIT]
    seg = re.compile(r'(?<![A-Za-z0-9])%s(?=[WKCX]+[a-z]*(?:_|\b))' % a)     # a group suffix segment's letter
    words = re.compile(r'(?<![A-Za-z0-9_])(%s)(?![A-Za-z0-9_])' % '|'.join(
        re.escape(w) for w in sorted(names, key=len, reverse=True)))
    packed = re.compile(r'(\(d,\)\s*\+\s*)%s(?=\s*\+)' % ('J' if direction == 'forward' else 'F_in'))
    shape = re.compile(BARE.format(a) + r'(?=\s*\+\s*(?:W|X|\.\.\.|stack_shape))')
    group_word = re.compile(r'(?<![A-Za-z0-9_])(?=[%sWKCSGX]*%s)(?=[%sWKCSGX]*W)[%sWKCSGX]{2,}[a-z]*(?![A-Za-z0-9_])'
                            % (a, a, a, a))         # with W: every group term of the library has it ('CP' does not)
    return dict(a=a, b=b, names=names, file_names=file_names, explicit=explicit, seg=seg, words=words,
                packed=packed, shape=shape, group_word=group_word, bare=re.compile(BARE.format(a)))


def ast_sites(src, letter):
    """(line, col) of every keyword `letter=` and attribute `.letter` (the GroupLengths field)."""
    sites = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Attribute) and node.attr == letter:
            sites.add(('attr', node.end_lineno, node.end_col_offset - len(letter)))
        if isinstance(node, ast.keyword) and node.arg == letter:
            sites.add(('kw', node.lineno, node.col_offset))
    return {(l, c) for _, l, c in sites}


def rename_text(text, R, kind, rel, line, log, in_group_file, skip_w3):
    """One token's text through the rules; each change logged."""
    a, b = R['a'], R['b']
    out = text

    def sub(pattern, repl, cls):
        nonlocal out
        def f(m):
            new = m.expand(repl) if isinstance(repl, str) else repl(m)
            if new != m.group(0):
                log.append((rel, line + m.string[:m.start()].count('\n'), cls, m.group(0), new))
            return new
        out = pattern.sub(f, out)

    if kind == 'STRING':
        try:
            value = ast.literal_eval(text)
        except Exception:
            value = None
        if isinstance(value, str) and SUBS.match(value.replace(' ', '')) and a in value:
            sub(re.compile(a), b, 'S1 subscripts')
            return out
        if value in ({'J', 'JW', 'JWKC'} if a == 'J' else {'P', 'PW', 'PWKC'}):
            sub(re.compile(a), b, 'S2 group literal')
            return out
    if kind == 'NAME':
        if text in R['names']:
            log.append((rel, line, 'N1 name', text, R['names'][text]))
            return R['names'][text]
        sub(R['seg'], b, 'N1 suffix')
        return out
    # strings and comments
    # the packed shape first: its `F_in` is not the size word `F_in` (the inverse would read it as `J_in`)
    sub(R['packed'], lambda m: m.group(1) + ('F_in' if a == 'J' else 'J'), 'W2 packed')
    sub(R['words'], lambda m: R['names'][m.group(1)], 'W1 size/name word')
    sub(re.compile(r'(?<![A-Za-z0-9])[A-Za-z0-9]*_[A-Za-z0-9_]*'),
        lambda m: R['seg'].sub(b, m.group(0)), 'W1 suffix word')
    sub(R['shape'], b, 'W2 shape')
    if in_group_file and not skip_w3:
        sub(R['bare'], b, 'W3 group file')
    sub(R['group_word'], lambda m: m.group(0).replace(a, b), 'W4 group word')
    return out


def explicit(src, rel, R, log):
    '''E: the explicit sites, by line. Forward runs them after the token rules, the inverse before (the mirror).'''
    lines = src.splitlines(keepends=True)
    for f, line, old, new in R['explicit']:
        if f != rel:
            continue
        if lines[line - 1].count(old) != 1:
            raise SystemExit('E: %s:%d does not hold %r exactly once: %r' % (f, line, old, lines[line - 1]))
        lines[line - 1] = lines[line - 1].replace(old, new)
        log.append((rel, line, 'E explicit', old, new))
    return ''.join(lines)


def rename_file(path, rel, R, log, dry):
    src = path.read_text()
    if R['a'] == 'P' and not dry:
        src = explicit(src, rel, R, log)
    lines = src.splitlines(keepends=True)
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))
    a, b = R['a'], R['b']
    in_group_file = rel in GROUP_FILES
    field_sites = ast_sites(src, a) if rel != 'tests/test_sampling_definitions.py' else set()
    edits = []      # (start, end, new)
    tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for i, tok in enumerate(tokens):
        kind = tokenize.tok_name[tok.type]
        if kind not in ('NAME', 'STRING', 'COMMENT'):
            continue
        (l0, c0), (l1, c1) = tok.start, tok.end
        start, end = offsets[l0 - 1] + c0, offsets[l1 - 1] + c1
        text = tok.string
        new = None
        for old, nw, kinds in R['file_names'].get(rel, ()):
            if kind in kinds:
                if kind == 'NAME' and text == old:
                    new = nw
                    log.append((rel, l0, 'F file name', old, nw))
                elif kind != 'NAME':
                    pat = re.compile(BARE.format(re.escape(old)))
                    if pat.search(text):
                        for _ in pat.findall(text):
                            log.append((rel, l0, 'F file word', old, nw))
                        text = pat.sub(nw, text)
                        new = text
        if new is not None and kind == 'NAME':
            edits.append((start, end, new))
            continue
        if kind == 'NAME' and text == a:
            if (l0, c0) in field_sites or in_group_file and not (rel, l0) in INVERSE_W3_SKIP:
                edits.append((start, end, b))
                log.append((rel, l0, 'N2 field' if (l0, c0) in field_sites else 'W3 group file', a, b))
            continue
        skip_w3 = R['a'] == 'P' and (rel, l0) in INVERSE_W3_SKIP
        out = rename_text(text, R, kind, rel, l0, log, in_group_file, skip_w3)
        if out != tok.string:
            edits.append((start, end, out))
    for start, end, new in sorted(edits, reverse=True):
        src = src[:start] + new + src[end:]
    if R['a'] == 'J' and not dry:
        src = explicit(src, rel, R, log)
    return src


def main():
    root, direction, log_path = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    dry = '--dry' in sys.argv
    R = rules(direction)
    log = []
    files = sorted(list((root / 't3toolbox').rglob('*.py')) + list((root / 'tests').rglob('*.py')))
    changed = 0
    for path in files:
        rel = str(path.relative_to(root))
        new = rename_file(path, rel, R, log, dry)
        if new != path.read_text():
            changed += 1
            if not dry:
                path.write_text(new)
    used = {(row[0], row[1]) for row in log if row[2] == 'E explicit'}
    missing = [(f, l) for f, l, _, _ in R['explicit'] if (f, l) not in used]
    if missing and not dry:
        raise SystemExit('E: sites never reached: %s' % missing)
    with open(log_path, 'w') as fh:
        fh.write('file\tline\tclass\told\tnew\n')
        for row in log:
            fh.write('\t'.join(str(x) for x in row) + '\n')
    print('%s: %d edits in %d files%s' % (direction, len(log), changed, ' (dry)' if dry else ''))


if __name__ == '__main__':
    main()
