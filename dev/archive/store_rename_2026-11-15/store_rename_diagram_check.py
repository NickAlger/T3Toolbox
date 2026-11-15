"""Store rename (plan §4A c, g; §5.3): the diagram/equation gates K10, K10b, K11 and K12.

One-off (D-9), in ``dev/``, never imported by the library. Reads the R1 inventory
``dev/store_rename_slices/diagrams_inventory.md`` and the fence list ``dev/store_rename_slices/fences.tsv``.

- **K10** (a block rewritten in place): the same line count, the same length per line, and every column where old
  and new differ holds a label character ``[A-Za-z0-9_']`` on both sides. Raw text: tabs are not expanded.
- **K10b** (a block re-laid out): the same line count, and every old alignment column -- a field start after two
  or more spaces in at least two lines, or a shared ``=``, ``:``, ``|``, ``#`` (not a ``:`` closing a label) -- maps
  to one consistent new column.
  Field-comment blocks also need every line of the new span to start its comment at the new start column.
- **K11** (fences untouched by scripts): every fenced span's text is byte-identical to ``v2026.4.0``, located by
  span TEXT, not by line; with ``--pass1-tree DIR`` also no sentinel ``ZQX\\d+_XQZ`` inside a fence there. A block
  rewritten by hand (its inventory new text recorded and different) is an exception, verified by K12 instead
  (``--list-exceptions`` prints them).
- **K12** (completion): every block's new span and new text are filled and equal the file verbatim; a block whose
  new text equals its old text carries a recorded ``same-reason``.

Usage (repo root, the t3toolbox env)::

    python dev/store_rename_diagram_check.py [all|k10|k10b|k11|k12|self-test] [--tree DIR] [--pass1-tree DIR]

``--tree`` is the tree under test (default: the working tree). Exit status 1 on any FAIL; PENDING (blocks not yet
rewritten) is not a failure. The functions ``k10``, ``k10b``, ``k11``, ``k12`` are importable by
``dev/store_rename_check.py``.
"""
import collections
import hashlib
import os
import re
import subprocess
import sys
import typing as typ

TAG = 'v2026.4.0'
R1_BASE = '88fbbedb'                   # the base for files absent at the tag (docs/glossary.md)
SLICES = 'dev/store_rename_slices'
FENCES = SLICES + '/fences.tsv'
INVENTORY = SLICES + '/diagrams_inventory.md'
LABEL_CHAR = re.compile(r"[A-Za-z0-9_']")
SENTINEL = re.compile(r'ZQX\d+_XQZ')
SEPARATORS = '=:|#'

Fence = collections.namedtuple('Fence', 'id file first last col_start col_end fenced kind sha1')
Entry = collections.namedtuple('Entry', 'id relayout new_span same_reason old_text new_text')


# ---------------------------------------------------------------------------------------------------------
# Reading the fences, the inventory and the trees
# ---------------------------------------------------------------------------------------------------------

def read_fences(path: str = FENCES) -> typ.List[Fence]:
    rows = []
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
        assert header == list(Fence._fields[:1]) + ['file', 'first_line', 'last_line', 'col_start', 'col_end',
                                                    'fenced', 'kind', 'anchor_sha1'], header
        for line in f:
            r = line.rstrip('\n').split('\t')
            rows.append(Fence(r[0], r[1], int(r[2]), int(r[3]), None if r[4] == '-' else int(r[4]),
                              None if r[5] == '-' else int(r[5]), r[6] == 'y', r[7], r[8]))
    return rows


def read_inventory(path: str = INVENTORY) -> typ.Dict[str, Entry]:
    """Parse each ``#### <id>`` section: its re-layout flag, new span, same-reason, old and new text."""
    text = open(path).read()
    out = {}
    for m in re.finditer(r'^#### (\S+)[^\n]*\n(.*?)(?=^#### |\Z)', text, re.S | re.M):
        bid, body = m.group(1), m.group(2)

        def field(name):
            mm = re.search(r'^- %s: (.*)$' % re.escape(name), body, re.M)
            return mm.group(1).strip() if mm else None

        blocks = re.findall(r'^(~{3,})\n(.*?)^\1$', body, re.S | re.M)
        old_text = blocks[0][1][:-1] if blocks and blocks[0][1] else ''
        new_text = None
        if len(blocks) > 1 and blocks[1][1]:
            new_text = blocks[1][1][:-1]
        new_span = field('new span')
        if new_span is not None and new_span.startswith('(R4/R5'):
            new_span = None
        same = field('same-reason')
        if same is not None and same.startswith('(R4/R5'):
            same = None
        out[bid] = Entry(bid, field('re-laid out') or '', new_span, same, old_text, new_text)
    return out


def _ref_for(path: str) -> str:
    r = subprocess.run(['git', 'cat-file', '-e', '%s:%s' % (TAG, path)], capture_output=True)
    return TAG if r.returncode == 0 else R1_BASE


_OLD_CACHE: typ.Dict[str, typ.List[str]] = {}


def old_lines(path: str) -> typ.List[str]:
    if path not in _OLD_CACHE:
        _OLD_CACHE[path] = subprocess.run(['git', 'show', '%s:%s' % (_ref_for(path), path)], check=True,
                                          capture_output=True, text=True).stdout.split('\n')
    return _OLD_CACHE[path]


def tree_lines(tree: str, path: str) -> typ.Optional[typ.List[str]]:
    p = os.path.join(tree, path)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return f.read().split('\n')


def cut(lines: typ.List[str], first: int, last: int, c0: typ.Optional[int], c1: typ.Optional[int]) -> typ.List[str]:
    seg = lines[first - 1:last]
    if c0 is not None:
        seg = [s[c0:(c1 if c1 is not None else len(s))] for s in seg]
    return seg


def fence_text(fc: Fence) -> str:
    return '\n'.join(cut(old_lines(fc.file), fc.first, fc.last, fc.col_start, fc.col_end))


def parse_span(span: str) -> typ.Tuple[int, int, typ.Optional[int], typ.Optional[int]]:
    """``'88–96'``, ``'88-96'``, ``'651'``, optionally ``', columns 48–end'`` / ``', columns 26–80'``."""
    m = re.fullmatch(r'\s*(\d+)(?:\s*[–-]\s*(\d+))?(?:\s*,\s*columns\s+(\d+)\s*[–-]\s*(end|\d+))?\s*', span)
    if not m:
        raise ValueError('unparsable new span %r' % span)
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    c0 = int(m.group(3)) if m.group(3) else None
    c1 = None if (m.group(4) in (None, 'end')) else int(m.group(4))
    return a, b, c0, c1


# ---------------------------------------------------------------------------------------------------------
# K10, K10b: alignment
# ---------------------------------------------------------------------------------------------------------

def k10(old: str, new: str) -> typ.List[str]:
    """In-place rewrite: same line count and lengths; differing columns are label characters on both sides."""
    errs = []
    a, b = old.split('\n'), new.split('\n')
    if len(a) != len(b):
        return ['line count %d -> %d' % (len(a), len(b))]
    for i, (x, y) in enumerate(zip(a, b)):
        if len(x) != len(y):
            errs.append('line %d: length %d -> %d' % (i + 1, len(x), len(y)))
            continue
        for c, (p, q) in enumerate(zip(x, y)):
            if p != q and not (LABEL_CHAR.match(p) and LABEL_CHAR.match(q)):
                errs.append('line %d col %d: %r -> %r (not a label character on both sides)' % (i + 1, c, p, q))
    return errs


def _alignment_columns(lines: typ.List[str]) -> typ.Dict[int, typ.Set[int]]:
    """Old alignment columns -> the lines that have them: a field start after >= 2 spaces (or the first
    non-blank of an indented line) in >= 2 lines, or a separator character shared by >= 2 lines."""
    starts = collections.defaultdict(set)
    seps = collections.defaultdict(set)
    for i, s in enumerate(lines):
        for m in re.finditer(r'(?:^|(?<=  ))(?! )\S', s):
            starts[m.start()].add(i)
        for c, ch in enumerate(s):
            if ch in SEPARATORS:
                if ch == ':' and c > 0 and s[c - 1] != ' ' and (c + 1 == len(s) or s[c + 1] == ' '):
                    continue          # R4a: a colon closing a label ("(all i):") moves with its label
                seps[(c, ch)].add(i)
    cols = collections.defaultdict(set)
    for c, ls in starts.items():
        if len(ls) >= 2:
            cols[c] |= ls
    for (c, ch), ls in seps.items():
        if len(ls) >= 2:
            cols[c] |= ls
    return cols


_TOKEN = re.compile(r"[A-Za-z0-9_']+|\S")


def _shape(tok: str) -> str:
    """A token's letter shape: a renamed label (``O_aib`` -> ``U_aib``, ``nU0`` -> ``nD0``) keeps its shape."""
    return re.sub(r'[A-Z]', 'A', re.sub(r'[a-z]', 'a', re.sub(r'[0-9]', '9', tok)))


def _column_map(x: str, y: str) -> typ.Dict[int, typ.Optional[int]]:
    """Old column -> new column through a token alignment on letter shapes (difflib), None where the old token was
    deleted. Inside a token the offset is kept (clipped to the new token)."""
    import difflib
    tx = [(m.start(), m.group(0)) for m in _TOKEN.finditer(x)]
    ty = [(m.start(), m.group(0)) for m in _TOKEN.finditer(y)]
    tok_map: typ.Dict[int, typ.Optional[int]] = {}
    sm = difflib.SequenceMatcher(None, [_shape(t) for _, t in tx], [_shape(t) for _, t in ty], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal' or (tag == 'replace' and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                tok_map[i1 + k] = j1 + k
        elif tag == 'replace':
            inner = difflib.SequenceMatcher(None, [t for _, t in tx[i1:i2]], [t for _, t in ty[j1:j2]], autojunk=False)
            for t2, a1, a2, b1, b2 in inner.get_opcodes():
                if t2 == 'equal' or (t2 == 'replace' and a2 - a1 == b2 - b1):
                    for k in range(a2 - a1):
                        tok_map[i1 + a1 + k] = j1 + b1 + k
        # 'delete': unmapped; 'insert': nothing old to map
    out: typ.Dict[int, typ.Optional[int]] = {}
    for k, (c, t) in enumerate(tx):
        j = tok_map.get(k)
        for off in range(len(t)):
            if j is None:
                out[c + off] = None
            else:
                out[c + off] = ty[j][0] + min(off, len(ty[j][1]) - 1)
    return out


def k10b(old: str, new: str) -> typ.List[str]:
    """Re-layout: same line count; every old alignment column maps to one consistent new column."""
    a, b = old.split('\n'), new.split('\n')
    if len(a) != len(b):
        return ['line count %d -> %d' % (len(a), len(b))]
    maps = [_column_map(x, y) for x, y in zip(a, b)]
    errs = []
    for c, ls in sorted(_alignment_columns(a).items()):
        targets = {maps[i].get(c) for i in ls} - {None}
        if len(targets) > 1:
            errs.append('old alignment column %d (lines %s) maps to new columns %s'
                        % (c, ','.join(str(i + 1) for i in sorted(ls)), sorted(targets)))
    return errs


# ---------------------------------------------------------------------------------------------------------
# K11: fences untouched
# ---------------------------------------------------------------------------------------------------------

def _locate(lines: typ.List[str], seg: typ.List[str], col) -> typ.List[int]:
    """1-based first lines where ``seg`` occurs as consecutive lines: whole lines (``col`` False); line suffixes
    (``col`` True or ``'suffix'``: a field comment shifts when the name beside it is renamed); or, for a side-by-side
    panel (``col == 'mid'``), the same text at one common column offset on every line."""
    hits = []
    n = len(seg)
    for i in range(len(lines) - n + 1):
        if col == 'mid':
            offs = [j for j in range(len(lines[i]) - len(seg[0]) + 1) if lines[i][j:j + len(seg[0])] == seg[0]]
            ok = any(all(lines[i + k][o:o + len(seg[k])] == seg[k] for k in range(n)) for o in offs)
        elif col:
            ok = all(lines[i + k].endswith(seg[k]) for k in range(n))
        else:
            ok = all(lines[i + k] == seg[k] for k in range(n))
        if ok:
            hits.append(i + 1)
    return hits


def k11_rewritten(fences: typ.Optional[typ.List[Fence]] = None) -> typ.List[str]:
    """The fenced blocks a slice has rewritten by hand (their inventory new text is recorded and differs from the
    old): K11's exceptions, each verified verbatim against the file by K12 instead (R4a)."""
    inv = read_inventory()
    return [f.id for f in (fences or read_fences()) if f.fenced and inv.get(f.id) is not None
            and inv[f.id].new_text is not None and inv[f.id].new_text != inv[f.id].old_text]


def k11(tree: str = '.', pass1_tree: typ.Optional[str] = None, fences: typ.Optional[typ.List[Fence]] = None
        ) -> typ.List[str]:
    rewritten = set(k11_rewritten()) if fences is None else set()      # a synthetic fence list: no exceptions
    fences = [f for f in (fences or read_fences()) if f.fenced and f.id not in rewritten]
    errs = []
    groups = collections.defaultdict(list)
    for fc in fences:
        seg = cut(old_lines(fc.file), fc.first, fc.last, fc.col_start, fc.col_end)
        if hashlib.sha1('\n'.join(seg).encode('utf-8')).hexdigest() != fc.sha1:
            errs.append('%s: the inventory anchor does not match %s at the base (fences.tsv is stale)' % (fc.id, fc.file))
        mode = False if fc.col_start is None else ('suffix' if fc.col_end is None else 'mid')
        groups[(fc.file, '\n'.join(seg), mode)].append(fc)
    for t in [tree] + ([pass1_tree] if pass1_tree else []):
        for (path, text, col), fcs in sorted(groups.items()):
            lines = tree_lines(t, path)
            if lines is None:
                errs.append('%s: %s missing in %s' % (','.join(f.id for f in fcs), path, t))
                continue
            hits = _locate(lines, text.split('\n'), col)
            if len(hits) < len(fcs):
                errs.append('%s (%s): fenced text found %d time(s) in %s, expected %d'
                            % (','.join(f.id for f in fcs), path, len(hits), t, len(fcs)))
        if t == pass1_tree:
            for fc in fences:
                lines = tree_lines(t, fc.file) or []
                for k in range(fc.first, fc.last + 1):
                    if k - 1 < len(lines) and SENTINEL.search(lines[k - 1]):
                        ln = lines[k - 1]
                        seg = ln if fc.col_start is None else (
                            ln[fc.col_start:fc.col_end] if fc.col_end is not None or '#' not in ln
                            else ln[ln.find('#'):])
                        if SENTINEL.search(seg):
                            errs.append('%s: sentinel inside the fence at %s:%d (pass-1 tree)' % (fc.id, fc.file, k))
    return errs


# ---------------------------------------------------------------------------------------------------------
# K12: completion
# ---------------------------------------------------------------------------------------------------------

def k12(tree: str = '.', fences: typ.Optional[typ.List[Fence]] = None,
        inventory: typ.Optional[typ.Dict[str, Entry]] = None) -> typ.Tuple[typ.List[str], int, int]:
    """(errors, filled, total). A block is filled when its new span and new text are recorded."""
    fences = fences or read_fences()
    inventory = inventory if inventory is not None else read_inventory()
    errs, filled = [], 0
    for fc in fences:
        e = inventory.get(fc.id)
        if e is None:
            errs.append('%s: no inventory section' % fc.id)
            continue
        if e.old_text != fence_text(fc):
            errs.append('%s: the inventory old text differs from the base text at its old span' % fc.id)
        if e.new_span is None or e.new_text is None:
            continue
        filled += 1
        lines = tree_lines(tree, fc.file)
        try:
            a, b, c0, c1 = parse_span(e.new_span)
        except ValueError as ex:
            errs.append('%s: %s' % (fc.id, ex))
            continue
        actual = '\n'.join(cut(lines or [], a, b, c0, c1))
        if actual != e.new_text:
            errs.append('%s: the new text is not the file verbatim at %s:%s' % (fc.id, fc.file, e.new_span))
        if e.new_text == e.old_text and not e.same_reason:
            errs.append('%s: new text equals old text but no same-reason is recorded' % fc.id)
    return errs, filled, len(fences)


# ---------------------------------------------------------------------------------------------------------
# Running K10 / K10b over the filled blocks
# ---------------------------------------------------------------------------------------------------------

def alignment_report(tree: str = '.') -> typ.Tuple[typ.List[str], int, int, int]:
    """(errors, k10 checked, k10b checked, pending) over the filled ascii / listing / relation blocks."""
    inv = read_inventory()
    errs, n10, n10b, pending = [], 0, 0, 0
    for fc in read_fences():
        e = inv.get(fc.id)
        if fc.kind in ('LaTeX equation line', 'LaTeX table or figure', 'markdown table', 'companion doctest'):
            continue                      # not column-aligned (LaTeX source, md tables) or code (doctest)
        if e is None or e.new_text is None:
            pending += 1
            continue
        relayout = e.relayout.lower()
        if relayout.startswith('y'):
            n10b += 1
            errs += ['%s K10b: %s' % (fc.id, m) for m in k10b(e.old_text, e.new_text)]
            if fc.col_start is not None and e.new_span:
                a, b, c0, _ = parse_span(e.new_span)
                lines = tree_lines(tree, fc.file) or []
                for k in range(a, b + 1):
                    if c0 is None or not lines[k - 1][c0:].startswith('#'):
                        errs.append('%s K10b: line %d does not start its comment at the new column %s'
                                    % (fc.id, k, c0))
        else:
            n10 += 1
            errs += ['%s K10: %s' % (fc.id, m) for m in k10(e.old_text, e.new_text)]
    return errs, n10, n10b, pending


# ---------------------------------------------------------------------------------------------------------
# Self-test: the approved sample-edit AFTERs pass, synthetic breakages fail
# ---------------------------------------------------------------------------------------------------------

_S5_OLD = '\n'.join([
    '           rL0       rL1       rR2      rR(d-1)         rRd',
    '        1 ------ L0 ------ D1 ------ ... ------ R(d-1) ------ 1',
    '                 |         |                    |',
    '                 | nU0     | nO1                | nU(d-1)',
    '                 |         |                    |',
    '                 U0       (V1)                   Ud',
    '                 |         |                    |'])
_S5_NEW = '\n'.join([
    '           rL0       rL1       rR2      rR(d-1)         rRd',
    '        1 ------ L0 ------ U1 ------ ... ------ R(d-1) ------ 1',
    '                 |         |                    |',
    '                 | nD0     | nU1                | nD(d-1)',
    '                 |         |                    |',
    '                 D0       (V1)                   Dd',
    '                 |         |                    |'])
_S6_OLD = '\n'.join([
    '        - up_tucker_cores   = (U0,...,U(d-1)), orthogonal:       U_ia U_ja = delta_ij',
    '        - down_tt_cores     = (O0,...,O(d-1)), outer-orthogonal  O_aib O_ajb = delta_ij',
    '        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij',
    '        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij'])
_S6_NEW = '\n'.join([
    '        - down_tucker_cores = (D0,...,D(d-1)), down-orthogonal:  D_ia D_ja = delta_ij',
    '        - up_tt_cores       = (U0,...,U(d-1)), up-orthogonal:    U_aib U_ajb = delta_ij',
    '        - left_tt_cores     = (L0,...,L(d-1)), left-orthogonal:  L_abi L_abj = delta_ij',
    '        - right_tt_cores    = (R0,...,R(d-1)), right-orthogonal  R_ibc R_jbc = delta_ij'])


def self_test() -> typ.List[str]:
    fails = []

    def expect(name, errs, ok):
        if bool(errs) == ok:
            fails.append('%s: expected %s, got %s' % (name, 'PASS' if ok else 'FAIL', errs or 'PASS'))

    # old vs old passes trivially, for every inventoried block
    for fc in read_fences():
        t = fence_text(fc)
        expect('k10 identity %s' % fc.id, k10(t, t), True)
        expect('k10b identity %s' % fc.id, k10b(t, t), True)
    # sample edit 5 (in place) passes K10; sample edit 6 (re-laid out) passes K10b and fails K10
    expect('k10 sample edit 5', k10(_S5_OLD, _S5_NEW), True)
    expect('k10b sample edit 6', k10b(_S6_OLD, _S6_NEW), True)
    expect('k10 sample edit 6 (not in place)', k10(_S6_OLD, _S6_NEW), False)
    # breakages
    shifted = _S5_NEW.replace('| nD0     | nU1', '| nD0    | nU1 ', 1)
    expect('k10 shifted pipe', k10(_S5_OLD, shifted), False)
    expect('k10 dropped line', k10(_S5_OLD, '\n'.join(_S5_NEW.split('\n')[:-1])), False)
    expect('k10 dash replaced by a letter', k10(_S5_OLD, _S5_NEW.replace('L0 ------ U1', 'L0 -----U U1')), False)
    expect('k10 expanded tab', k10('\tU0  U1', '        D0  D1'), False)
    bad6 = _S6_NEW.replace('up_tt_cores       = (U0', 'up_tt_cores = (U0', 1)
    expect('k10b misaligned =', k10b(_S6_OLD, bad6), False)
    bad6b = _S6_NEW.replace('up-orthogonal:    U_aib', 'up-orthogonal: U_aib', 1)
    expect('k10b misaligned equation column', k10b(_S6_OLD, bad6b), False)
    # sample edit 1 (the approved field comments, in place on the comment span) and sample batch 3 edit 8 (TABs)
    s1_old = '\n'.join(['# len=d. B_xo B_yo   = I_xy, Bi.shape = stack_shape+(nUi, Ni)',
                        '# len=d. R_ixj R_iyj = I_xy  Ri.shape = stack_shape+(rLi, nDi, rR(i+1))'])
    s1_new = '\n'.join(['# len=d. D_xo D_yo   = I_xy, Di.shape = stack_shape+(nDi, Ni)',
                        '# len=d. U_ixj U_iyj = I_xy  Ui.shape = stack_shape+(rLi, nUi, rR(i+1))'])
    expect('k10 sample edit 1', k10(s1_old, s1_new), True)
    expect('k10 sample batch 3 edit 8 (TABs kept)',
           k10('\tv  =     U0  U1  U2    +', '\tv  =     D0  D1  D2    +'), True)
    # parse_span
    expect('parse_span', [] if parse_span('156–159, columns 48–end') == (156, 159, 48, None) else ['bad'], True)
    # K11 / K12 on a synthetic tree: a copy of the base files, one fence edited, one sentinel planted
    import shutil
    import tempfile
    fences = read_fences()
    fvf = [f for f in fences if f.file == 't3toolbox/frame_variations_format.py']
    diagram = next(f for f in fvf if f.first == 88)
    comment = next(f for f in fvf if f.first == 156)
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, diagram.file)
        os.makedirs(os.path.dirname(dst))
        base = old_lines(diagram.file)
        with open(dst, 'w') as f:
            f.write('\n'.join(base))
        expect('k11 base copy', k11(tmp, fences=fvf), True)
        edited = list(base)
        edited[diagram.first + 2] = edited[diagram.first + 2].replace('nU0', 'nD0')
        renamed = edited[comment.first - 1].replace('up_tucker_cores:   ', 'down_tucker_cores: ')
        edited[comment.first - 1] = renamed            # a NAME rename beside a field comment: still located
        with open(dst, 'w') as f:
            f.write('\n'.join(edited))
        errs = k11(tmp, fences=fvf)
        expect('k11 edited diagram', errs, False)
        expect('k11 renamed field beside its comment', [e for e in errs if comment.id in e], True)
        p1 = os.path.join(tmp, 'p1')
        os.makedirs(os.path.join(p1, os.path.dirname(diagram.file)))
        planted = list(base)
        planted[diagram.first] = planted[diagram.first].replace('L0', 'ZQX0007_XQZ')
        with open(os.path.join(p1, diagram.file), 'w') as f:
            f.write('\n'.join(planted))
        with open(dst, 'w') as f:
            f.write('\n'.join(base))
        expect('k11 sentinel in a fence (pass-1 tree)', k11(tmp, p1, fences=fvf), False)
        # K12: one block filled with the file's own text and a reason passes; a wrong span fails
        inv = {f.id: Entry(f.id, 'n', None, None, fence_text(f), None) for f in fvf}
        inv[diagram.id] = inv[diagram.id]._replace(new_span='%d–%d' % (diagram.first, diagram.last),
                                                   new_text=fence_text(diagram), same_reason='test')
        errs, filled, total = k12(tmp, fences=fvf, inventory=inv)
        expect('k12 one filled block', errs + ([] if filled == 1 else ['filled %d' % filled]), True)
        inv[diagram.id] = inv[diagram.id]._replace(same_reason=None)
        expect('k12 new == old without a reason', k12(tmp, fences=fvf, inventory=inv)[0], False)
        inv[diagram.id] = inv[diagram.id]._replace(new_span='%d–%d' % (diagram.first + 1, diagram.last + 1),
                                                   same_reason='test')
        expect('k12 new text not at the new span', k12(tmp, fences=fvf, inventory=inv)[0], False)
    return fails


# ---------------------------------------------------------------------------------------------------------

def main(argv: typ.List[str]) -> int:
    what = argv[1] if len(argv) > 1 and not argv[1].startswith('--') else 'all'
    tree = argv[argv.index('--tree') + 1] if '--tree' in argv else '.'
    pass1 = argv[argv.index('--pass1-tree') + 1] if '--pass1-tree' in argv else None
    bad = False
    if what in ('all', 'self-test'):
        fails = self_test()
        print('self-test: %s' % ('PASS' if not fails else 'FAIL'))
        for m in fails:
            print('  ' + m)
        bad |= bool(fails)
    if what in ('all', 'k10', 'k10b'):
        errs, n10, n10b, pending = alignment_report(tree)
        status = 'FAIL' if errs else ('PENDING' if pending and not (n10 or n10b) else 'PASS')
        print('K10/K10b: %s (%d in place checked, %d re-laid out checked, %d blocks not yet rewritten)'
              % (status, n10, n10b, pending))
        for m in errs:
            print('  ' + m)
        bad |= bool(errs)
    if what in ('all', 'k11'):
        errs = k11(tree, pass1)
        rewritten = k11_rewritten()
        n = sum(1 for f in read_fences() if f.fenced) - len(rewritten)
        print('K11: %s (%d fenced spans located by text in %s%s; %d rewritten by hand, excepted and checked by K12)'
              % ('FAIL' if errs else 'PASS', n, tree, ' and %s' % pass1 if pass1 else '', len(rewritten)))
        if '--list-exceptions' in argv:
            print('  K11 exceptions: ' + ', '.join(rewritten))
        for m in errs:
            print('  ' + m)
        bad |= bool(errs)
    if what in ('all', 'k12'):
        errs, filled, total = k12(tree)
        status = 'FAIL' if errs else ('PASS' if filled == total else 'PENDING')
        print('K12: %s (%d of %d blocks filled)' % (status, filled, total))
        for m in errs:
            print('  ' + m)
        bad |= bool(errs)
    return 1 if bad else 0


if __name__ == '__main__':
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('run with the t3toolbox env (Python 3.11)')
    sys.exit(main(sys.argv))
