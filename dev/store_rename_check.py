"""The store rename's completeness checks K0-K17 (plan ``dev/store_rename_plan_2026-11-10.md`` §5.3; one-off, D-9).

Usage::

    python dev/store_rename_check.py [--tree DIR] [--hits DIR] [--pass1 DIR] [--stage R1|R2|R3|R4a|R4b|R5]
                                     [--only K1,K8,...] [--out FILE]

* ``--tree``: the tree under test (default: the working tree). R1 also runs it on the dry run's pass-2 tree
  (``<scratch>/pass2``), the preview of R2.
* ``--hits``: the directory holding ``R1_hit_table.tsv`` (K14 images, K15's positional check after R2).
* ``--pass1``: the pass-1 tree (K0 there is expected to FIND sentinels; K11 checks none lies inside a fence).
* ``--stage``: what the gates expect. Before R2 every pattern check is REPORT (its hits are the to-do list); K13
  skips fenced spans at R2/R3; K12 is PENDING until R4b/R5.

The old tree is always ``v2026.4.0`` (``git show``). Scope (§5.3): ``t3toolbox/``, ``tests/``, ``examples/`` for
K0-K16; the docs (``docs/*.md|rst``, ``docs/contributor/*``, ``README.md``, ``CLAUDE.md``) for all but K3's code
patterns and K13; the six tex notes for K2, K4, K7, K8. Carve-outs (§5.3): ``CHANGELOG.md``, ``docs/release_notes.md``,
the glossary's "(2026.4.0 ...)" alias groups and the T4S correspondence notes BY BLOCK, N17's alias
"outer-orthogonal" (``glossary.md:102``), the paper <-> code maps (``entries_apply_probe.md`` §8,
``batching_and_stacking.md`` §8), and this file.
"""
import argparse
import collections
import csv
import difflib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tokenize
import typing as typ

assert sys.version_info[:2] == (3, 11), 'store_rename_check.py needs Python 3.11 (the t3toolbox env)'

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'v2026.4.0'


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sra = _load('store_rename_apply', os.path.join(REPO, 'dev', 'store_rename_apply.py'))
sdc = _load('store_rename_diagram_check', os.path.join(REPO, 'dev', 'store_rename_diagram_check.py'))

TEX = tuple('docs/%s.tex' % n for n in ('padded_frame_cone', 'exact_hessian_two_hole', 'shared_t3_math', 'pad_safe_svd',
                                        'symmetric_probe_derivatives', 'ttm_t3m_ht_note'))
LETTER_API_FILES = (
    't3toolbox/backend/padded_frames.py', 't3toolbox/backend/uniform_padded_frames.py', 't3toolbox/manifold.py',
    't3toolbox/uniform_manifold.py', 't3toolbox/frame_variations_format.py', 't3toolbox/uniform_frame_variations_format.py',
    'tests/backend/test_padded_frame.py', 'tests/backend/test_uniform_padded_frame.py', 'tests/backend/test_padded_start.py',
    'tests/backend/_padded_frame_oracles.py', 'tests/backend/_uniform_padded_frame_testing.py', 'tests/test_dispatch.py',
    'tests/test_aligned_padded_frame_frontend.py')

# ===================================================================================================================
# The patterns (§5.3's table, transcribed)
# ===================================================================================================================

K1 = [re.compile(r'\b(?:masked_)?(?:up_tucker_cores|down_tt_cores|up_tucker_supercore|down_tt_supercore|up_tucker|'
                 r'down_tt|up_tucker_ranks|down_tucker_ranks|down_tt_ranks|up_orthogonalize_tucker_cores|'
                 r'down_orthogonalize_tt_cores)(?:_dot)?\b')]
K2 = [re.compile(r'(?<![\\\w])nO'), re.compile(r'n_\{O'), re.compile(r'n\^O'),
      re.compile(r'\b(?:n_up|n_down|r_left|r_right|k_up|k_down|m_up|m_down)\w*')]
K3_CODE = [re.compile(r'update\s*:\s*str\b'), re.compile(r'update\s*=\s*[\'"]'),
           re.compile(r'[\'"]O[\'"]\s+(?:not\s+)?in\s+update'),
           re.compile(r'[\'"](?=[DULR]*O)[DULRO]{2,4}[\'"]'),
           re.compile(r'\(\s*(?:[\'"][DULRO][\'"]\s*,\s*)*[\'"]O[\'"](?:\s*,\s*[\'"][DULRO][\'"])*\s*,?\s*\)')]
K3_DOCS = [re.compile(r"`'(?:ULRO|UL|RO)'`")]
K3_WORD_TUPLE = [re.compile(r"\('(?:down|up|left|right)'")]
# R3's extension: in the letter-API files, any string of 2-4 store letters (code, docstrings and comments): 2026.4.0's
# ('ULRO', 'UL', 'RO', ...) and new letters alike ('DL'), since form (c) takes a tuple.
K3_OLD_LETTERS = [re.compile(r'[\'"][DULRO]{2,4}[\'"]')]          # form (c) refuses every str of letters
# Carved out of K3 by function (R3): the refusal itself (its 'O' test and its message naming the 2026.4.0 form) and the
# tests that pass the 2026.4.0 forms on purpose, to see them refused.
K3_REFUSALS = {'t3toolbox/backend/padded_frames.py': ('_update_stores',),
               'tests/backend/test_padded_frame.py': ('test_bad_update',),
               'tests/backend/test_uniform_padded_frame.py': ('test_update',)}
K4 = [re.compile(p) for p in (r'\(U, ?O, ?[PL], ?[QR]\)', r'\((?:D|U), ?(?:U|O), ?P, ?Q\b', r'\(U, ?G, ?G, ?G\)',
                              r'\(D, ?G, ?G, ?G\)', r'\(?P, ?Q, ?O\)? ?(?:→|->)', r'Q = O = P = G', r'Q = O = G',
                              r'\b[OPQ] = G\b', r'no longer required orthogonal')]
K5 = [re.compile(r'\(up, down, left'), re.compile(r'up/down/left/right')]
K6 = [re.compile(p) for p in (r'Tucker up-(?:SVD|orth)', r'Tucker cores upward', r'TT cores downward',
                              r'down-orthogonal O\b', r'up-orthogonal tucker', r'Down TT core')]
K7 = [re.compile(p) for p in (r'\b[DU]\^[rku]\b', r'\[U\^', r'\b[DU] (?:x|z|jet)\b', r'\bDPhi\b', r'\bD\^2 ?Phi\b',
                              r'DΦ', r'D²Φ')]
K7_TEX = [re.compile(r'\\Dsh|\\Ush|D\\Phi|D\^2\\Phi|D_\\zeta')]
K8_O = re.compile(r'\bO(?:\d|_[a-z]|ᵢ|i\b|\(d-1\))')
K8 = [re.compile(p) for p in (
    r'\bO\[', r'\bO\.shape\b', r'\bfor\s+O\s+in\b', r'\((?:\s*[A-Z]\w?\s*,)*\s*O\s*(?:,\s*[A-Z]\w?\s*)*\)',
    r',\s*O\s*\)', r'\(mu\^\(\w+\) O\)', r'\b(?:mu\S*|nu) O\b', r'\bO (?:nu|sigma|tau)', r'``O``', r'(?<!`)`O`(?!`)',
    r'\\mathbf\s*\{?O', r'\b(?:O_i|O_aib|Oᵢ|δU|U_g)\b', r'U̇', r'outer[- ]orthogonal', r'\b[PQ]_(?:i|last|\{)',
    r'\b(?:Udot\w*|dU(?![a-z])\w*|gU|Ug|Ud)\b', r'<\s*O\s*,', r',\s*O\s*>')] + [K8_O]
K8_TEX = [re.compile(r'\\delta\s*U\b'), re.compile(r'\\dot\s*U\b'), re.compile(r'\\dot\{U\}')]
# draft 21: decorated letters with no trailing boundary (\dot US_i, \delta\widetilde{\mathbf U}_i), carved out in
# pad_safe_svd.tex (the SVD's \mathbf U, §1.5 item 2)
K8_TEX_DECOR = re.compile(r'\\(?:mathbf|dot|delta|widetilde|widehat|tilde|hat)\s*\{?\s*(?:\\mathbf\s*\{?\s*)?[UOPQ]')
K8B = [re.compile(r'\[[\'"][OPQ][\'"]\]')]
K0 = [re.compile(r'ZQX\d+_XQZ')]
K16 = [re.compile(r'n_\{[DUO],'), re.compile(r'n\^[DUO]'), re.compile(r'r_\{[LR],')]

STORE_WORD_RE = re.compile(r'\b(?:up|down|Up|Down|UP|DOWN)\b|\b(?:up|down)_[a-z]\b')

# ===================================================================================================================
# Files, carve-outs
# ===================================================================================================================


def code_paths() -> typ.List[str]:
    return sra.python_paths()


def doc_paths() -> typ.List[str]:
    return sra.doc_paths()


def read(tree: str, path: str) -> typ.Optional[str]:
    p = os.path.join(tree, path)
    if os.path.exists(p):
        return open(p, encoding='utf-8').read()
    if tree != REPO and os.path.exists(os.path.join(REPO, path)) and path.endswith('.tex'):
        return open(os.path.join(REPO, path), encoding='utf-8').read()      # the dry run writes no tex
    return None


_base_cache: typ.Dict[str, typ.Optional[str]] = {}


def read_base(path: str) -> typ.Optional[str]:
    if path not in _base_cache:
        r = subprocess.run(['git', '-C', REPO, 'show', '%s:%s' % (BASE, path)], capture_output=True)
        _base_cache[path] = r.stdout.decode('utf-8') if r.returncode == 0 else None
    return _base_cache[path]


def carved_lines(path: str, text: str, check: str) -> typ.Set[int]:
    """Line numbers carved out of a pattern check, by block (§5.3)."""
    lines = text.split('\n')
    out: typ.Set[int] = set()
    # blocks = paragraphs / list items (blank-line delimited)
    blocks, cur = [], []
    for i, ln in enumerate(lines, 1):
        if not ln.strip():
            if cur:
                blocks.append(cur)
            cur = []
        else:
            cur.append(i)
    if cur:
        blocks.append(cur)
    for b in blocks:
        t = '\n'.join(lines[i - 1] for i in b)
        if path == 'docs/glossary.md' and '(2026.4.0' in t and check in ('K1', 'K2', 'K4', 'K5', 'K8', 'K9'):
            out.update(b)
        if 'T4S writes U, O, P, Q' in t and check in ('K4', 'K8'):
            out.update(b)
    if path in ('docs/entries_apply_probe.md', 'docs/batching_and_stacking.md'):
        in8 = False
        for i, ln in enumerate(lines, 1):
            if re.match(r'#+ ', ln):
                in8 = bool(re.match(r'#+ (?:§\s*)?8[.\s]', ln))
            if in8:
                out.add(i)
    if path == 'docs/glossary.md' and check == 'K8':
        out.add(102)                                      # N17's own alias "outer-orthogonal"
    return out


def pattern_hits(tree: str, paths: typ.Sequence[str], pats, check: str, carve=None
                 ) -> typ.List[typ.Tuple[str, int, str, str]]:
    out = []
    for p in paths:
        text = read(tree, p)
        if text is None:
            continue
        skip = carved_lines(p, text, check)
        for i, ln in enumerate(text.split('\n'), 1):
            if i in skip:
                continue
            for pat in pats:
                for m in pat.finditer(ln):
                    if carve and carve(p, i, m, ln):
                        continue
                    out.append((p, i, m.group(0), ln.strip()))
    return out


# ===================================================================================================================
# The checks
# ===================================================================================================================

class Result:
    def __init__(self, k: str, status: str, summary: str, details: typ.Sequence[str] = ()):
        self.k, self.status, self.summary, self.details = k, status, summary, list(details)


def _rep(k, hits, stage, clean_from, what) -> Result:
    """A pattern check: REPORT before its gate slice, PASS/FAIL from it on."""
    det = ['%s:%d: `%s` -- %s' % (p, i, m, ln[:150]) for p, i, m, ln in hits]
    gated = _stage_ge(stage, clean_from)
    status = ('FAIL' if hits else 'PASS') if gated else ('REPORT' if hits else 'PASS')
    return Result(k, status, '%d hit(s) %s (gate from %s)' % (len(hits), what, clean_from), det)


ORDER = ('R1', 'R2', 'R3', 'R4a', 'R4b', 'R5', 'R6')


def _stage_ge(stage: str, ref: str) -> bool:
    return ORDER.index(stage) >= ORDER.index(ref)


def k0(tree, stage, pass1):
    hits = pattern_hits(tree, code_paths() + doc_paths(), K0, 'K0')
    out = [_rep('K0', hits, 'R6', 'R2', 'of a surviving sentinel (pass-2 / final tree)')]
    if pass1:
        n = sum(len(K0[0].findall(read(pass1, p) or '')) for p in code_paths() + doc_paths())
        out.append(Result('K0(pass1)', 'INFO', '%d sentinels in the pass-1 tree (expected: the hit count)' % n))
    return out


def k1(tree, stage):
    carve = lambda p, i, m, ln: False
    return [_rep('K1', pattern_hits(tree, code_paths() + doc_paths(), K1, 'K1', carve), stage, 'R4b',
                 'of an old spelling (code tree clean after R2 except prose, all after R5)')]


def k2(tree, stage):
    return [_rep('K2', pattern_hits(tree, code_paths() + doc_paths() + list(TEX), K2, 'K2'), stage, 'R4b',
                 'of a stale width spelling (everywhere, fences and notes included)')]


_prose_cache: typ.Dict[typ.Tuple[str, str], typ.Any] = {}


def _code_prose(tree: str, path: str):
    """(docstring line ranges, {line: comment column}, the K3_REFUSALS functions' lines) of a Python file."""
    if (tree, path) not in _prose_cache:
        import ast
        src = read(tree, path) or ''
        doc, refusal = [], set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    doc.append((first.lineno, first.end_lineno))
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in K3_REFUSALS.get(path, ())):
                refusal.update(range(node.lineno, node.end_lineno + 1))
        comments = {t.start[0]: t.start[1] for t in tokenize.generate_tokens(io.StringIO(src).readline)
                    if t.type == tokenize.COMMENT}
        _prose_cache[(tree, path)] = (doc, comments, refusal)
    return _prose_cache[(tree, path)]


def k3(tree, stage):
    """§5.3's K3, with R3's extension: the old letter strings in the letter-API files, and the word tuple read in the
    docs and in code files' docstrings and comments only (§5.3: "in docs and docstrings"; in code, store-word tuples
    name stores in messages and seeds, B2's literals). The K3_REFUSALS functions are carved out."""
    refused = lambda p, i, m, ln: i in _code_prose(tree, p)[2]
    hits = pattern_hits(tree, [p for p in code_paths() if p in LETTER_API_FILES], K3_CODE + K3_OLD_LETTERS, 'K3',
                        refused)

    def not_prose(p, i, m, ln):
        doc, comments, refusal = _code_prose(tree, p)
        prose = any(a <= i <= b for a, b in doc) or (i in comments and m.start() >= comments[i])
        return not prose or i in refusal
    hits += pattern_hits(tree, code_paths(), K3_WORD_TUPLE, 'K3', not_prose)
    doc_hits = pattern_hits(tree, doc_paths(), K3_DOCS + K3_WORD_TUPLE, 'K3')
    return [_rep('K3', hits, stage, 'R3', "of the update letter API's old form or a word tuple, in the code tree"),
            _rep('K3-docs', doc_hits, stage, 'R5', "of the update letter API's old form or a word tuple, in the docs "
                                                   "(their LETTER-API rows are R5's, §1.4 L7)")]


def k4_to_k8(tree, stage):
    res = []
    every = code_paths() + doc_paths()
    res.append(_rep('K4', pattern_hits(tree, every + list(TEX), K4, 'K4'), stage, 'R4b',
                    'of an old frame-letter tuple, hybrid or the cited substitution'))
    res.append(_rep('K5', pattern_hits(tree, every, K5, 'K5'), stage, 'R4b', 'of an old-order listing'))
    res.append(_rep('K6', pattern_hits(tree, every, K6, 'K6'), stage, 'R4b', 'of a backwards phrase'))
    res.append(_rep('K7', pattern_hits(tree, every + list(TEX), K7, 'K7')
                    + pattern_hits(tree, list(TEX), K7_TEX, 'K7'), stage, 'R4b',
                    'of the old shift / derivative notation'))

    def carve8(p, i, m, ln):
        g = m.group(0)
        if re.fullmatch(r'O[12]', g) and (p.startswith('docs/') or p in ('README.md', 'CLAUDE.md')):
            return True                                       # oracle / review codes (inv 3 §9)
        if p.endswith(('oracle_sweeps/o1_ragged.py', 'oracle_sweeps/o1_uniform.py')) and g == 'Ud':
            return True                                       # "u dense"
        if p == 'docs/shared_t3_math.tex' and i == 358 and 'O_g' in ln and g.startswith('O_g'):
            return True                                       # kept complement basis
        if (p, i) in _TEXT_KEEPS:
            return True                                       # §3.4 inventory keep
        return False
    res.append(_rep('K8', pattern_hits(tree, every, K8, 'K8', carve8)
                    + pattern_hits(tree, list(TEX), K8 + K8_TEX, 'K8', carve8)
                    + pattern_hits(tree, [t for t in TEX if not t.endswith('pad_safe_svd.tex')], [K8_TEX_DECOR], 'K8',
                                   carve8), stage, 'R4b',
                    'of a T4S letter left behind'))
    res.append(_rep('K8b', pattern_hits(tree, ['tests/test_jet_recurrence.py'], K8B, 'K8b'), stage, 'R2',
                    "of a T4S string key in test_jet_recurrence"))
    return res


_TEXT_KEEPS: typ.Set[typ.Tuple[str, int]] = set()


def _load_text_keeps():
    p = os.path.join(REPO, 'dev', 'store_rename_slices', 'pqu_text_keeps.tsv')
    if os.path.exists(p):
        for r in csv.DictReader(open(p), delimiter='\t'):
            _TEXT_KEEPS.add((r['file'], int(r['line'])))


def load_keeps() -> typ.Dict[typ.Tuple[str, int], str]:
    out = {}
    with open(os.path.join(REPO, 'dev', 'store_rename_keeps.txt')) as fh:
        for ln in fh:
            if ln.startswith('#') or ln.startswith('file\t'):
                continue
            c = ln.rstrip('\n').split('\t')
            if c[1] != '-':
                out[(c[0], int(c[1]))] = c[2]
    return out


def _text_tokens(src: str) -> typ.List[typ.Tuple[int, int, str]]:
    """(line, col, word) of every bare store word inside COMMENT and STRING tokens."""
    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        return out
    for t in toks:
        if t.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        for k, ln in enumerate(t.string.split('\n')):
            for m in STORE_WORD_RE.finditer(ln):
                out.append((t.start[0] + k, (t.start[1] if k == 0 else 0) + m.start(), m.group(0)))
    return out


def k9(tree, stage):
    """Residual store words (token-level) and the converse. On a tree equal to the base, (b) cannot hold, so every
    non-keep word is the to-do list (REPORT before R4b)."""
    keeps = load_keeps()
    opposite = {'up': 'down', 'down': 'up', 'Up': 'Down', 'Down': 'Up', 'UP': 'DOWN', 'DOWN': 'UP'}
    residual, converse = [], []
    for p in code_paths():
        new, old = read(tree, p), read_base(p)
        if new is None or old is None:
            continue
        nl, ol = new.split('\n'), old.split('\n')
        sm = difflib.SequenceMatcher(None, ol, nl, autojunk=False)
        line_map = {}
        for a, b, size in sm.get_matching_blocks():
            for k in range(size):
                line_map[b + k + 1] = a + k + 1
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'replace' and i2 - i1 == j2 - j1:
                for k in range(i2 - i1):
                    line_map[j1 + k + 1] = i1 + k + 1
        new_words = _text_tokens(new)
        old_words = collections.defaultdict(list)
        for ln, col, w in _text_tokens(old):
            old_words[ln].append((col, w))
        new_by_line = collections.defaultdict(list)
        for ln, col, w in new_words:
            new_by_line[ln].append((col, w))
        for ln, ws in new_by_line.items():
            oln = line_map.get(ln)
            kind = keeps.get((p, oln)) if oln else None
            if kind == 'KEEP':
                continue
            if kind == 'EXEMPT':
                continue
            ows = [w for _, w in sorted(old_words.get(oln, []))] if oln else []
            nws = [w for _, w in sorted(ws)]
            ok = len(ows) == len(nws) and all(o == opposite.get(n) for o, n in zip(ows, nws))
            if not ok:
                residual.append('%s:%d: %s' % (p, ln, ' '.join(nws)))
        for (kp, kl), kind in keeps.items():
            if kp != p or kind != 'KEEP':
                continue
            nln = next((n for n, o in line_map.items() if o == kl), None)
            ows = [w for _, w in sorted(old_words.get(kl, []))]
            nws = [w for _, w in sorted(new_by_line.get(nln, []))] if nln else None
            if ows and nws != ows:
                converse.append('%s:%d: keep words %s became %s' % (p, kl, ows, nws))
    gated = _stage_ge(stage, 'R4b')
    out = [Result('K9', ('FAIL' if residual else 'PASS') if gated else ('REPORT' if residual else 'PASS'),
                  '%d line(s) with a bare store word that is neither a keep, nor the opposite of its aligned old word '
                  '(the free-prose to-do list before R4b)' % len(residual), residual),
           Result('K9-converse', 'FAIL' if converse else 'PASS',
                  '%d keep line(s) whose store words did not survive' % len(converse), converse)]
    return out


def k10_k12(tree, stage, pass1):
    out = []
    errs, n10, n10b, pending = sdc.alignment_report(tree)
    out.append(Result('K10/K10b', 'FAIL' if errs else ('PENDING' if pending and not (n10 or n10b) else 'PASS'),
                      '%d in place, %d re-laid out checked; %d blocks not yet rewritten' % (n10, n10b, pending), errs))
    errs = sdc.k11(tree, pass1)
    out.append(Result('K11', 'FAIL' if errs else 'PASS', 'fenced spans byte-identical to %s (located by text)%s'
                      % (BASE, '; no sentinel inside a fence in the pass-1 tree' if pass1 else ''), errs))
    errs, filled, total = sdc.k12(tree)
    gated = _stage_ge(stage, 'R5')
    st = 'FAIL' if errs else ('PASS' if filled == total else ('FAIL' if gated else 'PENDING'))
    out.append(Result('K12', st, '%d of %d blocks filled' % (filled, total), errs))
    return out


K13_ND = {'down_tucker_cores', 'down_tucker_cores_dot', 'down_tucker_supercore', 'down_weights',
          'down_weight_supercore', 'down_mask', 'variations_down_mask', 'down_ranks', 'tt_variations', 'H'}
K13_NU = {'up_tt_cores', 'up_tt_cores_dot', 'up_tt_supercore', 'up_weights', 'up_weight_supercore', 'up_mask',
          'variations_up_mask', 'up_ranks', 'tucker_variations', 'V'}
K13_FORMS = (re.compile(r'^\s*(?:>>>\s*)?(\w+)\s*:\s*[^#\n]*#(?P<c>.*)$'),      # NAME: type  # ...
             re.compile(r'#\s*(\w+)\.\s(?P<c>.*)$'),                          # typ.Sequence[...],  # NAME. len=d, ...
             re.compile(r'^\s*-\s*(\w+)\s*=(?P<c>.*)$'))                      # - NAME = (...), elm_shape=(...)
WIDTH_TOK = re.compile(r'\bn[UD]\w*')


def k13(tree, stage):
    fences = [f for f in sra.load_fences() if f.fenced]
    skip_fenced = not _stage_ge(stage, 'R4a')
    fails, generic, unrec = [], [], []
    for p in code_paths():
        text = read(tree, p)
        if text is None:
            continue
        fi = sra.fence_intervals(fences, p) if skip_fenced else {}
        for i, ln in enumerate(text.split('\n'), 1):
            if not WIDTH_TOK.search(ln):
                continue
            for form in K13_FORMS:
                m = form.search(ln)
                if not m:
                    continue
                name, c0 = m.group(1), m.start('c')
                if fi and sra._in_fence(fi, i, c0, len(ln)):
                    break
                widths = WIDTH_TOK.findall(m.group('c'))
                if not widths:
                    break
                want = 'nD' if (name in K13_ND or name.startswith(('left_tt', 'right_tt'))) else \
                       'nU' if name in K13_NU else None
                if want is None:
                    (generic if name in ('tucker_cores', 'tt_cores') else unrec).append(
                        '%s:%d: %s %s' % (p, i, name, widths))
                else:
                    bad = [w for w in widths if not w.startswith(want)]
                    if bad:
                        fails.append('%s:%d: %s wants %s, has %s -- %s' % (p, i, name, want, bad, ln.strip()[:120]))
                break
    gated = _stage_ge(stage, 'R2')
    return [Result('K13', ('FAIL' if fails else 'PASS') if gated else ('REPORT' if fails else 'PASS'),
                   '%d shape line(s) whose width letter disagrees with the entry%s' % (
                       len(fails), ' (fenced spans skipped)' if skip_fenced else ''), fails),
            Result('K13-generic', 'INFO', '%d generic tucker_cores/tt_cores shape lines (reported to the reviewer, '
                                          '§4 item 12)' % len(generic), generic),
            Result('K13-unrecognised', 'INFO', '%d shape lines with widths but no recognised entry' % len(unrec),
                   unrec)]


def load_dnt():
    rows = []
    with open(os.path.join(REPO, 'dev', 'store_rename_dnt.txt')) as fh:
        for ln in fh:
            if ln.startswith('#') or ln.startswith('identifier\t'):
                continue
            c = ln.rstrip('\n').split('\t')
            rows.append((c[0], c[1], c[2]))
    return rows


def _names(path: str, src: str) -> typ.List[typ.Tuple[int, int, str, str]]:
    """Every NAME token (code + doctest examples), via the apply script's tokenizer (no hits kept)."""
    p1 = sra.Pass1(sra.Entries(), [], [])
    p1.record_names = []
    p1.python_file(path, src)
    return [(l, c, t, x) for _, l, c, t, x in p1.record_names]


def _outer(src):
    import ast
    out = []
    tree = ast.parse(src)

    def visit(node, inside):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not inside:
                    out.append((min([ch.lineno] + [d.lineno for d in ch.decorator_list]), ch.end_lineno, ch.name))
                visit(ch, True)
            else:
                visit(ch, inside)
    visit(tree, False)
    return out


def _in_scope(scope, path, line, outer):
    if scope == '*':
        return True
    if '::' in scope:
        f, fn = scope.split('::')
        return f == path and any(a <= line <= b and n == fn for a, b, n in outer)
    m = re.match(r'(.*):(\d+)-(\d+)$', scope)
    if m:
        return m.group(1) == path and int(m.group(2)) <= line <= int(m.group(3))
    return scope == path


def _contract_strings(src: str) -> collections.Counter:
    import ast
    c = collections.Counter()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', '')
            if name in ('contract', 'einsum'):
                for a in n.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        c[a.value] += 1
    return c


_ref_cache: typ.Dict[typ.Tuple[str, str], typ.Optional[str]] = {}


def read_ref(ref: str, path: str) -> typ.Optional[str]:
    if (ref, path) not in _ref_cache:
        r = subprocess.run(['git', '-C', REPO, 'show', '%s:%s' % (ref, path)], capture_output=True)
        _ref_cache[(ref, path)] = r.stdout.decode('utf-8') if r.returncode == 0 else None
    return _ref_cache[(ref, path)]


HUNK_FIELDS = ('file', 'tag', 'old_lines', 'new_lines', 'old', 'new')


def name_diff(tree: str, prev: str):
    """From R3 on (§5.3 K15): per code file, the NAME streams of the previous slice's commit and of the tree, and
    difflib's unequal opcodes between them."""
    out = []
    for p in code_paths():
        new, old = read(tree, p), read_ref(prev, p)
        if new is None or old is None or new == old:
            continue
        on, nn = _names(p, old), _names(p, new)
        sm = difflib.SequenceMatcher(None, [x[2] for x in on], [x[2] for x in nn], autojunk=False)
        out.append((p, old, new, on, nn, [o for o in sm.get_opcodes() if o[0] != 'equal']))
    return out


def name_hunks(diff) -> typ.List[typ.Dict[str, str]]:
    """The unequal NAME hunks as rows of HUNK_FIELDS (lines are first-last of the hunk's tokens; '-' when empty)."""
    rows = []
    for p, old, new, on, nn, ops in diff:
        for tag, i1, i2, j1, j2 in ops:
            rows.append(dict(file=p, tag=tag,
                             old_lines='%d-%d' % (on[i1][0], on[i2 - 1][0]) if i2 > i1 else '-',
                             new_lines='%d-%d' % (nn[j1][0], nn[j2 - 1][0]) if j2 > j1 else '-',
                             old=' '.join(x[2] for x in on[i1:i2]), new=' '.join(x[2] for x in nn[j1:j2])))
    return rows


def k15_cross_check(diff, hits_dir, names: typ.Sequence[str]) -> 'Result':
    """§5.3's K15 cross-check, from R3 on: over the replace and delete hunks (insert hunks are new code, read through
    the listed hunks), per file, every name in old - new must be a hit-table source and every name in new - old a
    hit-table image, unless the slice's handoff lists it (``names``)."""
    hits = list(csv.DictReader(open(os.path.join(hits_dir, 'R1_hit_table.tsv')), delimiter='\t')) if hits_dir else []
    sources = {h['old'] for h in hits if h['ctx'].endswith('NAME')}
    images = {h['new'] for h in hits if h['ctx'].endswith('NAME')}
    listed = set(names)
    errs, det = [], []
    for p, old, new, on, nn, ops in diff:
        a, b = collections.Counter(), collections.Counter()
        for tag, i1, i2, j1, j2 in ops:
            if tag in ('replace', 'delete'):
                a.update(x[2] for x in on[i1:i2])
                b.update(x[2] for x in nn[j1:j2])
        removed, added = a - b, b - a
        if not (removed or added):
            continue
        det.append('%s: old - new {%s}; new - old {%s}' % (p, ', '.join(sorted(removed)), ', '.join(sorted(added))))
        errs += ['%s: removed %s is neither a map source nor listed' % (p, n) for n in sorted(removed)
                 if n not in sources and n not in listed]
        errs += ['%s: added %s is neither a map image nor listed' % (p, n) for n in sorted(added)
                 if n not in images and n not in listed]
    return Result('K15-cross', 'FAIL' if errs else 'PASS', '%d name(s) outside the map and the listed names, over the '
                  'replace/delete hunks (listed: %s)' % (len(errs), ', '.join(sorted(listed)) or 'none'), errs + det)


def k14_k15_after_r2(tree, hits_dir, prev, hunks_file, write_hunks, names=()):
    """From R3 on. K15: every unequal NAME hunk against the previous slice's commit ``prev`` is one the slice lists
    (``hunks_file``, a TSV of HUNK_FIELDS that the handoff carries). K14: no DO-NOT-TOUCH token is lost or replaced
    in a hunk (the previous slice passed K14 against the tag, so every listed occurrence survives); DNT names that
    new code adds (public names in new tests) are counted, and the tag comparison of R2 is repeated with them
    subtracted."""
    dnt = load_dnt()
    diff = name_diff(tree, prev)
    rows = name_hunks(diff)
    if write_hunks:
        with open(write_hunks, 'w', newline='') as fh:
            w = csv.DictWriter(fh, HUNK_FIELDS, delimiter='\t', lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    by_ident = collections.defaultdict(list)
    for ident, scope, item in dnt:
        by_ident[ident].append((scope, item))

    def relevant(p, ident):
        return [(sc, it) for sc, it in by_ident.get(ident, ())
                if sc == '*' or sc == p or sc.startswith(p + '::') or sc.startswith(p + ':')]

    def count(p, names, outer, ident, scope):
        return sum(1 for l, c, t, x in names if t == ident and _in_scope(scope, p, l, outer))

    lost, added = [], collections.Counter()
    for p, old, new, on, nn, ops in diff:
        oo, no = _outer(old), _outer(new)
        for tag, i1, i2, j1, j2 in ops:
            a_names, b_names = on[i1:i2], nn[j1:j2]
            for ident in {x[2] for x in a_names} | {x[2] for x in b_names}:
                for scope, item in relevant(p, ident):
                    a, b = count(p, a_names, oo, ident, scope), count(p, b_names, no, ident, scope)
                    if a > b:
                        lost.append('%s: %s [%s, item %s] lost in the hunk at old lines %s (%d -> %d)'
                                    % (p, ident, scope, item, '%d-%d' % (on[i1][0], on[i2 - 1][0]), a, b))
                    elif b > a:
                        added[(p, ident, scope)] += b - a
    # the tag comparison (R2's K14) on the files this slice changed, with its additions subtracted; the others are
    # byte-identical to the previous slice, which passed K14 against the tag
    hits = list(csv.DictReader(open(os.path.join(hits_dir, 'R1_hit_table.tsv')), delimiter='\t')) if hits_dir else []
    images = collections.defaultdict(list)
    for h in hits:
        if h['ctx'].endswith('NAME'):
            images[(h['file'], h['new'])].append(int(h['line']))
    tag_errs, k14s = [], []
    for p, prev_src, new, _, nn, _ in diff:
        base = read_base(p)
        if base is None:
            continue
        bn, pn = _names(p, base), _names(p, prev_src)
        bo, po, no = _outer(base), _outer(prev_src), _outer(new)
        for ident in {x[2] for x in bn} | {x[2] for x in pn} | {x[2] for x in nn} | {n for f, n in images if f == p}:
            for scope, item in relevant(p, ident):
                co, cp, cn = count(p, bn, bo, ident, scope), count(p, pn, po, ident, scope), count(p, nn, no, ident, scope)
                ci = sum(1 for l in images.get((p, ident), ()) if _in_scope(scope, p, l, bo))
                if cp - ci != co:
                    tag_errs.append('%s: %s [%s, item %s] base %d, previous slice %d, images %d'
                                    % (p, ident, scope, item, co, cp, ci))
                if cn - cp != added[(p, ident, scope)]:
                    tag_errs.append('%s: %s [%s, item %s] previous slice %d, tree %d, additions in hunks %d'
                                    % (p, ident, scope, item, cp, cn, added[(p, ident, scope)]))
        if _contract_strings(base) != _contract_strings(new):
            k14s.append('%s: contract/einsum string arguments differ' % p)
    add_det = ['added: %s: %s [%s] +%d' % (p, i, s, v) for (p, i, s), v in sorted(added.items())]
    out = [Result('K14', 'FAIL' if (lost or tag_errs or k14s) else 'PASS',
                  '%d DNT token(s) lost in a hunk against %s, %d count difference(s) against %s after the hunks, '
                  '%d file(s) with different contract/einsum strings; %d DNT token(s) added by new code (%d DNT rows)'
                  % (len(lost), prev, len(tag_errs), BASE, len(k14s), sum(added.values()), len(dnt)),
                  lost + tag_errs + k14s + add_det)]
    if hunks_file:
        listed = list(csv.DictReader(open(hunks_file, newline=''), delimiter='\t'))
        key = lambda r: tuple(r[f] for f in HUNK_FIELDS)
        extra = [r for r in rows if key(r) not in {key(x) for x in listed}]
        missing = [r for r in listed if key(r) not in {key(x) for x in rows}]
        det = ['NOT LISTED: %s' % '\t'.join(key(r)) for r in extra] + ['LISTED, NOT FOUND: %s' % '\t'.join(key(r))
                                                                      for r in missing]
        out.append(Result('K15', 'FAIL' if det else 'PASS', '%d NAME hunk(s) against %s; %d not listed in %s, %d '
                          'listed and not found' % (len(rows), prev, len(extra), os.path.basename(hunks_file),
                                                    len(missing)), det))
    else:
        out.append(Result('K15', 'REPORT', '%d NAME hunk(s) against %s (pass --hunks to check them against the '
                          'listed ones)' % (len(rows), prev), ['\t'.join(r[f] for f in HUNK_FIELDS) for r in rows]))
    out.append(k15_cross_check(diff, hits_dir, names))
    return out


def k14_k15(tree, stage, hits_dir):
    """K14: DNT multisets per file and scope (images excluded); contract/einsum strings. K15: the NAME stream is
    closed (after R2: positional, against the hit table)."""
    dnt = load_dnt()
    hits = list(csv.DictReader(open(os.path.join(hits_dir, 'R1_hit_table.tsv')), delimiter='\t')) if hits_dir else []
    renamed = _stage_ge(stage, 'R2')          # images exist only on a renamed tree
    images = collections.Counter()
    hit_pos = {}
    for h in hits:
        if h['ctx'].endswith('NAME'):
            images[(h['file'], int(h['line']), h['new'])] += 1
            hit_pos[(h['file'], int(h['line']), int(h['col']))] = (h['old'], h['new'])
    k14, k14s, k15 = [], [], []
    for p in code_paths():
        new, old = read(tree, p), read_base(p)
        if new is None or old is None:
            continue
        on, nn = _names(p, old), _names(p, new)
        oo, no = _outer(old), _outer(new)
        for ident, scope, item in dnt:
            if scope != '*' and not (scope == p or scope.startswith(p + '::') or scope.startswith(p + ':')):
                continue
            co = sum(1 for l, c, t, x in on if t == ident and _in_scope(scope, p, l, oo))
            cn = sum(1 for l, c, t, x in nn if t == ident and _in_scope(scope, p, l, no))
            ci = sum(v for (f, l, n), v in images.items() if f == p and n == ident and _in_scope(scope, p, l, oo)) \
                if renamed else 0
            if cn - ci != co:
                k14.append('%s: %s [%s, item %s] base %d, tree %d, images %d' % (p, ident, scope, item, co, cn, ci))
        if _contract_strings(old) != _contract_strings(new):
            k14s.append('%s: contract/einsum string arguments differ' % p)
        # K15 after R2: positional
        if len(on) != len(nn):
            k15.append('%s: NAME stream length %d -> %d (not an in-place token rename)' % (p, len(on), len(nn)))
            continue
        for (l, c, t, x), (l2, c2, t2, x2) in zip(on, nn):
            if t != t2:
                exp = hit_pos.get((p, l, c))
                if not exp or exp != (t, t2):
                    k15.append('%s:%d:%d: %s -> %s not in the hit table' % (p, l, c, t, t2))
    return [Result('K14', 'FAIL' if (k14 or k14s) else 'PASS', '%d DNT multiset difference(s), %d file(s) with '
                   'different contract/einsum strings (%d DNT rows)' % (len(k14), len(k14s), len(dnt)), k14 + k14s),
            Result('K15', 'FAIL' if k15 else 'PASS', '%d NAME position(s) changed outside the hit table' % len(k15),
                   k15)]


def k15_realized(tree, hits_dir):
    """The converse of K15 on a renamed tree: every NAME hit of the table is realized (the image is in place)."""
    hits = list(csv.DictReader(open(os.path.join(hits_dir, 'R1_hit_table.tsv')), delimiter='\t'))
    missing = []
    for p in code_paths():
        new, old = read(tree, p), read_base(p)
        if new is None or old is None:
            continue
        on, nn = _names(p, old), _names(p, new)
        if len(on) != len(nn):
            continue
        pos = {(l, c): i for i, (l, c, t, x) in enumerate(on)}
        for h in hits:
            if h['file'] == p and h['ctx'].endswith('NAME'):
                i = pos.get((int(h['line']), int(h['col'])))
                if i is None or nn[i][2] != h['new']:
                    missing.append('%s:%s:%s %s -> %s not realized' % (p, h['line'], h['col'], h['old'], h['new']))
    return Result('K15-realized', 'FAIL' if missing else 'PASS', '%d hit(s) not realized' % len(missing), missing)


BARE_FRAME = re.compile(r'^[DULR](?:\d+|[a-ce-z]|_\w+)?$')        # D, D0, Di, D_r, Uc, ...; not Dd (a derivative)
VAR_NAMES = {'tucker_variations', 'tt_variations', 'V', 'H', 'tkv', 'ttv'}
VAR_ATTRS = {'variations', 'tucker_variations', 'tt_variations'}


def _is_variation(expr) -> bool:
    """The bound value IS a variation core (K17: "bound from ... directly"): follow the value's root through
    subscripts, ``.data``, attribute chains and ``tuple(<generator>)``; any other call's result is not judged by its
    arguments (``U1, ... = xmap(_up_edge, (..., V[i], ...))`` binds a new frame core)."""
    import ast
    if isinstance(expr, ast.Subscript):
        return _is_variation(expr.value)
    if isinstance(expr, ast.Attribute):
        return expr.attr in VAR_ATTRS or _is_variation(expr.value)
    if isinstance(expr, ast.Name):
        return expr.id in VAR_NAMES or expr.id.endswith(('_V', '_H'))
    if isinstance(expr, (ast.GeneratorExp, ast.ListComp)):
        return _is_variation(expr.elt)
    if isinstance(expr, ast.Call) and getattr(expr.func, 'id', None) in ('tuple', 'list') and len(expr.args) == 1:
        return _is_variation(expr.args[0])
    if isinstance(expr, (ast.Tuple, ast.List)):
        return any(_is_variation(e) for e in expr.elts)
    return False


def _bindings(tree_node, name):
    """(kind, expr) for every binding of ``name`` under ``tree_node``: the matching element of a tuple / zip, else the
    whole right-hand side or iterable; ``('param', None)`` for a parameter."""
    import ast
    out = []

    def match(target, value):
        if isinstance(target, ast.Name):
            if target.id == name:
                out.append(('bind', value))
        elif isinstance(target, (ast.Tuple, ast.List)):
            elts = None
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
                elts = value.elts
            elif (isinstance(value, ast.Call) and getattr(value.func, 'id', None) == 'zip'
                  and len(value.args) == len(target.elts)):
                elts = value.args                       # for a, b in zip(x, y): a <- x, b <- y
            for k, t in enumerate(target.elts):
                match(t, elts[k] if elts is not None else value)
        elif isinstance(target, ast.Starred):
            match(target.value, value)
    for n in ast.walk(tree_node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                match(t, n.value)
        elif isinstance(n, (ast.For, ast.comprehension)):
            match(n.target, n.iter)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            if name in {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}:
                out.append(('param', None))
    return out


def k17(tree, stage, hits_dir):
    """K17 (draft 22): no NAME whose value is a variation core is mapped to a bare frame letter."""
    import ast
    import doctest as _dt
    if not hits_dir:
        return [Result('K17', 'SKIP', 'needs --hits')]
    hits = [h for h in csv.DictReader(open(os.path.join(hits_dir, 'R1_hit_table.tsv')), delimiter='\t')
            if h['ctx'].endswith('NAME') and BARE_FRAME.match(h['new'])]
    bad, params, checked = [], collections.Counter(), 0
    cache = {}
    for h in hits:
        path, line = h['file'], int(h['line'])
        if path not in cache:
            src = read_base(path)
            cache[path] = (src, ast.parse(src))
        src, mod = cache[path]
        if h['ctx'].startswith('DOCTEST'):
            node = None
            for n in ast.walk(mod):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and '>>>' in n.value \
                        and n.lineno <= line <= n.end_lineno:
                    code = '\n'.join(ex.source for ex in _dt.DocTestParser().get_examples(n.value))
                    try:
                        node = ast.parse(code)
                    except SyntaxError:
                        node = None
                    break
            if node is None:
                continue
        else:
            outer = [n for n in ast.walk(mod) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.lineno <= line <= n.end_lineno]
            node = min(outer, key=lambda n: n.lineno) if outer else mod
            outer_top = [n for n in outer if not any(o is not n and o.lineno <= n.lineno and n.end_lineno <= o.end_lineno
                                                     for o in outer)]
            node = outer_top[0] if outer_top else mod
        checked += 1
        for kind, expr in _bindings(node, h['old']):
            if kind == 'param':
                params[(path, h['old'])] += 1
            elif _is_variation(expr):
                bad.append('%s:%d: %s -> %s, bound from `%s`' % (path, line, h['old'], h['new'],
                                                               ast.unparse(expr)[:90]))
    bad = sorted(set(bad))
    gated = _stage_ge(stage, 'R2')
    return [Result('K17', ('FAIL' if bad else 'PASS') if gated else ('REPORT' if bad else 'PASS'),
                   '%d NAME hit(s) to a bare frame letter checked; %d bound from a variation core' % (checked, len(bad)),
                   bad),
            Result('K17-params', 'INFO', '%d renamed parameters (binding resolved by the caller, not by K17)'
                   % len(params), ['%s: %s' % k for k in sorted(params)])]


# K17's shape half (the maintainer, 2026-11-11: "shape decides the letter"): on a tree in the NEW letters, a NAME whose
# shape comment is (nU.., N..) (V-family) or (rL.., nD.., rR..) (H-family) may not carry D, dD, U or dU
SHAPE_FORMS = (re.compile(r'^\s*(?:>>>\s*)?(\w+)\s*:\s*[^#\n]*#(?P<c>.*)$'),             # NAME: type  # ...
               re.compile(r'#\s*(\w+)\.\s(?P<c>.*)$'),                                   # ...,  # NAME. len=d, ...
               re.compile(r'^\s*(?:>>>\s*)?(\w+)\s*=[^=#\n][^#\n]*#(?P<c>.*)$'))         # NAME = ...  # (shape)
_GROUP = re.compile(r'\(([^()]*(?:\([^()]*\)[^()]*)*)\)')
V_SHAPE = re.compile(r'^\s*nU\w*\s*,\s*N\w*\s*$')
H_SHAPE = re.compile(r'^\s*rL[\w()+-]*\s*,\s*nD\w*(?:\([\w+-]*\))?\s*,\s*rR[\w()+-]*\s*$')   # no composite nD+nU
# core-shaped arrays: a frame down core (nD, N), a frame up core (rL, nU, rR), a point's own core (n, N) / (rL, n, rR)
CORE_SHAPES = (re.compile(r'^\s*nD\w*\s*,\s*N\w*\s*$'),
               re.compile(r'^\s*rL[\w()+-]*\s*,\s*nU\w*(?:\([\w+-]*\))?\s*,\s*rR[\w()+-]*\s*$'),
               re.compile(r'^\s*n(?:i|_?\w*)?\s*,\s*N\w*\s*$'),
               re.compile(r'^\s*rL[\w()+-]*\s*,\s*n(?:i|_?\w*)?\s*,\s*rR[\w()+-]*\s*$'))
FRAME_LETTERED = re.compile(r'^(?:d|g)?[DU](?:$|[0-9_]|dot|d$|i$|s$)')
POINT_LETTERED = re.compile(r'^(?:d|g|o|k|K)?[BG](?:$|[0-9_]|dot|d$|i$|s$)')     # R1 review MINOR-7
VH_LETTERED = re.compile(r'^g?[VH](?:$|[0-9_]|dot|d$|i$|s$)')                    # R1 review NIT (third pass)


def shape_family(comment: str) -> typ.Optional[str]:
    """'V' / 'H' (variation-shaped), 'core' (a frame or point core's own shape), or None."""
    for g in _GROUP.findall(comment):
        g = g.replace(' ', '')
        if V_SHAPE.match(g):
            return 'V'
        if H_SHAPE.match(g):
            return 'H'
        if any(c.match(g) for c in CORE_SHAPES):
            return 'core'
    return None


def _shape_verdict(name: str, fam: str) -> typ.Optional[str]:
    """'frame' (D/dD/U/dU on a V/H array: gated from R2), 'point' (B/G forms on a V/H array: gated from R4b, where
    item 12's n makes the corewise ones core-shaped), 'converse' (a V/H name on a core-shaped array: gated from R2)."""
    if fam in ('V', 'H'):
        if FRAME_LETTERED.match(name):
            return 'frame'
        if POINT_LETTERED.match(name):
            return 'point'
    elif fam == 'core' and VH_LETTERED.match(name):
        return 'converse'
    return None


def _return_shapes(src: str) -> typ.List[typ.Tuple[int, str, typ.List[typ.Optional[str]]]]:
    """(return line, returned NAME, family) for every ``return a`` / ``return (a, b, ...)`` whose function declares one
    shape comment per returned element on its ``-> ...:`` lines (the tuple-element comments, in order)."""
    import ast
    out = []
    lines = src.split('\n')
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.returns is None or not fn.body:
            continue
        a, b = fn.returns.lineno, fn.body[0].lineno - 1
        fams = []
        for k in range(a, b + 1):
            ln = lines[k - 1]
            if '#' in ln:
                fam = shape_family(ln[ln.index('#'):])
                if fam or _GROUP.search(ln[ln.index('#'):]):
                    fams.append(fam)
        if not fams:
            continue
        for r in ast.walk(fn):
            if not isinstance(r, ast.Return) or r.value is None:
                continue
            elts = r.value.elts if isinstance(r.value, ast.Tuple) else [r.value]
            if len(elts) == len(fams):
                for e, fam in zip(elts, fams):
                    if isinstance(e, ast.Name) and fam:
                        out.append((r.lineno, e.id, fam))
    return out


def k17_shape(tree, stage):
    """K17's shape half (the maintainer: "shape decides the letter"), on a tree in the NEW letters. A NAME's array is
    resolved from its own shape comment (a parameter, a single-name assignment) or, for a returned NAME, from its
    function's return-element comment by position. Verdicts: ``frame`` (D/dD/U/dU on a V/H array; gated from R2),
    ``point`` (B/G forms on a V/H array; gated from R4b, after item 12's generic n; R1 review MINOR-7), ``converse`` (a
    V/H name on a core-shaped array; gated from R2). Return-element labels (``# NAME. len=d, ...``) are comments,
    listed for R4 (K17-labels)."""
    fences = [f for f in sra.load_fences() if f.fenced]
    skip_fenced = not _stage_ge(stage, 'R4a')
    found = collections.defaultdict(list)
    base_names: typ.Dict[str, typ.Set[str]] = {}

    def introduced(p, name):           # the converse judges only names a slice introduced (not pre-existing H, H0)
        if p not in base_names:
            b = read_base(p)
            base_names[p] = {t for _, _, t, _ in _names(p, b)} if b is not None else set()
        return name not in base_names[p]
    labels, n = [], 0
    for p in code_paths():
        text = read(tree, p)
        if text is None:
            continue
        fi = sra.fence_intervals(fences, p) if skip_fenced else {}
        for i, ln in enumerate(text.split('\n'), 1):
            if '#' not in ln:
                continue
            for k, form in enumerate(SHAPE_FORMS):
                m = form.search(ln)
                if not m:
                    continue
                if fi and sra._in_fence(fi, i, m.start('c'), len(ln)):
                    break
                fam = shape_family(m.group('c'))
                if fam:
                    n += 1
                    v = _shape_verdict(m.group(1), fam)
                    if v == 'converse' and not introduced(p, m.group(1)):
                        v = None
                    if v:
                        msg = '%s:%d: %s [%s] is on a %s-shaped array -- %s' % (p, i, m.group(1), v, fam,
                                                                              ln.strip()[:110])
                        (labels if k == 1 else found[v]).append(msg)
                break
        try:
            rets = _return_shapes(text)
        except SyntaxError:
            rets = []
        for line, name, fam in rets:
            n += 1
            v = _shape_verdict(name, fam)
            if v == 'converse' and not introduced(p, name):
                v = None
            if v:
                found[v].append('%s:%d: returned %s [%s] is %s-shaped by its function\'s return comment'
                                % (p, line, name, v, fam))
    out = []
    for v, gate, what in (('frame', 'R2', 'D/dD/U/dU on a V/H-shaped array'),
                          ('point', 'R4b', 'B/G forms on a V/H-shaped array (item 12\'s n makes the corewise ones '
                                           'core-shaped in R4b)'),
                          ('converse', 'R2', 'a V/H name on a core-shaped array')):
        bad = found[v]
        gated = _stage_ge(stage, gate)
        meaningful = _stage_ge(stage, 'R2')
        st = ('FAIL' if bad else 'PASS') if gated else ('REPORT' if (bad or not meaningful) else 'PASS')
        out.append(Result('K17-shape-' + v, st, '%d: %s (gate from %s; %d NAME/array pairs resolved%s)' % (
            len(bad), what, gate, n, '' if meaningful else '; meaningful on a renamed tree'), bad))
    out.append(Result('K17-labels', 'INFO', '%d return-element comment labels with a shape verdict (comments: R4 '
                                            'rewrites them)' % len(labels), labels))
    return out


def k16(tree, stage):
    hits = pattern_hits(tree, list(TEX), K16, 'K16')
    return [Result('K16', 'REPORT', '%d LaTeX width form(s) in the notes, each for explicit sign-off at R5' % len(hits),
                   ['%s:%d: `%s` -- %s' % h for h in hits])]


# ===================================================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', default=REPO)
    ap.add_argument('--hits', default=None)
    ap.add_argument('--pass1', default=None)
    ap.add_argument('--stage', default='R1', choices=ORDER)
    ap.add_argument('--only', default='')
    ap.add_argument('--out', default=None)
    ap.add_argument('--realized', action='store_true', help='K15 converse: every NAME hit is in place (R2 on)')
    ap.add_argument('--prev', default='HEAD', help='R3 on: the previous slice\'s commit, for K14/K15')
    ap.add_argument('--hunks', default=None, help='R3 on: the TSV of the NAME hunks the handoff lists (K15)')
    ap.add_argument('--write-hunks', default=None, help='R3 on: write the NAME hunks found to this TSV')
    ap.add_argument('--names', default='', help='R3 on: comma-separated names the handoff lists as removed or added '
                                                'by its replace/delete hunks (K15 cross-check)')
    a = ap.parse_args(argv)
    _load_text_keeps()
    only = set(a.only.split(',')) if a.only else None
    want = lambda k: only is None or k in only
    res: typ.List[Result] = []
    if want('K0'):
        res += k0(a.tree, a.stage, a.pass1)
    if want('K1'):
        res += k1(a.tree, a.stage)
    if want('K2'):
        res += k2(a.tree, a.stage)
    if want('K3'):
        res += k3(a.tree, a.stage)
    if want('K4') or want('K5') or want('K6') or want('K7') or want('K8'):
        res += k4_to_k8(a.tree, a.stage)
    if want('K9'):
        res += k9(a.tree, a.stage)
    if want('K10') or want('K11') or want('K12'):
        res += k10_k12(a.tree, a.stage, a.pass1)
    if want('K13'):
        res += k13(a.tree, a.stage)
    if want('K14') or want('K15'):
        if _stage_ge(a.stage, 'R3'):
            res += k14_k15_after_r2(a.tree, a.hits, a.prev, a.hunks, a.write_hunks,
                                    [n for n in a.names.split(',') if n])
        else:
            res += k14_k15(a.tree, a.stage, a.hits)
    if a.realized and a.hits:
        res.append(k15_realized(a.tree, a.hits))
    if want('K16'):
        res += k16(a.tree, a.stage)
    if want('K17'):
        res += k17(a.tree, a.stage, a.hits)
        res += k17_shape(a.tree, a.stage)
    lines = ['# store_rename_check: tree `%s`, stage %s' % (a.tree, a.stage), '',
             '| check | status | summary |', '|---|---|---|']
    lines += ['| %s | %s | %s |' % (r.k, r.status, r.summary) for r in res]
    for r in res:
        if r.details:
            lines += ['', '## %s (%s, %d)' % (r.k, r.status, len(r.details)), ''] + ['- ' + d for d in r.details]
    text = '\n'.join(lines) + '\n'
    if a.out:
        open(a.out, 'w').write(text)
    for r in res:
        print('%-16s %-8s %s' % (r.k, r.status, r.summary))
    return 1 if any(r.status == 'FAIL' for r in res) else 0


if __name__ == '__main__':
    sys.exit(main())
