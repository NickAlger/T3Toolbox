"""Store rename R1 (plan §4A a): the diagram / equation candidate scanner.

Proposes every ASCII diagram and every letter- or width-bearing equation in ``t3toolbox/``, ``tests/``,
``examples/``, ``docs/*.md|rst|tex``, ``docs/contributor/*``, ``README.md`` and ``CLAUDE.md`` as read at
``v2026.4.0`` (``docs/glossary.md``, absent at the tag, is read at the R1 base ``88fbbedb``), groups the
candidate lines into proposed blocks by structure, then applies the hand curation (``CURATION`` below).

Writes, under ``dev/store_rename_slices/``:

- ``diagram_candidates.tsv``: every candidate line, with the block it belongs to or ``dismissed`` + reason;
- ``fences.tsv``: one row per inventoried block (consumed by ``dev/store_rename_apply.py``);
- ``diagrams_inventory.md``: one section per block, the §4A(a) columns.

Run from the repo root: ``<env-python> dev/store_rename_slices/diagram_scanner.py``. One-off (D-9), never
imported by the library. Python 3.11 (the f-string tokenization the rename script also pins).
"""
import collections
import hashlib
import io
import re
import subprocess
import sys
import tokenize
import typing as typ

TAG = 'v2026.4.0'
R1_BASE = '88fbbedb'          # HEAD when R1 ran; the base for files absent at the tag (docs/glossary.md)
OUT_DIR = 'dev/store_rename_slices'

# ---------------------------------------------------------------------------------------------------------
# Reading the tree at the tag
# ---------------------------------------------------------------------------------------------------------

def git_show(ref: str, path: str) -> str:
    return subprocess.run(['git', 'show', '%s:%s' % (ref, path)], check=True, capture_output=True,
                          text=True).stdout


def base_ref(path: str) -> str:
    """The ref a path's old spans are read at: the tag, or the R1 base for files the tag lacks."""
    r = subprocess.run(['git', 'cat-file', '-e', '%s:%s' % (TAG, path)], capture_output=True)
    return TAG if r.returncode == 0 else R1_BASE


def in_scope(path: str) -> bool:
    if path.startswith(('t3toolbox/', 'tests/', 'examples/')) and path.endswith('.py'):
        return True
    if path in ('README.md', 'CLAUDE.md'):
        return True
    m = re.fullmatch(r'docs/([^/]+)\.(md|rst|tex)', path)
    if m:
        return True
    return bool(re.fullmatch(r'docs/contributor/[^/]+\.md', path))


def scope_files() -> typ.List[typ.Tuple[str, str]]:
    names = set()
    for ref in (TAG, R1_BASE):
        out = subprocess.run(['git', 'ls-tree', '-r', '--name-only', ref], check=True, capture_output=True,
                             text=True).stdout.split('\n')
        names.update(p for p in out if p and in_scope(p))
    return sorted((p, base_ref(p)) for p in names)


# ---------------------------------------------------------------------------------------------------------
# Regions: which lines are text (comment, docstring prose, expected output, docs) and which are code
# ---------------------------------------------------------------------------------------------------------

Line = collections.namedtuple('Line', 'no raw region text col')
# region: lit (docstring literal block after ::) | disp (displayed line in a docstring) | code | cmt (full-line comment) | tcmt (trailing comment) | prose | ex (doctest example) | out
#         (doctest output) | str (other string) | md | mdcode | mdtable | rst | rstlit | tex | texmath
# text:   the text part of the line the rules read; col: its start column in raw (0-based)


def python_lines(src: str) -> typ.List[Line]:
    raw = src.split('\n')
    region = ['code'] * (len(raw) + 2)
    text = [''] * (len(raw) + 2)
    col = [0] * (len(raw) + 2)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for t in toks:
        if t.type == tokenize.COMMENT:
            ln = t.start[0]
            before = raw[ln - 1][:t.start[1]]
            region[ln] = 'tcmt' if before.strip() else 'cmt'
            text[ln] = t.string
            col[ln] = t.start[1]
        elif t.type == tokenize.STRING:
            (l0, c0), (l1, c1) = t.start, t.end
            multi = l1 > l0 or t.string.lstrip('rRbBuUfF')[:3] in ('"""', "'''")
            in_doc_example = False
            lit_indent = None
            prev_blank = False
            body = [raw[k - 1] for k in range(l0 + 1, l1 + 1) if raw[k - 1].strip()]
            base_indent = min((len(b) - len(b.lstrip()) for b in body), default=None) if multi else None
            for ln in range(l0, l1 + 1):
                seg = raw[ln - 1]
                a = c0 if ln == l0 else 0
                b = c1 if ln == l1 else len(seg)
                s = seg[a:b]
                if not multi:
                    if region[ln] == 'code':
                        region[ln], text[ln], col[ln] = 'str', s, a
                    else:
                        text[ln] += ' ' + s
                    continue
                st = s.lstrip()
                if st.startswith('>>>') or (in_doc_example and st.startswith('...')):
                    region[ln] = 'ex'
                    in_doc_example = True
                elif in_doc_example and st and not st.startswith(('"""', "'''")):
                    region[ln] = 'out'
                else:
                    in_doc_example = False
                    region[ln] = 'prose'
                    ind = len(s) - len(st)
                    if lit_indent is not None and st and ind <= lit_indent:
                        lit_indent = None
                    if lit_indent is not None and st:
                        region[ln] = 'lit'
                    if st.rstrip().endswith('::'):
                        lit_indent = ind
                    if (region[ln] == 'prose' and st and base_indent is not None and ind > base_indent
                            and (prev_blank or region[ln - 1] == 'disp') and '=' in st):
                        region[ln] = 'disp'
                prev_blank = not st
                if not st:
                    in_doc_example = False
                text[ln], col[ln] = s, a
    return [Line(i + 1, raw[i], region[i + 1], text[i + 1], col[i + 1]) for i in range(len(raw))]


def md_lines(src: str) -> typ.List[Line]:
    out, fence = [], None
    for i, r in enumerate(src.split('\n')):
        st = r.lstrip()
        if fence is None and (st.startswith('```') or st.startswith('~~~')):
            fence = st[:3]
            out.append(Line(i + 1, r, 'mdcode', '', 0))
            continue
        if fence is not None:
            if st.startswith(fence):
                fence = None
                out.append(Line(i + 1, r, 'mdcode', '', 0))
                continue
            reg = 'ex' if st.startswith(('>>>', '... ')) else 'mdcode'
            out.append(Line(i + 1, r, reg, r, 0))
            continue
        reg = 'mdtable' if r.startswith('|') else 'md'
        out.append(Line(i + 1, r, reg, r, 0))
    return out


def rst_lines(src: str) -> typ.List[Line]:
    out, lit_indent, pending = [], None, False
    for i, r in enumerate(src.split('\n')):
        st = r.lstrip()
        ind = len(r) - len(st)
        if lit_indent is not None and st and ind <= lit_indent:
            lit_indent = None
        if pending and st:
            lit_indent, pending = ind - 1, False
        reg = 'rstlit' if lit_indent is not None else 'rst'
        if st.startswith(('>>>', '... ')):
            reg = 'ex'
        out.append(Line(i + 1, r, reg, r, 0))
        if r.rstrip().endswith('::'):
            pending, lit_indent = True, None
    return out


MATH_ENVS = r'(?:equation|align|gather|multline|eqnarray|displaymath|tikzpicture|tabular|array|split|cases)\*?'


def tex_lines(src: str) -> typ.List[Line]:
    out, depth, display = [], 0, False
    for i, r in enumerate(src.split('\n')):
        code = re.sub(r'(?<!\\)%.*', '', r)
        begins = len(re.findall(r'\\begin\{' + MATH_ENVS + r'\}', code))
        ends = len(re.findall(r'\\end\{' + MATH_ENVS + r'\}', code))
        opens_display = bool(re.search(r'\\\[|\$\$', code)) and not display
        in_math = depth > 0 or display or begins > 0 or opens_display or '$' in code or r'\(' in code
        out.append(Line(i + 1, r, 'texmath' if in_math else 'tex', code, 0))
        depth = max(0, depth + begins - ends)
        if re.search(r'\\\[', code):
            display = True
        if re.search(r'\\\]', code):
            display = False
        if code.count('$$') % 2 == 1:
            display = not display
    return out


# ---------------------------------------------------------------------------------------------------------
# Candidate rules (on the text part of each line)
# ---------------------------------------------------------------------------------------------------------

LABEL = r'(?:\(\s*\w{0,6}\s*\)|[A-Za-z]\w{0,7}(?:\([\w+-]+\))?|1|\.\.\.|\?)'
EDGE = re.compile(r'(?=' + LABEL + r'\s*-{2,}\s*' + LABEL + r')')
PIPE_LINE = re.compile(r'^\s*(?:#\s*)?(?:[A-Za-z]\w{0,3}\s+=\s+)?(?:\|\s*(?:[A-Za-z(]\w{0,7}(?:\([\w+-]+\))?\)?)?\s*)+$')
LABEL_ITEM = r'(?:[=+]|\(\s*\w{0,6}\s*\)|\(?[A-Za-z]\w{0,6}(?:\([\w+-]+\))?\)?)'
LABEL_LINE = re.compile(r'^\s*(?:#\s*)?(?:' + LABEL_ITEM + r'\s{2,})+' + LABEL_ITEM + r'\s*$')
WIDTH = re.compile(r'(?<![\\\w])(?:n[UDO]|r[LR])(?=\d|[ijkdn]\b|_|\(|\b)')
WIDTH_CHANGING = re.compile(r'(?<![\\\w])n[UDO](?=\d|[ijkdn]\b|_|\(|\b)')
CORE = re.compile(r'(?<![\w\\])(?:[UODPQ](?:\d+\b|_[a-z]{1,4}\b|_\{|\(d-1\)|\(i\)|i\b|ᵢ|d\b)|[LR](?:\d+\b|\(d-1\)))')
INDEX = re.compile(r'(?:(?<![\w\\])[UODPQLRBGVH][A-Za-z]?_[a-z]{2,4}\b.*=\s*(?:delta_?\{?[a-z]{2}\}?|I_?[a-z]{2,3}\b|I\b|0\b)'
                   r'|δ_?\{?[a-z]{2}|delta_ij|einsum\(\s*[\'"]'
                   r'|(?<![\w\\])[UODPQLRB](?:_\w{1,3}|ᵢ|\d|i\b|\^[TLR]|ᵀ)[^=`]{0,25}=\s*(?:0|I)\b)')
WORDIDX = re.compile(r'\b(?:up|down)_[a-z]\b')
LISTING = re.compile(r'^\s*-\s+[\w.]+\s*=\s*\(')
STORE_LABEL = re.compile(r'\b(?:up_tucker|down_tt|left_tt|right_tt|outer_tt)\w*')
TEX_K16 = re.compile(r'n_\{[DUO],|n\^\{?[DUO]|r_\{[LR],|r\^\{?[LR]')
# a decorated store letter, with no trailing-boundary requirement (\dot US_i, \delta\widetilde{\mathbf U}_i): R1 review,
# MAJOR-2
TEX_DECOR = re.compile(r'\\(?:mathbf|dot|delta|widetilde|widehat)\s*\{?\s*(?:\\(?:mathbf|dot|widetilde|widehat)\s*\{?\s*)*[UOPQ]')
# an aligned shape listing: a line made only of `X = (shape)` entries (one level of nested parentheses): R1 review,
# MAJOR-1 (tests/test_frame_variations_format.py:25-27); a block needs two or more such lines in a row
_ENTRY = r'[A-Za-z]\w{0,6}\s*=\s*\((?:[^()]|\([^()]*\))*\)'
ALIGNED = re.compile(r'^\s*(?:#\s*)?(?:' + _ENTRY + r'\s*)+$')
TEX_LET = re.compile(r'(?<![\\A-Za-z])(?:U|O(?!\()|P|Q|D|L|R|B)(?![A-Za-z])|\\(?:Dsh|Ush)\b')   # big-O O(...) is not a letter (§1.5 item 13)


def rules(line: Line, path: str) -> typ.Set[str]:
    s = line.text
    if not s or line.region == 'code':
        return set()
    tags = set()
    if line.region in ('tex', 'texmath'):
        if TEX_K16.search(s):
            tags.add('K16')
        if line.region == 'texmath' and TEX_LET.search(_tex_math_part(s)):
            tags.add('TEXLET')
        elif re.search(r'(?<![\\A-Za-z])(?:U|O|P|Q|D|L|R|B)(?=[_^])', s):
            tags.add('TEXLET')
        elif TEX_DECOR.search(s):
            tags.add('TEXLET')
        return tags
    if len(EDGE.findall(s)) >= 2 or re.search(LABEL + r'\s*-{3,}\s*' + LABEL, s):
        tags.add('BOX')
    if '|' in s and PIPE_LINE.match(s) and line.region not in ('mdtable',):
        tags.add('PIPE')
    if LABEL_LINE.match(s) and line.region not in ('str',):
        tags.add('LABELS')
    if WIDTH.search(s):
        tags.add('WIDTH')
    if CORE.search(s):
        tags.add('CORE')
    if INDEX.search(s):
        tags.add('INDEX')
    if WORDIDX.search(s):
        tags.add('WORDIDX')
    if LISTING.match(s):
        tags.add('LISTING')
    if ALIGNED.match(s):
        tags.add('ALIGNED')
    return tags


def _tex_math_part(s: str) -> str:
    """The math segments of a line (everything when no ``$`` delimits it, i.e. inside an environment)."""
    if '$' not in s:
        return re.sub(r'\\(?:text|textbf|emph|code|label|ref|eqref|cite|mathrm)\{[^}]*\}', '', s)
    parts = s.split('$')
    return ' '.join(parts[1::2])


# ---------------------------------------------------------------------------------------------------------
# Grouping candidate lines into proposed blocks
# ---------------------------------------------------------------------------------------------------------

Block = collections.namedtuple('Block', 'file first last col_start col_end kind fenced')

STRUCT = {'BOX', 'PIPE', 'LABELS'}
LETTERISH = {'CORE', 'WIDTH', 'INDEX', 'WORDIDX', 'TEXLET', 'K16'}


def propose(path: str, lines: typ.List[Line], tags: typ.Dict[int, typ.Set[str]]
            ) -> typ.Tuple[typ.List[Block], typ.Dict[int, str]]:
    """Auto-proposed blocks, and a dismissal reason for every other candidate line."""
    blocks: typ.List[Block] = []
    dismissed: typ.Dict[int, str] = {}
    taken: typ.Set[int] = set()
    by_no = {l.no: l for l in lines}
    n = len(lines)

    def text_region(no):
        return by_no[no].region not in ('code',)

    # 1. ASCII diagrams: maximal runs of structural lines (blank lines break a diagram).
    i = 1
    while i <= n:
        if tags.get(i, set()) & STRUCT and text_region(i):
            j = i
            while j + 1 <= n and tags.get(j + 1, set()) & STRUCT and by_no[j + 1].region == by_no[i].region:
                j += 1
            run = range(i, j + 1)
            if j > i and any(tags.get(k, set()) & {'BOX', 'PIPE'} for k in run):
                letters = any(tags.get(k, set()) & {'CORE', 'WIDTH'} for k in run)
                letters = letters or any(re.search(r'(?<![\w])[UDOLR]\d', by_no[k].text) for k in run)
                if letters:
                    blocks.append(Block(path, i, j, None, None, 'ascii diagram', True))
                elif any('PIPE' in tags.get(k, set()) for k in run):
                    for k in run:
                        dismissed[k] = 'DIAGRAM_NO_STORE: a diagram with no store letter or width (B/G/H/V/N/r/n/u labels only)'
                else:
                    for k in run:
                        dismissed[k] = 'NO_PIPE_RUN: consecutive lines with "--" dashes or wide-spaced words but no pipe (|) line: prose or aligned text, not a diagram'
                taken.update(run)
            i = j + 1
        else:
            i += 1

    # 2. Python trailing / tuple-element comments carrying an orthogonality relation: field-comment blocks.
    i = 1
    while i <= n:
        l = by_no[i]
        if i not in taken and l.region in ('tcmt', 'cmt') and 'INDEX' in tags.get(i, set()):
            j = i
            while (j + 1 <= n and by_no[j + 1].region in ('tcmt', 'cmt')
                   and 'INDEX' in tags.get(j + 1, set())):
                j += 1
            c = min(by_no[k].col for k in range(i, j + 1))
            kind = 'field-comment relation'
            blocks.append(Block(path, i, j, c if l.region == 'tcmt' else None, None, kind, True))
            taken.update(range(i, j + 1))
            i = j + 1
        else:
            i += 1

    # 3. Docstring / doc listings and displayed relations: runs of LISTING / INDEX / WORDIDX lines.
    i = 1
    while i <= n:
        t = tags.get(i, set())
        l = by_no[i]
        lead = ('LISTING' in t and t & {'CORE', 'WIDTH'}) or (t & {'INDEX', 'WORDIDX'})
        if (i not in taken and lead and l.region in ('prose', 'lit', 'disp', 'out', 'md', 'mdcode', 'rst', 'rstlit', 'cmt', 'str')):
            j = i
            while j + 1 <= n and (j + 1) not in taken:
                t2 = tags.get(j + 1, set())
                l2 = by_no[j + 1]
                same = l2.region == l.region
                if not same:
                    break
                if 'LISTING' in t and 'LISTING' in t2:
                    j += 1
                elif not ('LISTING' in t) and t2 & {'INDEX', 'WORDIDX'}:
                    j += 1
                else:
                    break
            kind = 'shape listing' if 'LISTING' in t else 'index equation'
            blocks.append(Block(path, i, j, None, None, kind, True))
            taken.update(range(i, j + 1))
            i = j + 1
        else:
            i += 1

    # 3a. Aligned shape listings: two or more consecutive lines made only of `X = (shape)` entries that carry a
    #     store letter or a width (keyed on the alignment, not on the region: a docstring listing is not prose).
    i = 1
    while i <= n:
        l = by_no[i]
        t = tags.get(i, set())
        if i not in taken and 'ALIGNED' in t and l.region not in ('code', 'tex', 'texmath'):
            j = i
            while (j + 1 <= n and (j + 1) not in taken and 'ALIGNED' in tags.get(j + 1, set())
                   and by_no[j + 1].region == l.region):
                j += 1
            run = range(i, j + 1)
            if j > i and any(tags.get(k, set()) & {'CORE', 'WIDTH'} for k in run):
                blocks.append(Block(path, i, j, None, None, 'shape listing', True))
                taken.update(run)
            i = j + 1
        else:
            i += 1

    # 3b. Displayed equations: runs of literal-block lines (docstring ``::`` blocks, rst literal blocks, md
    #     code fences) that carry a store letter, a width or a relation and are not doctest code.
    i = 1
    while i <= n:
        l = by_no[i]
        t = tags.get(i, set())
        if i not in taken and l.region in ('lit', 'disp', 'rstlit', 'mdcode') and t & {'CORE', 'WIDTH', 'INDEX', 'WORDIDX'}:
            j = i
            while (j + 1 <= n and (j + 1) not in taken and by_no[j + 1].region == l.region
                   and by_no[j + 1].raw.strip()):
                j += 1
            blocks.append(Block(path, i, j, None, None, 'index equation', True))
            taken.update(range(i, j + 1))
            i = j + 1
        else:
            i += 1

    # 4. LaTeX: a display environment with a tagged line is one block; text-mode tagged lines group by
    #    consecutive runs (a blank or untagged line breaks them).
    if path.endswith('.tex'):
        spans = _tex_spans(lines)
        span_of = {}
        for (a_, b_, env) in spans:
            for k in range(a_, b_ + 1):
                span_of.setdefault(k, (a_, b_, env))
        i = 1
        while i <= n:
            if i in taken or not (tags.get(i, set()) & {'K16', 'TEXLET'}):
                i += 1
                continue
            if i in span_of:
                first, last, env = span_of[i]
            else:
                first = last = i
                env = None
                while (last + 1 <= n and last + 1 not in span_of and last + 1 not in taken
                       and tags.get(last + 1, set()) & {'K16', 'TEXLET'}):
                    last += 1
            kind = ('LaTeX table or figure' if env and re.match(r'(tikzpicture|tabular|table|figure)', env)
                    else 'LaTeX equation line')
            blocks.append(Block(path, first, last, None, None, kind, True))
            taken.update(range(first, last + 1))
            i = last + 1

    # 5. Everything else that a rule flagged: dismissed, with the class reason.
    for no, t in tags.items():
        if no in taken or no in dismissed or not t:
            continue
        dismissed[no] = classify_dismissal(path, by_no[no], t)
    return blocks, dismissed


TEX_SPAN_ENVS = r'(?:equation|align|gather|multline|eqnarray|displaymath|tikzpicture|tabular|table|figure|algorithmic|algorithm)\*?'


def _tex_spans(lines: typ.List[Line]) -> typ.List[typ.Tuple[int, int, str]]:
    """Outermost display spans: math/tikz/tabular/table/figure/algorithm environments, \\[ \\] and $$ $$."""
    spans, stack, disp = [], [], None
    for l in lines:
        code = l.text
        for m in re.finditer(r'\\(begin|end)\{(' + TEX_SPAN_ENVS + r')\}|\\\[|\\\]|\$\$', code):
            tok = m.group(0)
            if m.group(1) == 'begin':
                stack.append((l.no, m.group(2)))
            elif m.group(1) == 'end':
                if stack:
                    a_, env = stack.pop()
                    if not stack:
                        spans.append((a_, l.no, env))
            elif tok == '\\[' and not stack:
                disp = l.no
            elif tok == '\\]' and disp is not None and not stack:
                spans.append((disp, l.no, 'display'))
                disp = None
            elif tok == '$$' and not stack:
                if disp is None:
                    disp = l.no
                else:
                    spans.append((disp, l.no, 'display'))
                    disp = None
    return spans


def classify_dismissal(path: str, l: Line, t: typ.Set[str]) -> str:
    s = l.text
    if t == {'ALIGNED'}:
        return ('ALIGNED_NO_STORE: an `X = (...)` annotation with no store letter or width (`shape=(...)`, `K = (3,)`, '
                'a tuple of names), alone or in a run without one: not a store listing (R1 review MAJOR-1 rule)')
    if l.region == 'tcmt':
        if STORE_LABEL.search(s) or 'WIDTH' in t:
            return 'SHAPE_COMMENT: signature/field/tuple-element/body shape comment; widths and labels scripted by A1/A3 in R2, checked by K13'
        return 'TRAILING_COMMENT_LETTER: a letter in a code comment (a local named for its core, or prose): §3.4 / tier C/E row, not a block'
    if l.region == 'ex':
        return 'DOCTEST_CODE: a doctest example line (code); its letters are letter_sites rows (R2, K15), not a block'
    if l.region == 'out':
        return 'DOCTEST_OUTPUT: expected output (unchanged by §4A e)'
    if l.region == 'mdtable':
        return 'MD_TABLE_ROW: a markdown table row in prose (glossary/symbol/compat tables): tier A + tier E rows (R5), not a diagram'
    if l.region == 'str':
        return 'STRING_LITERAL: a message or label string; tier A/B/E rows, not a block'
    if t & {'WIDTH'} and not t & {'CORE', 'INDEX'}:
        return 'PROSE_WIDTH: a width symbol inside a prose sentence or inline shape; scripted by A3 (K13 where an entry)'
    if 'LISTING' in t and not t & {'CORE', 'WIDTH'}:
        return 'LISTING_NO_STORE: a core listing of B/G/V/H only (no store letter, width or store word); nothing to rewrite'
    if 'TEXLET' in t or 'K16' in t:
        return 'TEX_OTHER'
    if t & {'LABELS'} and not t & {'CORE', 'WIDTH'}:
        return 'LABEL_ONLY: a line of short words with wide spacing, not part of a diagram (no box/pipe neighbour)'
    if t & {'BOX', 'PIPE', 'LABELS'} and not t & {'CORE', 'WIDTH'}:
        return 'PROSE_DASHES: "--" used as a dash in prose, or a pipe in text; not a diagram'
    if 'CORE' in t:
        return 'PROSE_LETTER: a store letter inside a prose sentence or a one-line inline formula; tier D/E or §3.4 row (R4a/R4b/R5), not a block (sample edit 7 treats jet-shift prose this way)'
    return 'OTHER'


# ---------------------------------------------------------------------------------------------------------
# Hand curation (R1): overrides of the automatic proposal, each with its reason
# ---------------------------------------------------------------------------------------------------------

from diagram_curation import CURATION_BLOCKS, CURATION_DROP, CURATION_NOTES, SEED_RESOLUTION  # noqa: E402
from diagram_curation import CURATION_PANELS  # noqa: E402


def span_text(raw_lines: typ.List[str], b: Block) -> str:
    seg = raw_lines[b.first - 1:b.last]
    if b.col_start is not None:
        seg = [s[b.col_start:(b.col_end if b.col_end is not None else len(s))] for s in seg]
    return '\n'.join(seg)


def letters_in(text: str, path: str) -> str:
    if path.endswith('.tex'):
        found = sorted(set(m.group(0) for m in re.finditer(
            r'(?<![\\A-Za-z])(?:U|O|P|Q|D|L|R|B)(?:_\{?[\w,]+\}?|\^\{?[\w,]+\}?)?(?![A-Za-z])|\\(?:Dsh|Ush)|'
            r'n_\{[DUO],\w\}|n\^\{?[DUO]\}?(?:_\w)?|r_\{[LR],[\w+-]+\}|r\^\{?[LR]\}?(?:_\{?[\w+-]+\}?)?', text)))
        return ', '.join(found[:40]) + (' …' if len(found) > 40 else '')
    cores = sorted(set(re.findall(r'(?<![\w])[UODPQLRBGVHE](?:\d+|\(d-1\)|d|i|_[a-z]{1,4}|ᵢ)?(?![\w])', text)))
    widths = sorted(set(m.group(0) for m in re.finditer(r'(?<![\\\w])(?:n[UDO]|r[LR])\w*(?:\([\w+-]+\))?', text)))
    words = sorted(set(re.findall(r'\b(?:up|down|left|right|outer)_\w+|\b(?:up|down)\b', text)))
    parts = []
    if cores:
        parts.append('letters: ' + ' '.join(cores))
    if widths:
        parts.append('widths: ' + ' '.join(widths))
    if words:
        parts.append('store words: ' + ' '.join(words))
    return '; '.join(parts) if parts else '(none)'


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()


_ABBREV: typ.Dict[str, str] = {}


def _abbrevs(paths: typ.Iterable[str]) -> None:
    """A short, unique id prefix per file: the stem's initials, two letters per word where initials collide."""
    def initials(path, k):
        stem = path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        a = ''.join(w[:k] for w in stem.split('_') if w).upper()
        if path.startswith('docs/contributor/'):
            return 'C-' + a
        if path.startswith('docs/') or path in ('README.md', 'CLAUDE.md'):
            return ('TEX-' if path.endswith('.tex') else 'DOC-') + a
        if path.startswith('tests/'):
            return 'T-' + a
        if path.startswith('examples/'):
            return 'X-' + a
        return a
    paths = sorted(set(paths))
    k = {p: 1 for p in paths}
    for _ in range(12):
        cur = {p: initials(p, k[p]) for p in paths}
        count = collections.Counter(cur.values())
        clash = [p for p in paths if count[cur[p]] > 1]
        if not clash:
            break
        for p in clash:
            k[p] += 1
    count = collections.Counter(cur.values())
    for p in paths:          # the same stem in the frontend and the backend: mark the backend one
        if count[cur[p]] > 1 and '/backend/' in p:
            cur[p] = 'B-' + cur[p]
    _ABBREV.update(cur)
    assert len(set(_ABBREV.values())) == len(_ABBREV), 'id prefixes collide'


def block_id(path: str, first: int, col_start: typ.Optional[int] = None) -> str:
    """``FVF-0058``; a side-by-side panel (curation's CURATION_PANELS) takes a suffix, ``FVF-0058a``."""
    return '%s-%04d%s' % (_ABBREV[path], first, CURATION_PANELS.get((path, first, col_start), ''))


def _overlap(x: 'Block', b: 'Block') -> bool:
    """Two blocks overlap when their lines AND their column ranges do (a side-by-side panel owns a column range)."""
    if x.file != b.file or not (x.first <= b.last and b.first <= x.last):
        return False
    xa, xb = x.col_start or 0, float('inf') if x.col_end is None else x.col_end
    ba, bb = b.col_start or 0, float('inf') if b.col_end is None else b.col_end
    return xa < bb and ba < xb


def main() -> None:
    files = scope_files()
    _abbrevs(p for p, _ in files)
    all_cands = []
    all_blocks: typ.List[Block] = []
    sources: typ.Dict[str, typ.List[str]] = {}
    refs: typ.Dict[str, str] = {}
    for path, ref in files:
        src = git_show(ref, path)
        refs[path] = ref
        sources[path] = src.split('\n')
        if path.endswith('.py'):
            lines = python_lines(src)
        elif path.endswith('.md'):
            lines = md_lines(src)
        elif path.endswith('.rst'):
            lines = rst_lines(src)
        else:
            lines = tex_lines(src)
        tags = {l.no: rules(l, path) for l in lines}
        tags = {k: v for k, v in tags.items() if v}
        # doctest example lines are candidates only through the companion rule (curation)
        blocks, dismissed = propose(path, lines, tags)
        all_blocks.extend(blocks)
        for no, t in sorted(tags.items()):
            all_cands.append([path, no, lines[no - 1].region, ','.join(sorted(t)), dismissed.get(no), lines[no - 1].raw])

    # curation: drop auto blocks, add hand blocks
    drop = {(f, a) for f, a in CURATION_DROP}
    blocks = [b for b in all_blocks if (b.file, b.first) not in drop]
    seen = {(b.file, b.first) for b in blocks}
    hand = [Block(*spec) for spec in CURATION_BLOCKS]
    for b in hand:                         # a hand block replaces every automatic block whose LINES it overlaps
        blocks = [x for x in blocks if x in hand or not (x.file == b.file and x.first <= b.last and b.first <= x.last)]
        blocks.append(b)
    blocks.sort(key=lambda b: (b.file, b.first, b.col_start or 0))
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            assert not _overlap(a, b), ('overlap', a, b)

    # each candidate line: its block, else its dismissal
    def owner(path, no):
        for b in blocks:
            if b.file == path and b.first <= no <= b.last:
                return block_id(b.file, b.first, b.col_start)
        return None

    with open(OUT_DIR + '/diagram_candidates.tsv', 'w') as f:
        f.write('file\tline\tregion\trules\tdisposition\treason\ttext\n')
        n_in = n_dis = 0
        for path, no, reg, t, dis, raw in all_cands:
            bid = owner(path, no)
            if bid:
                disp, reason = bid, ''
                n_in += 1
            else:
                disp = 'dismissed'
                reason = CURATION_NOTES.get((path, no)) or dis or 'CURATION: dropped from a proposed block (see the inventory notes)'
                n_dis += 1
            f.write('%s\t%d\t%s\t%s\t%s\t%s\t%s\n' % (path, no, reg, t, disp, reason, raw.replace('\t', '\\t')))

    with open(OUT_DIR + '/fences.tsv', 'w') as f:
        f.write('id\tfile\tfirst_line\tlast_line\tcol_start\tcol_end\tfenced\tkind\tanchor_sha1\n')
        for b in blocks:
            txt = span_text(sources[b.file], b)
            f.write('%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n' % (
                block_id(b.file, b.first, b.col_start), b.file, b.first, b.last,
                '-' if b.col_start is None else b.col_start, '-' if b.col_end is None else b.col_end,
                'y' if b.fenced else 'n', b.kind, sha1(txt)))

    write_inventory(blocks, sources, refs, len(all_cands), n_in, n_dis)
    print('candidates %d (in blocks %d, dismissed %d); blocks %d' % (len(all_cands), n_in, n_dis, len(blocks)))
    kinds = collections.Counter(b.kind for b in blocks)
    print(dict(kinds))


def write_inventory(blocks, sources, refs, n_cands, n_in, n_dis) -> None:
    from diagram_curation import BLOCK_NOTES, BLOCK_NOTES_BY_FILE, PREAMBLE
    groups = collections.Counter(
        'tex' if b.file.endswith('.tex') else ('docs' if b.file.startswith('docs/') or b.file in ('README.md', 'CLAUDE.md')
                                               else 'code') for b in blocks)
    kinds = collections.Counter(b.kind for b in blocks)
    k16 = [(b, no, sources[b.file][no - 1]) for b in blocks if b.file.endswith('.tex')
           for no in range(b.first, b.last + 1) if TEX_K16.search(sources[b.file][no - 1])]
    out = [PREAMBLE % dict(n_cands=n_cands, n_in=n_in, n_dis=n_dis, n_blocks=len(blocks),
                           groups=', '.join('%s %d' % kv for kv in sorted(groups.items())),
                           kinds=', '.join('%s %d' % kv for kv in sorted(kinds.items())), n_k16=len(k16))]
    out.append('\n## Seed list resolution (§4A a)\n')
    out.append('| seed entry | resolved as |\n|---|---|')
    ids = {(b.file, b.first): block_id(b.file, b.first, b.col_start) for b in blocks}

    def resolve(m):
        f, n = m.group(1).rsplit(':', 1)
        return '**%s**' % ids[(f, int(n))]
    for seed, res in SEED_RESOLUTION:
        out.append('| %s | %s |' % (seed, re.sub(r'<<([^>]+)>>', resolve, res)))
    out.append('\n## K16: every LaTeX width form in the notes (sign-off list for R5)\n')
    out.append('| block | file:line | text |\n|---|---|---|')
    for b, no, raw in k16:
        out.append('| %s | `%s:%d` | `%s` |' % (block_id(b.file, b.first, b.col_start), b.file, no, raw.strip().replace('|', '\\|')))
    out.append('\n## Blocks\n')
    cur_file = None
    for b in blocks:
        if b.file != cur_file:
            cur_file = b.file
            out.append('\n### `%s`%s\n' % (b.file, '' if refs[b.file] == TAG else ' (read at %s: absent at %s)' % (refs[b.file], TAG)))
        bid = block_id(b.file, b.first, b.col_start)
        txt = span_text(sources[b.file], b)
        note = dict(BLOCK_NOTES_BY_FILE.get(b.file, {}))
        note.update(BLOCK_NOTES.get((b.file, b.first), {}))
        span = '%d–%d' % (b.first, b.last) if b.last > b.first else '%d' % b.first
        if b.col_start is not None:
            span += ', columns %d–%s' % (b.col_start, 'end' if b.col_end is None else b.col_end)
        k16_here = [no for (bb, no, _) in k16 if bb is b]
        out.append('#### %s%s\n' % (bid, '  (K16)' if k16_here else ''))
        out.append('- file: `%s`; old span: %s; kind: %s; fenced: %s' % (b.file, span, b.kind, 'y' if b.fenced else 'n'))
        out.append('- letters and widths: %s' % note.get('letters', letters_in(txt, b.file)))
        if note.get('roles'):
            out.append('- roles: %s' % note['roles'])
        out.append('- re-laid out: %s' % note.get('relayout', _default_relayout(b, txt)))
        out.append('- new == old possible: %s' % note.get('same', _default_same(b, txt)))
        if k16_here:
            out.append('- K16 width lines: %s' % ', '.join(str(x) for x in k16_here))
        if note.get('note'):
            out.append('- note: %s' % note['note'])
        out.append('- new span: (R4/R5)')
        out.append('- same-reason: (R4/R5: required only when the new text equals the old)')
        fence = '~~~~' if '~~~' in txt else '~~~'
        out.append('\nold text:\n\n%s\n%s\n%s\n' % (fence, txt, fence))
        out.append('new text (R4/R5, verbatim the file at the new span):\n\n%s\n%s\n' % (fence, fence))
    path = OUT_DIR + '/diagrams_inventory.md'
    try:
        current = open(path).read()
    except FileNotFoundError:
        current = ''
    if re.search(r'^- new span: (?!\(R4/R5\))', current, re.M) and '--force' not in sys.argv:
        raise SystemExit('diagrams_inventory.md has filled blocks (R4/R5); not overwriting (pass --force)')
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


def _default_relayout(b: Block, txt: str) -> str:
    if b.kind == 'ascii diagram':
        return 'n expected: labels keep their length (U0→D0, nU0→nD0, nO1→nU1); K10 in place'
    if b.kind in ('shape listing', 'field-comment relation') and STORE_LABEL.search(txt):
        return 'y expected: the store identifiers change length (up_tucker_* → down_tucker_*, down_tt_* → up_tt_*); K10b'
    if b.file.endswith('.tex'):
        return 'n/a: LaTeX source (no column alignment); reviewed by contraction sizes (§4A f)'
    return 'n expected unless a label grows; K10 in place'


def _default_same(b: Block, txt: str) -> str:
    if re.search(r'(?<![\w\\])[UO](?:\d|_|i\b|\(d-1\)|d\b)|(?<![\\\w])n[UDO]|\b(?:up|down)_', txt):
        return 'no: it carries a changing letter, width or store word'
    if b.file.endswith('.tex'):
        return 'possible: classify each letter by §4.8 / §3.4 at R5; record the reason if nothing changes'
    return 'possible: record the reason at R4 if nothing changes'


if __name__ == '__main__':
    sys.path.insert(0, OUT_DIR)
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('run with the t3toolbox env (Python 3.11)')
    main()
