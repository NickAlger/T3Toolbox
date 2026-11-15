"""R6's alignment scan (store rename; one-off, in dev/ like the other gate tools).

Over every .py/.md/.rst file the arc touched (``git diff --name-only v2026.4.0 HEAD``, ``dev/`` excluded), three column
features per line: a trailing comment after code (code lines, doctest source lines, markdown code blocks), the type
column of a ``name:   type`` signature/field line, and the ``=`` of a padded assignment. Pairs are consecutive lines
with the same feature (a doctest comment pairs across up to two output lines; a type line across its own
multi-line type). A pair is an INTENDED alignment when one of its lines is padded (a comment gap >= 3 spaces, a
type/``=`` gap >= 2). Reports:

* ``drift``: adjacent lines that shared a column at the tag and no longer do (the arc broke an alignment);
* ``NEAR``: intended pairs 1-2 columns apart (a slipped alignment);
* ``FIT``: intended pairs >= 3 apart where both lines could take the other's column (a judgement: separate groups
  or a slip);
* ``LONG`` (counted only): >= 3 apart where one line's code is too long for the other's column (a stick-out).

All of the above pair ADJACENT lines only. ``--bridge`` (report only) adds the comment columns split by one or two
comment-less non-blank lines (the R6 reviewer's bridge scan): ``BRIDGE`` pairs 1-2 columns apart in the tree, with
whether the two lines share an indent, and ``BRIDGE-DRIFT`` pairs that shared a column at the tag and no longer do.
A bridged pair is often two groups (a statement and a nested block, a return line, a call's continuation), so it is
a list to judge, not a gate.

Usage: ``python dev/store_rename_slices/r6_align_scan.py [--list NEAR,FIT] [--bridge]`` (the t3toolbox env).
"""
import argparse, difflib, io, os, re, subprocess, sys, tokenize, collections

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = 'v2026.4.0'
PROMPT = re.compile(r'^(\s*)(>>> |\.\.\. )')
TYPECOL = re.compile(r'^(\s*)([A-Za-z_]\w*):(\s+)(\S)')
EQCOL = re.compile(r'^(\s*)([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)(\s+)=(?!=)\s')


def git(*a):
    return subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True).stdout


def frag_comment_col(s):
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(s + '\n').readline))
    except Exception:
        m = re.search(r'\S(\s+)#', s)
        if not m:
            return None
        b = s[:m.end() - 1]
        return None if (b.count("'") % 2 or b.count('"') % 2) else m.end() - 1
    code = False
    for t in toks:
        if t.type == tokenize.COMMENT:
            return t.start[1] if code else None
        if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
            code = True
    return None


def features(path, text):
    lines = text.split('\n')
    feats = {}
    string_lines, code_comment, code_lines = set(), {}, set()
    if path.endswith('.py'):
        seen = {}
        for t in tokenize.generate_tokens(io.StringIO(text).readline):
            if t.type == tokenize.STRING and t.start[0] != t.end[0]:
                string_lines.update(range(t.start[0], t.end[0] + 1))
            if t.type == tokenize.COMMENT:
                if seen.get(t.start[0]):
                    code_comment[t.start[0]] = t.start[1]
            elif t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING):
                seen[t.start[0]] = True
                code_lines.add(t.start[0])
    in_md = False
    for i, line in enumerate(lines, 1):
        fl = []
        if i in code_comment:
            fl.append(('comment', code_comment[i]))
        m = PROMPT.match(line)
        if m and (path.endswith(('.md', '.rst')) or i in string_lines):
            c = frag_comment_col(line[m.end():])
            if c is not None:
                fl.append(('doctest', m.end() + c))
        elif path.endswith('.md') and line.lstrip().startswith('```'):
            in_md = not in_md
        elif path.endswith('.md') and in_md:
            c = frag_comment_col(line)
            if c is not None:
                fl.append(('mdcode', c))
        if path.endswith('.py') and i in code_lines and not (i in string_lines and i not in code_comment):
            mt = TYPECOL.match(line)
            if mt and not line.rstrip().endswith(':'):
                fl.append(('type', mt.start(4)))
            me = EQCOL.match(line)
            if me:
                fl.append(('assign', me.end(3)))
        if fl:
            feats[i] = dict(fl)
    return lines, feats


def gap(line, kind, col):
    if kind == 'assign':
        return len(re.match(r'^(.*?)(\s*)=', line).group(2))
    return col - len(line[:col].rstrip())


def partner(lines, feats, i, kind):
    if kind == 'doctest':
        k = i + 1
        while k <= len(lines) and k - i <= 3 and lines[k - 1].strip() and not PROMPT.match(lines[k - 1]):
            k += 1
        return k
    if kind == 'type':
        ind0 = len(lines[i - 1]) - len(lines[i - 1].lstrip())
        k = i + 1
        while k <= len(lines) and k - i <= 12:
            ln = lines[k - 1]
            ind = len(ln) - len(ln.lstrip())
            if 'type' in feats.get(k, {}) and ind == ind0:
                break
            if ln.strip() and (ind > ind0 or (ind == ind0 and re.match(r'^\s*[\])],', ln))):
                k += 1
                continue
            break
        return k
    return i + 1


def pairs(lines, feats):
    for i in sorted(feats):
        for kind, ci in feats[i].items():
            j = partner(lines, feats, i, kind)
            cj = feats.get(j, {}).get(kind)
            if cj is None:
                continue
            gi, gj = gap(lines[i - 1], kind, ci), gap(lines[j - 1], kind, cj)
            if max(gi, gj) < (3 if kind in ('comment', 'doctest', 'mdcode') else 2):
                continue
            yield i, j, kind, ci, cj


def bridged(lines, feats):
    """Comment pairs (i, j) with one or two comment-less non-blank lines between them, one line padded."""
    for i in sorted(feats):
        for kind, ci in feats[i].items():
            if kind not in ('comment', 'doctest', 'mdcode'):
                continue
            for n in (1, 2):
                j = i + n + 1
                mids = range(i + 1, j)
                if j > len(lines) or any(kind in feats.get(m, {}) for m in mids) \
                        or any(not lines[m - 1].strip() for m in mids):
                    break
                cj = feats.get(j, {}).get(kind)
                if cj is None:
                    continue
                if max(gap(lines[i - 1], kind, ci), gap(lines[j - 1], kind, cj)) >= 3:
                    indent = lambda k: len(lines[k - 1]) - len(lines[k - 1].lstrip())
                    yield i, j, kind, ci, cj, indent(i) == indent(j)
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='NEAR,FIT')
    ap.add_argument('--bridge', action='store_true', help='also list comment columns split by 1-2 comment-less lines')
    a = ap.parse_args()
    show = set(a.list.split(','))
    files = [f for f in git('diff', '--name-only', BASE, 'HEAD').split()
             if f.endswith(('.py', '.md', '.rst')) and not f.startswith('dev/') and os.path.exists(os.path.join(REPO, f))]
    count, drift, checked = collections.Counter(), 0, 0
    for f in files:
        new = open(os.path.join(REPO, f), encoding='utf-8').read()
        lines, feats = features(f, new)
        for i, j, kind, ci, cj in pairs(lines, feats):
            if ci == cj:
                continue
            ei, ej = len(lines[i - 1][:ci].rstrip()), len(lines[j - 1][:cj].rstrip())
            cls = 'NEAR' if abs(ci - cj) <= 2 else ('FIT' if max(ei, ej) + 1 <= min(ci, cj) else 'LONG')
            count[cls] += 1
            if cls in show:
                print('%s\t%s:%d-%d\t%s\t%d/%d' % (cls, f, i, j, kind, ci + 1, cj + 1))
        if a.bridge:
            for i, j, kind, ci, cj, same in bridged(lines, feats):
                if 1 <= abs(ci - cj) <= 2:
                    count['BRIDGE'] += 1
                    print('BRIDGE\t%s:%d-%d\t%s\t%d/%d\tsame_indent=%s' % (f, i, j, kind, ci + 1, cj + 1, same))
        old = git('show', '%s:%s' % (BASE, f))
        if not old:
            continue
        olines, ofeats = features(f, old)
        sm = difflib.SequenceMatcher(None, olines, lines, autojunk=False)
        mp = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal' or (tag == 'replace' and i2 - i1 == j2 - j1):
                mp.update({i1 + k + 1: j1 + k + 1 for k in range(i2 - i1)})
        for i, j, kind, ci, cj in pairs(olines, ofeats):
            if ci != cj or j != i + 1 or i not in mp or j not in mp or mp[j] != mp[i] + 1:
                continue
            checked += 1
            ni, nj = feats.get(mp[i], {}).get(kind), feats.get(mp[j], {}).get(kind)
            if ni is not None and nj is not None and ni != nj:
                drift += 1
                print('DRIFT\t%s:%d-%d\t%s\ttag %d -> %d/%d' % (f, mp[i], mp[j], kind, ci + 1, ni + 1, nj + 1))
        if a.bridge:
            for i, j, kind, ci, cj, same in bridged(olines, ofeats):
                if ci != cj or i not in mp or j not in mp or mp[j] - mp[i] != j - i:
                    continue
                ni, nj = feats.get(mp[i], {}).get(kind), feats.get(mp[j], {}).get(kind)
                if ni is not None and nj is not None and ni != nj:
                    count['BRIDGE-DRIFT'] += 1
                    print('BRIDGE-DRIFT\t%s:%d-%d\t%s\ttag %d -> %d/%d' % (f, mp[i], mp[j], kind, ci + 1, ni + 1, nj + 1))
    print('files %d; tag-aligned adjacent pairs %d, drifted %d; NEAR %d, FIT %d, LONG %d (adjacent pairs)'
          % (len(files), checked, drift, count['NEAR'], count['FIT'], count['LONG']), file=sys.stderr)
    if a.bridge:
        print('bridged (report only): BRIDGE %d, BRIDGE-DRIFT %d' % (count['BRIDGE'], count['BRIDGE-DRIFT']),
              file=sys.stderr)


if __name__ == '__main__':
    main()
