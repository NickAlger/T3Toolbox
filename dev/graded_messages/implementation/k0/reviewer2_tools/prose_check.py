"""Check the built tree against k0/prose_edits.md, block by block.

    python prose_check.py ROOT [PROSE_MD]

prose_edits.md quotes each text K0 rewrites as indented blocks, labelled by the prose before them ("Before:",
"After:"). For each block: the file it belongs to (the last `path:lines` named in a heading or a bullet before
it), and whether its lines occur in that file of ROOT as one contiguous run:
  - EXACT: line for line, with the block's own indentation removed and the file's kept (the relative indentation,
    the aligned columns and the banners must match);
  - CONTENT: the same lines once every line is stripped of its leading blanks (indentation differs);
  - ABSENT.
An "After" block must be EXACT; a "Before" block must be ABSENT, unless the text says it stays (then read it).
"""
import os
import re
import sys

DEFAULT = '/home/nick/repos/T3Toolbox/dev/graded_messages/implementation/k0/prose_edits.md'
PATH = re.compile(r'`((?:t3toolbox/|docs/)?[\w/]*?\w+\.(?:py|rst|md)):\d')
SHORT = {'probing.py', 'apply.py', 'entries.py', 'ut3_sampling.py', 'utv_sampling.py', 'fitting.py',
         'uniform_fitting.py', 'common.py'}


def resolve(name, current_section):
    if name.startswith(('t3toolbox/', 'docs/')):
        return name
    if name in SHORT:
        return 't3toolbox/backend/' + name
    return name


def blocks(md_lines):
    """(label, file, start_line, [lines]) for every indented block."""
    out, prose, cur_file, i = [], [], None, 0
    section_api = False
    while i < len(md_lines):
        line = md_lines[i]
        if line.startswith('## 3.'):
            section_api = True
        if line.startswith('## 4.'):
            section_api = False
        indent = len(line) - len(line.lstrip(' '))
        is_block_line = line.strip() and indent >= 4
        prev_blank = i > 0 and not md_lines[i - 1].strip()
        if is_block_line and prev_blank:
            start = i
            blk = []
            while i < len(md_lines) and (not md_lines[i].strip() or
                                         len(md_lines[i]) - len(md_lines[i].lstrip(' ')) >= indent):
                blk.append(md_lines[i])
                i += 1
            while blk and not blk[-1].strip():
                blk.pop()
            text = ' '.join(prose)
            ia, ib = text.rfind('After'), text.rfind('Before')
            label = 'after' if ia > ib else ('before' if ib >= 0 else '?')
            dedent = min(len(b) - len(b.lstrip(' ')) for b in blk if b.strip())
            out.append((label, cur_file, start + 1, [b[dedent:] if b.strip() else '' for b in blk]))
            prose = []
            continue
        m = PATH.findall(line)
        if line.startswith('### ') and section_api:
            cur_file = 'docs/api_reference.rst'
        elif m and (line.startswith('### ') or line.lstrip().startswith('- **')):
            cur_file = resolve(m[0], None)
        if line.strip():
            prose.append(line)
        i += 1
    return out


def find(file_lines, blk):
    n = len(blk)
    for k in range(len(file_lines) - n + 1):
        window = file_lines[k:k + n]
        if all((not b and not w.strip()) or (b and w.endswith(b) and not w[:len(w) - len(b)].strip())
               for b, w in zip(blk, window)):
            # the same relative indentation: one common leading prefix for the whole run
            pref = {w[:len(w) - len(b)] for b, w in zip(blk, window) if b}
            if len(pref) == 1:
                return 'EXACT', k + 1
    stripped = [w.strip() for w in file_lines]
    sb = [b.strip() for b in blk]
    for k in range(len(stripped) - n + 1):
        if stripped[k:k + n] == sb:
            return 'CONTENT', k + 1
    return 'ABSENT', None


def main(argv):
    root = argv[0]
    md = argv[1] if len(argv) > 1 else DEFAULT
    for label, f, start, blk in blocks(open(md).read().split('\n')):
        path = os.path.join(root, f) if f else None
        if not path or not os.path.exists(path):
            print('%-6s md:%-4d %-45s  FILE MISSING' % (label, start, f))
            continue
        status, at = find(open(path).read().split('\n'), blk)
        flag = ''
        if label == 'after' and status != 'EXACT':
            flag = '   <-- after-text not found exactly'
        if label == 'before' and status != 'ABSENT':
            flag = '   <-- before-text still present (check whether it is meant to stay)'
        print('%-6s md:%-4d %-45s %d lines  %-7s %s%s' % (label, start, f, len(blk), status,
                                                          'at %d' % at if at else '', flag))


if __name__ == '__main__':
    main(sys.argv[1:])
