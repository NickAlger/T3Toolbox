"""Generates the drawings of the "Variable diagrams" tab of the Claude doc (prototype of 2026-11-15, 23:02).

The maintainer asked for a table of tensor network diagrams, one row per sampling variable: the name, the
subnetwork that is contracted at one representative edge in the middle of the network, and the shape. Each row
is one drawing (a widget module: one line of JSX that returns an SVG). This script writes the modules as a JSON
map from a slug to its module, which were then published to the doc by hand. Seven rows so far: mu, nu, nu_L,
xi, eta, mu_R_dot, mu_tilde, in the names of the four-letters-and-a-basis scheme (TYPE_CATALOG.md, section 14).

Inside a drawing a variable is written as in a formula (the maintainer, 23:10): the Greek letter, a tilde or a
dot drawn above it, and the basis and the site as subscripts. The left column and the title keep the code spelling.

Every text in a module is a literal with its own id, as the doc requires.

Usage:  python3 variable_diagrams_gen.py out.json [preview_folder]
With a preview folder, each drawing is also written there as a plain SVG, and as a PNG if rsvg-convert is installed.
"""
import json, sys

XS = [226, 296, 366, 436, 506]          # site centres
YT, YD, YW = 96, 152, 204               # TT row, Tucker row, input row
BW, BH, DW, DH, RW = 44, 30, 40, 26, 13

HEAD = ("export default () => { const edge = 'var(--cds-chart-axis)', grid = 'var(--cds-chart-grid)', accent = 'var(--cds-chart-categorical-1)', "
        "ink = 'var(--cds-text-primary)', quiet = 'var(--cds-text-secondary)'; "
        "const core = (cx, cy, w, h, kind) => <rect x={cx - w / 2} y={cy - h / 2} width={w} height={h} rx='6' fill={kind === 'hot' ? accent : 'none'} "
        "fillOpacity={kind === 'hot' ? 0.12 : 1} stroke={kind === 'hot' ? accent : edge} strokeWidth={kind === 'hot' ? 2 : 1.25} "
        "strokeDasharray={kind === 'ghost' ? '4 4' : 'none'} strokeOpacity={kind === 'ghost' ? 0.55 : 1}/>; "
        "const dot = (cx, cy, hot) => <circle cx={cx} cy={cy} r='13' fill={hot ? accent : 'none'} fillOpacity={hot ? 0.12 : 1} stroke={hot ? accent : edge} strokeWidth={hot ? 2 : 1.25}/>; "
        "const wire = (x1, y1, x2, y2) => <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={edge} strokeWidth='1.25'/>; ")


def text(tid, x, y, words, size=None, weight=None, anchor=None, fill='ink'):
    assert "'" not in words and '{' not in words and '<' not in words, words
    attrs = "data-claude-text-id='%s' x='%s' y='%s'" % (tid, x, y)
    if anchor: attrs += " textAnchor='%s'" % anchor
    if size: attrs += " fontSize='%s'" % size
    if weight: attrs += " fontWeight='%s'" % weight
    return "<text %s fill={%s}>%s</text>" % (attrs, fill, words)


def sub_text(tid, x, y, main, sub, size, anchor='middle', weight=600, italic=False):
    """A letter with a subscript, as in a formula: two tspans, each with its own id (the text element has none)."""
    assert all(c not in main + sub for c in "'{<")
    attrs = "x='%s' y='%s' fontSize='%s' fontWeight='%s'" % (x, y, size, weight)
    if anchor != 'start': attrs += " textAnchor='%s'" % anchor
    style = " fontStyle='italic'" if italic else ''
    return ("<text %s fill={ink}><tspan data-claude-text-id='%s'%s>%s</tspan><tspan data-claude-text-id='%s-sub' dy='4' fontSize='10.5'>%s</tspan></text>"
            % (attrs, tid, style, main, tid, sub))


def accent(kind, cx, y):
    """A dot or a tilde drawn above a letter whose baseline is y and whose centre is cx."""
    if kind == 'dot':
        return "<circle cx='%.1f' cy='%.1f' r='1.5' fill={ink}/>" % (cx, y - 12.5)
    if kind == 'tilde':
        return "<path d='M%.1f %.1f q2 -2.6 4 0 t4 0' fill='none' stroke={ink} strokeWidth='1.3'/>" % (cx - 4, y - 11.5)
    assert kind is None
    return ''


GREEK_W, SUB_W = 8.6, 5.4            # rough widths at 15 italic 600 and at 10.5, used only to place a label


def module(spec):
    slug, H = spec['slug'], spec.get('H', 240)
    out = [HEAD, "return <svg viewBox='0 0 760 %d' role='img' aria-label='%s' fontSize='13'>" % (H, spec['title'])]
    out.append("<defs><marker id='%s-arrow' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto-start-reverse'>"
               "<path d='M0 0L10 5L0 10z' fill={accent}/></marker></defs>" % slug)
    out.append(text('title', 24, 24, spec['title'], size=13, weight=600))
    out.append("<line x1='184' y1='40' x2='184' y2='%d' stroke={grid}/><line x1='572' y1='40' x2='572' y2='%d' stroke={grid}/>" % (H - 14, H - 14))
    # name column
    out.append("<g data-claude-anchor='name-column'>")
    out.append(text('name', 24, 100, spec['name'], size=18, weight=600))
    out.append(text('side', 24, 124, spec['side'], size=11.5, fill='quiet'))
    out.append(text('basis', 24, 140, spec['basis'], size=11.5, fill='quiet'))
    out.append("</g>")
    # site labels
    out.append("<g data-claude-anchor='site-labels'>")
    for i, x in enumerate(XS):
        out.append(text('site-%d' % i, x, 58, 'site %d' % i, size=11.5, anchor='middle', fill='quiet'))
    out.append("</g>")
    # network
    out.append("<g data-claude-anchor='network'>")
    sites = spec['sites']                      # i -> dict(tt=label or None, kind, leaf=(Dlabel, wlabel, hot_w) or None)
    for i, x in enumerate(XS):
        s = sites.get(i)
        if s is None or s.get('tt') is None:
            out.append("{core(%d, %d, %d, %d, 'ghost')}" % (x, YT, BW, BH))
        else:
            out.append("{core(%d, %d, %d, %d, '%s')}" % (x, YT, BW, BH, s.get('kind', 'plain')))
            out.append(sub_text('tt-%d' % i, x, YT + 5, s['tt'][0], s['tt'][1:], 14))
        if s is not None and s.get('leaf'):
            dl, wl, hot_w, hot_d = s['leaf']
            if s.get('tt') is not None:
                out.append("{wire(%d, %d, %d, %d)}" % (x, YT + BH // 2, x, YD - DH // 2))
            out.append("{core(%d, %d, %d, %d, '%s')}" % (x, YD, DW, DH, 'hot' if hot_d else 'plain'))
            out.append(sub_text('tucker-%d' % i, x, YD + 5, dl[0], dl[1:], 14))
            out.append("{wire(%d, %d, %d, %d)}" % (x, YD + DH // 2, x, YW - RW))
            out.append("{dot(%d, %d, %s)}" % (x, YW, 'true' if hot_w else 'false'))
            out.append(sub_text('input-%d' % i, x, YW + 3, wl[0], wl[1:], 12.5, weight=500, italic=True))
            if hot_w:                                  # the seed: w with a tilde
                out.append("<path d='M%.1f %.1f q1.6 -2.2 3.2 0 t3.2 0' fill='none' stroke={ink} strokeWidth='1.2'/>" % (x - 6.2, YW - 7.2))
        elif s is not None and s.get('ghost_leaf'):
            out.append("{core(%d, %d, %d, %d, 'ghost')}" % (x, YD, DW, DH))
    for (i, j) in spec.get('bonds', []):       # contracted bonds between sites i and j = i + 1
        out.append("{wire(%d, %d, %d, %d)}" % (XS[i] + BW // 2, YT, XS[j] - BW // 2, YT))
    out.append("</g>")
    # the open edge
    x1, y1, x2, y2 = spec['open']
    out.append("<g data-claude-anchor='open-edge'><line x1='%d' y1='%d' x2='%d' y2='%d' stroke={accent} strokeWidth='2' markerEnd='url(#%s-arrow)'/>" % (x1, y1, x2, y2, slug))
    lx, ly, la = spec['open_label_at']
    letter, sub, mark = spec['open_label']
    if la == 'middle':                                 # place by a start anchor, so that the accent can be put over the letter
        lx = lx - (GREEK_W + SUB_W * len(sub)) / 2
    out.append(sub_text('open-edge-label', '%.1f' % lx, ly, letter, sub, 15, anchor='start', italic=True))
    out.append(accent(mark, lx + GREEK_W / 2 + 1.5, ly))          # the italic letter leans to the right
    out.append("</g>")
    # notes under the network
    for k, (y, words) in enumerate(spec.get('notes', [])):
        out.append(text('note-%d' % k, 196, y, words, size=11.5, fill='quiet'))
    # shape column
    out.append("<g data-claude-anchor='shape-column'>")
    out.append(text('shape-head', 584, 82, 'shape', size=11.5, fill='quiet'))
    out.append(text('shape', 584, 102, spec['shape'], size=12.5, weight=600))
    for k, words in enumerate(spec['shape_notes']):
        out.append(text('shape-note-%d' % k, 584, 124 + 16 * k, words, size=11.5, fill='quiet'))
    out.append("</g></svg>; };")
    code = ''.join(out)
    assert '"' not in code and '\n' not in code
    return code


leaf = lambda i, hot_w=False, hot_d=False, d='D': ('%s%d' % (d, i), 'w%d' % i, hot_w, hot_d)
right_of = lambda i: XS[i] + BW // 2
left_of = lambda i: XS[i] - BW // 2

SPECS = [
    dict(slug='mu', name='mu', side='left of a bond', basis='L basis',
         title='mu[2] is everything left of bond 2, through the L cores',
         sites={0: dict(tt='L0', leaf=leaf(0)), 1: dict(tt='L1', leaf=leaf(1))}, bonds=[(0, 1)],
         open=(right_of(1), YT, left_of(2), YT), open_label=('\u03bc', '2', None), open_label_at=((right_of(1) + left_of(2)) // 2, 76, 'middle'),
         shape='T + W + C + (rL2,)', shape_notes=['rL2: bond 2, L basis', 'a jet']),
    dict(slug='nu', name='nu', side='right of a bond', basis='R basis',
         title='nu[3] is everything right of bond 3, through the R cores',
         sites={3: dict(tt='R3', leaf=leaf(3)), 4: dict(tt='R4', leaf=leaf(4))}, bonds=[(3, 4)],
         open=(left_of(3), YT, right_of(2), YT), open_label=('\u03bd', '3', None), open_label_at=((left_of(3) + right_of(2)) // 2, 76, 'middle'),
         shape='T + W + C + (rR3,)', shape_notes=['rR3: bond 3, R basis', 'a jet']),
    dict(slug='nu-l', name='nu_L', side='right of a bond', basis='L basis',
         title='nu_L[3] is everything right of bond 3, through the L cores',
         sites={3: dict(tt='L3', leaf=leaf(3)), 4: dict(tt='L4', leaf=leaf(4))}, bonds=[(3, 4)],
         open=(left_of(3), YT, right_of(2), YT), open_label=('\u03bd', 'L,3', None), open_label_at=((left_of(3) + right_of(2)) // 2, 76, 'middle'),
         shape='T + W + C + (rL3,)', shape_notes=['rL3: bond 3, L basis', 'a jet']),
    dict(slug='xi', name='xi', side='leaf below a Tucker edge', basis='D basis',
         title='xi[2] is the leaf below Tucker edge 2: the input through D',
         sites={2: dict(tt=None, leaf=leaf(2))}, bonds=[],
         open=(XS[2], YD - DH // 2, XS[2], YT + BH // 2), open_label=('\u03be', '2', None), open_label_at=(XS[2] + 10, 128, 'start'),
         shape='A + W + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'affine: A has two slots']),
    dict(slug='eta', name='eta', side='trunk above a Tucker edge', basis='U basis',
         title='eta[2] is everything above Tucker edge 2, with U at site 2',
         sites={0: dict(tt='L0', leaf=leaf(0)), 1: dict(tt='L1', leaf=leaf(1)), 2: dict(tt='U2', ghost_leaf=True),
                3: dict(tt='R3', leaf=leaf(3)), 4: dict(tt='R4', leaf=leaf(4))}, bonds=[(0, 1), (1, 2), (2, 3), (3, 4)],
         open=(XS[2], YT + BH // 2, XS[2], YD - DH // 2), open_label=('\u03b7', '2', None), open_label_at=(XS[2] + 10, 128, 'start'),
         shape='T + W + C + (nU2,)', shape_notes=['nU2: Tucker edge 2 in U', 'a jet']),
    dict(slug='mu-r-dot', name='mu_R_dot', side='left of a bond', basis='R basis', H=288,
         title='mu_R_dot[3] is the left of bond 3 with the direction at one site',
         sites={0: dict(tt='L0', leaf=leaf(0)), 1: dict(tt='H1', kind='hot', leaf=leaf(1)), 2: dict(tt='R2', leaf=leaf(2))},
         bonds=[(0, 1), (1, 2)],
         open=(right_of(2), YT, left_of(3), YT), open_label=('\u03bc', 'R,3', 'dot'), open_label_at=((right_of(2) + left_of(3)) // 2, 76, 'middle'),
         notes=[(238, 'One term of a sum over where the direction sits:'),
                (254, 'site 0, 1 or 2, as H, or as U with V in place of D.'),
                (270, 'Cores before it are L, cores after it are R.')],
         shape='T + W + K + C + (rR3,)', shape_notes=['rR3: bond 3, R basis', 'a jet', 'K: one per direction']),
    dict(slug='mu-tilde', name='mu_tilde', side='left of a bond', basis='L basis', H=272,
         title='mu_tilde[2] is mu[2] with one input replaced by the seed',
         sites={0: dict(tt='L0', leaf=('D0', 'w0', True, False)), 1: dict(tt='L1', leaf=leaf(1))}, bonds=[(0, 1)],
         open=(right_of(1), YT, left_of(2), YT), open_label=('\u03bc', '2', 'tilde'), open_label_at=((right_of(1) + left_of(2)) // 2, 76, 'middle'),
         notes=[(238, 'One term of a sum over which input is the seed.'),
                (254, 'The seed is drawn as w with a tilde.')],
         shape='T + W + C + (rL2,)', shape_notes=['rL2: bond 2, L basis', 'dual jet, order turned']),
]

# ---- a local preview: the same module as a plain SVG file, and a PNG if rsvg-convert is installed -------
COLOURS = dict(edge='#8a8a86', grid='#d9d8d3', accent='#2f6fd0', ink='#1f1f1e', quiet='#6b6a66')
KEBAB = dict(textAnchor='text-anchor', fontSize='font-size', fontWeight='font-weight', fontStyle='font-style',
             strokeWidth='stroke-width', strokeDasharray='stroke-dasharray', strokeOpacity='stroke-opacity',
             fillOpacity='fill-opacity', markerEnd='marker-end')


def preview_svg(code):
    """Translate a module into a standalone SVG.  Only for looking at a drawing locally; the doc renders the module."""
    import re
    assert code.startswith(HEAD) and code.endswith('; };')
    body = code[len(HEAD):-len('; };')]
    assert body.startswith('return ')
    body = body[len('return '):]
    c = COLOURS

    def core(m):
        cx, cy, w, h, kind = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
        hot, ghost = kind == 'hot', kind == 'ghost'
        return ("<rect x='%g' y='%g' width='%d' height='%d' rx='6' fill='%s' fill-opacity='%s' stroke='%s' stroke-width='%s'%s/>"
                % (cx - w / 2, cy - h / 2, w, h, c['accent'] if hot else 'none', 0.12 if hot else 1, c['accent'] if hot else c['edge'],
                   2 if hot else 1.25, " stroke-dasharray='4 4' stroke-opacity='0.55'" if ghost else ''))

    def dot(m):
        hot = m[3] == 'true'
        return ("<circle cx='%s' cy='%s' r='13' fill='%s' fill-opacity='%s' stroke='%s' stroke-width='%s'/>"
                % (m[1], m[2], c['accent'] if hot else 'none', 0.12 if hot else 1, c['accent'] if hot else c['edge'], 2 if hot else 1.25))

    body = re.sub(r"\{core\((\d+), (\d+), (\d+), (\d+), '(\w+)'\)\}", core, body)
    body = re.sub(r"\{dot\((\d+), (\d+), (true|false)\)\}", dot, body)
    body = re.sub(r"\{wire\((\d+), (\d+), (\d+), (\d+)\)\}",
                  lambda m: "<line x1='%s' y1='%s' x2='%s' y2='%s' stroke='%s' stroke-width='1.25'/>" % (m[1], m[2], m[3], m[4], c['edge']), body)
    for name, colour in c.items():
        body = body.replace('={%s}' % name, "='%s'" % colour)
    for camel, kebab in KEBAB.items():
        body = body.replace(' %s=' % camel, ' %s=' % kebab)
    assert '{' not in body, body[body.index('{') - 60:body.index('{') + 60]       # a helper this translator does not know
    view = re.search(r"viewBox='0 0 (\d+) (\d+)'", body)
    return body.replace("<svg ", "<svg xmlns='http://www.w3.org/2000/svg' font-family='Helvetica, Arial, sans-serif' ", 1).replace(
        '>', "><rect width='%s' height='%s' fill='white'/>" % (view[1], view[2]), 1)


if __name__ == '__main__':
    mods = {s['slug']: module(s) for s in SPECS}
    json.dump(mods, open(sys.argv[1], 'w'), ensure_ascii=False)
    for k, v in mods.items():
        print(k, len(v))
    if len(sys.argv) > 2:                      # a folder for the previews
        import os, shutil, subprocess
        os.makedirs(sys.argv[2], exist_ok=True)
        for k, v in mods.items():
            path = os.path.join(sys.argv[2], k + '.svg')
            open(path, 'w').write(preview_svg(v))
            if shutil.which('rsvg-convert'):
                subprocess.run(['rsvg-convert', '-z', '1.5', path, '-o', path[:-4] + '.png'], check=True)
