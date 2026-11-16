"""Generates the drawings of the "Variable diagrams" tab of the Claude doc.

The maintainer asked (2026-11-15, 23:02) for a table of tensor network diagrams, one row per sampling variable:
the name, the subnetwork that is contracted at one representative edge in the middle of the network, and the
shape. Each row is one drawing (a widget module: one line of JSX that returns an SVG). This script writes the
modules as a JSON map from a slug to its module; they are published to the doc from that file. The names and the
definitions are those of SAMPLE_VARIABLES.md (the notation decided on 2026-11-15, 23:23).

28 rows. The first seven (mu, nu, nu_L, xi, eta, mu_R_dot, mu_tilde) were the prototype shown to the maintainer.
The other twenty-one (the remaining fifteen variables of SAMPLE_VARIABLES.md section 3, and the outputs and
blocks of its section 6) were drawn on 2026-11-16 by an agent that was given only that page, as a test of the
page, and checked against it (see the comment above the second SPECS list).

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
    if kind == 'tilde_dot':                            # both: the tilde next to the letter, the dot above it
        return accent('tilde', cx, y) + "<circle cx='%.1f' cy='%.1f' r='1.5' fill={ink}/>" % (cx, y - 17.5)
    assert kind is None
    return ''


HOLE_DH = 32                          # a Tucker core taken out is drawn a little taller than a Tucker core, to leave room for two accents
CORE_W, CORE_SUB_W = 9.6, 5.8        # rough widths of a core letter at 14 bold and of its subscript at 10.5


def core_label(tid, x, y, label, mark=None, size=14):
    """A core's label (letter and site subscript, centred on x), with a dot or a tilde above the letter if marked."""
    out = sub_text(tid, x, y, label[0], label[1:], size)
    if mark:
        cx = x - (CORE_W + CORE_SUB_W * len(label[1:])) / 2 + CORE_W / 2
        if mark in ('tilde', 'tilde_dot'):             # an upright capital is taller than a Greek letter: lift the tilde
            out += accent('tilde', cx, y - 1.5)
            if mark == 'tilde_dot':
                out += "<circle cx='%.1f' cy='%.1f' r='1.5' fill={ink}/>" % (cx, y - 18)
        else:
            out += accent(mark, cx, y)
    return out


def hole(tid, cx, cy, w, h, label, mark):
    """A core taken out of the network: a dashed outline in the accent colour, named by the block that stands for it."""
    return ("<rect x='%g' y='%g' width='%d' height='%d' rx='6' fill='none' stroke={accent} strokeWidth='1.5' strokeDasharray='4 3'/>"
            % (cx - w / 2, cy - h / 2, w, h)) + core_label(tid, cx, cy + (7 if mark == 'tilde_dot' else 5), label, mark)


def ghost_input(cx, cy):
    """Where an input would be, on a mode edge that is left open."""
    return "<circle cx='%d' cy='%d' r='13' fill='none' stroke={edge} strokeWidth='1.25' strokeDasharray='4 4' strokeOpacity='0.55'/>" % (cx, cy)


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
        if s is not None and s.get('tt_hole'):         # the TT core of this site is taken out: (label, mark)
            out.append(hole('tt-hole-%d' % i, x, YT, BW, BH, *s['tt_hole']))
        elif s is None or s.get('tt') is None:
            out.append("{core(%d, %d, %d, %d, 'ghost')}" % (x, YT, BW, BH))
        else:
            out.append("{core(%d, %d, %d, %d, '%s')}" % (x, YT, BW, BH, s.get('kind', 'plain')))
            if s.get('tt_mark'):                       # a dotted TT core: the raw direction L_dot
                out.append(core_label('tt-%d' % i, x, YT + 5, s['tt'], s['tt_mark']))
            else:
                out.append(sub_text('tt-%d' % i, x, YT + 5, s['tt'][0], s['tt'][1:], 14))
        if s is not None and s.get('leaf'):
            dl, wl, hot_w, hot_d = s['leaf']
            if s.get('tt') is not None:
                out.append("{wire(%d, %d, %d, %d)}" % (x, YT + BH // 2, x, YD - DH // 2))
            out.append("{core(%d, %d, %d, %d, '%s')}" % (x, YD, DW, DH, 'hot' if hot_d else 'plain'))
            if s.get('leaf_mark'):                     # a dotted Tucker core: the raw direction D_dot
                out.append(core_label('tucker-%d' % i, x, YD + 5, dl, s['leaf_mark']))
            else:
                out.append(sub_text('tucker-%d' % i, x, YD + 5, dl[0], dl[1:], 14))
            if s.get('ghost_input'):                   # mode edge left open: the open edge is drawn instead of the wire
                out.append(ghost_input(x, YW))
            else:
                out.append("{wire(%d, %d, %d, %d)}" % (x, YD + DH // 2, x, YW - RW))
                out.append("{dot(%d, %d, %s)}" % (x, YW, 'true' if hot_w else 'false'))
                out.append(sub_text('input-%d' % i, x, YW + 3, wl[0], wl[1:], 12.5, weight=500, italic=True))
                if hot_w:                              # the seed: w with a tilde
                    out.append("<path d='M%.1f %.1f q1.6 -2.2 3.2 0 t3.2 0' fill='none' stroke={ink} strokeWidth='1.2'/>" % (x - 6.2, YW - 7.2))
        elif s is not None and s.get('ghost_leaf'):
            out.append("{core(%d, %d, %d, %d, 'ghost')}" % (x, YD, DW, DH))
        elif s is not None and s.get('hole_leaf'):     # the Tucker core of this site is taken out: (label, mark, input, seed?)
            hl, hm, wl, hot_w = s['hole_leaf']
            out.append(hole('tucker-hole-%d' % i, x, YD, DW, HOLE_DH, hl, hm))
            out.append("{dot(%d, %d, %s)}" % (x, YW, 'true' if hot_w else 'false'))
            out.append(sub_text('input-%d' % i, x, YW + 3, wl[0], wl[1:], 12.5, weight=500, italic=True))
            if hot_w:
                out.append("<path d='M%.1f %.1f q1.6 -2.2 3.2 0 t3.2 0' fill='none' stroke={ink} strokeWidth='1.2'/>" % (x - 6.2, YW - 7.2))
    for (i, j) in spec.get('bonds', []):       # contracted bonds between sites i and j = i + 1
        out.append("{wire(%d, %d, %d, %d)}" % (XS[i] + BW // 2, YT, XS[j] - BW // 2, YT))
    out.append("</g>")
    # the open edge
    if 'opens' in spec:                                # a block: several open legs, all pointing into the hole, which carries the name
        out.append("<g data-claude-anchor='open-edges'>")
        for (x1, y1, x2, y2) in spec['opens']:
            out.append("<line x1='%d' y1='%d' x2='%d' y2='%d' stroke={accent} strokeWidth='2' markerEnd='url(#%s-arrow)'/>" % (x1, y1, x2, y2, slug))
        out.append("</g>")
        return finish(spec, out)
    x1, y1, x2, y2 = spec['open']
    if spec['open_label'][2] == 'tilde_dot' and spec['open_label_at'][2] == 'start':
        lx, ly, la = spec['open_label_at']             # beside a vertical edge: lower, so that the dot clears the box above
        spec = dict(spec, open_label_at=(lx, ly + 5, la))
    out.append("<g data-claude-anchor='open-edge'><line x1='%d' y1='%d' x2='%d' y2='%d' stroke={accent} strokeWidth='2' markerEnd='url(#%s-arrow)'/>" % (x1, y1, x2, y2, slug))
    lx, ly, la = spec['open_label_at']
    letter, sub, mark = spec['open_label']
    if la == 'middle':                                 # place by a start anchor, so that the accent can be put over the letter
        lx = lx - (GREEK_W + SUB_W * len(sub)) / 2
    out.append(sub_text('open-edge-label', '%.1f' % lx, ly, letter, sub, 15, anchor='start', italic=True))
    out.append(accent(mark, lx + GREEK_W / 2 + 1.5, ly))          # the italic letter leans to the right
    out.append("</g>")
    return finish(spec, out)


def finish(spec, out):
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
    dict(slug='mu-tilde', name='mu_tilde', side='left of a bond', basis='L basis', H=288,
         title='mu_tilde[2] is one array: the sum over the inputs of mu[2], each replaced in turn',
         sites={0: dict(tt='L0', leaf=('D0', 'w0', True, False)), 1: dict(tt='L1', leaf=leaf(1))}, bonds=[(0, 1)],
         open=(right_of(1), YT, left_of(2), YT), open_label=('\u03bc', '2', 'tilde'), open_label_at=((right_of(1) + left_of(2)) // 2, 76, 'middle'),
         notes=[(238, 'One term of the sum is drawn: here input 0 is replaced, by'),
                (254, 'the residual-like vector of mode 0 (w with a tilde).'),
                (270, 'mu_tilde[2] combines the residuals of modes 0 and 1.')],
         shape='T + W + C + (rL2,)', shape_notes=['rL2: bond 2, L basis', 'dual jet, order turned']),
]

# ---- the other fifteen variables, and the outputs and blocks (added 2026-11-16, from SAMPLE_VARIABLES.md alone) ----
# Devices added for these rows (none changes the seven rows above):
#   tt_mark / leaf_mark = 'dot'   a dot over a core's letter: the raw direction L_dot or D_dot (a hot core, like H and V)
#   ghost_input                   a dashed circle where the input of an open mode edge would be (z, Jp)
#   tt_hole / hole_leaf           a core taken out (a block): a dashed outline in the accent colour, named by the block;
#                                 'opens' then lists the block's legs, every arrow pointing into the hole
#   accent 'tilde_dot'            a tilde next to the letter and a dot above it


def notes_rows(*lines):
    """The notes under the network, one every 16 units from y = 238, and the height of the drawing that fits them."""
    return dict(notes=[(238 + 16 * k, words) for k, words in enumerate(lines)], H=256 + 16 * (len(lines) - 1))


def open_left_to_right(k):                 # bond k, travelling left-to-right (a mu)
    return dict(open=(right_of(k - 1), YT, left_of(k), YT), open_label_at=((right_of(k - 1) + left_of(k)) // 2, 76, 'middle'))


def open_right_to_left(k):                  # bond k, travelling right-to-left (a nu)
    return dict(open=(left_of(k), YT, right_of(k - 1), YT), open_label_at=((left_of(k) + right_of(k - 1)) // 2, 76, 'middle'))


def open_bottom_to_top(i):                    # Tucker edge i, bottom-to-top (a xi)
    return dict(open=(XS[i], YD - DH // 2, XS[i], YT + BH // 2), open_label_at=(XS[i] + 10, 128, 'start'))


def open_top_to_bottom(i):                  # Tucker edge i, top-to-bottom (an eta)
    return dict(open=(XS[i], YT + BH // 2, XS[i], YD - DH // 2), open_label_at=(XS[i] + 10, 128, 'start'))


def open_mode(i):                      # mode edge i, top-to-bottom (an output)
    return dict(open=(XS[i], YD + DH // 2, XS[i], YW - RW), open_label_at=(XS[i] + 10, 186, 'start'))


plain = lambda tt, i: dict(tt='%s%d' % (tt, i), leaf=leaf(i))                       # a TT core over its D core and input
seeded = lambda tt, i: dict(tt='%s%d' % (tt, i), leaf=leaf(i, hot_w=True))          # the same with the seed as input
l_dot = lambda i: dict(tt='L%d' % i, kind='hot', tt_mark='dot', leaf=leaf(i))       # raw direction L_dot over D
d_dot = lambda i, hot_w=False: dict(tt='L%d' % i, leaf=leaf(i, hot_w=hot_w, hot_d=True, d='D'), leaf_mark='dot')   # L over D_dot
v_dir = lambda i: dict(tt='U%d' % i, leaf=leaf(i, hot_d=True, d='V'))               # frame direction: U over V
ALL_BONDS = [(0, 1), (1, 2), (2, 3), (3, 4)]

SPECS += [
    # -- the left of a bond --
    dict(slug='mu-dot', name='mu_dot', side='left of a bond', basis='L basis',
         title='mu_dot[3] is the left of bond 3 with the raw direction at one site, through L cores',
         sites={0: plain('L', 0), 1: d_dot(1), 2: plain('L', 2)}, bonds=[(0, 1), (1, 2)],
         open_label=('μ', '3', 'dot'), **open_left_to_right(3),
         **notes_rows('One term of a sum over where the direction sits: site 0,',
                      '1 or 2, as L_dot over D, or as L with D_dot in place of D.',
                      'Raw form only: every other TT core is L.',
                      'The same side in the R basis, frame form: mu_R_dot.'),
         shape='T + W + K + C + (rL3,)', shape_notes=['rL3: bond 3, L basis', 'a jet', 'K: one per direction']),
    dict(slug='mu-tilde-dot', name='mu_tilde_dot', side='left of a bond', basis='L basis',
         title='mu_tilde_dot[3] is the left of bond 3 with the raw direction at one site and the seed',
         sites={0: seeded('L', 0), 1: l_dot(1), 2: plain('L', 2)}, bonds=[(0, 1), (1, 2)],
         open_label=('μ', '3', 'tilde_dot'), **open_left_to_right(3),
         **notes_rows('One term of a sum over where the direction sits (site 0,',
                      '1 or 2) and over which input is the seed (0, 1 or 2).',
                      'Both may be at one site. Raw form: L_dot over D, or L',
                      'with D_dot in place of D; every other TT core is L.'),
         shape='T + W + K + C + (rL3,)', shape_notes=['rL3: bond 3, L basis', 'dual jet, order turned', 'K: one per direction']),
    # -- the right of a bond --
    dict(slug='nu-tilde', name='nu_tilde', side='right of a bond', basis='R basis',
         title='nu_tilde[2] is one array: the sum over the inputs of nu[2], each replaced in turn',
         sites={2: plain('R', 2), 3: plain('R', 3), 4: seeded('R', 4)}, bonds=[(2, 3), (3, 4)],
         open_label=('ν', '2', 'tilde'), **open_right_to_left(2),
         **notes_rows('One term of the sum is drawn: here input 4 is replaced, by',
                      'the residual-like vector of mode 4 (w with a tilde).',
                      'nu_tilde[2] combines the residuals of modes 2, 3 and 4.'),
         shape='T + W + C + (rR2,)', shape_notes=['rR2: bond 2, R basis', 'dual jet, order turned']),
    dict(slug='nu-l-dot', name='nu_L_dot', side='right of a bond', basis='L basis',
         title='nu_L_dot[2] is the right of bond 2 with the direction at one site',
         sites={2: plain('L', 2), 3: dict(tt='H3', kind='hot', leaf=leaf(3)), 4: plain('R', 4)}, bonds=[(2, 3), (3, 4)],
         open_label=('ν', 'L,2', 'dot'), **open_right_to_left(2),
         **notes_rows('One term of a sum over where the direction sits:',
                      'site 2, 3 or 4, as H, or as U with V in place of D.',
                      'Cores before it are L, cores after it are R. Drawn in',
                      'the frame form; the raw form (all other cores L) gives',
                      'the same arrays for the lifted direction, with no damping.'),
         shape='T + W + K + C + (rL2,)', shape_notes=['rL2: bond 2, L basis', 'a jet', 'K: one per direction']),
    dict(slug='nu-l-tilde', name='nu_L_tilde', side='right of a bond', basis='L basis',
         title='nu_L_tilde[2] is one array: the sum over the inputs of nu_L[2], each replaced in turn',
         sites={2: plain('L', 2), 3: seeded('L', 3), 4: plain('L', 4)}, bonds=[(2, 3), (3, 4)],
         open_label=('ν', 'L,2', 'tilde'), **open_right_to_left(2),
         **notes_rows('One term of the sum is drawn: here input 3 is replaced, by',
                      'the residual-like vector of mode 3 (w with a tilde).',
                      'nu_L_tilde[2] combines the residuals of modes 2, 3 and 4.'),
         shape='T + W + C + (rL2,)', shape_notes=['rL2: bond 2, L basis', 'dual jet, order turned']),
    dict(slug='nu-l-tilde-dot', name='nu_L_tilde_dot', side='right of a bond', basis='L basis',
         title='nu_L_tilde_dot[2] is the right of bond 2 with the raw direction at one site and the seed',
         sites={2: d_dot(2), 3: seeded('L', 3), 4: plain('L', 4)}, bonds=[(2, 3), (3, 4)],
         open_label=('ν', 'L,2', 'tilde_dot'), **open_right_to_left(2),
         **notes_rows('One term of a sum over where the direction sits (site 2,',
                      '3 or 4) and over which input is the seed (2, 3 or 4).',
                      'Both may be at one site. Raw form: L_dot over D, or L',
                      'with D_dot in place of D; every other TT core is L.'),
         shape='T + W + K + C + (rL2,)', shape_notes=['rL2: bond 2, L basis', 'dual jet, order turned', 'K: one per direction']),
    # -- the leaf below a Tucker edge --
    dict(slug='xi-dot', name='xi_dot', side='leaf below a Tucker edge', basis='D basis',
         title='xi_dot[2] is the leaf below Tucker edge 2 with the raw direction D_dot in place of D',
         sites={2: dict(tt=None, leaf=leaf(2, hot_d=True, d='D'), leaf_mark='dot')}, bonds=[],
         open_label=('ξ', '2', 'dot'), **open_bottom_to_top(2),
         **notes_rows('Raw form only: D_dot has the shape of D.',
                      'The same side in the U basis, frame form: xi_U_dot.'),
         shape='A + W + K + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'affine: A has two slots', 'K: one per direction']),
    dict(slug='xi-tilde', name='xi_tilde', side='leaf below a Tucker edge', basis='D basis',
         title='xi_tilde[2] is the leaf below Tucker edge 2 with the residual-like vector as its input',
         sites={2: dict(tt=None, leaf=leaf(2, hot_w=True))}, bonds=[],
         open_label=('ξ', '2', 'tilde'), **open_bottom_to_top(2),
         **notes_rows('A single term: the leaf has one input. w with a tilde is the',
                      'residual-like vector of mode 2 (the ztilde of the library).'),
         shape='T + W + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'dual jet, order turned', 'not affine: T, not A']),
    dict(slug='xi-tilde-dot', name='xi_tilde_dot', side='leaf below a Tucker edge', basis='D basis',
         title='xi_tilde_dot[2] is the raw direction D_dot applied to the seed',
         sites={2: dict(tt=None, leaf=leaf(2, hot_w=True, hot_d=True, d='D'), leaf_mark='dot')}, bonds=[],
         open_label=('ξ', '2', 'tilde_dot'), **open_bottom_to_top(2),
         **notes_rows('A single term. Raw form only.'),
         shape='T + W + K + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'dual jet, order turned', 'K: one per direction']),
    dict(slug='xi-u-dot', name='xi_U_dot', side='leaf below a Tucker edge', basis='U basis',
         title='xi_U_dot[2] is the leaf below Tucker edge 2 with the direction V in place of D',
         sites={2: dict(tt=None, leaf=leaf(2, hot_d=True, d='V'))}, bonds=[],
         open_label=('ξ', 'U,2', 'dot'), **open_bottom_to_top(2),
         **notes_rows('Frame form only: V has its Tucker edge in the U basis.',
                      'The same side in the D basis, raw form: xi_dot.'),
         shape='A + W + K + C + (nU2,)', shape_notes=['nU2: Tucker edge 2 in U', 'affine: A has two slots', 'K: one per direction']),
    # -- the trunk above a Tucker edge --
    dict(slug='eta-d', name='eta_D', side='trunk above a Tucker edge', basis='D basis',
         title='eta_D[2] is everything above Tucker edge 2, with every TT core an L core',
         sites={0: plain('L', 0), 1: plain('L', 1), 2: dict(tt='L2', ghost_leaf=True), 3: plain('L', 3), 4: plain('L', 4)},
         bonds=ALL_BONDS, open_label=('η', 'D,2', None), **open_top_to_bottom(2),
         shape='T + W + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'a jet']),
    dict(slug='eta-d-dot', name='eta_D_dot', side='trunk above a Tucker edge', basis='D basis',
         title='eta_D_dot[2] is the trunk above Tucker edge 2 with the direction at one site',
         sites={0: plain('L', 0), 1: v_dir(1), 2: dict(tt='R2', ghost_leaf=True), 3: plain('R', 3), 4: plain('R', 4)},
         bonds=ALL_BONDS, open_label=('η', 'D,2', 'dot'), **open_top_to_bottom(2),
         **notes_rows('One term of a sum over where the direction sits: any site,',
                      'as H, or as U with V in place of D; at site 2 only as H2.',
                      'Cores before it are L, cores after it are R. Drawn in',
                      'the frame form; the raw form (all other cores L) gives',
                      'the same arrays for the lifted direction, with no damping.'),
         shape='T + W + K + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'a jet', 'K: one per direction']),
    dict(slug='eta-tilde', name='eta_tilde', side='trunk above a Tucker edge', basis='U basis',
         title='eta_tilde[2] is one array: the sum over the inputs of eta[2], each replaced in turn',
         sites={0: seeded('L', 0), 1: plain('L', 1), 2: dict(tt='U2', ghost_leaf=True), 3: plain('R', 3), 4: plain('R', 4)},
         bonds=ALL_BONDS, open_label=('η', '2', 'tilde'), **open_top_to_bottom(2),
         **notes_rows('One term of the sum is drawn: here input 0 is replaced, by',
                      'the residual-like vector of mode 0 (w with a tilde).',
                      'eta_tilde[2] combines the residuals of every mode but 2.'),
         shape='T + W + C + (nU2,)', shape_notes=['nU2: Tucker edge 2 in U', 'dual jet, order turned']),
    dict(slug='eta-d-tilde', name='eta_D_tilde', side='trunk above a Tucker edge', basis='D basis',
         title='eta_D_tilde[2] is one array: the sum over the inputs of eta_D[2], each replaced in turn',
         sites={0: plain('L', 0), 1: plain('L', 1), 2: dict(tt='L2', ghost_leaf=True), 3: plain('L', 3), 4: seeded('L', 4)},
         bonds=ALL_BONDS, open_label=('η', 'D,2', 'tilde'), **open_top_to_bottom(2),
         **notes_rows('One term of the sum is drawn: here input 4 is replaced, by',
                      'the residual-like vector of mode 4 (w with a tilde).',
                      'eta_D_tilde[2] combines the residuals of every mode but 2.',
                      'Read only by the gradient on the corewise frame.'),
         shape='T + W + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'dual jet, order turned']),
    dict(slug='eta-d-tilde-dot', name='eta_D_tilde_dot', side='trunk above a Tucker edge', basis='D basis',
         title='eta_D_tilde_dot[2] is the trunk above Tucker edge 2 with the raw direction and the seed',
         sites={0: seeded('L', 0), 1: plain('L', 1), 2: dict(tt='L2', ghost_leaf=True), 3: d_dot(3), 4: plain('L', 4)},
         bonds=ALL_BONDS, open_label=('η', 'D,2', 'tilde_dot'), **open_top_to_bottom(2),
         **notes_rows('One term of a sum over where the direction sits (any site;',
                      'at site 2 only as L_dot2) and over which input is the',
                      'seed (any but input 2). Both may be at one site.',
                      'Raw form only: every other TT core is L.'),
         shape='T + W + K + C + (nD2,)', shape_notes=['nD2: Tucker edge 2 in D', 'dual jet, order turned', 'K: one per direction']),
    # -- outputs --
    dict(slug='z', name='z', side='output', basis='mode edge, no basis',
         title='z[2] is the whole network at the base point, with mode edge 2 open',
         sites={0: plain('L', 0), 1: plain('L', 1), 2: dict(tt='L2', leaf=leaf(2), ghost_input=True), 3: plain('L', 3), 4: plain('L', 4)},
         bonds=ALL_BONDS, open_label=('z', '2', None), **open_mode(2),
         **notes_rows('z[2] = z_term(eta_D[2], D[2]): the trunk eta_D[2]',
                      'through D2. The dashed circle is where input 2 would be.'),
         shape='T + W + C + (N2,)', shape_notes=['N2: mode edge 2', 'a covector']),
    dict(slug='jp', name='Jp', side='output', basis='mode edge, no basis',
         title='Jp[2] is the whole network with the direction at one site, with mode edge 2 open',
         sites={0: plain('L', 0), 1: plain('L', 1), 2: dict(tt='U2', leaf=leaf(2, hot_d=True, d='V'), ghost_input=True),
                3: plain('R', 3), 4: plain('R', 4)},
         bonds=ALL_BONDS, open_label=('Jp', '2', None), **open_mode(2),
         **notes_rows('One term of a sum over where the direction sits: any site,',
                      'as H, or as U with V in place of D. Here V2, the term',
                      'z_term(eta[2], V[2]); a direction elsewhere, or H2, is',
                      'z_term(eta_D_dot[2], D[2]). Frame form of the direction.'),
         shape='T + W + K + C + (N2,)', shape_notes=['N2: mode edge 2', 'a covector', 'K: one per direction']),
    # -- blocks: a core taken out, its legs open --
    dict(slug='v-tilde', name='V_tilde', side='block, shape of V', basis='legs: U basis, mode',
         title='V_tilde[2] is the whole network with the leaf core under U2 taken out, and the seed',
         sites={0: plain('L', 0), 1: plain('L', 1), 2: dict(tt='U2', hole_leaf=('V2', 'tilde', 'w2', False)),
                3: seeded('R', 3), 4: plain('R', 4)},
         bonds=ALL_BONDS, opens=[(XS[2], YT + BH // 2, XS[2], YD - HOLE_DH // 2), (XS[2], YW - RW, XS[2], YD + HOLE_DH // 2)],
         **notes_rows('One term of a sum over which input is the seed (any input,',
                      'input 2 too), summed over the samples; the top coefficient.',
                      'The arrows are the legs of the block: Tucker edge 2 in',
                      'the U basis, and mode edge 2. For JtJp the seed is Jp.'),
         shape='C + (nU2, N2)', shape_notes=['the shape of V[2]', 'no T, no W', 'JtJp: K in front']),
    dict(slug='h-tilde', name='H_tilde', side='block, shape of H', basis='legs: L, D, R bases',
         title='H_tilde[2] is the whole network with the TT core of site 2 taken out, and the seed',
         sites={0: plain('L', 0), 1: seeded('L', 1), 2: dict(tt_hole=('H2', 'tilde'), leaf=leaf(2)), 3: plain('R', 3), 4: plain('R', 4)},
         bonds=[(0, 1), (3, 4)],
         opens=[(right_of(1), YT, left_of(2), YT), (left_of(3), YT, right_of(2), YT), (XS[2], YD - DH // 2, XS[2], YT + BH // 2)],
         **notes_rows('One term of a sum over which input is the seed (any input),',
                      'summed over the samples; the top coefficient. The arrows',
                      'are the legs of the block: bond 2 in L, Tucker edge 2',
                      'in D, bond 3 in R. For JtJp the seed is Jp.'),
         shape='C + (rL2, nD2, rR3)', shape_notes=['the shape of H[2]', 'no T, no W', 'JtJp: K in front']),
    dict(slug='v-tilde-dot', name='V_tilde_dot', side='block, shape of D', basis='legs: D basis, mode',
         title='V_tilde_dot[2] is the network at the base point with D2 taken out, the direction and the seed',
         sites={0: plain('L', 0), 1: l_dot(1), 2: dict(tt='L2', hole_leaf=('V2', 'tilde_dot', 'w2', False)),
                3: seeded('L', 3), 4: plain('L', 4)},
         bonds=ALL_BONDS, opens=[(XS[2], YT + BH // 2, XS[2], YD - HOLE_DH // 2), (XS[2], YW - RW, XS[2], YD + HOLE_DH // 2)],
         **notes_rows('One term of a sum over where the raw direction sits (any',
                      'site; at site 2 only as L_dot2) and over which input is the',
                      'seed (any, input 2 too). Every other TT core is L.',
                      'Summed over the samples; the top coefficient.',
                      'Legs: Tucker edge 2 in the D basis, and mode edge 2.'),
         shape='K + C + (nD2, N2)', shape_notes=['the shape of D[2]', 'no T, no W', 'K: one per direction']),
    dict(slug='h-tilde-dot', name='H_tilde_dot', side='block, shape of L', basis='legs: L, D, L bases',
         title='H_tilde_dot[2] is the network at the base point with L2 taken out, the direction and the seed',
         sites={0: plain('L', 0), 1: plain('L', 1), 2: dict(tt_hole=('H2', 'tilde_dot'), leaf=leaf(2, hot_w=True)),
                3: plain('L', 3), 4: l_dot(4)},
         bonds=[(0, 1), (3, 4)],
         opens=[(right_of(1), YT, left_of(2), YT), (left_of(3), YT, right_of(2), YT), (XS[2], YD - DH // 2, XS[2], YT + BH // 2)],
         **notes_rows('One term of a sum over where the raw direction sits (any',
                      'site; at site 2 only as D_dot2) and over which input is the',
                      'seed (any). Every other TT core is L.',
                      'Summed over the samples; the top coefficient.',
                      'Legs: bond 2 in L, Tucker edge 2 in D, bond 3 in L.'),
         shape='K + C + (rL2, nD2, rL3)', shape_notes=['the shape of L[2]', 'no T, no W', 'K: one per direction']),
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
