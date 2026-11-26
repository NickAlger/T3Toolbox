"""K1 part 1: the scripted half of the dense reference's rewrite (the signatures and the seed's calls). Run once on
the tree of part 0; it refuses a file whose counts differ. The helpers, the three sums' bodies and the docstrings are
edited by hand. (The block weight's comment is the note's
§4.1 text with its last words on a continuation line, the house form, to keep the line within 120 columns.) Each replacement is a whole line (or a fixed call prefix), so no other line can move."""
import sys

PATH = 't3toolbox/backend/sampling_definitions.py'
OLD_W = ('        weight:           typ.Optional[typ.Any] = None,      # of the type of z, each axis 1 or z\'s size; '
         'None = ones\n')
NEW_W = ('        block_weight:     typ.Optional[typ.Any] = None,      # one per block: probe (d,)+F, apply F, entries ();\n'
         '                                                             #   None = ones\n'
         '        full_weight:      typ.Optional[typ.Any] = None,      # z\'s shape, each W axis 1 or W\'s size; '
         'None = ones\n')
OLD_Y = '        y:                typ.Optional[typ.Any] = None,      # the data, of the type of z\n'
NEW_Y = '        y:                typ.Optional[typ.Any] = None,      # the data, z\'s shape exactly: probe len=d, F+W+(Ni,)\n'
OLD_SEED, NEW_SEED = '_the_seed(S, sample, y, weight, w_tilde, ', '_the_seed(S, sample, y, block_weight, full_weight, w_tilde, '

lines = open(PATH).readlines()
counts = {'weight': sum(l == OLD_W for l in lines), 'y': sum(l == OLD_Y for l in lines),
          'seed': sum(l.count(OLD_SEED) for l in lines)}
if counts != {'weight': 37, 'y': 37, 'seed': 17}:
    sys.exit('refused: counts %s' % counts)
out = []
for l in lines:
    if l == OLD_W:
        out.append(NEW_W)
    elif l == OLD_Y:
        out.append(NEW_Y)
    else:
        out.append(l.replace(OLD_SEED, NEW_SEED))
open(PATH, 'w').write(''.join(out))
print('rewritten: %s' % counts)
