"""M0's residual scan: every word with a capital J left in t3toolbox/ and tests/ (or any ROOT), by class.

    python residual.py ROOT

Words of the known non-group families (the Jacobian `Jp`, `JtJp`, `Jacobian`, `jax`, `Jet` words, ...) are counted;
every other word with a `J` -- the bare `J` above all -- is printed with its line, for reading by eye. After M0 none of
them may be the jet group.
"""
import collections
import io
import pathlib
import re
import sys
import tokenize

KNOWN = re.compile(r'^(_?Jp\w*|\w*_Jp\w*|JtJp\w*|\w*JtJp\w*|Jacobians?|\w*JAX\w*|Jax\w*|Jit\w*|Jitting|_?Jet\w*|'
                   r'\w*JET\w*|Test\w*|test_\w*|Joint|Junk|JVP|ADJOINT|OBJECT|REJECT\w*|_JP_PATHS|_HELD_JP|'
                   r'_\w*_JP_ROWS|_JAX_WARNED|_HAVE_JAX|HAS_JAX|NO_JETS|WITH_JETS|IJKLMNOPQRSTUVWXY|'
                   r'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ|g_of_Jp|paired_with_Jp|held_Jp)$')
root = pathlib.Path(sys.argv[1])
known, other = collections.Counter(), collections.defaultdict(list)
for path in sorted(list((root / 't3toolbox').rglob('*.py')) + list((root / 'tests').rglob('*.py'))):
    src = path.read_text()
    lines = src.splitlines()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.COMMENT) or 'J' not in tok.string:
            continue
        for m in re.finditer(r'[A-Za-z0-9_]*J[A-Za-z0-9_]*', tok.string):
            w = m.group(0)
            ln = tok.start[0] + tok.string[:m.start()].count('\n')
            if KNOWN.match(w):
                known[w] += 1
            else:
                other[w].append('%s:%d: %s' % (path.relative_to(root), ln, lines[ln - 1].strip()[:140]))
print('known non-group words: %d uses of %d words' % (sum(known.values()), len(known)))
for w in sorted(other, key=lambda w: (-len(other[w]), w)):
    print('==== %r: %d' % (w, len(other[w])))
    for site in other[w]:
        print('   ', site)
