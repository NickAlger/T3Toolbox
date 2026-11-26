"""M2's diff audit: every changed line of a converted (pre-existing) test file since 85c6e54d, classified.
A line is FLAGGED when, on its removed or added side, it touches an assertion or a tolerance; flagged lines are read."""
import re, subprocess, sys, collections
BASE = '85c6e54d'
names = subprocess.run(['git', 'diff', '--name-status', BASE, '--', 'tests/'], capture_output=True, text=True).stdout.split('\n')
files = [l.split('\t')[1] for l in names if l.startswith('M\t')]
new = [l.split('\t')[1] for l in names if l.startswith('A\t')]
ASSERT = re.compile(r'assert\w*\(|self\.fail|assertRaises|subTest')
TOL = re.compile(r'\b(rtol|atol|places|delta)\b|\b1e-\d+|allclose')
counts = collections.Counter(); flagged = []
for f in files:
    diff = subprocess.run(['git', 'diff', '-U0', BASE, '--', f], capture_output=True, text=True).stdout.split('\n')
    for line in diff:
        if not line or line[0] not in '+-' or line.startswith(('+++', '---')):
            continue
        body = line[1:]
        if ASSERT.search(body) or TOL.search(body):
            flagged.append((f, line)); counts[f, 'flagged'] += 1
        else:
            counts[f, 'form'] += 1
print('new files (additions, not conversions):', new)
for f in files:
    print('%-55s changed lines: %4d not touching an assertion or tolerance, %3d flagged' % (f, counts[f, 'form'], counts[f, 'flagged']))
print('\nFLAGGED LINES')
for f, l in flagged:
    print(f, '|', l[:230])
