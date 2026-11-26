"""K0's per-test count comparison of two `pytest -v` logs (xdist or not).

    python tally.py BASELINE_LOG NEW_LOG

For every test node: its outcome and its number of passed (and failed) subtests. Prints the totals of each log and
every node whose counts differ, or that is in one log only (the tests K0 renames are read under their new names).
After K0 the differences are the deleted tests (in the baseline only), the new tests, and the subtests of the
tests K0 edits; each one is accounted for in K0_HANDOVER.md.
"""
import collections
import re
import sys

LINE = re.compile(r'^(?:\[gw\d+\] \[\s*\d+%\] )?(SUBPASSED|SUBFAILED|SUBSKIPPED|PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)'
                  r'(?:\[.*?\])?(?: ?\(.*?\))? (tests/\S+)')
LINE_NOXDIST = re.compile(r'^(tests/\S+) (?:\(.*?\) |\[.*?\] )?(SUBPASSED|SUBFAILED|SUBSKIPPED|PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)')


def tally(path):
    per = collections.defaultdict(collections.Counter)
    for line in open(path, errors='replace'):
        m = LINE.match(line)
        if m:
            per[m.group(2)][m.group(1)] += 1
            continue
        m = LINE_NOXDIST.match(line)
        if m:
            per[m.group(1)][m.group(2)] += 1
    return per


def totals(per):
    t = collections.Counter()
    for c in per.values():
        t.update(c)
    return dict(sorted(t.items()))


RENAMED = {  # K0's renames (the design note, section 4): old name -> new name
    'test_jit_probe_derivatives': 'test_jit_the_interpreter_on_multi_group_strings',
    'test_uniform_probe_derivatives_two_hole_is_the_jvp_of_its_transpose':
        'test_uniform_probe_two_hole_at_jets_is_the_jvp_of_its_transpose'}


def renamed(node):
    head, _, name = node.rpartition('::')
    return head + '::' + RENAMED.get(name, name)


a, b = tally(sys.argv[1]), tally(sys.argv[2])
a = {renamed(node): c for node, c in a.items()}      # the tests K0 renames, under their new names
print('baseline:', len(a), 'nodes', totals(a))
print('new:     ', len(b), 'nodes', totals(b))
diff = 0
for node in sorted(set(a) | set(b)):
    if a.get(node) != b.get(node):
        diff += 1
        print('  %s\n      baseline %s\n      new      %s' % (node, dict(a.get(node, {})), dict(b.get(node, {}))))
print('%d nodes differ' % diff)
