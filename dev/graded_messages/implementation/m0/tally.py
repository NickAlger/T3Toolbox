"""M0's per-test count comparison of two `pytest -v` logs (xdist or not).

    python tally.py BASELINE_LOG NEW_LOG

For every test node: its outcome and its number of passed (and failed) subtests. Prints the totals of each log and
every node whose counts differ, or that is in one log only (the two tests M0 renames are read under their new names).
After a pure rename the only differences are new tests and new subtests of the tests M0 extends.
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


RENAMED = {'test_every_J_literal_is_a_jet_contract_literal': 'test_every_P_literal_is_a_jet_contract_literal',
           'test_counts_are_nonnegative_ints_and_J_any_count': 'test_counts_are_nonnegative_ints_and_P_any_count'}


def renamed(node):
    head, _, name = node.rpartition('::')
    return head + '::' + RENAMED.get(name, name)


a, b = tally(sys.argv[1]), tally(sys.argv[2])
a = {renamed(node): c for node, c in a.items()}      # the two tests M0 renames, under their new names
print('baseline:', len(a), 'nodes', totals(a))
print('new:     ', len(b), 'nodes', totals(b))
diff = 0
for node in sorted(set(a) | set(b)):
    if a.get(node) != b.get(node):
        diff += 1
        print('  %s\n      baseline %s\n      new      %s' % (node, dict(a.get(node, {})), dict(b.get(node, {}))))
print('%d nodes differ' % diff)
