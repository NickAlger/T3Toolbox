"""K1's per-test count comparison (from k0/tally.py) of two `pytest -v` logs (xdist or not).

    python tally.py BASELINE_LOG NEW_LOG

For every test node: its outcome and its number of passed (and failed) subtests. Prints the totals of each log and
every node whose counts differ, or that is in one log only (the tests K1 renames are read under their new names).
After a part the differences are the renamed tests (read under their new names), the new tests, and the
subtests of the tests the part edits; each one is accounted for in the part's handover.
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


RENAMED = {  # K1 part 0's renames: old node name -> new
    'TestTheBatch::test_check_batch': 'TestTheBatch::test_check',
    'TestFrontendOnlyAndUserOracles::test_a_user_oracle_without_a_draw':
        'TestFrontendOnlyAndUserOracles::test_a_user_batch_without_a_draw',
    'TestTheOracle::test_order_is_required_and_entries_has_no_jets':
        'TestTheBatchConstructor::test_the_kind_and_the_order'}


def renamed(node):
    for old, new in RENAMED.items():
        if node.endswith('::' + old) or ('::' + old + '[') in node:
            return node.replace('::' + old, '::' + new)
    return node


a, b = tally(sys.argv[1]), tally(sys.argv[2])
a = {renamed(node): c for node, c in a.items()}      # the tests the part renames, under their new names

print('baseline:', len(a), 'nodes', totals(a))
print('new:     ', len(b), 'nodes', totals(b))
diff = 0
for node in sorted(set(a) | set(b)):
    if a.get(node) != b.get(node):
        diff += 1
        print('  %s\n      baseline %s\n      new      %s' % (node, dict(a.get(node, {})), dict(b.get(node, {}))))
print('%d nodes differ' % diff)
