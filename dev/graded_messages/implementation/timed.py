#!/usr/bin/env python3
"""Run a command and report its wall time and peak memory.   Usage: timed.py [--limit SECONDS] STATS_FILE -- command [args...]

This stands in for `/usr/bin/time -v`, which is not installed in the agents' workspace. It needs only the
standard library and Linux's /proc.

The command runs with this process's stdin, stdout and stderr (so the caller redirects them). Twice a second the
script adds up the resident memory (RSS) of the command and all its descendants (pytest-xdist workers, sphinx
workers, ...), and reads the system's MemAvailable. At the end it writes one line of `key=value` pairs to
STATS_FILE and exits with the command's exit code:

    exit=0 seconds=123.4 peak_tree_rss_mb=2345 min_mem_available_mb=4567 mem_total_mb=8024

What the numbers are:
  peak_tree_rss_mb       the largest sampled sum of RSS over the process tree. RSS counts shared pages once per
                         process, so with several workers it is an upper bound on what the tree really holds.
  min_mem_available_mb   the smallest sampled MemAvailable of the whole machine: how close the run came to
                         running out, whatever else was running. This is the number to watch.
Both are sampled, so a spike shorter than half a second can be missed.

With `--limit SECONDS` (0 = none) a command still running after that time is ended, with all its descendants
(SIGTERM, then SIGKILL ten seconds later), and the stats line gains `timed_out=1`.

The command stays in this process's process group, so that whoever signals the group (run_gate.sh stop) reaches
the command and its workers too.

Every five seconds it also appends a line `seconds tree_rss_mb mem_available_mb` to STATS_FILE.trace and syncs it
to disk, so that the memory history survives if the machine goes down during the run (it has: see P0_BASELINE.md).
"""
import os
import signal
import subprocess
import sys
import threading
import time

PERIOD = 0.5                                             # seconds between samples
TRACE_PERIOD = 5.0                                       # seconds between lines of the trace file
PAGE_MB = os.sysconf('SC_PAGE_SIZE') / 2**20


def _meminfo_mb(key):
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith(key + ':'):
                return int(line.split()[1]) / 1024.0
    return float('nan')


def _tree(root_pid):
    """{pid: RSS in MB} of root_pid and its descendants, from one pass over /proc."""
    parent, rss = {}, {}
    for name in os.listdir('/proc'):
        if not name.isdigit():
            continue
        try:
            with open('/proc/%s/stat' % name) as f:
                fields = f.read().rsplit(')', 1)[1].split()      # after the "(comm)" field, which may hold spaces
            parent[int(name)] = int(fields[1])                   # ppid
            rss[int(name)] = int(fields[21]) * PAGE_MB           # rss, in pages
        except (OSError, IndexError, ValueError):
            continue                                             # the process ended while we were reading
    children = {}
    for pid, ppid in parent.items():
        children.setdefault(ppid, []).append(pid)
    tree, todo = {}, [root_pid]
    while todo:
        pid = todo.pop()
        if pid in tree:
            continue
        tree[pid] = rss.get(pid, 0.0)
        todo.extend(children.get(pid, []))
    return tree


def _signal_tree(root_pid, signum):
    for pid in _tree(root_pid):
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def main(argv):
    limit = 0.0
    if len(argv) > 2 and argv[1] == '--limit':
        limit, argv = float(argv[2]), argv[:1] + argv[3:]
    if len(argv) < 4 or argv[2] != '--':
        sys.stderr.write(__doc__)
        return 2
    stats_file, command = argv[1], argv[3:]
    peak = {'rss': 0.0, 'avail': _meminfo_mb('MemAvailable'), 'timed_out': False}
    start = time.time()
    proc = subprocess.Popen(command)

    def forward(signum, _frame):                                 # a signal sent to us alone still ends the tree
        _signal_tree(proc.pid, signum)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    done = threading.Event()

    def sample():
        last_trace, killed_at = 0.0, None
        with open(stats_file + '.trace', 'w') as trace:
            while not done.is_set():
                now = time.time()
                rss, avail = sum(_tree(proc.pid).values()), _meminfo_mb('MemAvailable')
                peak['rss'] = max(peak['rss'], rss)
                peak['avail'] = min(peak['avail'], avail)
                if now - last_trace >= TRACE_PERIOD:
                    last_trace = now
                    trace.write('%.0f %.0f %.0f\n' % (now - start, rss, avail))
                    trace.flush()
                    os.fsync(trace.fileno())
                if limit and killed_at is None and now - start >= limit:
                    peak['timed_out'], killed_at = True, now
                    _signal_tree(proc.pid, signal.SIGTERM)
                elif killed_at is not None and now - killed_at >= 10.0:
                    _signal_tree(proc.pid, signal.SIGKILL)
                done.wait(PERIOD)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    code = proc.wait()
    done.set()
    sampler.join()
    line = 'exit=%d seconds=%.1f peak_tree_rss_mb=%.0f min_mem_available_mb=%.0f mem_total_mb=%.0f%s\n' % (
        code, time.time() - start, peak['rss'], peak['avail'], _meminfo_mb('MemTotal'),
        ' timed_out=1' if peak['timed_out'] else '')
    with open(stats_file, 'w') as f:
        f.write(line)
    if peak['timed_out']:
        return 124
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
