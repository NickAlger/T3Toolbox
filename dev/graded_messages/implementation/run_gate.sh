#!/usr/bin/env bash
# The gate of Part 1 (PART1_PLAN.md section 5.1; .github/workflows/tests.yaml and build-sphinx-docs.yaml).
#
#   run_gate.sh [--detach] [-n N] [-t S] [-l LOGDIR] [STEP ...] the whole gate, or only the named steps
#   run_gate.sh [--detach] [-n N] [-t S] [-l LOGDIR] quick TARGET ... only the tests in the given files (or node ids)
#   run_gate.sh wait LOGDIR [SECONDS]                             wait for a detached run (default 570 s), then report
#   run_gate.sh stop LOGDIR                                       end a detached run and everything it started
#
# The steps, always in this order, strictly one after another:
#   suite               pytest tests/ -q                      (with -n N when N > 1)
#   doctest-modules     pytest --doctest-modules t3toolbox -q
#   doctest-quickstart  python -m doctest docs/getting_started.rst
#   doctest-pages       python -m doctest on every docs/*.md and docs/contributor/*.md except doctest_style.md
#   examples            every examples/*.py, each must exit 0
#   sphinx              python -m sphinx -W --keep-going -b html docs LOGDIR/sphinx_html
# One more step, never part of the default list, run only when named (the slices that need it say so: P16, P17, P20):
#   slow                the slow tier, a release-gate step (CLAUDE.md), always serial:
#                       T3TOOLBOX_SLOW_TESTS=1 pytest tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py -q
#
# Options:
#   -n N        test workers for `suite` and `quick` (default 1 = plain pytest, no xdist). P0_BASELINE.md says what
#               is safe on the agents' machine. The other steps are single-process.
#   -t SECONDS  a time limit for each step (each example, in `examples`); 0, the default, is no limit. A step that
#               reaches it is ended with its whole process tree and counts as FAIL ("stopped at the time limit").
#   -l LOGDIR   where the logs go (default: a fresh directory under /tmp, printed first). Each step writes
#               LOGDIR/<step>.log; `examples` also writes LOGDIR/examples/<name>.log. results.tsv holds the table.
#   --detach    start the run in the background and return at once. A foreground command is cut off after ten
#               minutes in the agents' workspace, and the suite alone takes longer, so run the gate like this:
#                   run_gate.sh --detach -n 1 -l /tmp/gate_P5        # returns at once
#                   run_gate.sh wait /tmp/gate_P5                    # repeat until it prints the summary
#               `wait` exits 0 when the run finished and every step passed, 1 when it finished and a step failed,
#               3 when it is still running after SECONDS (call it again), 2 when the run is gone without having
#               finished: the machine restarted under it (this happened once in P0, see P0_BASELINE.md) or it was
#               stopped. Start it again; LOGDIR/<step>.stats.trace has the memory history up to the end.
#
# Memory: only one run at a time. A second run refuses to start while another holds /tmp/t3toolbox_gate.lock.
# The lock and LOGDIR/PID name a run by its pid, the start time of that process and the machine's boot id, so a
# lock or a log directory left over from before a restart of the machine is recognised as dead, and `stop` signals
# only the run of the LOGDIR it is given, and only when that run is provably still this one. If the gate process
# itself is killed with -9, the step it was running goes on: the lock then still refuses a new run, `wait` says so
# (exit 2), and `stop LOGDIR` ends the step and frees the lock.
# Never start anything else heavy while a run is going (the maintainer has run a 13 GB machine out of memory with
# sphinx and a parallel test run together).
#
# Each step runs under timed.py (beside this script), which records its wall time, the peak resident memory of its
# process tree and the lowest MemAvailable of the machine while it ran. The summary table at the end:
#   step, result, counts, seconds, peak RSS (MB), lowest MemAvailable (MB).
# Exit code of a foreground run: 0 if every step passed, 1 otherwise.
#
# Nothing is written into the repository: pytest runs with its cache off, and sphinx builds under LOGDIR (the
# docs/autoapi folder that sphinx-autoapi makes while it builds is removed afterwards if this run created it).

{   # in braces: bash reads the whole block before it runs any of it, so editing this file cannot disturb a run
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
TIMED="$HERE/timed.py"
LOCK=/tmp/t3toolbox_gate.lock
ALL_STEPS="suite doctest-modules doctest-quickstart doctest-pages examples sphinx"
KNOWN_STEPS="$ALL_STEPS slow"                               # slow runs only when named, and last
export PYTHONPATH="$REPO"

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# ---------------------------------------------------------------- the summary table

print_summary() {            # $1 = LOGDIR
  echo
  printf '%-19s %-6s %-58s %8s %9s %10s\n' step result counts seconds peak_MB minavl_MB
  while IFS=$'\t' read -r step result counts seconds peak avail; do
    printf '%-19s %-6s %-58s %8s %9s %10s\n' "$step" "$result" "$counts" "$seconds" "$peak" "$avail"
  done < "$1/results.tsv"
  echo "logs: $1"
}

all_passed() { ! cut -f2 "$1/results.tsv" | grep -qv '^PASS$'; }      # $1 = LOGDIR

# ---------------------------------------------------------------- which run is which
#
# A run is its gate process: a session and process-group leader, so its pid is also the group of everything it
# starts (timed.py and the step keep that group). A pid alone does not identify a run: /tmp outlives a restart
# of the machine, and pids start again from low numbers. So LOGDIR/PID and the lock hold
#     <pid> <start time of that process, from /proc/<pid>/stat> <boot id of the machine> [<LOGDIR>]
# and a run counts as going only if the boot id is this boot's and the process with that pid started at that time.

BOOT="$(cat /proc/sys/kernel/random/boot_id)"

start_time() {               # start_time PID -> field 22 of /proc/PID/stat; fails if there is no such process
  local s; s="$(cat "/proc/$1/stat" 2>/dev/null)" || return 1
  s="${s##*) }"; set -- $s; echo "${20}"
}

run_state() {                # run_state FILE -> alive | orphans | dead
  local pid start boot rest now
  [ -r "$1" ] || { echo dead; return; }
  read -r pid start boot rest < "$1" || { echo dead; return; }
  if ! [[ "$pid" =~ ^[0-9]+$ ]] || [ "$boot" != "$BOOT" ]; then echo dead; return; fi   # another boot: not ours
  if now="$(start_time "$pid")"; then
    if [ "$now" = "$start" ]; then echo alive; else echo dead; fi      # same pid, other start: the pid was reused
    return
  fi
  # The gate process is gone. A group id is not given out again while the group has members, so in this boot
  # whatever is left in group <pid> was started by that run (a step that outlived a kill -9 of the gate).
  if [ -n "$(pgrep -g "$pid")" ]; then echo orphans; else echo dead; fi
}

same_run() { [ -e "$1" ] && [ -e "$2" ] && [ "$(cut -d' ' -f1-3 "$1")" = "$(cut -d' ' -f1-3 "$2")" ]; }

# ---------------------------------------------------------------- wait and stop

if [ "${1:-}" = wait ]; then
  LOGDIR="${2:?usage: run_gate.sh wait LOGDIR [SECONDS]}"; LIMIT="${3:-570}"; t0=$SECONDS
  while [ ! -e "$LOGDIR/DONE" ]; do
    state="$(run_state "$LOGDIR/PID")"
    if [ "$state" = dead ]; then
      sleep 2                                              # the run may have just finished
      [ -e "$LOGDIR/DONE" ] && break
      echo "the run in $LOGDIR is not running and did not finish (see $LOGDIR/gate.out)"; exit 2
    fi
    if [ "$state" = orphans ]; then
      echo "the gate process of $LOGDIR is gone, but a step it started is still running. End it with:  $0 stop $LOGDIR"
      exit 2
    fi
    if [ $((SECONDS - t0)) -ge "$LIMIT" ]; then
      echo "still running after ${LIMIT}s. Finished so far:"; print_summary "$LOGDIR"
      echo "now running: $(cat "$LOGDIR/CURRENT" 2>/dev/null)"
      echo "free memory now: $(awk '/MemAvailable/ {printf "%d MB", $2/1024}' /proc/meminfo)"
      exit 3
    fi
    sleep 5
  done
  print_summary "$LOGDIR"
  all_passed "$LOGDIR"; exit $?
fi

if [ "${1:-}" = stop ]; then                               # acts on the run of LOGDIR only, and only if it is provably it
  LOGDIR="${2:?usage: run_gate.sh stop LOGDIR}"
  state="$(run_state "$LOGDIR/PID")"
  if [ "$state" = dead ]; then
    echo "no run is going in $LOGDIR: nothing was signalled"
  else
    read -r pid _ < "$LOGDIR/PID"
    kill -TERM -- "-$pid" 2>/dev/null                      # the whole group: the gate, timed.py, the step, its workers
    for _ in $(seq 20); do [ -z "$(pgrep -g "$pid")" ] && break; sleep 0.5; done
    if [ -n "$(pgrep -g "$pid")" ]; then kill -KILL -- "-$pid" 2>/dev/null; sleep 1; fi
    echo "stopped the run in $LOGDIR$( [ "$state" = orphans ] && echo ' (its gate process was already gone)')"
  fi
  if same_run "$LOCK" "$LOGDIR/PID"; then rm -f "$LOCK"; fi   # only this run's lock; another run's lock stays
  exit 0
fi

# ---------------------------------------------------------------- options

WORKERS=1; LOGDIR=""; DETACH=0; STEP_LIMIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    -n|-l|-t)                              # each takes a value; without one `shift 2` fails and the loop never ends
      [ $# -ge 2 ] || { echo "option $1 needs a value" >&2; exit 2; }
      case "$1" in
        -n) WORKERS="$2" ;;
        -l) LOGDIR="$2" ;;
        -t) STEP_LIMIT="$2" ;;
      esac
      shift 2 ;;
    --detach) DETACH=1; shift ;;
    -h|--help) usage ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) break ;;
  esac
done
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "-n needs a whole number of workers, 1 or more; got '$WORKERS'" >&2; exit 2; }
[[ "$STEP_LIMIT" =~ ^[0-9]+$ ]] || { echo "-t needs a whole number of seconds; got '$STEP_LIMIT'" >&2; exit 2; }

MODE=gate; QUICK_TARGETS=()
if [ "${1:-}" = quick ]; then
  MODE=quick; shift; QUICK_TARGETS=("$@")
  [ ${#QUICK_TARGETS[@]} -gt 0 ] || { echo "quick needs at least one test file" >&2; exit 2; }
  STEPS=quick
else
  STEPS="${*:-$ALL_STEPS}"
  for s in $STEPS; do
    case " $KNOWN_STEPS " in *" $s "*) ;; *) echo "unknown step '$s'. Steps: $KNOWN_STEPS" >&2; exit 2 ;; esac
  done
  ordered=""; for s in $KNOWN_STEPS; do case " $STEPS " in *" $s "*) ordered="$ordered $s" ;; esac; done
  STEPS="$ordered"                                         # always the gate's order, whatever order was typed
fi

[ -n "$LOGDIR" ] || LOGDIR="$(mktemp -d /tmp/t3toolbox_gate.XXXXXX)"
mkdir -p "$LOGDIR"; LOGDIR="$(cd "$LOGDIR" && pwd)"

# Every run, detached or not, is started once more as the leader of a session of its own (see "which run is which").
if [ -z "${T3TOOLBOX_GATE_INNER:-}" ]; then
  if [ -e "$LOGDIR/PID" ] && [ "$(run_state "$LOGDIR/PID")" != dead ]; then
    echo "a run is going in $LOGDIR: give another directory, or wait for it or stop it" >&2; exit 2
  fi
  args=(-n "$WORKERS" -t "$STEP_LIMIT" -l "$LOGDIR")
  if [ "$MODE" = quick ]; then args+=(quick "${QUICK_TARGETS[@]}"); else args+=($STEPS); fi
  rm -f "$LOGDIR/DONE" "$LOGDIR/PID"
  if [ "$DETACH" = 1 ]; then
    T3TOOLBOX_GATE_INNER=1 setsid nohup "$0" "${args[@]}" > "$LOGDIR/gate.out" 2>&1 < /dev/null &
    for _ in $(seq 20); do [ -e "$LOGDIR/PID" ] && break; sleep 0.5; done
    [ -e "$LOGDIR/PID" ] || { echo "the run did not start:" >&2; cat "$LOGDIR/gate.out" >&2; exit 2; }
    echo "detached. Logs: $LOGDIR"
    echo "wait with:  $0 wait $LOGDIR"
    exit 0
  fi
  T3TOOLBOX_GATE_INNER=1 setsid "$0" "${args[@]}" &        # in the foreground: wait for it, and pass on Ctrl-C
  child=$!
  trap 'kill -TERM -- "-$child" 2>/dev/null' INT TERM
  while :; do wait "$child"; code=$?; kill -0 "$child" 2>/dev/null || break; done
  exit "$code"
fi

# ---------------------------------------------------------------- one run at a time

if [ -e "$LOCK" ]; then
  state="$(run_state "$LOCK")"
  if [ "$state" != dead ]; then
    read -r pid _ _ other < "$LOCK"
    echo "another gate run is going (pid $pid, logs $other). Only one at a time: wait for it or stop it." >&2
    [ "$state" = orphans ] && echo "(its gate process is gone but a step of it is still running:  $0 stop $other)" >&2
    exit 2
  fi
fi
echo "$$ $(start_time $$) $BOOT $LOGDIR" > "$LOCK"; cp "$LOCK" "$LOGDIR/PID"
rm -f "$LOGDIR/DONE"; : > "$LOGDIR/results.tsv"
trap 'if same_run "$LOCK" "$LOGDIR/PID"; then rm -f "$LOCK"; fi' EXIT
cd "$REPO"
echo "logs: $LOGDIR    workers: $WORKERS    steps:$( [ "$MODE" = quick ] && echo " quick ${QUICK_TARGETS[*]}" || echo "$STEPS")"
python3 "$HERE/versions.py" | tee "$LOGDIR/versions.txt"

# ---------------------------------------------------------------- how a step is run and recorded

stat_of() { tr ' ' '\n' < "$1" | sed -n "s/^$2=//p"; }     # stat_of FILE KEY

record() {                   # record STEP EXIT_CODE COUNTS
  local step="$1" code="$2" counts="$3" stats="$LOGDIR/$1.stats" result=PASS
  [ "$code" = 0 ] || result=FAIL
  [ "$(stat_of "$stats" timed_out)" = 1 ] && counts="stopped at the time limit of ${STEP_LIMIT}s. $counts"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$step" "$result" "$counts" \
    "$(stat_of "$stats" seconds)" "$(stat_of "$stats" peak_tree_rss_mb)" "$(stat_of "$stats" min_mem_available_mb)" \
    >> "$LOGDIR/results.tsv"
  echo "$step: $result   $counts"
}

timed() {                    # timed STEP command...   -> LOGDIR/STEP.log, LOGDIR/STEP.stats; returns the exit code
  local step="$1"; shift
  echo "$step (started $(date +%H:%M:%S))" > "$LOGDIR/CURRENT"
  python3 "$TIMED" --limit "$STEP_LIMIT" "$LOGDIR/$step.stats" -- "$@" > "$LOGDIR/$step.log" 2>&1
}

pytest_counts() {            # the last line of a pytest -q log that carries the totals, without the '=' rules
  grep -E '(passed|failed|error|skipped|no tests ran)' "$1" | tail -1 | sed -E 's/^=+ *//; s/ *=+$//; s/ in [0-9.]+s.*$//'
}

xdist_args() { if [ "$WORKERS" -gt 1 ]; then echo "-n $WORKERS"; fi; }

# ---------------------------------------------------------------- the steps

step_suite() {
  timed suite python3 -m pytest tests/ -q -rfEs -p no:cacheprovider $(xdist_args)
  record suite $? "$(pytest_counts "$LOGDIR/suite.log")"
}

step_slow() {                # serial whatever -n says: the release checklist runs it so
  timed slow env T3TOOLBOX_SLOW_TESTS=1 python3 -m pytest tests/test_oracle_sweep.py tests/backend/test_uniform_curvature.py \
    -q -rfEs -p no:cacheprovider
  record slow $? "$(pytest_counts "$LOGDIR/slow.log")"
}

step_quick() {
  timed quick python3 -m pytest "${QUICK_TARGETS[@]}" -q -rfEs -p no:cacheprovider $(xdist_args)
  record quick $? "$(pytest_counts "$LOGDIR/quick.log")"
}

step_doctest-modules() {
  timed doctest-modules python3 -m pytest --doctest-modules t3toolbox -q -rfEs -p no:cacheprovider
  record doctest-modules $? "$(pytest_counts "$LOGDIR/doctest-modules.log")"
}

doctest_counts() {           # doctest prints nothing when all pass; failures as "***Test Failed*** N failures."
  local log="$1" files="$2" failed
  failed=$(grep -c '^\*\*\*Test Failed\*\*\*' "$log")
  echo "$files file(s), $failed with failures, $(grep -c '^Failed example:' "$log") failed example(s)"
}

step_doctest-quickstart() {
  timed doctest-quickstart python3 -m doctest docs/getting_started.rst
  record doctest-quickstart $? "$(doctest_counts "$LOGDIR/doctest-quickstart.log" 1)"
}

step_doctest-pages() {
  local pages; pages=$(ls docs/*.md docs/contributor/*.md 2> "$LOGDIR/doctest-pages.ls-errors" | grep -v 'doctest_style.md')
  timed doctest-pages python3 -m doctest $pages          # CI's command. The count of pages is in the summary:
  record doctest-pages $? "$(doctest_counts "$LOGDIR/doctest-pages.log" "$(echo "$pages" | wc -l)")"   # 48 at P0
}

step_examples() {            # every example runs even if one fails, as in CI; one log each
  mkdir -p "$LOGDIR/examples"
  local n=0 failed="" start=$SECONDS peak=0 avail=999999 f name code
  : > "$LOGDIR/examples.log"
  for f in examples/*.py; do
    name="examples/$(basename "$f" .py)"; n=$((n + 1))
    timed "$name" python3 "$f"; code=$?
    [ "$code" = 0 ] || failed="$failed $(basename "$f")"
    printf '%-4s %7ss %6s MB  %s\n' "$( [ "$code" = 0 ] && echo PASS || echo FAIL)" \
      "$(stat_of "$LOGDIR/$name.stats" seconds)" "$(stat_of "$LOGDIR/$name.stats" peak_tree_rss_mb)" "$f" >> "$LOGDIR/examples.log"
    [ "$(stat_of "$LOGDIR/$name.stats" peak_tree_rss_mb)" -gt "$peak" ] && peak=$(stat_of "$LOGDIR/$name.stats" peak_tree_rss_mb)
    [ "$(stat_of "$LOGDIR/$name.stats" min_mem_available_mb)" -lt "$avail" ] && avail=$(stat_of "$LOGDIR/$name.stats" min_mem_available_mb)
  done
  echo "exit=0 seconds=$((SECONDS - start)) peak_tree_rss_mb=$peak min_mem_available_mb=$avail" > "$LOGDIR/examples.stats"
  local nfailed; nfailed=$(echo $failed | wc -w)
  record examples "$nfailed" "$n examples, $nfailed failed$( [ -n "$failed" ] && echo ":$failed")"
}

step_sphinx() {
  local had_autoapi=0; [ -e docs/autoapi ] && had_autoapi=1
  rm -rf "$LOGDIR/sphinx_html" "$LOGDIR/sphinx_doctrees"
  timed sphinx python3 -m sphinx -W --keep-going -b html -d "$LOGDIR/sphinx_doctrees" docs "$LOGDIR/sphinx_html"
  local code=$?
  [ "$had_autoapi" = 0 ] && rm -rf docs/autoapi
  record sphinx $code "$(grep -c -E '(WARNING|ERROR):' "$LOGDIR/sphinx.log") warning/error line(s)"
}

# ---------------------------------------------------------------- run

for s in $STEPS; do "step_$s"; done
rm -f "$LOGDIR/CURRENT"
print_summary "$LOGDIR"
all_passed "$LOGDIR"; code=$?
echo "$code" > "$LOGDIR/DONE"
exit $code
}
