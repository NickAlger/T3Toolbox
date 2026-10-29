#!/bin/bash
# Driver: all (seed, m, geom) cells for one d, 2 workers, skipping cells already on disk.
# usage (snapshot root): bash run_scaling_all.sh D "SEEDS"      e.g. bash run_scaling_all.sh 6 "1 2 3 4 5"
D=$1; SEEDS=${2:-"1 2 3 4 5"}
for s in $SEEDS; do for m in $(seq 1 $D); do
  if [ $m -eq 1 ]; then gs="unshared"; else gs="shared unshared"; fi
  for g in $gs; do [ -f results_scaling/d${D}_m${m}_${g}_${s}.json ] || echo "$D $s $m $g"; done
done; done | OMP_NUM_THREADS=1 PYTHONPATH=$PWD xargs -P 2 -L 1 python run_scaling.py
