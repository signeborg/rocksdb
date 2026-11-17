#!/bin/bash
OUT=${1:-/cluster/work/signehbo/results/2026-11-17_balanced}
mkdir -p "$OUT"
cd /cluster/work/signehbo/rocksdb_pgm
DB=/cluster/work/signehbo/testdb_bal
BASE="--benchmarks=fillrandom,readrandom,seekrandom --num=5000000 --reads=2000000 --seek_nexts=10 --key_size=16 --value_size=100 --compression_type=none --statistics --seed=1 --db=$DB"
NOCOMP="--disable_auto_compactions=true --level0_slowdown_writes_trigger=100000 --level0_stop_writes_trigger=100000"
run() { ./db_bench $BASE $2 $3 > "$OUT/$1_rep$4.txt" 2> /dev/null; }
for rep in 1 2 3; do
  if [ $((rep % 2)) = 1 ]; then ORDER="std pgm"; else ORDER="pgm std"; fi
  for idx in $ORDER; do
    if [ $idx = pgm ]; then IDX="--use_pgm_index=true"; else IDX=""; fi
    run leveling_$idx "--compaction_style=0" "$IDX" $rep
    run tiering_$idx "--compaction_style=1" "$IDX" $rep
    run nocompaction_$idx "$NOCOMP" "$IDX" $rep
  done
done
echo klar > "$OUT/DONE"
