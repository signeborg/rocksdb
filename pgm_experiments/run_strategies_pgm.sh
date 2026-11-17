#!/bin/bash
OUT=${1:-/cluster/work/signehbo/results/2026-11-17_pgm}
mkdir -p "$OUT"
cd /cluster/work/signehbo/rocksdb_pgm
DB=/cluster/work/signehbo/testdb_pgm
BASE="--benchmarks=fillrandom,stats,readrandom --num=5000000 --key_size=16 --value_size=100 --compression_type=none --statistics --seed=1 --db=$DB"
NOCOMP="--disable_auto_compactions=true --level0_slowdown_writes_trigger=100000 --level0_stop_writes_trigger=100000"
for idx in std pgm; do
  if [ $idx = pgm ]; then IDX="--use_pgm_index=true"; else IDX=""; fi
  ./db_bench $BASE --compaction_style=0 $IDX > "$OUT/leveling_$idx.txt"
  ./db_bench $BASE --compaction_style=1 $IDX > "$OUT/tiering_$idx.txt"
  ./db_bench $BASE $NOCOMP $IDX > "$OUT/nocompaction_$idx.txt"
done
grep -E "fillrandom|readrandom|^ *Sum" "$OUT"/*.txt
