#!/bin/bash
OUT=/cluster/work/signehbo/results/2026-11-17_train50m
mkdir -p "$OUT"
cd /cluster/work/signehbo/rocksdb_pgm
DB=/cluster/work/signehbo/testdb_train50m
BASE="--benchmarks=fillrandom,waitforcompaction --num=50000000 --key_size=16 --value_size=100 --compression_type=none --seed=1 --use_pgm_index=true --db=$DB"
NOCOMP="--disable_auto_compactions=true --level0_slowdown_writes_trigger=100000 --level0_stop_writes_trigger=100000"
for s in leveling tiering nocompaction; do
  case $s in
    leveling) X="--compaction_style=0";;
    tiering) X="--compaction_style=1";;
    nocompaction) X="$NOCOMP";;
  esac
  ./db_bench $BASE $X > "$OUT/$s.txt" 2> "$OUT/$s.log"
done
for s in leveling tiering nocompaction; do
  echo "== $s"
  grep "^fillrandom" "$OUT/$s.txt"
  grep -o "PGM_TRAIN keys=[0-9]* segments=[0-9]* bytes=[0-9]* micros=[0-9]*" "$OUT/$s.log" | sed 's/[a-z_]*=//g' | awk '{n++;k+=$2;g+=$3;b+=$4;t+=$5} END{print "models="n, "keys="k, "segments="g, "model_bytes="b, "train_ms="t/1000}'
done > "$OUT/summary.txt"
rm -rf /cluster/work/signehbo/testdb_train50m
echo klar > "$OUT/DONE"
