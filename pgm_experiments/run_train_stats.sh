#!/bin/bash
OUT=/cluster/work/signehbo/results/2026-11-17_train
mkdir -p "$OUT"
cd /cluster/work/signehbo/rocksdb_pgm
DB=/cluster/work/signehbo/testdb_train
BASE="--benchmarks=fillrandom --num=5000000 --key_size=16 --value_size=100 --compression_type=none --seed=1 --use_pgm_index=true --db=$DB"
NOCOMP="--disable_auto_compactions=true --level0_slowdown_writes_trigger=100000 --level0_stop_writes_trigger=100000"
./db_bench $BASE --compaction_style=0 > "$OUT/leveling.txt" 2> "$OUT/leveling.log"
./db_bench $BASE --compaction_style=1 > "$OUT/tiering.txt" 2> "$OUT/tiering.log"
./db_bench $BASE $NOCOMP > "$OUT/nocompaction.txt" 2> "$OUT/nocompaction.log"
for s in leveling tiering nocompaction; do
  echo "== $s"
  grep "^fillrandom" "$OUT/$s.txt"
  grep PGM_TRAIN "$OUT/$s.log" | awk '{split($2,a,"=");split($3,b,"=");split($4,c,"=");split($5,d,"=");n++;k+=a[2];g+=b[2];by+=c[2];t+=d[2]} END{print "models="n, "keys="k, "segments="g, "model_bytes="by, "train_ms="t/1000}'
done
