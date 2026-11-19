// Writes keys from different distributions with the PGM UDI and reads them back.
// Build: g++ -std=c++20 -O1 -DROCKSDB_PLATFORM_POSIX -DOS_LINUX -I. -Iinclude \
//   -DROCKSDB_NAMESPACE=rocksdb pgm_dist_test.cc librocksdb.a -lpthread -ldl -lz -o pgm_dist_test
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "utilities/pgm_index/pgm_index_factory.h"

using namespace ROCKSDB_NAMESPACE;

static std::string MakeKey(uint64_t v) {
  std::string k(16, 'p');
  for (int i = 0; i < 8; i++) k[i] = static_cast<char>((v >> (56 - 8 * i)) & 0xff);
  return k;
}

static std::vector<uint64_t> Gen(const std::string& d, size_t n, std::mt19937_64& rng) {
  std::vector<uint64_t> v;
  v.reserve(n);
  if (d == "uniform") {
    for (size_t i = 0; i < n; i++) v.push_back(rng());
  } else if (d == "lognormal") {
    std::lognormal_distribution<double> ln(0.0, 2.0);
    for (size_t i = 0; i < n; i++) v.push_back(static_cast<uint64_t>(ln(rng) * 1e12));
  } else if (d == "clustered") {
    std::vector<uint64_t> centers;
    for (int i = 0; i < 100; i++) centers.push_back(rng() >> 1);
    for (size_t i = 0; i < n; i++) v.push_back(centers[rng() % 100] + rng() % (1ULL << 32));
  } else if (d == "bursty") {
    uint64_t cur = rng() >> 16;
    for (size_t i = 0; i < n; i++) {
      uint64_t gap = (rng() % 100 == 0) ? 1000000 + rng() % 1000000 : 1 + rng() % 10;
      cur += gap;
      v.push_back(cur);
    }
  } else {
    fprintf(stderr, "unknown dist %s\n", d.c_str());
    exit(1);
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  std::shuffle(v.begin(), v.end(), rng);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s dist [n] [path]\n", argv[0]); return 1; }
  const std::string dist = argv[1];
  const size_t n = argc > 2 ? strtoull(argv[2], nullptr, 10) : 5000000;
  const std::string path = argc > 3 ? argv[3] : "/cluster/work/signehbo/testdb_dist";

  Options options;
  options.create_if_missing = true;
  options.compression = kNoCompression;
  options.disable_auto_compactions = true;
  options.level0_slowdown_writes_trigger = 100000;
  options.level0_stop_writes_trigger = 100000;
  BlockBasedTableOptions table_options;
  auto factory = std::make_shared<pgm_index::PgmIndexFactory>();
  table_options.user_defined_index_factory = factory;
  table_options.use_udi_as_primary_index = true;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  DestroyDB(path, options);
  std::unique_ptr<DB> db;
  Status s = DB::Open(options, path, &db);
  if (!s.ok()) { fprintf(stderr, "open failed: %s\n", s.ToString().c_str()); return 1; }

  std::mt19937_64 rng(12345);
  std::vector<uint64_t> vals = Gen(dist, n, rng);
  const std::string value(100, 'x');
  auto t0 = std::chrono::steady_clock::now();
  for (uint64_t x : vals) {
    s = db->Put(WriteOptions(), MakeKey(x), value);
    if (!s.ok()) { fprintf(stderr, "put failed: %s\n", s.ToString().c_str()); return 1; }
  }
  s = db->Flush(FlushOptions());
  if (!s.ok()) { fprintf(stderr, "flush failed: %s\n", s.ToString().c_str()); return 1; }
  double fill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::string nfiles;
  db->GetProperty("rocksdb.num-files-at-level0", &nfiles);

  const size_t reads = 1000000;
  size_t found = 0;
  t0 = std::chrono::steady_clock::now();
  for (size_t i = 0; i < reads; i++) {
    std::string got;
    if (db->Get(ReadOptions(), MakeKey(vals[rng() % vals.size()]), &got).ok()) found++;
  }
  double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("DIST %s keys=%zu files=%s fill_s=%.1f reads=%zu found=%zu read_s=%.1f\n", dist.c_str(),
         vals.size(), nfiles.c_str(), fill_s, reads, found, read_s);
  db.reset();
  DestroyDB(path, options);
  return 0;
}
