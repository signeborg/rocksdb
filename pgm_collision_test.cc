// Standalone check of the PGM UDI: compares Get/Seek/Next against std::map.
// Build (from the rocksdb_pgm directory, after "make static_lib" or db_bench):
//   g++ -std=c++20 -O1 -DROCKSDB_PLATFORM_POSIX -DOS_LINUX -I. -Iinclude \
//     -DROCKSDB_NAMESPACE=rocksdb pgm_collision_test.cc librocksdb.a \
//     -lpthread -ldl -lz -o pgm_collision_test
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/table.h"
#include "utilities/pgm_index/pgm_index_factory.h"

using namespace ROCKSDB_NAMESPACE;

static int failures = 0;
#define CHECK(cond, msg)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      ++failures;                                                     \
      if (failures < 20) fprintf(stderr, "FAIL: %s (line %d)\n", msg, \
                                 __LINE__);                           \
    }                                                                 \
  } while (0)

static std::string Show(const std::string& s) {
  std::string r;
  for (unsigned char c : s) {
    char buf[8];
    snprintf(buf, sizeof(buf), (c >= 32 && c < 127) ? "%c" : "\\x%02x", c);
    r += buf;
  }
  return r;
}

// Builds the key set that stresses 8-byte prefix collisions.
static std::vector<std::string> MakeKeys() {
  std::vector<std::string> keys;
  // 1) Many keys sharing the same 8-byte prefix (collision group).
  for (int i = 0; i < 400; i++) {
    char buf[32];
    snprintf(buf, sizeof(buf), "SAMEPREF%05d", i);
    keys.push_back(buf);
  }
  // 2) Keys that are prefixes of other keys, and keys shorter than 8 bytes.
  for (std::string k : {"", "a", "ab", "abc", "abcdefg", "abcdefgh",
                        "abcdefghi", "abcdefghij", "abcdefgh\0", "b", "bb"}) {
    keys.push_back(k);
  }
  keys.push_back(std::string("abcdefgh\0", 9));
  keys.push_back(std::string("abcdefgh\0\0", 10));
  keys.push_back(std::string("\0", 1));
  keys.push_back(std::string("\0\0\0\0\0\0\0\0", 8));
  keys.push_back(std::string("\0\0\0\0\0\0\0\0x", 9));
  // 3) 0xFF keys (the all-ones prefix is clamped in KeyToU64).
  keys.push_back(std::string(7, '\xff'));
  keys.push_back(std::string(8, '\xff'));
  keys.push_back(std::string(8, '\xff') + "a");
  keys.push_back(std::string(9, '\xff'));
  keys.push_back(std::string(8, '\xff') + std::string(8, '\xff'));
  keys.push_back(std::string("\xff\xff\xff\xff\xff\xff\xff\xfe", 8));
  // 4) Random keys with random length and few distinct prefixes.
  std::mt19937 rng(42);
  for (int i = 0; i < 3000; i++) {
    std::string k = "PRE" + std::to_string(rng() % 5) + "XXXX";
    int extra = rng() % 12;
    for (int j = 0; j < extra; j++) k.push_back(static_cast<char>(rng() % 256));
    keys.push_back(k);
  }
  return keys;
}

static void RunMode(bool primary) {
  const std::string path =
      std::string("/tmp/pgm_collision_db_") + (primary ? "primary" : "secondary");
  Options options;
  options.create_if_missing = true;
  options.compression = kNoCompression;
  options.write_buffer_size = 64 * 1024;  // many small SSTs
  BlockBasedTableOptions table_options;
  table_options.block_size = 128;  // many blocks per SST
  auto factory = std::make_shared<pgm_index::PgmIndexFactory>();
  table_options.user_defined_index_factory = factory;
  table_options.use_udi_as_primary_index = primary;
  options.table_factory.reset(NewBlockBasedTableFactory(table_options));

  DestroyDB(path, options);
  std::unique_ptr<DB> db;
  Status s = DB::Open(options, path, &db);
  if (!s.ok()) {
    fprintf(stderr, "open failed: %s\n", s.ToString().c_str());
    exit(1);
  }

  std::map<std::string, std::string> ref;
  std::vector<std::string> keys = MakeKeys();
  std::mt19937 rng(7);
  // Write each key, overwrite a third of them several times (same user key
  // with many seqnos, which can straddle block boundaries), delete a few.
  for (const auto& k : keys) {
    std::string v = "v0_" + std::to_string(rng());
    CHECK(db->Put(WriteOptions(), k, v).ok(), "put");
    ref[k] = v;
  }
  for (int round = 1; round <= 4; round++) {
    for (size_t i = 0; i < keys.size(); i += 3) {
      std::string v = "v" + std::to_string(round) + "_" + std::to_string(rng());
      CHECK(db->Put(WriteOptions(), keys[i], v).ok(), "overwrite");
      ref[keys[i]] = v;
    }
  }
  for (size_t i = 5; i < keys.size(); i += 29) {
    CHECK(db->Delete(WriteOptions(), keys[i]).ok(), "delete");
    ref.erase(keys[i]);
  }
  CHECK(db->Flush(FlushOptions()).ok(), "flush");
  db->CompactRange(CompactRangeOptions(), nullptr, nullptr);

  ReadOptions ro;
  if (!primary) ro.table_index_factory = factory.get();

  // Get on every key plus some absent keys.
  for (const auto& k : keys) {
    std::string got;
    Status gs = db->Get(ro, k, &got);
    auto it = ref.find(k);
    if (it == ref.end()) {
      CHECK(gs.IsNotFound(), "deleted key must be NotFound");
    } else {
      CHECK(gs.ok() && got == it->second, "Get value mismatch");
    }
  }
  for (std::string k : {"SAMEPREF", "SAMEPREF99999", "SAMEPREE", "zzz",
                        "\x01", "abcdefgh\x01"}) {
    std::string got;
    bool in_ref = ref.count(k) > 0;
    Status gs = db->Get(ro, k, &got);
    CHECK(in_ref == gs.ok(), "absent key presence mismatch");
  }

  // Seek to every key, to key+"\0", to key minus last byte, and random keys.
  std::unique_ptr<Iterator> it(db->NewIterator(ro));
  std::vector<std::string> targets = keys;
  for (const auto& k : keys) {
    targets.push_back(k + std::string(1, '\0'));
    if (!k.empty()) targets.push_back(k.substr(0, k.size() - 1));
  }
  for (int i = 0; i < 2000; i++) {
    std::string k;
    int n = rng() % 14;
    for (int j = 0; j < n; j++) k.push_back(static_cast<char>(rng() % 256));
    targets.push_back(k);
    targets.push_back("SAMEPREF" + std::to_string(rng() % 100000));
  }
  size_t seeks = 0;
  for (const auto& t : targets) {
    it->Seek(t);
    auto r = ref.lower_bound(t);
    ++seeks;
    if (r == ref.end()) {
      CHECK(!it->Valid(), "Seek past end must be invalid");
    } else if (!it->Valid() || it->key().ToString() != r->first) {
      CHECK(false, "Seek landed on wrong key");
      if (failures < 20)
        fprintf(stderr, "  target=%s expected=%s got=%s\n", Show(t).c_str(),
                Show(r->first).c_str(),
                it->Valid() ? Show(it->key().ToString()).c_str() : "<end>");
    } else {
      CHECK(it->value().ToString() == r->second, "Seek value mismatch");
      // Walk a few steps with Next.
      auto r2 = r;
      for (int step = 0; step < 3; step++) {
        it->Next();
        ++r2;
        if (r2 == ref.end()) {
          CHECK(!it->Valid(), "Next past end must be invalid");
          break;
        }
        CHECK(it->Valid() && it->key().ToString() == r2->first,
              "Next landed on wrong key");
        if (!it->Valid()) break;
      }
    }
  }

  // Full forward scan and full backward scan.
  size_t n = 0;
  auto r = ref.begin();
  for (it->SeekToFirst(); it->Valid(); it->Next(), ++r, ++n) {
    CHECK(r != ref.end() && it->key().ToString() == r->first, "scan mismatch");
    if (r == ref.end()) break;
  }
  CHECK(n == ref.size(), "forward scan count");
  n = 0;
  auto rr = ref.rbegin();
  for (it->SeekToLast(); it->Valid(); it->Prev(), ++rr, ++n) {
    CHECK(rr != ref.rend() && it->key().ToString() == rr->first,
          "reverse scan mismatch");
    if (rr == ref.rend()) break;
  }
  CHECK(n == ref.size(), "backward scan count");
  CHECK(it->status().ok(), "iterator status");

  // Snapshots: same user key with several versions.
  const Snapshot* snap = db->GetSnapshot();
  std::string probe = "SAMEPREF00007";
  std::string old_val;
  CHECK(db->Get(ro, probe, &old_val).ok(), "probe present");
  for (int i = 0; i < 300; i++) {
    CHECK(db->Put(WriteOptions(), probe, "new" + std::to_string(i)).ok(), "put");
  }
  db->Flush(FlushOptions());
  ReadOptions snap_ro = ro;
  snap_ro.snapshot = snap;
  std::string got;
  CHECK(db->Get(snap_ro, probe, &got).ok() && got == old_val,
        "snapshot must still see old value");
  CHECK(db->Get(ro, probe, &got).ok() && got == "new299", "latest value");
  db->ReleaseSnapshot(snap);

  printf("mode=%s: %zu keys in ref, %zu seeks checked\n",
         primary ? "primary" : "secondary", ref.size(), seeks);
  it.reset();
  db.reset();
  DestroyDB(path, options);
}

int main() {
  RunMode(true);
  RunMode(false);
  if (failures == 0) {
    printf("ALL OK\n");
    return 0;
  }
  printf("%d FAILURES\n", failures);
  return 1;
}

