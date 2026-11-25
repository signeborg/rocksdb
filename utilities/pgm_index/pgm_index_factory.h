//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
//  EXPERIMENTAL: per-block PGM-index as a User Defined Index.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "rocksdb/comparator.h"
#include "rocksdb/slice.h"
#include "rocksdb/user_defined_index.h"

namespace ROCKSDB_NAMESPACE {
namespace pgm_index {

class PgmIndexBuilder final : public UserDefinedIndexBuilder {
 public:
  explicit PgmIndexBuilder(const Comparator* comparator);
  Slice AddIndexEntry(const Slice& last_key_in_current_block,
                      const Slice* first_key_in_next_block,
                      const BlockHandle& block_handle,
                      std::string* separator_scratch,
                      const IndexEntryContext& context) override;
  Status Finish(Slice* index_contents) override;
  uint64_t EstimatedSize() const override;

 private:
  struct Entry {
    std::string separator;
    uint64_t tag;
    uint64_t offset;
    uint64_t size;
    bool next_same_prefix;
  };
  const Comparator* comparator_;
  std::vector<Entry> entries_;
  std::string buffer_;
  bool finished_ = false;
};

// Maps the first 8 bytes of a key to a uint64_t (big-endian, zero padded).
// Preserves bytewise order: a < b implies KeyToU64(a) <= KeyToU64(b).
uint64_t KeyToU64(const Slice& key);

// The PGM model type lives in the .cc file (pimpl) so the third-party header
// does not leak into every includer of this file.
struct PgmModelHolder;

class PgmIndexReader final : public UserDefinedIndexReader {
 public:
  explicit PgmIndexReader(const Comparator* comparator);
  ~PgmIndexReader() override;
  Status InitFromSlice(const Slice& data);
  std::unique_ptr<UserDefinedIndexIterator> NewIterator(
      const ReadOptions& read_options) override;
  size_t ApproximateMemoryUsage() const override;

  struct Entry {
    Slice separator;
    uint64_t prefix;
    uint64_t tag;
    uint64_t offset;
    uint64_t size;
  };
  size_t NumEntries() const { return entries_.size(); }
  const Entry& GetEntry(size_t i) const { return entries_[i]; }
  Slice SeparatorOf(size_t i, std::string* scratch) const;
  size_t NumSegments() const;

  // Index of the first entry whose (separator, tag) is >= (target,
  // target_tag) in internal key order, or NumEntries() if there is none.
  size_t Seek(const Slice& target, uint64_t target_tag) const;

 private:
  // True if entry i sorts strictly before (target, target_tag).
  bool EntryBefore(size_t i, const Slice& target, uint64_t target_tag) const;

  const Comparator* comparator_;
  std::vector<Entry> entries_;
  size_t data_size_ = 0;
  // Distinct KeyToU64(separator) values, ascending, and for each the index
  // of its first entry. group_start_[prefixes_.size()] == entries_.size().
  std::vector<uint64_t> prefixes_;
  std::vector<uint32_t> group_start_;
  std::unique_ptr<PgmModelHolder> pgm_;
};

class PgmIndexIterator final : public UserDefinedIndexIterator {
 public:
  PgmIndexIterator(const PgmIndexReader* reader, const Comparator* comparator);
  void Prepare(const ScanOptions scan_opts[], size_t num_opts) override;
  Status SeekToFirstAndGetResult(IterateResult* result) override;
  Status SeekToLastAndGetResult(IterateResult* result) override;
  Status PrevAndGetResult(IterateResult* result) override;
  Status SeekAndGetResult(const Slice& target, IterateResult* result,
                          const SeekContext& context) override;
  Status NextAndGetResult(IterateResult* result) override;
  UserDefinedIndexBuilder::BlockHandle value() override;

 private:
  IterBoundCheck CheckBounds(const Slice& reference_key) const;
  // Fills result for pos_; returns false (and an invalid result) if pos_ is
  // past the last entry.
  bool SetResult(IterateResult* result);

  const PgmIndexReader* reader_;
  const Comparator* comparator_;
  size_t pos_;
  std::vector<ScanOptions> scan_opts_;
  size_t current_scan_idx_ = 0;
  bool prepared_ = false;
  std::string key_buf_;
};

class PgmIndexFactory : public UserDefinedIndexFactory {
 public:
  static const char* kClassName() { return "pgm_index"; }
  const char* Name() const override { return kClassName(); }

  UserDefinedIndexBuilder* NewBuilder() const override {
    abort();
    return nullptr;
  }
  std::unique_ptr<UserDefinedIndexReader> NewReader(
      Slice& /*index_block*/) const override {
    abort();
    return nullptr;
  }
  Status NewBuilder(
      const UserDefinedIndexOption& option,
      std::unique_ptr<UserDefinedIndexBuilder>& builder) const override;
  Status NewReader(
      const UserDefinedIndexOption& option, Slice& index_block,
      std::unique_ptr<UserDefinedIndexReader>& reader) const override;
};

}  // namespace pgm_index
}  // namespace ROCKSDB_NAMESPACE
