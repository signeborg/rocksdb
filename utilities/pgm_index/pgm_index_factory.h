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
  };
  const Comparator* comparator_;
  std::vector<Entry> entries_;
  std::string buffer_;
  bool finished_ = false;
};

class PgmIndexReader final : public UserDefinedIndexReader {
 public:
  explicit PgmIndexReader(const Comparator* comparator);
  Status InitFromSlice(const Slice& data);
  std::unique_ptr<UserDefinedIndexIterator> NewIterator(
      const ReadOptions& read_options) override;
  size_t ApproximateMemoryUsage() const override;

 private:
  const Comparator* comparator_;
  size_t data_size_ = 0;
};

class PgmIndexIterator final : public UserDefinedIndexIterator {
 public:
  PgmIndexIterator(const PgmIndexReader* reader, const Comparator* comparator);
  void Prepare(const ScanOptions scan_opts[], size_t num_opts) override;
  Status SeekAndGetResult(const Slice& target, IterateResult* result,
                          const SeekContext& context) override;
  Status NextAndGetResult(IterateResult* result) override;
  UserDefinedIndexBuilder::BlockHandle value() override;

 private:
  const PgmIndexReader* reader_;
  const Comparator* comparator_;
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
