//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "utilities/pgm_index/pgm_index_factory.h"

#include "util/coding.h"

namespace ROCKSDB_NAMESPACE {
namespace pgm_index {

// ---- Builder ----
PgmIndexBuilder::PgmIndexBuilder(const Comparator* comparator)
    : comparator_(comparator) {}

Slice PgmIndexBuilder::AddIndexEntry(const Slice& last_key_in_current_block,
                                     const Slice* first_key_in_next_block,
                                     const BlockHandle& block_handle,
                                     std::string* /*separator_scratch*/,
                                     const IndexEntryContext& context) {
  // A real tag is only needed when the same user key continues in the next
  // block (or this is the last block). Otherwise 0 = "any seqno of this key".
  uint64_t tag = 0;
  if (first_key_in_next_block == nullptr ||
      comparator_->Compare(last_key_in_current_block,
                           *first_key_in_next_block) == 0) {
    tag = context.last_key_tag;
  }
  entries_.push_back({last_key_in_current_block.ToString(), tag,
                      block_handle.offset, block_handle.size});
  return Slice(entries_.back().separator);
}

Status PgmIndexBuilder::Finish(Slice* index_contents) {
  buffer_.clear();
  PutFixed32(&buffer_, 1);  // format version
  PutVarint64(&buffer_, entries_.size());
  for (const Entry& e : entries_) {
    PutLengthPrefixedSlice(&buffer_, Slice(e.separator));
    PutVarint64(&buffer_, e.tag);
    PutVarint64(&buffer_, e.offset);
    PutVarint64(&buffer_, e.size);
  }
  *index_contents = Slice(buffer_);
  finished_ = true;
  return Status::OK();
}

uint64_t PgmIndexBuilder::EstimatedSize() const {
  if (finished_) {
    return buffer_.size();
  }
  // Upper bound: version + count + per entry (length + 3 varints <= 35 B).
  uint64_t total = 4 + 10;
  for (const Entry& e : entries_) {
    total += e.separator.size() + 35;
  }
  return total;
}

// ---- Reader (stub) ----
PgmIndexReader::PgmIndexReader(const Comparator* comparator)
    : comparator_(comparator) {}

Status PgmIndexReader::InitFromSlice(const Slice& data) {
  data_size_ = data.size();
  return Status::OK();
}

std::unique_ptr<UserDefinedIndexIterator> PgmIndexReader::NewIterator(
    const ReadOptions& /*read_options*/) {
  return std::make_unique<PgmIndexIterator>(this, comparator_);
}

size_t PgmIndexReader::ApproximateMemoryUsage() const { return data_size_; }

// ---- Iterator (stub) ----
PgmIndexIterator::PgmIndexIterator(const PgmIndexReader* reader,
                                   const Comparator* comparator)
    : reader_(reader), comparator_(comparator) {}

void PgmIndexIterator::Prepare(const ScanOptions /*scan_opts*/[],
                               size_t /*num_opts*/) {}

Status PgmIndexIterator::SeekAndGetResult(const Slice& /*target*/,
                                          IterateResult* /*result*/,
                                          const SeekContext& /*context*/) {
  return Status::NotSupported("TODO");
}

Status PgmIndexIterator::NextAndGetResult(IterateResult* /*result*/) {
  return Status::NotSupported("TODO");
}

UserDefinedIndexBuilder::BlockHandle PgmIndexIterator::value() {
  return UserDefinedIndexBuilder::BlockHandle{0, 0};
}

// ---- Factory ----
Status PgmIndexFactory::NewBuilder(
    const UserDefinedIndexOption& option,
    std::unique_ptr<UserDefinedIndexBuilder>& builder) const {
  const Comparator* cmp =
      option.comparator ? option.comparator : BytewiseComparator();
  if (cmp != BytewiseComparator()) {
    return Status::NotSupported(
        "PgmIndexFactory requires BytewiseComparator; got: ", cmp->Name());
  }
  builder = std::make_unique<PgmIndexBuilder>(cmp);
  return Status::OK();
}

Status PgmIndexFactory::NewReader(
    const UserDefinedIndexOption& option, Slice& index_block,
    std::unique_ptr<UserDefinedIndexReader>& reader) const {
  const Comparator* cmp =
      option.comparator ? option.comparator : BytewiseComparator();
  if (cmp != BytewiseComparator()) {
    return Status::NotSupported(
        "PgmIndexFactory requires BytewiseComparator; got: ", cmp->Name());
  }
  auto pgm_reader = std::make_unique<PgmIndexReader>(cmp);
  Status s = pgm_reader->InitFromSlice(index_block);
  if (!s.ok()) {
    return s;
  }
  reader = std::move(pgm_reader);
  return Status::OK();
}

}  // namespace pgm_index
}  // namespace ROCKSDB_NAMESPACE
