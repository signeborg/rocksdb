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

// ---- Reader ----
PgmIndexReader::PgmIndexReader(const Comparator* comparator)
    : comparator_(comparator) {}

Status PgmIndexReader::InitFromSlice(const Slice& data) {
  data_size_ = data.size();
  Slice input = data;

  uint32_t version = 0;
  if (!GetFixed32(&input, &version) || version != 1) {
    return Status::Corruption("PGM index: bad format version");
  }
  uint64_t count = 0;
  // Each entry takes at least 4 bytes, so a larger count must be corrupt.
  if (!GetVarint64(&input, &count) || count > input.size()) {
    return Status::Corruption("PGM index: bad entry count");
  }

  entries_.clear();
  entries_.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    Entry e;
    if (!GetLengthPrefixedSlice(&input, &e.separator) ||
        !GetVarint64(&input, &e.tag) || !GetVarint64(&input, &e.offset) ||
        !GetVarint64(&input, &e.size)) {
      return Status::Corruption("PGM index: truncated entry");
    }
    entries_.push_back(e);
  }
  return Status::OK();
}

std::unique_ptr<UserDefinedIndexIterator> PgmIndexReader::NewIterator(
    const ReadOptions& /*read_options*/) {
  return std::make_unique<PgmIndexIterator>(this, comparator_);
}

size_t PgmIndexReader::ApproximateMemoryUsage() const {
  return data_size_ + entries_.capacity() * sizeof(Entry);
}

// ---- Reader::Seek (plain binary search, PGM comes later) ----
size_t PgmIndexReader::Seek(const Slice& target, uint64_t target_tag) const {
  size_t lo = 0;
  size_t hi = entries_.size();
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    const Entry& e = entries_[mid];
    const int c = comparator_->Compare(e.separator, target);
    // Higher tag sorts first, so an equal key with a higher tag is before.
    const bool before = c < 0 || (c == 0 && target_tag < e.tag);
    if (before) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// ---- Iterator ----
PgmIndexIterator::PgmIndexIterator(const PgmIndexReader* reader,
                                   const Comparator* comparator)
    : reader_(reader), comparator_(comparator), pos_(reader->NumEntries()) {}

void PgmIndexIterator::Prepare(const ScanOptions scan_opts[], size_t num_opts) {
  scan_opts_.assign(scan_opts, scan_opts + num_opts);
  current_scan_idx_ = 0;
  prepared_ = true;
}

bool PgmIndexIterator::SetResult(IterateResult* result) {
  if (pos_ >= reader_->NumEntries()) {
    pos_ = reader_->NumEntries();
    // Running out of entries says nothing about the upper bound.
    result->bound_check_result = IterBoundCheck::kUnknown;
    result->key = Slice();
    return false;
  }
  result->key = reader_->GetEntry(pos_).separator;
  return true;
}

Status PgmIndexIterator::SeekToFirstAndGetResult(IterateResult* result) {
  pos_ = 0;
  if (SetResult(result)) {
    result->bound_check_result = IterBoundCheck::kInbound;
  }
  return Status::OK();
}

Status PgmIndexIterator::SeekToLastAndGetResult(IterateResult* result) {
  pos_ = reader_->NumEntries() == 0 ? 0 : reader_->NumEntries() - 1;
  if (SetResult(result)) {
    result->bound_check_result = IterBoundCheck::kInbound;
  }
  return Status::OK();
}

Status PgmIndexIterator::PrevAndGetResult(IterateResult* result) {
  if (pos_ == 0 || pos_ > reader_->NumEntries()) {
    pos_ = reader_->NumEntries();
  } else {
    --pos_;
  }
  if (SetResult(result)) {
    result->bound_check_result = IterBoundCheck::kInbound;
  }
  return Status::OK();
}

Status PgmIndexIterator::SeekAndGetResult(const Slice& target,
                                          IterateResult* result,
                                          const SeekContext& context) {
  if (prepared_) {
    while (current_scan_idx_ < scan_opts_.size()) {
      const auto& opts = scan_opts_[current_scan_idx_];
      if (opts.range.limit.has_value() &&
          comparator_->Compare(target, opts.range.limit.value()) >= 0) {
        current_scan_idx_++;
      } else {
        break;
      }
    }
  }
  pos_ = reader_->Seek(target, context.target_tag);
  if (SetResult(result)) {
    result->bound_check_result = CheckBounds(target);
  }
  return Status::OK();
}

Status PgmIndexIterator::NextAndGetResult(IterateResult* result) {
  if (pos_ >= reader_->NumEntries()) {
    SetResult(result);
    return Status::OK();
  }
  const Slice prev = reader_->GetEntry(pos_).separator;
  ++pos_;
  if (SetResult(result)) {
    result->bound_check_result = CheckBounds(prev);
  }
  return Status::OK();
}

UserDefinedIndexBuilder::BlockHandle PgmIndexIterator::value() {
  const auto& e = reader_->GetEntry(pos_);
  return UserDefinedIndexBuilder::BlockHandle{e.offset, e.size};
}

IterBoundCheck PgmIndexIterator::CheckBounds(const Slice& reference_key) const {
  if (!prepared_ || scan_opts_.empty()) {
    return IterBoundCheck::kInbound;
  }
  if (current_scan_idx_ >= scan_opts_.size()) {
    return IterBoundCheck::kOutOfBound;
  }
  const auto& opts = scan_opts_[current_scan_idx_];
  // Separators are upper bounds of block contents, so compare the reference
  // key (seek target / previous separator), not the current separator.
  if (opts.range.limit.has_value() &&
      comparator_->Compare(reference_key, opts.range.limit.value()) >= 0) {
    return IterBoundCheck::kOutOfBound;
  }
  return IterBoundCheck::kInbound;
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
