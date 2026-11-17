//  Copyright (c) Meta Platforms, Inc. and affiliates.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "utilities/pgm_index/pgm_index_factory.h"

#include "util/coding.h"
#include <algorithm>
#include <cassert>
#include <exception>
#include <limits>

#include "pgm/pgm_index.hpp"

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

// ---- Helpers ----
uint64_t KeyToU64(const Slice& key) {
  uint64_t v = 0;
  const size_t n = std::min<size_t>(key.size(), 8);
  for (size_t i = 0; i < 8; ++i) {
    // Zero pad on the right so shorter keys sort before their extensions.
    v = (v << 8) | (i < n ? static_cast<unsigned char>(key[i]) : 0u);
  }
  // PGM reserves numeric_limits<uint64_t>::max() as a sentinel.
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  return v == kMax ? kMax - 1 : v;
}

// PGM model: error at most 16 positions at the bottom level, 4 above.
struct PgmModelHolder {
  using Model = pgm::PGMIndex<uint64_t, 16, 4, float>;
  explicit PgmModelHolder(const std::vector<uint64_t>& keys)
      : model(keys.begin(), keys.end()) {}
  Model model;
};

// ---- Reader ----
PgmIndexReader::PgmIndexReader(const Comparator* comparator)
    : comparator_(comparator) {}

PgmIndexReader::~PgmIndexReader() = default;

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

  // Distinct 8-byte prefixes, and where each group of equal prefixes starts.
  prefixes_.clear();
  group_start_.clear();
  for (size_t i = 0; i < entries_.size(); ++i) {
    const uint64_t k = KeyToU64(entries_[i].separator);
    if (prefixes_.empty() || prefixes_.back() != k) {
      if (!prefixes_.empty() && k < prefixes_.back()) {
        return Status::Corruption("PGM index: separators not sorted");
      }
      prefixes_.push_back(k);
      group_start_.push_back(static_cast<uint32_t>(i));
    }
  }
  group_start_.push_back(static_cast<uint32_t>(entries_.size()));

  // Train the PGM model on the distinct prefixes.
  pgm_.reset();
  if (!prefixes_.empty()) {
    try {
      pgm_ = std::make_unique<PgmModelHolder>(prefixes_);
    } catch (const std::exception& ex) {
      return Status::Corruption(std::string("PGM index: training failed: ") +
                                ex.what());
    }
  }
  return Status::OK();
}

size_t PgmIndexReader::NumSegments() const {
  return pgm_ ? pgm_->model.segments_count() : 0;
}

std::unique_ptr<UserDefinedIndexIterator> PgmIndexReader::NewIterator(
    const ReadOptions& /*read_options*/) {
  return std::make_unique<PgmIndexIterator>(this, comparator_);
}

size_t PgmIndexReader::ApproximateMemoryUsage() const {
  return data_size_ + entries_.capacity() * sizeof(Entry) +
         prefixes_.capacity() * sizeof(uint64_t) +
         group_start_.capacity() * sizeof(uint32_t) +
         (pgm_ ? sizeof(PgmModelHolder) + pgm_->model.size_in_bytes() : 0);
}

bool PgmIndexReader::EntryBefore(size_t i, const Slice& target,
                                 uint64_t target_tag) const {
  const Entry& e = entries_[i];
  const int c = comparator_->Compare(e.separator, target);
  // Higher tag sorts first, so an equal key with a higher tag is before.
  return c < 0 || (c == 0 && target_tag < e.tag);
}

// ---- Reader::Seek (PGM window + search inside the prefix group) ----
size_t PgmIndexReader::Seek(const Slice& target, uint64_t target_tag) const {
  const size_t num_groups = prefixes_.size();
  if (num_groups == 0) {
    return 0;
  }
  const uint64_t k = KeyToU64(target);

  // 1) The model gives an approximate position and a small window [lo, hi)
  //    that must contain the first prefix >= k.
  const auto approx = pgm_->model.search(k);
  const auto first = prefixes_.begin() + approx.lo;
  const auto last = prefixes_.begin() + approx.hi;
  size_t g =
      static_cast<size_t>(std::lower_bound(first, last, k) - prefixes_.begin());
  // The model guarantees the window. Guard anyway, so a model bug can never
  // return a wrong block.
  if ((g > 0 && prefixes_[g - 1] >= k) ||
      (g < num_groups && prefixes_[g] < k)) {
    assert(false);
    g = static_cast<size_t>(
        std::lower_bound(prefixes_.begin(), prefixes_.end(), k) -
        prefixes_.begin());
  }
  if (g == num_groups) {
    return entries_.size();
  }

  // 2) Groups before g have prefix < k, groups after g have prefix > k. If
  //    the prefix of group g is larger than k, the whole group is after the
  //    target. Otherwise compare full keys (and tags) inside the group.
  size_t lo = group_start_[g];
  size_t hi = group_start_[g + 1];
  if (prefixes_[g] != k) {
    return lo;
  }
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (EntryBefore(mid, target, target_tag)) {
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
