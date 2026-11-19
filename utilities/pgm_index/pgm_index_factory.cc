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
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pgm/pgm_index.hpp"

namespace ROCKSDB_NAMESPACE {
namespace pgm_index {

// PGM model that can be written to and read from the index block.
class PgmModel : public pgm::PGMIndex<uint64_t, 16, 4, float> {
 public:
  PgmModel() = default;
  explicit PgmModel(const std::vector<uint64_t>& keys)
      : pgm::PGMIndex<uint64_t, 16, 4, float>(keys.begin(), keys.end()) {}

  void EncodeTo(std::string* dst) const {
    PutVarint64(dst, n);
    PutFixed64(dst, first_key);
    PutVarint64(dst, segments.size());
    dst->append(reinterpret_cast<const char*>(segments.data()),
                segments.size() * sizeof(Segment));
    PutVarint64(dst, levels_offsets.size());
    for (size_t off : levels_offsets) {
      PutVarint64(dst, off);
    }
  }

  bool DecodeFrom(Slice* input) {
    uint64_t n64 = 0, first = 0, num_segments = 0, num_levels = 0;
    if (!GetVarint64(input, &n64) || !GetFixed64(input, &first) ||
        !GetVarint64(input, &num_segments) ||
        num_segments > input->size() / sizeof(Segment)) {
      return false;
    }
    segments.resize(num_segments);
    memcpy(segments.data(), input->data(), num_segments * sizeof(Segment));
    input->remove_prefix(num_segments * sizeof(Segment));
    if (!GetVarint64(input, &num_levels) || num_levels > input->size()) {
      return false;
    }
    levels_offsets.assign(num_levels, 0);
    for (uint64_t i = 0; i < num_levels; ++i) {
      uint64_t off = 0;
      if (!GetVarint64(input, &off)) return false;
      levels_offsets[i] = off;
    }
    n = n64;
    first_key = first;
    return true;
  }
};

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
  PutFixed32(&buffer_, 2);  // format version 2: entries, then the model
  PutVarint64(&buffer_, entries_.size());
  for (const Entry& e : entries_) {
    PutLengthPrefixedSlice(&buffer_, Slice(e.separator));
    PutVarint64(&buffer_, e.tag);
    PutVarint64(&buffer_, e.offset);
    PutVarint64(&buffer_, e.size);
  }
  // Distinct 8-byte prefixes (the same as the reader builds).
  std::vector<uint64_t> prefix_keys;
  prefix_keys.reserve(entries_.size());
  for (const Entry& e : entries_) {
    uint64_t k = KeyToU64(Slice(e.separator));
    if (prefix_keys.empty() || prefix_keys.back() != k) {
      prefix_keys.push_back(k);
    }
  }
  if (prefix_keys.empty()) {
    buffer_.push_back(0);  // no model
  } else {
    buffer_.push_back(1);  // a model follows
    auto t0 = std::chrono::steady_clock::now();
    PgmModel model(prefix_keys);
    auto t1 = std::chrono::steady_clock::now();
    model.EncodeTo(&buffer_);
    fprintf(stderr, "PGM_TRAIN keys=%zu segments=%zu bytes=%zu micros=%lld\n",
            prefix_keys.size(), model.segments_count(), model.size_in_bytes(),
            static_cast<long long>(
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                    .count()));
  }
  *index_contents = Slice(buffer_);
  finished_ = true;
  return Status::OK();
}

uint64_t PgmIndexBuilder::EstimatedSize() const {
  if (finished_) {
    return buffer_.size();
  }
  // Upper bound: version + count + per entry (length + 3 varints <= 35 B),
  // plus the model flag and at most 48 B per entry for the model.
  uint64_t total = 4 + 10 + 1 + 64;
  for (const Entry& e : entries_) {
    total += e.separator.size() + 35 + 48;
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
  using Model = PgmModel;
  PgmModelHolder() = default;
  explicit PgmModelHolder(const std::vector<uint64_t>& keys) : model(keys) {}
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
  if (!GetFixed32(&input, &version) || (version != 1 && version != 2)) {
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

  // Version 2: the trained model follows the entries.
  std::unique_ptr<PgmModelHolder> stored_model;
  if (version == 2) {
    if (input.empty()) {
      return Status::Corruption("PGM index: missing model flag");
    }
    const char has_model = input[0];
    input.remove_prefix(1);
    if (has_model == 1) {
      stored_model = std::make_unique<PgmModelHolder>();
      if (!stored_model->model.DecodeFrom(&input)) {
        return Status::Corruption("PGM index: bad model");
      }
    }
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

  // Use the stored model if there is one, otherwise train on the prefixes.
  pgm_.reset();
  if (stored_model) {
    pgm_ = std::move(stored_model);
  } else if (!prefixes_.empty()) {
    try {
      pgm_ = std::make_unique<PgmModelHolder>(prefixes_);
    } catch (const std::exception& ex) {
      return Status::Corruption(std::string("PGM index: training failed: ") +
                                ex.what());
    }
  }
  if (std::getenv("PGM_MEM_LOG") != nullptr) {
    fprintf(stderr,
            "PGM_MEM data=%zu entries=%zu prefixes=%zu groups=%zu model=%zu "
            "total=%zu\n",
            data_size_, entries_.capacity() * sizeof(Entry),
            prefixes_.capacity() * sizeof(uint64_t),
            group_start_.capacity() * sizeof(uint32_t),
            pgm_ ? pgm_->model.size_in_bytes() : size_t{0},
            ApproximateMemoryUsage());
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
