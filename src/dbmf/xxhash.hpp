#ifndef DBINFER_DBMF_XXHASH_HPP
#define DBINFER_DBMF_XXHASH_HPP

#include <cstddef>
#include <cstdint>

namespace dbinfer::dbmf {

// XXH64 (public-domain algorithm by Yann Collet), one-shot. reads are
// memcpy-based so an unaligned mmap pointer is safe to pass.
std::uint64_t xxhash64(const std::byte* data, std::size_t len, std::uint64_t seed = 0);

}  // namespace dbinfer::dbmf

#endif  // DBINFER_DBMF_XXHASH_HPP
