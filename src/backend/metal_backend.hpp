#ifndef DBINFER_BACKEND_METAL_BACKEND_HPP
#define DBINFER_BACKEND_METAL_BACKEND_HPP

#include "backend/backend.hpp"

#include <cstddef>

namespace dbinfer::backend {

// process singleton, or nullptr when no Metal device is present. the first call
// creates the device, command queue, and compiles the mul_mat shader from
// source. only declared when DBINFER_METAL is defined.
Backend *metal_backend();

// true when a page-aligned pointer of this size can back a zero-copy Metal
// buffer. exposed so a test can prove DBMF's 16 KiB alignment wraps no-copy
// while an unaligned mmap pointer falls back to a copy.
bool metal_can_wrap_nocopy(const void *ptr, std::size_t bytes);

} // namespace dbinfer::backend

#endif // DBINFER_BACKEND_METAL_BACKEND_HPP
