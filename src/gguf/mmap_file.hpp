#ifndef DBINFER_GGUF_MMAP_FILE_HPP
#define DBINFER_GGUF_MMAP_FILE_HPP

#include <cstddef>
#include <expected>
#include <string>
#include <string_view>

namespace dbinfer::gguf {

struct Error;

// read-only mmap of a model file. Move-only: the mapping is unmapped exactly
// once, on destruction of whichever MappedFile currently owns it. Model
// weights are read straight out of data() for the file's whole lifetime, so
// nothing may outlive the MappedFile it came from.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  // opens path read-only and maps its full contents. Rejects empty files
  // (mmap of a zero-length file is undefined) up front.
  static std::expected<MappedFile, Error> open(std::string_view path);

  // writable anonymous mapping, for tensor bytes decoded once at load time from
  // a compressed container. filled via data_mut() before any reader sees it.
  static std::expected<MappedFile, Error> anonymous(std::size_t size);

  const std::byte* data() const { return static_cast<const std::byte*>(data_); }
  std::byte* data_mut() { return static_cast<std::byte*>(data_); }
  std::size_t size() const { return size_; }

 private:
  MappedFile(void* data, std::size_t size) : data_(data), size_(size) {}

  void reset() noexcept;

  void* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace dbinfer::gguf

#endif
