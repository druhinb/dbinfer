#include "gguf/gguf.hpp"

#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dbinfer::gguf {

MappedFile::MappedFile(MappedFile &&other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile &MappedFile::operator=(MappedFile &&other) noexcept {
  if (this != &other) {
    reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

MappedFile::~MappedFile() { reset(); }

void MappedFile::reset() noexcept {
  if (data_ != nullptr) {
    ::munmap(data_, size_);
    data_ = nullptr;
    size_ = 0;
  }
}

std::expected<MappedFile, Error> MappedFile::open(std::string_view path) {
  std::string path_str(path);

  int fd = ::open(path_str.c_str(), O_RDONLY);
  if (fd < 0) {
    return std::unexpected(Error{std::string("cannot open: ") + std::strerror(errno), path_str, 0});
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    Error e{std::string("fstat failed: ") + std::strerror(errno), path_str, 0};
    ::close(fd);
    return std::unexpected(std::move(e));
  }

  if (st.st_size == 0) {
    ::close(fd);
    return std::unexpected(Error{"file is empty", path_str, 0});
  }

  const auto size = static_cast<std::size_t>(st.st_size);
  void *addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    Error e{std::string("mmap failed: ") + std::strerror(errno), path_str, 0};
    ::close(fd);
    return std::unexpected(std::move(e));
  }

  ::close(fd);

  // weights are read in essentially random order during the forward pass, so
  // hint the kernel against readahead it would waste.
  ::posix_madvise(addr, size, POSIX_MADV_RANDOM);

  return MappedFile(addr, size);
}

} // namespace dbinfer::gguf
