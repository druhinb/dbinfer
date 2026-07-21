#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "gguf/gguf.hpp"

namespace dbinfer::gguf {

using std::string;
using std::unexpected;

MappedFile::MappedFile(MappedFile&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
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
  string path_str(path);

  int fd = ::open(path_str.c_str(), O_RDONLY);
  if (fd < 0) {
    return unexpected(Error{string("cannot open: ") + std::strerror(errno), path_str, 0});
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    Error e{string("fstat failed: ") + std::strerror(errno), path_str, 0};
    ::close(fd);
    return unexpected(std::move(e));
  }

  if (st.st_size == 0) {
    ::close(fd);
    return unexpected(Error{"file is empty", path_str, 0});
  }

  const auto size = static_cast<std::size_t>(st.st_size);
  void* addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (addr == MAP_FAILED) {
    Error e{string("mmap failed: ") + std::strerror(errno), path_str, 0};
    ::close(fd);
    return unexpected(std::move(e));
  }

  ::close(fd);

  // weights are read in essentially random order during the forward pass, so
  // hint the kernel against readahead it would waste.
  ::posix_madvise(addr, size, POSIX_MADV_RANDOM);

  return MappedFile(addr, size);
}

std::expected<MappedFile, Error> MappedFile::anonymous(std::size_t size) {
  if (size == 0) return unexpected(Error{"anonymous mapping of zero bytes", "", 0});
  void* addr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (addr == MAP_FAILED)
    return unexpected(Error{string("anonymous mmap failed: ") + std::strerror(errno), "", 0});
  return MappedFile(addr, size);
}

}  // namespace dbinfer::gguf
