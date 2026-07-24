#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "dbmf/dbmf.hpp"
#include "gguf/gguf.hpp"

namespace {

std::uint64_t file_size(const char* path) {
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) return 0;
  std::fseek(f, 0, SEEK_END);
  const std::int64_t n = std::ftell(f);
  std::fclose(f);
  return n < 0 ? 0 : static_cast<std::uint64_t>(n);
}

}  // namespace

int main(int argc, char** argv) {
  bool compress = false;
  const char* in = nullptr;
  const char* out = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--compress") == 0)
      compress = true;
    else if (in == nullptr)
      in = argv[i];
    else if (out == nullptr)
      out = argv[i];
  }

  if (in == nullptr || out == nullptr) {
    std::fprintf(stderr, "usage: %s [--compress] <in.gguf> <out.dbmf>\n", argv[0]);
    return 2;
  }

  auto loaded = dbinfer::gguf::load(in);
  if (!loaded) {
    std::fprintf(stderr, "error: load gguf: %s\n",
                 dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }

  dbinfer::dbmf::ConvertOptions opts;
  opts.compress = compress;
  auto ok = dbinfer::dbmf::convert(*loaded, out, opts);
  if (!ok) {
    std::fprintf(stderr, "error: convert: %s\n", dbinfer::gguf::to_string(ok.error()).c_str());
    return 1;
  }

  const std::uint64_t gsz = file_size(in);
  const std::uint64_t dsz = file_size(out);
  std::printf("wrote %s (%llu tensors, %s)\n", out,
              static_cast<std::uint64_t>(loaded->tensors.size()), compress ? "compressed" : "raw");
  std::printf("gguf %llu bytes -> dbmf %llu bytes (%.2f%%)\n", static_cast<std::uint64_t>(gsz),
              static_cast<std::uint64_t>(dsz),
              gsz > 0 ? 100.0 * static_cast<double>(dsz) / static_cast<double>(gsz) : 0.0);
  return 0;
}
