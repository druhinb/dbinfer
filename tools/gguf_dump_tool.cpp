#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <variant>

#include "gguf/gguf.hpp"

namespace {

using dbinfer::gguf::MetaArray;
using dbinfer::gguf::MetaValue;

std::string scalar_to_string(const MetaValue& mv) {
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, bool>) {
          return v ? "true" : "false";
        } else if constexpr (std::is_same_v<T, MetaArray>) {
          return "<array>";
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
          return std::to_string(v);
        } else {
          return std::to_string(static_cast<std::int64_t>(v));
        }
      },
      mv.value);
}

void print_meta_value(const MetaValue& mv) {
  if (const auto* arr = std::get_if<MetaArray>(&mv.value)) {
    std::printf("[%s] x %zu", dbinfer::gguf::to_string(arr->elem_type), arr->values.size());
    const std::size_t preview = arr->values.size() < 6 ? arr->values.size() : 6;
    if (preview > 0) {
      std::printf(" = [");
      for (std::size_t i = 0; i < preview; ++i) {
        if (i > 0) std::printf(", ");
        std::printf("%s", scalar_to_string(arr->values[i]).c_str());
      }
      if (arr->values.size() > preview) std::printf(", ...");
      std::printf("]");
    }
  } else {
    std::printf("%s", scalar_to_string(mv).c_str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
    return 2;
  }

  auto loaded = dbinfer::gguf::load(argv[1]);
  if (!loaded) {
    std::fprintf(stderr, "%s\n", dbinfer::gguf::to_string(loaded.error()).c_str());
    return 1;
  }
  const dbinfer::gguf::GgufFile& f = *loaded;

  std::printf("version: %u\n", f.version);
  std::printf("alignment: %llu\n", static_cast<std::uint64_t>(f.alignment));
  std::printf("tensor_count: %zu\n", f.tensors.size());
  std::printf("kv_count: %zu\n", f.metadata.size());

  std::printf("\nmetadata:\n");
  for (const auto& kv : f.metadata) {
    std::printf("  %s = ", kv.first.c_str());
    print_meta_value(kv.second);
    std::printf("\n");
  }

  std::printf("\ntensors:\n");
  for (const auto& t : f.tensors) {
    std::printf("  %-32s %-6s shape=[", t.name.c_str(), dbinfer::gguf::to_string(t.type));
    for (std::uint32_t d = 0; d < t.n_dims; ++d) {
      if (d > 0) std::printf(", ");
      std::printf("%llu", static_cast<std::uint64_t>(t.shape[d]));
    }
    std::printf("] offset=%llu nbytes=%llu\n", static_cast<std::uint64_t>(t.offset),
                static_cast<std::uint64_t>(t.nbytes));
  }

  return 0;
}
