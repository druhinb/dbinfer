#include "backend/backend.hpp"

#ifdef DBINFER_METAL
#include "backend/metal_backend.hpp"
#endif

#include <cstdlib>
#include <string_view>

namespace dbinfer::backend {

Backend *active_backend() {
  static Backend *const selected = [] -> Backend * {
    const char *sel = std::getenv("DBINFER_BACKEND");
    const std::string_view name = sel != nullptr ? sel : "";
#ifdef DBINFER_METAL
    if (name == "metal")
      return metal_backend();
#endif
    return nullptr;
  }();
  return selected;
}

} // namespace dbinfer::backend
