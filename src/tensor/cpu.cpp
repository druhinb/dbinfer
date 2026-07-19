#include "tensor/cpu.hpp"

#include <cstddef>
#include <cstdint>
#include <sys/sysctl.h>

namespace dbinfer::tensor {

namespace {

bool sysctl_flag(const char *name) {
  int value = 0;
  std::size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0)
    return false;
  return value != 0;
}

std::uint32_t sysctl_u32(const char *name) {
  std::uint32_t value = 0;
  std::size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0)
    return 0;
  return value;
}

} // namespace

const CpuFeatures &cpu_features() {
  static const CpuFeatures features{
      sysctl_flag("hw.optional.arm.FEAT_DotProd"),
      sysctl_flag("hw.optional.arm.FEAT_I8MM"),
  };
  return features;
}

std::size_t p_core_count() {
  const std::uint32_t n = sysctl_u32("hw.perflevel0.physicalcpu");
  return n == 0 ? 1 : n;
}

} // namespace dbinfer::tensor
