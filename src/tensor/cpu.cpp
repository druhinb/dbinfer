#include "tensor/cpu.hpp"

#include <cstddef>
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

} // namespace

const CpuFeatures &cpu_features() {
  static const CpuFeatures features{
      sysctl_flag("hw.optional.arm.FEAT_DotProd"),
      sysctl_flag("hw.optional.arm.FEAT_I8MM"),
  };
  return features;
}

} // namespace dbinfer::tensor
