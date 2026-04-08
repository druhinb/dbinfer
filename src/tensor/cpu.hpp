#ifndef DBINFER_TENSOR_CPU_HPP
#define DBINFER_TENSOR_CPU_HPP

#include <cstddef>

namespace dbinfer::tensor {

struct CpuFeatures {
  bool dotprod;
  bool i8mm;
};

// runtime arm feature detection, computed once on first call.
const CpuFeatures &cpu_features();

// physical performance-core count from hw.perflevel0.physicalcpu, 1 on failure.
std::size_t p_core_count();

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_CPU_HPP
