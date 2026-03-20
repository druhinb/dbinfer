#ifndef DBINFER_TENSOR_CPU_HPP
#define DBINFER_TENSOR_CPU_HPP

namespace dbinfer::tensor {

struct CpuFeatures {
  bool dotprod;
  bool i8mm;
};

// runtime arm feature detection, computed once on first call.
const CpuFeatures &cpu_features();

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_CPU_HPP
