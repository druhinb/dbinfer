#ifndef DBINFER_TENSOR_DEQUANT_HPP
#define DBINFER_TENSOR_DEQUANT_HPP

#include <cstdint>

namespace dbinfer::tensor {

// converts an IEEE-754 binary16 bit pattern to float32, exactly (including
// zero, subnormals, infinities, and NaN).
float f16_to_f32(std::uint16_t h);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_DEQUANT_HPP
