#ifndef DBINFER_TENSOR_TENSOR_HPP
#define DBINFER_TENSOR_TENSOR_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dbinfer::tensor {

// non-owning view over a Tensor's storage, for passing shape+data together
// without copying.
struct TensorView {
  const float* data = nullptr;
  std::array<std::size_t, 4> shape{1, 1, 1, 1};
  std::uint32_t n_dims = 0;

  std::size_t n_elements() const {
    std::size_t n = 1;
    for (std::uint32_t d = 0; d < n_dims; ++d) n *= shape[d];
    return n;
  }
};

// owning, up-to-4D float buffer sized from shape/n_dims at construction and
// zero-initialized.
class Tensor {
 public:
  Tensor() = default;

  Tensor(std::array<std::size_t, 4> shape, std::uint32_t n_dims) : shape_(shape), n_dims_(n_dims) {
    std::size_t n = 1;
    for (std::uint32_t d = 0; d < n_dims_; ++d) n *= shape_[d];
    data_.assign(n, 0.0f);
  }

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
  std::size_t n_elements() const { return data_.size(); }
  const std::array<std::size_t, 4>& shape() const { return shape_; }
  std::uint32_t n_dims() const { return n_dims_; }

  TensorView view() const { return TensorView{data_.data(), shape_, n_dims_}; }

 private:
  std::vector<float> data_;
  std::array<std::size_t, 4> shape_{1, 1, 1, 1};
  std::uint32_t n_dims_ = 0;
};

}  // namespace dbinfer::tensor

#endif  // DBINFER_TENSOR_TENSOR_HPP
