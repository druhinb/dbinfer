#ifndef DBINFER_TRY_HPP
#define DBINFER_TRY_HPP

#include <expected>
#include <utility>

// Unwrap a std::expected, or return its error from the enclosing function.
// Statement-expression is a clang/gcc extension; this codebase is clang-only.
#define TRY(expr)                                                                                  \
  ({                                                                                               \
    auto _dbinfer_try = (expr);                                                                    \
    if (!_dbinfer_try)                                                                             \
      return std::unexpected(std::move(_dbinfer_try).error());                                     \
    *std::move(_dbinfer_try);                                                                      \
  })

#endif // DBINFER_TRY_HPP
