#include "tensor/thread_pool.hpp"

#include "tensor/cpu.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <utility>

namespace dbinfer::tensor {

namespace {

std::pair<std::size_t, std::size_t> split(std::size_t idx, std::size_t count, std::size_t n,
                                          std::size_t align) {
  const std::size_t blocks = (n + align - 1) / align;
  const std::size_t base = blocks / count;
  const std::size_t rem = blocks % count;
  const std::size_t b0 = idx * base + std::min(idx, rem);
  const std::size_t b1 = b0 + base + (idx < rem ? 1u : 0u);
  return {std::min(b0 * align, n), std::min(b1 * align, n)};
}

void run_range(std::size_t idx, std::size_t count, ThreadPool::TaskFn fn, void *ctx, std::size_t n,
               std::size_t align) {
  const auto [begin, end] = split(idx, count, n, align);
  if (begin < end)
    fn(ctx, begin, end);
}

std::size_t default_thread_count() {
  if (const char *e = std::getenv("DBINFER_THREADS")) {
    char *end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end != e && *end == '\0' && v > 0)
      return static_cast<std::size_t>(v);
  }
  return p_core_count();
}

std::unique_ptr<ThreadPool> &pool_slot() {
  static std::unique_ptr<ThreadPool> slot;
  return slot;
}

} // namespace

ThreadPool::ThreadPool(std::size_t thread_count) : count_(thread_count == 0 ? 1 : thread_count) {
  const std::size_t spawn = count_ - 1;
  workers_.reserve(spawn);
  for (std::size_t i = 0; i < spawn; ++i)
    workers_.emplace_back([this, idx = i + 1] { worker_loop(idx); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard lk(mtx_);
    stop_ = true;
  }
  work_cv_.notify_all();
}

void ThreadPool::worker_loop(std::size_t idx) {
  std::size_t seen = 0;
  while (true) {
    TaskFn fn = nullptr;
    void *ctx = nullptr;
    std::size_t n = 0;
    std::size_t align = 0;
    std::size_t count = 0;
    {
      std::unique_lock lk(mtx_);
      work_cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
      if (stop_)
        return;
      seen = generation_;
      fn = fn_;
      ctx = ctx_;
      n = n_;
      align = align_;
      count = count_;
    }
    run_range(idx, count, fn, ctx, n, align);
    {
      std::lock_guard lk(mtx_);
      if (--active_ == 0)
        done_cv_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(std::size_t n, std::size_t align, void *ctx, TaskFn fn) {
  if (align == 0)
    align = 1;
  if (count_ == 1 || n <= align) {
    fn(ctx, 0, n);
    return;
  }
  {
    std::lock_guard lk(mtx_);
    fn_ = fn;
    ctx_ = ctx;
    n_ = n;
    align_ = align;
    active_ = workers_.size();
    ++generation_;
  }
  work_cv_.notify_all();
  run_range(0, count_, fn, ctx, n, align);
  {
    std::unique_lock lk(mtx_);
    done_cv_.wait(lk, [&] { return active_ == 0; });
  }
}

ThreadPool &thread_pool() {
  std::unique_ptr<ThreadPool> &slot = pool_slot();
  if (!slot)
    slot = std::make_unique<ThreadPool>(default_thread_count());
  return *slot;
}

void configure_thread_count(std::size_t n) {
  std::unique_ptr<ThreadPool> &slot = pool_slot();
  slot.reset();
  slot = std::make_unique<ThreadPool>(n == 0 ? default_thread_count() : n);
}

} // namespace dbinfer::tensor
