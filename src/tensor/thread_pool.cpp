#include "tensor/thread_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>

#include "tensor/cpu.hpp"

namespace dbinfer::tensor {

namespace {

// hint to the core inside a wait loop. keeps a worker hot between decode
// barriers without a lock.
inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  asm volatile("pause" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

// spin budget before parking. sized to cover the gap between the ~8 matvec
// barriers of one decode layer (single-digit microseconds each) while still
// parking a pool left idle between forward passes. tune via BENCH.log.
constexpr int kSpinBudget = 8192;

std::pair<std::size_t, std::size_t> split(std::size_t idx, std::size_t count, std::size_t n,
                                          std::size_t align) {
  const std::size_t blocks = (n + align - 1) / align;
  const std::size_t base = blocks / count;
  const std::size_t rem = blocks % count;
  const std::size_t b0 = idx * base + std::min(idx, rem);
  const std::size_t b1 = b0 + base + (idx < rem ? 1u : 0u);
  return {std::min(b0 * align, n), std::min(b1 * align, n)};
}

void run_range(std::size_t idx, std::size_t count, ThreadPool::TaskFn fn, void* ctx, std::size_t n,
               std::size_t align) {
  const auto [begin, end] = split(idx, count, n, align);
  if (begin < end) fn(ctx, begin, end);
}

std::size_t default_thread_count() {
  if (const char* e = std::getenv("DBINFER_THREADS")) {
    char* end = nullptr;
    const std::int64_t v = std::strtol(e, &end, 10);
    if (end != e && *end == '\0' && v > 0) return static_cast<std::size_t>(v);
  }
  return p_core_count();
}

std::unique_ptr<ThreadPool>& pool_slot() {
  static std::unique_ptr<ThreadPool> slot;
  return slot;
}

}  // namespace

ThreadPool::ThreadPool(std::size_t thread_count) : count_(thread_count == 0 ? 1 : thread_count) {
  const std::size_t spawn = count_ - 1;
  workers_.reserve(spawn);
  for (std::size_t i = 0; i < spawn; ++i)
    workers_.emplace_back([this, idx = i + 1] { worker_loop(idx); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard lk(mtx_);
    stop_.store(true, std::memory_order_release);
  }
  work_cv_.notify_all();
}

// blocks until generation_ advances past seen (or stop_), spinning first so a
// worker catches the next barrier without a syscall, then parking on work_cv_.
std::size_t ThreadPool::wait_for_work(std::size_t seen) {
  for (int spins = 0; spins < kSpinBudget; ++spins) {
    const std::size_t g = generation_.load(std::memory_order_acquire);
    if (g != seen || stop_.load(std::memory_order_acquire)) return g;
    cpu_relax();
  }

  std::unique_lock lk(mtx_);
  work_cv_.wait(lk, [&] {
    return stop_.load(std::memory_order_acquire) ||
           generation_.load(std::memory_order_acquire) != seen;
  });
  return generation_.load(std::memory_order_acquire);
}

void ThreadPool::worker_loop(std::size_t idx) {
  std::size_t seen = 0;
  while (true) {
    seen = wait_for_work(seen);
    if (stop_.load(std::memory_order_acquire)) return;
    // published under the caller's lock before the generation_ release above.
    run_range(idx, count_, fn_, ctx_, n_, align_);
    if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // last worker out: wake a caller that has parked on done_cv_. taking the
      // lock here serializes against the caller's park so the wake is not lost.
      std::lock_guard lk(mtx_);
      done_cv_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(std::size_t n, std::size_t align, void* ctx, TaskFn fn) {
  if (align == 0) align = 1;
  if (count_ == 1 || n <= align) {
    fn(ctx, 0, n);
    return;
  }

  {
    // publish the task under the lock so a worker mid-park either sees the new
    // generation before waiting or is woken by the notify below.
    std::lock_guard lk(mtx_);
    fn_ = fn;
    ctx_ = ctx;
    n_ = n;
    align_ = align;
    active_.store(workers_.size(), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
  }

  work_cv_.notify_all();
  run_range(0, count_, fn, ctx, n, align);

  // join: spin on the worker count before parking, matching the worker side.
  for (int spins = 0; spins < kSpinBudget; ++spins) {
    if (active_.load(std::memory_order_acquire) == 0) return;
    cpu_relax();
  }
  std::unique_lock lk(mtx_);
  done_cv_.wait(lk, [&] { return active_.load(std::memory_order_acquire) == 0; });
}

ThreadPool& thread_pool() {
  std::unique_ptr<ThreadPool>& slot = pool_slot();
  if (!slot) slot = std::make_unique<ThreadPool>(default_thread_count());
  return *slot;
}

void configure_thread_count(std::size_t n) {
  std::unique_ptr<ThreadPool>& slot = pool_slot();
  slot.reset();
  slot = std::make_unique<ThreadPool>(n == 0 ? default_thread_count() : n);
}

}  // namespace dbinfer::tensor
