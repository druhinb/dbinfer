#ifndef DBINFER_TENSOR_THREAD_POOL_HPP
#define DBINFER_TENSOR_THREAD_POOL_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace dbinfer::tensor {

// fixed pool of worker threads that spin on an atomic generation counter until
// a task is published, then divide one index range among themselves and the
// caller. workers fall back to a condition variable only after a bounded spin,
// so the tightly-spaced barriers of one decode step stay lock-free while an
// idle pool still parks instead of burning cores.
class ThreadPool {
public:
  using TaskFn = void (*)(void *, std::size_t, std::size_t);

  explicit ThreadPool(std::size_t thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  std::size_t worker_count() const { return count_; }

  // splits [0,n) into aligned contiguous ranges run across the pool, blocking.
  void parallel_for(std::size_t n, std::size_t align, void *ctx, TaskFn fn);

private:
  void worker_loop(std::size_t idx);
  std::size_t wait_for_work(std::size_t seen);

  std::size_t count_;
  // task fields are published before the generation_ release store and read
  // after the matching acquire load, so workers see them without the lock.
  TaskFn fn_ = nullptr;
  void *ctx_ = nullptr;
  std::size_t n_ = 0;
  std::size_t align_ = 0;
  std::atomic<std::size_t> generation_{0};
  std::atomic<std::size_t> active_{0};
  std::atomic<bool> stop_{false};
  std::mutex mtx_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::jthread> workers_;
};

// fn lives on the caller stack for the blocking duration of parallel_for.
template <class Fn> void parallel_for(ThreadPool &pool, std::size_t n, std::size_t align, Fn &&fn) {
  auto trampoline = [](void *ctx, std::size_t begin, std::size_t end) {
    (*static_cast<std::remove_reference_t<Fn> *>(ctx))(begin, end);
  };
  pool.parallel_for(n, align, &fn, trampoline);
}

// lazily built global pool resolving count from env then p_core_count.
ThreadPool &thread_pool();

// rebuilds the global pool. call between forward runs, not concurrency-safe.
void configure_thread_count(std::size_t n);

} // namespace dbinfer::tensor

#endif // DBINFER_TENSOR_THREAD_POOL_HPP
