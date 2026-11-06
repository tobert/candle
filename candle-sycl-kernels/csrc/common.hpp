// Shared helpers for the elementwise SYCL kernels.
#pragma once
#include "candle_sycl.h"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <atomic>
#include <cstdio>
#include <exception>
#include <mutex>

using bf16 = sycl::ext::oneapi::bfloat16;
using f16 = sycl::half;

// Real definition of the handle that candle_sycl.h forward-declares for C.
struct CandleSyclQueue {
  // Execution errors that a wait saw but could not return to anyone: the
  // internal flushes (`candle_sycl_wait`) cannot propagate an error, and an
  // async device-to-device copy or memset fails only at a later wait. The
  // first such status is kept here until a reporting point (synchronize, a
  // host copy) hands it back, once.
  std::atomic<int> pending{0};
  // Set by the queue's async handler, which runs inside wait_and_throw, so
  // the wait that invoked it can return the failure. Without a handler SYCL's
  // default one terminates the process.
  std::atomic<int> async_err{0};
  // Test hook (candle_sycl_debug_fail_next_wait): the next wait fails with
  // this status.
  std::atomic<int> inject{0};
  sycl::queue q;
  // Device-side guard: two int32 slots. `guard[0]` is the flag: kernels that
  // validate an index (the indexed MoE expert id) store a reason code there
  // instead of reading out of bounds. `guard[1]` is the snapshot a
  // read-and-clear kernel exchanges the flag into, so the read and the clear
  // are one device-side atomic and a trip landing between them is never wiped
  // unread. Reported (and cleared) by candle_sycl_synchronize and by every
  // device-to-host copy, so the failure surfaces at the next host-visible point
  // without a host round trip per launch. Null if allocation failed, in which
  // case kernels skip the guard.
  int32_t *guard;
  // Serializes the read-and-clear: `guard[1]` is shared by every caller.
  std::mutex guard_mu;
  explicit CandleSyclQueue(const sycl::device &d)
      : q(d,
          [this](sycl::exception_list errs) {
            for (auto &e : errs) {
              try {
                std::rethrow_exception(e);
              } catch (const std::exception &ex) {
                std::fprintf(stderr, "candle-sycl: async error: %s\n", ex.what());
              } catch (...) {
                std::fprintf(stderr, "candle-sycl: async error (non-standard exception)\n");
              }
            }
            int zero = 0;
            async_err.compare_exchange_strong(zero, CANDLE_SYCL_ERR_EXCEPTION);
          },
          sycl::property::queue::in_order{}),
        guard(static_cast<int32_t *>(sycl::malloc_device(2 * sizeof(int32_t), q))) {
    if (guard) {
      q.memset(guard, 0, 2 * sizeof(int32_t)).wait();
    }
  }
  ~CandleSyclQueue() {
    if (guard) {
      sycl::free(guard, q);
    }
  }
};

// Compute the flat index into a strided source for dense output position `i`.
inline int64_t strided_index(int64_t i, const CandleSyclLayout &l) {
  if (l.num_dims == 0) {
    return i;
  }
  int64_t idx = l.offset;
  for (int d = (int)l.num_dims - 1; d >= 0; --d) {
    int64_t dim = l.dims[d];
    int64_t coord = i % dim;
    i /= dim;
    idx += coord * l.strides[d];
  }
  return idx;
}

inline bool is_dense(const CandleSyclLayout &l) { return l.num_dims == 0; }

// Accumulator type for elementwise math: widen fp16/bf16 to float.
template <typename T> struct Acc {
  using type = float;
};
template <> struct Acc<double> {
  using type = double;
};
template <typename T> inline auto to_acc(T v) {
  return static_cast<typename Acc<T>::type>(v);
}

// Dispatch a templated functor over the runtime dtype. F is a generic lambda
// `[&]<typename T>() { ... }`.
template <typename F>
int dispatch_dtype(CandleSyclDType dt, F &&f) {
  switch (dt) {
  case CANDLE_SYCL_U8: return f.template operator()<uint8_t>();
  case CANDLE_SYCL_U32: return f.template operator()<uint32_t>();
  case CANDLE_SYCL_I64: return f.template operator()<int64_t>();
  case CANDLE_SYCL_F16: return f.template operator()<f16>();
  case CANDLE_SYCL_BF16: return f.template operator()<bf16>();
  case CANDLE_SYCL_F32: return f.template operator()<float>();
  case CANDLE_SYCL_F64: return f.template operator()<double>();
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}

// Float-only dispatch (for ops undefined on integers).
template <typename F>
int dispatch_float(CandleSyclDType dt, F &&f) {
  switch (dt) {
  case CANDLE_SYCL_F16: return f.template operator()<f16>();
  case CANDLE_SYCL_BF16: return f.template operator()<bf16>();
  case CANDLE_SYCL_F32: return f.template operator()<float>();
  case CANDLE_SYCL_F64: return f.template operator()<double>();
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
