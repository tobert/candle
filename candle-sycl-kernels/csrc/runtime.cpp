// Host-side SYCL runtime: queue, USM, memcpy, device info.
#include "common.hpp"
#include <cstdlib>
#include <cstring>
#include <vector>

using sycl::queue;

namespace {
std::vector<sycl::device> gpu_devices() {
  std::vector<sycl::device> out;
  for (const auto &p : sycl::platform::get_platforms()) {
    for (const auto &d : p.get_devices(sycl::info::device_type::gpu)) {
      out.push_back(d);
    }
  }
  return out;
}
} // namespace

extern "C" {

int candle_sycl_device_count(void) {
  try {
    return (int)gpu_devices().size();
  } catch (...) {
    return 0;
  }
}

CandleSyclQueue *candle_sycl_queue_new(int ordinal) {
  try {
    auto devs = gpu_devices();
    if (ordinal < 0 || (size_t)ordinal >= devs.size()) {
      return nullptr;
    }
    return new CandleSyclQueue(devs[ordinal]);
  } catch (...) {
    return nullptr;
  }
}

void candle_sycl_queue_free(CandleSyclQueue *q) { delete q; }

void *candle_sycl_queue_native(CandleSyclQueue *q) { return &q->q; }

} // extern "C"

namespace {
// Atomically read and clear the guard flag, returning the status it encodes.
// Call after a host wait has shown the flag nonzero (`peeked`): the common
// clean case then costs one 4-byte copy riding on a wait the caller pays
// anyway, and only a trip pays for the exchange kernel. The exchange may find
// 0 if another thread took the code first; that thread reports it, so this
// one returns OK and each trip is reported exactly once.
int take_guard(CandleSyclQueue *q) {
  std::lock_guard<std::mutex> lock(q->guard_mu);
  int32_t *g = q->guard;
  q->q.single_task([=] {
    g[1] = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(g[0])
               .exchange(0);
  });
  int32_t code = 0;
  q->q.memcpy(&code, g + 1, sizeof(code)).wait_and_throw();
  return code ? CANDLE_SYCL_ERR_DEVICE_GUARD_BASE + code : CANDLE_SYCL_OK;
}
} // namespace

extern "C" {

int candle_sycl_synchronize(CandleSyclQueue *q) {
  // A device-side guard (e.g. an indexed MoE expert id outside the stack)
  // stores its reason code instead of reading out of bounds. Report it once,
  // clearing it so the queue stays usable.
  try {
    int32_t peek = 0;
    if (q->guard) {
      q->q.memcpy(&peek, q->guard, sizeof(peek));
    }
    q->q.wait_and_throw();
    return peek ? take_guard(q) : CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

int candle_sycl_wait(CandleSyclQueue *q) {
  // Wait without reading the guard flag: internal flushes (defer_free,
  // drain_pool, queue drop) must not consume a guard error the caller's
  // synchronize has not seen yet.
  try {
    q->q.wait_and_throw();
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

int candle_sycl_device_info(CandleSyclQueue *q, CandleSyclDeviceInfo *out) {
  try {
    auto d = q->q.get_device();
    auto name = d.get_info<sycl::info::device::name>();
    std::memset(out, 0, sizeof(*out));
    std::strncpy(out->name, name.c_str(), sizeof(out->name) - 1);
    auto driver = d.get_info<sycl::info::device::driver_version>();
    std::strncpy(out->driver_version, driver.c_str(), sizeof(out->driver_version) - 1);
    out->global_mem_bytes = d.get_info<sycl::info::device::global_mem_size>();
    out->max_compute_units = d.get_info<sycl::info::device::max_compute_units>();
    out->max_clock_khz = d.get_info<sycl::info::device::max_clock_frequency>() * 1000u;
    out->is_integrated = d.get_info<sycl::info::device::host_unified_memory>() ? 1 : 0;
    out->supports_fp16 = d.has(sycl::aspect::fp16) ? 1 : 0;
    out->supports_fp64 = d.has(sycl::aspect::fp64) ? 1 : 0;
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

namespace {
// oneAPI 2026.x defaults to the Level-Zero V2 adapter, which fails the first
// kernel submission from an in-order USM queue on Xe2 with
// UR_RESULT_ERROR_UNSUPPORTED_FEATURE. The UR loader reads this when it first
// loads adapters, so it must be set before any SYCL call; a static initialiser
// runs early enough, and `overwrite = 0` leaves a caller's setting alone.
struct UrLoaderDefaults {
  UrLoaderDefaults() { setenv("UR_LOADER_USE_LEVEL_ZERO_V2", "0", 0); }
};
const UrLoaderDefaults ur_loader_defaults;
} // namespace

void *candle_sycl_malloc(CandleSyclQueue *q, size_t bytes) {
  try {
    if (bytes == 0) {
      bytes = 1;
    }
    return sycl::malloc_device(bytes, q->q);
  } catch (...) {
    return nullptr;
  }
}

void candle_sycl_free(CandleSyclQueue *q, void *ptr) {
  if (ptr) {
    sycl::free(ptr, q->q);
  }
}

int candle_sycl_memcpy_htod(CandleSyclQueue *q, void *dst, const void *src, size_t bytes) {
  try {
    if (bytes) q->q.memcpy(dst, src, bytes).wait_and_throw();
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}
int candle_sycl_memcpy_dtoh(CandleSyclQueue *q, void *dst, const void *src, size_t bytes) {
  return candle_sycl_memcpy_htod(q, dst, src, bytes);
}
int candle_sycl_memcpy_dtod(CandleSyclQueue *q, void *dst, const void *src, size_t bytes) {
  // Device-to-device: the in-order queue orders this against every later
  // kernel, so waiting on the host here only serializes submits. Host copies
  // must still wait — the host buffer need not outlive the call.
  try {
    if (bytes) q->q.memcpy(dst, src, bytes);
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

int candle_sycl_memset(CandleSyclQueue *q, void *dst, int value, size_t bytes) {
  // As `candle_sycl_memcpy_dtod`: in-order, no host wait.
  try {
    if (bytes) q->q.memset(dst, value, bytes);
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

} // extern "C"
