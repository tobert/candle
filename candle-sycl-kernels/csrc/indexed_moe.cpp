// Fused indexed MoE mat-vec (the decode path of the LFM2.5-8B-A1B opinion
// engine). The weight is a stack of per-expert quantized matrices
// `[num_experts, n, k]`; each (batch, topk) pair routes to one expert whose id
// sits in `ids`. The activation for each task is quantized to int8 blocks
// (per-256 for K-quants, per-32 for Q4_0/Q8_0) exactly as `mmvq.cpp` does, and
// each weight row is dotted against it, so the numerics match the single
// matrix `candle_sycl_mmvq_q8` path and the CPU `vec_dot` reference.
//
// Layout notes:
//  - weights: `expert e` starts at byte `e * (n * k) / qk * sizeof(Blk)`, where
//    `qk` is 32 for Q4_0/Q8_0 and QK_K (256) for the K-quants — i.e. the
//    per-expert stride is `n * (k / BLK)` blocks.
//  - activations: quantized as `m` rows of `k` (one row per (batch, topk)
//    task), the same `q8/d8/s32` scratch layout `mmvq.cpp` uses, so
//    `quantize_act` is shared verbatim.
//  - output: `out[task * n + row]`, task = batch * topk + topk_idx.
//
// The kernel is correctness-first (GLM review, step 3): one work-item per
// (task, row, chunk) pair like `launch_chunked`, no Xe-specific tuning. The
// B70 has no hardware integer divide, so indices are passed in exactly as
// `launch_chunked` does — no `/ n` recovery anywhere.

#include "mmvq_internal.hpp"

namespace {

// One expert-weight row dotted against one quantized activation row.
// `w` is the stack base; `row` indexes rows within one expert; `expert`
// selects the matrix; `b` indexes blocks along k. Signature mirrors the
// `item(mi, row, b)` lambda in mmvq.cpp so the same `dot_blk` helpers apply.
template <typename Blk, int BLK, typename Dot>
int run_indexed_moe_q8(CandleSyclQueue *cq, const void *w, const void *act, bool act_f16,
                       const uint32_t *ids, void *out, bool out_f16, size_t n, size_t k,
                       size_t batch, size_t topk, size_t num_experts, int8_t *q8, float *d8,
                       int32_t *s32, float *tmp, size_t ch, Dot dot) {
  const size_t m = batch * topk; // one activation row per routed task
  try {
    quantize_act(cq->q, BLK, act, act_f16, k, m, q8, d8, s32);
    const Blk *wb = static_cast<const Blk *>(w);
    const size_t nblk = k / BLK;                 // blocks per weight row
    const size_t row_stride = n * nblk;          // blocks per expert matrix
    const size_t nch = (nblk + ch - 1) / ch;
    const float *d8f = d8;
    const int32_t *s32p = s32;
    int32_t *guard = cq->guard;
    auto item = [=](size_t mi, size_t row, size_t c) {
      // Guard the id here, not on the host: the ids are produced on device and
      // a host check would sync the queue once per MoE layer per token. A bad
      // id stores reason code 1 in the queue's guard flag (reported at the
      // next synchronize or device-to-host copy) and reads expert 0 rather than out of bounds.
      const uint32_t expert_raw = ids[mi];
      uint32_t expert = expert_raw;
      if (expert_raw >= num_experts) {
        if (guard) {
          sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                           sycl::access::address_space::global_space>(*guard)
              .store(1);
        }
        expert = 0;
      }
      const int8_t *y = q8 + mi * nblk * BLK;
      const float *dy = d8f + mi * nblk;
      const int32_t *sy = s32p + mi * nblk * 8;
      float acc = 0.f;
      size_t b1 = sycl::min((c + 1) * ch, nblk);
      for (size_t b = c * ch; b < b1; ++b)
        acc += dot(wb[expert * row_stride + row * nblk + b], y + b * BLK, dy[b], sy + b * 8);
      return acc;
    };
    // Two passes exactly like `launch_chunked`: partial sums into `tmp`
    // (m*n*nch), then a reduction per (task, row).
    cq->q.parallel_for(sycl::range<3>(m, n, nch), [=](sycl::id<3> id) {
      size_t mi = id[0], row = id[1], c = id[2];
      tmp[(mi * n + row) * nch + c] = item(mi, row, c);
    });
    cq->q.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> id) {
      size_t mr = id[0] * n + id[1];
      float acc = 0.f;
      for (size_t c = 0; c < nch; ++c) acc += tmp[mr * nch + c];
      if (out_f16)
        static_cast<f16 *>(out)[mr] = (f16)acc;
      else
        static_cast<float *>(out)[mr] = acc;
    });
    return CANDLE_SYCL_OK;
  } catch (...) {
    return CANDLE_SYCL_ERR_LAUNCH;
  }
}

} // namespace

// Indexed MoE integer mat-vec. `act`/`out` are f32, or f16 when `act_f16` /
// `out_f16`. `ids` is `batch * topk` u32 expert ids (one per task, layout
// matches the CUDA `indexed_moe_forward` `indices` argument). `num_experts`
// bounds the ids: an id outside `[0, num_experts)` stores reason code 1 in the
// queue's device guard flag (reported by `candle_sycl_synchronize` or
// the next `candle_sycl_memcpy_dtoh`) and reads
// expert 0 instead of out of bounds. Scratch has the
// same shape `candle_sycl_mmvq_q8` takes: `q8` `m*k` bytes, `d8` `m*(k/blk)`
// f32, `s32` `m*(k/blk)*8` i32, `tmp` `m*n*ceil((k/blk)/ch)` f32 with
// `m = batch * topk`. Returns CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE for dtypes
// without an integer kernel.
extern "C" int candle_sycl_indexed_moe_q8(CandleSyclQueue *q, uint32_t dt, const void *w,
                                          const void *act, int act_f16, const uint32_t *ids,
                                          void *out, int out_f16, size_t n, size_t k,
                                          size_t batch, size_t topk, size_t num_experts,
                                          int8_t *q8, float *d8, int32_t *s32, float *tmp,
                                          size_t ch) {
  const bool af = act_f16 != 0, of = out_f16 != 0;
  auto dot = [](const auto &x, const int8_t *y, float d, const int32_t *s) {
    return dot_blk(x, y, d, s);
  };
  switch (dt) {
  case G_Q4_0: return run_indexed_moe_q8<BQ4_0, QK>(q, w, act, af, ids, out, of, n, k, batch, topk, num_experts, q8, d8, s32, tmp, ch, dot);
  case G_Q8_0: return run_indexed_moe_q8<BQ8_0, QK>(q, w, act, af, ids, out, of, n, k, batch, topk, num_experts, q8, d8, s32, tmp, ch, dot);
  case G_Q4K: return run_indexed_moe_q8<BQ4K, QK_K>(q, w, act, af, ids, out, of, n, k, batch, topk, num_experts, q8, d8, s32, tmp, ch, dot);
  case G_Q5K: return run_indexed_moe_q8<BQ5K, QK_K>(q, w, act, af, ids, out, of, n, k, batch, topk, num_experts, q8, d8, s32, tmp, ch, dot);
  case G_Q6K: return run_indexed_moe_q8<BQ6K, QK_K>(q, w, act, af, ids, out, of, n, k, batch, topk, num_experts, q8, d8, s32, tmp, ch, dot);
  default: return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
