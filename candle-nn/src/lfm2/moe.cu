#include "cuda_utils.cuh"
#include <hip/hip_runtime.h>
// Preserve the rounding boundaries of separate Tensor multiply/add operations.
#pragma clang fp contract(off)

extern "C" __global__ void lfm2_route_32_4(
    const float *logits, const float *bias, uint32_t *ids, float *weights) {
    const unsigned lane = threadIdx.x;
    const size_t row = blockIdx.x;
    __shared__ float scores[32], ranked[32], sum[4];
    __shared__ unsigned order[32];
    scores[lane] = recipg(1.0f + expg(-logits[row * 32 + lane]));
    ranked[lane] = scores[lane] + bias[lane];
    order[lane] = lane;
    __syncthreads();
    // Same bitonic compare/exchange network as candle-kernels/src/sort.cu.
    // Using a different tie-break would change which experts run on equal scores.
    for (unsigned k = 2; k <= 32; k *= 2) {
        for (unsigned j = k / 2; j > 0; j /= 2) {
            unsigned other = lane ^ j;
            if (other > lane) {
                bool swap = (lane & k) == 0
                    ? ranked[order[lane]] < ranked[order[other]]
                    : ranked[order[lane]] > ranked[order[other]];
                if (swap) {
                    unsigned tmp = order[lane];
                    order[lane] = order[other];
                    order[other] = tmp;
                }
            }
            __syncthreads();
        }
    }
    if (lane < 4) sum[lane] = 0.0f + scores[order[lane]];
    __syncthreads();
    if (lane < 2) sum[lane] += sum[lane + 2];
    __syncthreads();
    if (lane == 0) sum[0] += sum[1];
    __syncthreads();
    if (lane < 4) {
        ids[row * 4 + lane] = order[lane];
        weights[row * 4 + lane] = scores[order[lane]] / (sum[0] + 1e-6f);
    }
}

extern "C" __global__ void lfm2_combine_4(
    const float *x, const float *weights, float *out, size_t n, size_t hidden) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += size_t(blockDim.x) * gridDim.x) {
        const size_t row = i / hidden, col = i % hidden;
        const float *v = x + row * 4 * hidden + col;
        const float *w = weights + row * 4;
        // fast_sum's four lanes each start at zero, then reduce (0+2)+(1+3).
        float a = 0.0f + v[0] * w[0];
        float b = 0.0f + v[hidden] * w[1];
        float c = 0.0f + v[2 * hidden] * w[2];
        float d = 0.0f + v[3 * hidden] * w[3];
        out[i] = (a + c) + (b + d);
    }
}
