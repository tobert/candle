#include <hip/hip_runtime.h>
#include <math.h>
// Keep separate f32 multiply/add rounding, matching the unfused Tensor path.
// FMA contraction would alter router choices after repeated decode steps.
#pragma clang fp contract(off)
extern "C" __global__ void lfm2_conv_step(
    const float *input,const float *weight,const float *old,float *out,
    size_t batch,size_t h,size_t k) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=batch*h) return;
    size_t b=i/h,c=i%h;
    float bx=input[b*3*h+c]*input[b*3*h+2*h+c];
    float sum=0;
    for(size_t tap=0;tap<k;tap++) {
        float v=tap+1<k ? old[i*k+tap+1] : bx;
        float product=v*weight[c*k+tap];
        sum=sum+product;
        out[batch*h+i*k+tap]=v;
    }
    out[i]=input[b*3*h+h+c]*sum;
}
extern "C" __global__ void lfm2_swiglu(const float *input,float *out,size_t n,size_t h) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    size_t offset=(i/h)*2*h+i%h;
    float gate=input[offset];
    float activated=gate/(1.0f+expf(-gate));
    out[i]=activated*input[offset+h];
}
