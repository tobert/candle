// Two-stage greedy reduction. Unlike arbitrary argmax tie behavior, our
// contract selects the smallest ID and rejects every raw nonfinite logit.
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
static constexpr unsigned WIDTH=256;
__device__ bool better(float a,unsigned ai,float b,unsigned bi) {
    return a>b || (a==b && ai<bi);
}
extern "C" __global__ void greedy_tiles(
    const float *logits,const unsigned *seen,float *values,unsigned *ids,
    unsigned *errors,unsigned vocab,float penalty) {
    __shared__ float v[WIDTH];
    __shared__ unsigned id[WIDTH],bad[WIDTH];
    unsigned t=threadIdx.x;
    float best=-INFINITY; unsigned which=UINT32_MAX,err=0;
    size_t begin=(size_t)blockIdx.x*1024;
    size_t end=begin+1024<(size_t)vocab ? begin+1024 : (size_t)vocab;
    for(size_t i=begin+t;i<end;i+=WIDTH) {
        float x=logits[i]; err|=!isfinite(x);
        if(seen[i]) x=x<0 ? x*penalty : x/penalty;
        if(better(x,i,best,which)) {best=x;which=i;}
    }
    v[t]=best;id[t]=which;bad[t]=err;__syncthreads();
    for(unsigned stride=WIDTH/2;stride;stride/=2) {
        if(t<stride) {
            if(better(v[t+stride],id[t+stride],v[t],id[t])) {v[t]=v[t+stride];id[t]=id[t+stride];}
            bad[t]|=bad[t+stride];
        }
        __syncthreads();
    }
    if(t==0) {values[blockIdx.x]=v[0];ids[blockIdx.x]=id[0];errors[blockIdx.x]=bad[0];}
}
extern "C" __global__ void greedy_finish(
    const float *values,const unsigned *ids,const unsigned *errors,
    unsigned *seen,unsigned *out,unsigned tiles) {
    __shared__ float v[WIDTH];
    __shared__ unsigned id[WIDTH],bad[WIDTH];
    unsigned t=threadIdx.x;
    float best=-INFINITY;unsigned which=UINT32_MAX,err=0;
    for(unsigned i=t;i<tiles;i+=WIDTH) {
        err|=errors[i];
        if(better(values[i],ids[i],best,which)) {best=values[i];which=ids[i];}
    }
    v[t]=best;id[t]=which;bad[t]=err;__syncthreads();
    for(unsigned stride=WIDTH/2;stride;stride/=2) {
        if(t<stride) {
            if(better(v[t+stride],id[t+stride],v[t],id[t])) {v[t]=v[t+stride];id[t]=id[t+stride];}
            bad[t]|=bad[t+stride];
        }
        __syncthreads();
    }
    if(t==0) {
        out[0]=bad[0] ? UINT32_MAX : id[0];
        if(!bad[0] && id[0]!=UINT32_MAX) seen[id[0]]=1;
    }
}
