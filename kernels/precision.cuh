// Storage precision is independent of reduction/softmax accumulation (FP32).
// PTX conversions avoid requiring system CUDA headers for NVRTC compilation.
#if defined(BLOCKFORGE_FP16)
typedef unsigned short scalar_t;
__device__ __forceinline__ float read_scalar(scalar_t x) {
    float out; asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(x)); return out;
}
__device__ __forceinline__ scalar_t write_scalar(float x) {
    scalar_t out; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(out) : "f"(x)); return out;
}
#elif defined(BLOCKFORGE_BF16)
typedef unsigned short scalar_t;
__device__ __forceinline__ float read_scalar(scalar_t x) {
    return __uint_as_float((unsigned int)x << 16);
}
__device__ __forceinline__ scalar_t write_scalar(float x) {
    unsigned int bits=__float_as_uint(x);
    if ((bits & 0x7fffffffU)>0x7f800000U) return (scalar_t)((bits>>16)|0x40U);
    return (scalar_t)((bits+0x7fffU+((bits>>16)&1U))>>16);
}
#else
typedef float scalar_t;
__device__ __forceinline__ float read_scalar(scalar_t x) { return x; }
__device__ __forceinline__ scalar_t write_scalar(float x) { return x; }
#endif
