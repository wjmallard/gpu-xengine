#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// --- System parameters ---
#define N_ANTENNAS  3  // A, B, C
#define N_BASELINES 3  // AB, BC, CA
#define N_CHANNELS  64

#define CHANNEL_WIDTH_HZ  45000000
#define N_SPECTRA_PER_SEC CHANNEL_WIDTH_HZ  // critically sampled: one spectrum per Nyquist interval
#define INTEGRATION_TIME_MS 1

// Number of spectra per integration
#define N_SPECTRA  (N_SPECTRA_PER_SEC * INTEGRATION_TIME_MS / 1000)

// Total bytes per spectrum from all antennas: 3 x 64 = 192
#define BYTES_PER_SPECTRUM (N_ANTENNAS * N_CHANNELS)

// 3 auto (real) + 3 cross (real+imag) = 9 int32 values
#define N_PRODUCTS 9
// --- Output product layout per channel ---
// [0] AA*  [1] BB*  [2] CC*  (auto, real)
// [3] AB*_re  [4] AB*_im     (cross, complex)
// [5] BC*_re  [6] BC*_im
// [7] CA*_re  [8] CA*_im

// Output: 64 channels x 9 products x 4 bytes = 2304 bytes
#define OUTPUT_INTS (N_CHANNELS * N_PRODUCTS)

// --- Unpacked sample type ---
struct Sample {
    int8_t re;
    int8_t im;
};
static_assert(sizeof(Sample) == 2, "Sample must be 2 bytes (no padding)");

// --- CUDA error checking ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- Utilities ---
static __host__ __device__ inline int ceil_div(int n, int d) {
    return (n + d - 1) / d;
}

// --- Nibble pack/unpack ---
static __host__ __device__ inline uint8_t pack_sample(int re, int im) {
    if (re < -7) re = -7; if (re > 7) re = 7;
    if (im < -7) im = -7; if (im > 7) im = 7;
    return (uint8_t)(((re & 0xF) << 4) | (im & 0xF));
}

static __host__ __device__ inline void unpack_sample(uint8_t byte, int8_t* re, int8_t* im) {
    *re = ((int8_t)(byte & 0xF0)) >> 4;
    *im = ((int8_t)(byte << 4)) >> 4;
}

#endif // COMMON_H
