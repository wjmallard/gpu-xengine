#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

// --- System parameters ---
#define N_ANTENNAS  3
#define N_BASELINES (N_ANTENNAS * (N_ANTENNAS + 1) / 2)
#define N_CHANNELS  64

#define CHANNEL_WIDTH_HZ  45000000
#define N_SPECTRA_PER_SEC CHANNEL_WIDTH_HZ  // critically sampled: one spectrum per Nyquist interval
#define INTEGRATION_TIME_MS 1

// Number of spectra per integration
#define N_SPECTRA  (N_SPECTRA_PER_SEC * INTEGRATION_TIME_MS / 1000)

// Total bytes per spectrum from all antennas
#define BYTES_PER_SPECTRUM (N_ANTENNAS * N_CHANNELS)

// Each baseline produces (re, im); autos have im=0
#define N_PRODUCTS (N_BASELINES * 2)
// --- Output product layout per channel (row-major upper triangle) ---
// bl 0: (0,0) AA*  → (power, 0)
// bl 1: (0,1) AB*  → (re, im)
// bl 2: (0,2) AC*  → (re, im)
// bl 3: (1,1) BB*  → (power, 0)
// bl 4: (1,2) BC*  → (re, im)
// bl 5: (2,2) CC*  → (power, 0)

#define OUTPUT_INTS (N_CHANNELS * N_PRODUCTS)

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

/*
 * Baseline helpers for a row-major upper triangle
 * (0,0), (0,1), (0,2), (1,1), (1,2), (2,2), ...
 *
 * Forward map:
 *   Row i has (N-i) entries and starts at index T(i) = i*(2N-i-1)/2 + i,
 *   so bl(i,j) = i*(2N-i-1)/2 + j.
 *
 * Inverse map:
 *   Row i starts at T(i) = i*(2N-i+1)/2.  Solving T(i)=bl for i gives
 *   i = ((2N+1) - sqrt((2N+1)^2 - 8*bl)) / 2, then j = bl - (bl(i,i) - i).
 *   See NRAO CASA memo "Baseline Indexing" or Thompson, Moran & Swenson S4.
 *   The fixup handles exact boundaries where sqrtf rounds the wrong way.
 */

static __host__ __device__ inline int ants_to_baseline(int ant_i, int ant_j) {
    return ant_i * (2 * N_ANTENNAS - ant_i - 1) / 2 + ant_j;
}

static __host__ __device__ inline void baseline_to_ants(int bl, int *ant_i, int *ant_j) {
    int n2 = 2 * N_ANTENNAS + 1;
    int i = (int)((n2 - sqrtf((float)(n2 * n2 - 8 * bl))) * 0.5f);
    if ((i + 1) * (2 * N_ANTENNAS - i) / 2 <= bl) i++;
    *ant_i = i;
    *ant_j = bl - i * (2 * N_ANTENNAS - i - 1) / 2;
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
