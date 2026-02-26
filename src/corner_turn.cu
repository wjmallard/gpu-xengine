#include "corner_turn.h"

#define THREADS_PER_BLOCK 256
#define N_BLOCKS(n) ceil_div(n, THREADS_PER_BLOCK)

/*
 * Grid:   (N_BLOCKS(n_spectra), N_ANTENNAS, N_CHANNELS)
 * Block:  THREADS_PER_BLOCK threads
 * Thread: (spectrum, antenna, channel)
 * 
 * Result: (channel, antenna, spectrum)
 */
__global__ void corner_turn_kernel(
    const uint8_t *packed,
    Sample *unpacked,
    int n_spectra
) {
    int s  = blockIdx.x * blockDim.x + threadIdx.x;  // spectrum
    int a  = blockIdx.y;                             // antenna
    int ch = blockIdx.z;                             // channel

    if (s >= n_spectra)
        return;

    // Input: [spectrum][antenna][channel]
    int in_idx = s * (N_ANTENNAS * N_CHANNELS) + a * N_CHANNELS + ch;
    uint8_t byte = packed[in_idx];

    // Unpack
    Sample sample;
    unpack_sample(byte, &sample.re, &sample.im);

    // Output: [channel][antenna][spectrum]
    int out_idx = ch * (N_ANTENNAS * n_spectra) + a * n_spectra + s;
    unpacked[out_idx] = sample;
}

void launch_corner_turn(
    const uint8_t *d_packed,
    Sample *d_unpacked,
    int n_spectra
) {
    dim3 grid(N_BLOCKS(n_spectra), N_ANTENNAS, N_CHANNELS);
    dim3 block(THREADS_PER_BLOCK);
    corner_turn_kernel<<<grid, block>>>(d_packed, d_unpacked, n_spectra);
    CUDA_CHECK(cudaGetLastError());
}
