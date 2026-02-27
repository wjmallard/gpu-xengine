#include "corner_turn.h"

#define THREADS_PER_BLOCK 256
#define SPECTRA_PER_TILE 256

/*
 * Grid:   (ceil_div(n_spectra, SPECTRA_PER_TILE))
 * Block:  THREADS_PER_BLOCK threads
 * Thread: (spectrum, antenna, channel)
 * 
 * Result: (channel, antenna, spectrum)
 */
__global__ void corner_turn_kernel(
    const uint8_t *packed,
    uint8_t *transposed,
    int n_spectra
) {
    int tile = blockIdx.x;  // tile [0, ceil_div(n_spectra, SPECTRA_PER_TILE))
    int tid  = threadIdx.x; // thread [0, THREADS_PER_BLOCK)

    int s_start = tile * SPECTRA_PER_TILE;
    int s_end   = min(s_start + SPECTRA_PER_TILE, n_spectra);
    int n_tile = s_end - s_start; // number of spectra in this tile
    int tile_offset = s_start * BYTES_PER_SPECTRUM;
    int tile_bytes = n_tile * BYTES_PER_SPECTRUM;

    __shared__ uint8_t smem[SPECTRA_PER_TILE * BYTES_PER_SPECTRUM];

    /*
     * Contiguous load: gmem → smem
     * Coalesced gmem reads, sequential smem writes.
     *
     * Load tile from global memory into shared memory.
     * Data stays in input layout: [spectrum][antenna][channel].
     * THREADS_PER_BLOCK adjacent threads read adjacent bytes.
     */
    for (int i = tid; i < tile_bytes; i += THREADS_PER_BLOCK) {
        smem[i] = packed[tile_offset + i];
    }

    __syncthreads();

    /*
     * Transposed write: smem → gmem
     * Sequential smem reads, scattered gmem writes.
     *
     * Decompose each byte's position into (spectrum, antenna, channel),
     * then write to output layout: [channel][antenna][spectrum].
     */
    for (int i = tid; i < tile_bytes; i += THREADS_PER_BLOCK) {
        int local_s  = i / BYTES_PER_SPECTRUM;
        int col      = i % BYTES_PER_SPECTRUM;
        int ant      = col / N_CHANNELS;
        int ch       = col % N_CHANNELS;

        int global_s = s_start + local_s;
        int gmem_idx = ch * N_ANTENNAS * n_spectra + ant * n_spectra + global_s;

        transposed[gmem_idx] = smem[i];
    }
}

void launch_corner_turn(
    const uint8_t *d_packed,
    uint8_t *d_transposed,
    int n_spectra
) {
    dim3 grid(ceil_div(n_spectra, SPECTRA_PER_TILE));
    dim3 block(THREADS_PER_BLOCK);
    corner_turn_kernel<<<grid, block>>>(d_packed, d_transposed, n_spectra);
    CUDA_CHECK(cudaGetLastError());
}
