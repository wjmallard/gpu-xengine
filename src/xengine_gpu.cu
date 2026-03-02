#include "xengine_gpu.h"

#define THREADS_PER_BLOCK 256

/*
 * Grid:   (N_CHANNELS, N_BASELINES) — one block per (channel, baseline)
 * Block:  THREADS_PER_BLOCK threads
 * Thread: integrates a stripe of spectra for one baseline
 *
 * Shared-memory reduction combines per-thread results.
 */
__global__ void xcorr_integrate_kernel(
    const uint8_t *d_transposed,
    int32_t *output,
    int n_spectra
) {
    int ch = blockIdx.x;   // channel  [0, N_CHANNELS)
    int bl = blockIdx.y;   // baseline [0, N_BASELINES)
    int tid = threadIdx.x; // thread   [0, THREADS_PER_BLOCK)

    // Decode baseline into antenna pair
    int ant_i, ant_j;
    baseline_to_ants(bl, &ant_i, &ant_j);

    // Pointers to this channel's data for each antenna: [antenna][spectrum]
    const uint8_t *ch_data = d_transposed + ch * N_ANTENNAS * n_spectra;
    const uint8_t *row_i = ch_data + ant_i * n_spectra;
    const uint8_t *row_j = ch_data + ant_j * n_spectra;

    // Divide spectra evenly across threads
    int spectra_per_thread = ceil_div(n_spectra, THREADS_PER_BLOCK);
    int s_start = tid * spectra_per_thread;
    int s_end = s_start + spectra_per_thread;
    if (s_end > n_spectra)
        s_end = n_spectra;

    // Accumulate x * conj(y) for this thread's stripe
    int32_t acc_re = 0;
    int32_t acc_im = 0;

    for (int s = s_start; s < s_end; s++) {
        int8_t xi_re, xi_im, xj_re, xj_im;
        unpack_sample(row_i[s], &xi_re, &xi_im);
        unpack_sample(row_j[s], &xj_re, &xj_im);

        acc_re += xi_re * xj_re + xi_im * xj_im;
        acc_im += xi_im * xj_re - xi_re * xj_im;
    }

    // Shared-memory reduction (2 products per block)
    __shared__ int32_t smem_re[THREADS_PER_BLOCK];
    __shared__ int32_t smem_im[THREADS_PER_BLOCK];

    smem_re[tid] = acc_re;
    smem_im[tid] = acc_im;
    __syncthreads();

    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_re[tid] += smem_re[tid + stride];
            smem_im[tid] += smem_im[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the 2 output values for this (channel, baseline)
    if (tid == 0) {
        int base = ch * N_PRODUCTS + bl * 2;
        output[base + 0] = smem_re[0];
        output[base + 1] = smem_im[0];
    }
}

void launch_xcorr_integrate(
    const uint8_t *d_transposed,
    int32_t *d_output,
    int n_spectra
) {
    dim3 grid(N_CHANNELS, N_BASELINES);
    dim3 block(THREADS_PER_BLOCK);
    xcorr_integrate_kernel<<<grid, block>>>(d_transposed, d_output, n_spectra);
    CUDA_CHECK(cudaGetLastError());
}
