#include "xengine_gpu.h"

#define THREADS_PER_BLOCK 256

/*
 * Grid:   N_CHANNELS blocks
 * Block:  THREADS_PER_BLOCK threads
 * Thread: integrates a stripe of spectra
 *
 * Shared-memory reduction combines per-thread results.
 */
__global__ void xcorr_integrate_kernel(
    const uint8_t *d_transposed,
    int32_t *output,
    int n_spectra
) {
    int ch = blockIdx.x;   // channel [0, N_CHANNELS)
    int tid = threadIdx.x; // thread  [0, THREADS_PER_BLOCK)

    // Pointer to this channel's data: [antenna][spectrum]
    const uint8_t *ch_data = d_transposed + ch * N_ANTENNAS * n_spectra;

    // Divide spectra evenly across threads
    int spectra_per_thread = ceil_div(n_spectra, THREADS_PER_BLOCK);
    int s_start = tid * spectra_per_thread;
    int s_end = s_start + spectra_per_thread;
    if (s_end > n_spectra)
        s_end = n_spectra;

    // Initialize accumulation registers
    int32_t acc_aa = 0;
    int32_t acc_bb = 0;
    int32_t acc_cc = 0;
    int32_t acc_ab_re = 0;
    int32_t acc_ab_im = 0;
    int32_t acc_bc_re = 0;
    int32_t acc_bc_im = 0;
    int32_t acc_ca_re = 0;
    int32_t acc_ca_im = 0;

    // Fill accumulation registers
    for (int s = s_start; s < s_end; s++) {
        uint8_t a = ch_data[0 * n_spectra + s];
        uint8_t b = ch_data[1 * n_spectra + s];
        uint8_t c = ch_data[2 * n_spectra + s];

        // Unpack bytes
        int8_t a_re, a_im, b_re, b_im, c_re, c_im;
        unpack_sample(a, &a_re, &a_im);
        unpack_sample(b, &b_re, &b_im);
        unpack_sample(c, &c_re, &c_im);

        // Auto-correlations: |x|^2
        acc_aa += a_re * a_re + a_im * a_im;
        acc_bb += b_re * b_re + b_im * b_im;
        acc_cc += c_re * c_re + c_im * c_im;

        // Cross-correlations: x * conj(y)
        acc_ab_re += a_re * b_re + a_im * b_im;
        acc_ab_im += a_im * b_re - a_re * b_im;

        acc_bc_re += b_re * c_re + b_im * c_im;
        acc_bc_im += b_im * c_re - b_re * c_im;

        acc_ca_re += c_re * a_re + c_im * a_im;
        acc_ca_im += c_im * a_re - c_re * a_im;
    }

    // Move per-thread results into shared memory for cross-thread reduction
    __shared__ int32_t smem[N_PRODUCTS][THREADS_PER_BLOCK];

    smem[0][tid] = acc_aa;
    smem[1][tid] = acc_bb;
    smem[2][tid] = acc_cc;
    smem[3][tid] = acc_ab_re;
    smem[4][tid] = acc_ab_im;
    smem[5][tid] = acc_bc_re;
    smem[6][tid] = acc_bc_im;
    smem[7][tid] = acc_ca_re;
    smem[8][tid] = acc_ca_im;
    __syncthreads();

    // Tree reduction
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[0][tid] += smem[0][tid + stride];
            smem[1][tid] += smem[1][tid + stride];
            smem[2][tid] += smem[2][tid + stride];
            smem[3][tid] += smem[3][tid + stride];
            smem[4][tid] += smem[4][tid + stride];
            smem[5][tid] += smem[5][tid + stride];
            smem[6][tid] += smem[6][tid + stride];
            smem[7][tid] += smem[7][tid + stride];
            smem[8][tid] += smem[8][tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the 9 output values for this channel
    if (tid == 0) {
        int base = ch * N_PRODUCTS;
        output[base + 0] = smem[0][0];
        output[base + 1] = smem[1][0];
        output[base + 2] = smem[2][0];
        output[base + 3] = smem[3][0];
        output[base + 4] = smem[4][0];
        output[base + 5] = smem[5][0];
        output[base + 6] = smem[6][0];
        output[base + 7] = smem[7][0];
        output[base + 8] = smem[8][0];
    }
}

void launch_xcorr_integrate(
    const uint8_t *d_transposed,
    int32_t *d_output,
    int n_spectra
) {
    dim3 grid(N_CHANNELS); // one block per channel
    dim3 block(THREADS_PER_BLOCK);
    xcorr_integrate_kernel<<<grid, block>>>(d_transposed, d_output, n_spectra);
    CUDA_CHECK(cudaGetLastError());
}
