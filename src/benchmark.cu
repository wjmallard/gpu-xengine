#include "common.h"
#include "fengine_sim.h"
#include "corner_turn.h"
#include "xengine_gpu.h"

static void run_benchmark(int n_spectra, int n_iterations = 20) {

    // Host allocation
    uint8_t* h_packed = NULL;
    int32_t h_output[OUTPUT_INTS];
    fengine_sim(&h_packed, n_spectra, PATTERN_NOISE);

    // Device allocation
    uint8_t* d_packed;
    Sample*  d_unpacked;
    int32_t* d_output;

    size_t packed_size   = (size_t)n_spectra * BYTES_PER_SPECTRUM;
    size_t unpacked_size = (size_t)N_CHANNELS * N_ANTENNAS * n_spectra * sizeof(Sample);
    size_t output_size   = OUTPUT_INTS * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_packed, packed_size));
    CUDA_CHECK(cudaMalloc(&d_unpacked, unpacked_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    // CUDA events
    cudaEvent_t ev_start, ev_h2d, ev_unpack, ev_xcorr, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_unpack));
    CUDA_CHECK(cudaEventCreate(&ev_xcorr));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    // Warmup (2 iterations)
    for (int w = 0; w < 2; w++) {
        CUDA_CHECK(cudaMemcpy(d_packed, h_packed, packed_size, cudaMemcpyHostToDevice));
        launch_corner_turn(d_packed, d_unpacked, n_spectra);
        launch_xcorr_integrate(d_unpacked, d_output, n_spectra);
        CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time iterations
    float sum_h2d = 0, sum_unpack = 0, sum_xcorr = 0, sum_d2h = 0, sum_total = 0;

    for (int i = 0; i < n_iterations; i++) {
        CUDA_CHECK(cudaEventRecord(ev_start));

        CUDA_CHECK(cudaMemcpy(d_packed, h_packed, packed_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_h2d));

        launch_corner_turn(d_packed, d_unpacked, n_spectra);
        CUDA_CHECK(cudaEventRecord(ev_unpack));

        launch_xcorr_integrate(d_unpacked, d_output, n_spectra);
        CUDA_CHECK(cudaEventRecord(ev_xcorr));

        CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(ev_end));

        CUDA_CHECK(cudaEventSynchronize(ev_end));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_h2d));    sum_h2d    += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_h2d, ev_unpack));   sum_unpack += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_unpack, ev_xcorr)); sum_xcorr  += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_xcorr, ev_end));    sum_d2h    += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));    sum_total  += ms;
    }

    float avg_h2d    = sum_h2d    / n_iterations;
    float avg_unpack = sum_unpack / n_iterations;
    float avg_xcorr  = sum_xcorr  / n_iterations;
    float avg_d2h    = sum_d2h    / n_iterations;
    float avg_total  = sum_total  / n_iterations;

    float budget_ms  = (float)n_spectra / (float)(N_SPECTRA_PER_SEC / 1000);

    printf("\n");
    printf("=== Benchmark (%d iterations, %d spectra) ===\n", n_iterations, n_spectra);
    printf("\n");
    printf("  Stage                Time (ms)\n");
    printf("  ------------------------------\n");
    printf("  H2D transfer         %8.3f\n", avg_h2d);
    printf("  Corner turn          %8.3f\n", avg_unpack);
    printf("  Xcorr + integrate    %8.3f\n", avg_xcorr);
    printf("  D2H transfer         %8.3f\n", avg_d2h);
    printf("  ------------------------------\n");
    printf("  Total pipeline       %8.3f\n", avg_total);
    printf("\n");
    printf("  Input:      %.2f MB  (%d spectra x %d B/spectrum)\n",
           packed_size / 1e6, n_spectra, BYTES_PER_SPECTRUM);
    printf("  Output:     %d bytes  (%d channels x %d products x 4 B)\n",
           (int)output_size, N_CHANNELS, N_PRODUCTS);
    printf("  H2D BW:     %.2f GB/s\n",
           packed_size / (avg_h2d * 1e6));
    printf("  Throughput: %.1f M spectra/sec\n",
           n_spectra / (avg_total * 1e3));
    printf("  FPGA rate:  %.1f M spectra/sec\n",
           N_SPECTRA_PER_SEC / 1e6);
    printf("  Real-time:  %s  (pipeline %.3f ms vs budget %.3f ms = %.1f%%)\n",
           avg_total < budget_ms ? "YES" : "NO",
           avg_total, budget_ms,
           avg_total / budget_ms * 100.0f);
    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d));
    CUDA_CHECK(cudaEventDestroy(ev_unpack));
    CUDA_CHECK(cudaEventDestroy(ev_xcorr));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_unpacked));
    CUDA_CHECK(cudaFree(d_output));
    free(h_packed);
}

int main() {
    int n_spectra = N_SPECTRA;

    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1e3);

    run_benchmark(n_spectra);
    return 0;
}
