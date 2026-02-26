#include "common.h"
#include "corner_turn.h"
#include "fengine_sim.h"
#include "xengine_cpu.h"
#include "xengine_gpu.h"

static void print_channel(const int32_t *output, int ch) {
    const int32_t *p = output + ch * N_PRODUCTS;
    printf(
        "  ch %2d:  AA*=%-8d BB*=%-8d CC*=%-8d AB*=(%d,%d)  BC*=(%d,%d)  CA*=(%d,%d)\n",
        ch, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
    );
}

// ---------------------------------------------------------------------------
// Validate GPU output against CPU reference (exact integer match)
// Returns number of mismatches.
// ---------------------------------------------------------------------------
static int validate(const int32_t *gpu, const int32_t *cpu) {
    int mismatches = 0;
    for (int i = 0; i < OUTPUT_INTS; i++) {
        if (gpu[i] != cpu[i]) {
            if (mismatches < 10) {
                int ch = i / N_PRODUCTS;
                int p = i % N_PRODUCTS;
                printf(
                    "  MISMATCH ch=%d prod=%d: gpu=%d cpu=%d\n",
                    ch, p, gpu[i], cpu[i]
                );
            }
            mismatches++;
        }
    }
    return mismatches;
}

// ---------------------------------------------------------------------------
// Run one test pattern: generate data, CPU ref, GPU pipeline, compare.
// ---------------------------------------------------------------------------
static bool run_pattern(TestPattern pattern, int n_spectra) {
    const char *name = pattern_names[pattern];
    printf("--- %s ---\n", name);

    // Generate packed F-engine data
    uint8_t *h_packed = NULL;
    fengine_sim(&h_packed, n_spectra, pattern);

    // Run CPU pipeline
    int32_t cpu_output[OUTPUT_INTS];
    xengine_cpu(h_packed, cpu_output, n_spectra);

    // Run GPU pipeline
    uint8_t *d_packed;
    Sample *d_unpacked;
    int32_t *d_output;

    size_t packed_size   = (size_t)n_spectra * BYTES_PER_SPECTRUM;
    size_t unpacked_size = (size_t)N_CHANNELS * N_ANTENNAS * n_spectra * sizeof(Sample);
    size_t output_size   = OUTPUT_INTS * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_packed, packed_size));
    CUDA_CHECK(cudaMalloc(&d_unpacked, unpacked_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));

    CUDA_CHECK(
        cudaMemcpy(d_packed, h_packed, packed_size, cudaMemcpyHostToDevice)
    );
    launch_corner_turn(d_packed, d_unpacked, n_spectra);
    launch_xcorr_integrate(d_unpacked, d_output, n_spectra);

    int32_t gpu_output[OUTPUT_INTS];
    CUDA_CHECK(
        cudaMemcpy(gpu_output, d_output, output_size, cudaMemcpyDeviceToHost)
    );

    // Display a few channels
    for (int ch = 0; ch < 8; ch++)
        print_channel(gpu_output, ch);

    // Validate GPU output against CPU output
    int mismatches = validate(gpu_output, cpu_output);
    if (mismatches == 0) {
        printf("  PASS %s (%d values match)\n", name, OUTPUT_INTS);
    } else {
        printf("  FAIL %s (%d / %d mismatches)\n", name, mismatches, OUTPUT_INTS);
    }

    printf("\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_unpacked));
    CUDA_CHECK(cudaFree(d_output));
    free(h_packed);

    return mismatches == 0;
}

int main() {
    int n_spectra = N_SPECTRA;
    printf("Test GPU X-engine (%d spectra)\n\n", n_spectra);

    bool all_pass = true;
    for (int p = 0; p < PATTERN_COUNT; p++) {
        all_pass &= run_pattern((TestPattern)p, n_spectra);
    }

    printf("=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
