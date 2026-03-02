#include "common.h"
#include "fengine_sim.h"
#include "xengine_cpu.h"
#include <cmath>

static void print_channel(const int32_t *output, int ch) {
    const int32_t *p = output + ch * N_PRODUCTS;
    printf("  ch %2d:", ch);
    for (int bl = 0; bl < N_BASELINES; bl++) {
        int ant_i, ant_j;
        baseline_to_ants(bl, &ant_i, &ant_j);
        if (ant_i == ant_j)
            printf("  %c%c*=%-8d", 'A'+ant_i, 'A'+ant_j, p[bl*2]);
        else
            printf("  %c%c*=(%d,%d)", 'A'+ant_i, 'A'+ant_j, p[bl*2], p[bl*2+1]);
    }
    printf("\n");
}

static void print_channels(const int32_t *output) {
    for (int ch = 0; ch < 8; ch++)
        print_channel(output, ch);
}

/*
 * Normalized correlation coefficient for a given baseline, averaged over all
 * channels.
 *
 * corr_coeff = |XY*| / sqrt(XX* * YY*)
 */
static double corr_coeff(const int32_t *output, int bl) {
    int ant_i, ant_j;
    baseline_to_ants(bl, &ant_i, &ant_j);
    int auto_i = ants_to_baseline(ant_i, ant_i);
    int auto_j = ants_to_baseline(ant_j, ant_j);

    double sum = 0;
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const int32_t *p = output + ch * N_PRODUCTS;

        double xx = p[auto_i * 2];
        double yy = p[auto_j * 2];
        double re = p[bl * 2];
        double im = p[bl * 2 + 1];

        sum += sqrt(re * re + im * im) / sqrt(xx * yy);
    }
    return sum / N_CHANNELS;
}

// Check that all channels have the same expected values
static bool check_uniform(
    const int32_t *output,
    int32_t exp_auto,
    int32_t exp_cross_re,
    int32_t exp_cross_im,
    const char *name
) {
    int ch;
    for (ch = 0; ch < N_CHANNELS; ch++) {
        const int32_t *p = output + ch * N_PRODUCTS;
        bool ok = true;
        for (int bl = 0; bl < N_BASELINES && ok; bl++) {
            int ant_i, ant_j;
            baseline_to_ants(bl, &ant_i, &ant_j);
            if (ant_i == ant_j)
                ok = (p[bl*2] == exp_auto && p[bl*2+1] == 0);
            else
                ok = (p[bl*2] == exp_cross_re && p[bl*2+1] == exp_cross_im);
        }
        if (!ok)
            break;
    }

    if (ch < N_CHANNELS) {
        printf("  FAIL %s at ch %d\n", name, ch);
        return false;
    }
    else {
        printf("  PASS %s\n", name);
        return true;
    }
}

static bool test_constant(int n_spectra) {
    uint8_t *data;
    int32_t output[OUTPUT_INTS];

    printf("--- constant ---\n");
    fengine_sim(&data, n_spectra, PATTERN_CONSTANT);
    xengine_cpu(data, output, n_spectra);
    print_channels(output);

    // (1,0): auto = 1, cross = (1, 0)
    bool pass = check_uniform(output, n_spectra, n_spectra, 0, "constant");
    printf("\n");

    cudaFreeHost(data);
    return pass;
}

static bool test_impulse(int n_spectra) {
    uint8_t *data;
    int32_t output[OUTPUT_INTS];

    printf("--- impulse ---\n");
    fengine_sim(&data, n_spectra, PATTERN_IMPULSE);
    xengine_cpu(data, output, n_spectra);
    print_channels(output);

    // (3,4): auto = 9+16 = 25, cross = (25, 0)
    bool pass = check_uniform(output, 25 * n_spectra, 25 * n_spectra, 0, "impulse");
    printf("\n");

    cudaFreeHost(data);
    return pass;
}

static bool test_tone(int n_spectra) {
    uint8_t *data;
    int32_t output[OUTPUT_INTS];

    printf("--- tone ---\n");
    fengine_sim(&data, n_spectra, PATTERN_TONE);
    xengine_cpu(data, output, n_spectra);
    print_channels(output);

    bool pass = true;

    // Channel 7 should match impulse values: all baselines = (25*n, 0)
    const int32_t *p7 = output + 7 * N_PRODUCTS;
    for (int bl = 0; bl < N_BASELINES; bl++) {
        if (p7[bl*2] != 25 * n_spectra || p7[bl*2+1] != 0) {
            printf("  FAIL tone: ch 7 bl %d has wrong values (%d,%d)\n",
                   bl, p7[bl*2], p7[bl*2+1]);
            pass = false;
        }
    }

    // All other channels should be zero
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        if (ch == 7)
            continue;
        const int32_t *p = output + ch * N_PRODUCTS;
        for (int i = 0; i < N_PRODUCTS; i++) {
            if (p[i] != 0) {
                printf(
                    "  FAIL tone: ch %d prod %d = %d (expected 0)\n",
                    ch, i, p[i]
                );
                pass = false;
                break;
            }
        }
    }

    if (pass)
        printf("  PASS tone\n");
    printf("\n");

    cudaFreeHost(data);
    return pass;
}

static bool test_noise(int n_spectra) {
    uint8_t *data;
    int32_t output[OUTPUT_INTS];

    printf("--- noise ---\n");
    fengine_sim(&data, n_spectra, PATTERN_NOISE);
    xengine_cpu(data, output, n_spectra);
    print_channels(output);

    // r = |XY*| / sqrt(XX* * YY*), averaged over channels
    int bl_ab = ants_to_baseline(0, 1);
    int bl_bc = ants_to_baseline(1, 2);
    int bl_ac = ants_to_baseline(0, 2);

    double r_ab = corr_coeff(output, bl_ab);
    double r_bc = corr_coeff(output, bl_bc);
    double r_ac = corr_coeff(output, bl_ac);

    printf(
        "  r_AB=%.4f  r_BC=%.4f  r_AC=%.4f"
        "  --  Expected: all ~0.0\n",
        r_ab, r_bc, r_ac
    );
    printf("\n");

    cudaFreeHost(data);
    return true;
}

static bool test_correlated(int n_spectra) {
    uint8_t *data;
    int32_t output[OUTPUT_INTS];

    printf("--- correlated ---\n");
    fengine_sim(&data, n_spectra, PATTERN_CORRELATED);
    xengine_cpu(data, output, n_spectra);
    print_channels(output);

    // Antennas 0,1 are correlated; antenna 2 is independent
    int bl_ab = ants_to_baseline(0, 1);
    int bl_bc = ants_to_baseline(1, 2);
    int bl_ac = ants_to_baseline(0, 2);

    double r_ab = corr_coeff(output, bl_ab);
    double r_bc = corr_coeff(output, bl_bc);
    double r_ac = corr_coeff(output, bl_ac);

    printf(
        "  r_AB=%.4f  r_BC=%.4f  r_AC=%.4f"
        "  --  Expected: AB~1.0, BC~0.0, AC~0.0\n",
        r_ab, r_bc, r_ac
    );

    cudaFreeHost(data);
    return true;
}

int main() {
    int n_spectra = N_SPECTRA;
    printf("Test CPU X-engine (%d spectra)\n\n", n_spectra);

    bool all_pass = true;
    all_pass &= test_constant(n_spectra);
    all_pass &= test_impulse(n_spectra);
    all_pass &= test_tone(n_spectra);
    all_pass &= test_noise(n_spectra);
    all_pass &= test_correlated(n_spectra);

    printf("\n=== %s ===\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
