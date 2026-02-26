#include "common.h"
#include "fengine_sim.h"
#include "xengine_cpu.h"
#include <cmath>

static void print_channel(const int32_t *output, int ch) {
    const int32_t *p = output + ch * N_PRODUCTS;
    printf(
        "  ch %2d:  AA*=%-8d BB*=%-8d CC*=%-8d AB*=(%d,%d)  BC*=(%d,%d)  CA*=(%d,%d)\n",
        ch, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]
    );
}

static void print_channels(const int32_t *output) {
    for (int ch = 0; ch < 8; ch++)
        print_channel(output, ch);
}

/*
 * Normalized correlation coefficient, averaged over all channels.
 *
 * corr_coeff = |XY*| / sqrt(XX* * YY*)
 *
 * auto_x, auto_y: product indices for the two autos (0=AA, 1=BB, 2=CC)
 * cross_re, cross_im: product indices for the cross
 */
static double corr_coeff(
    const int32_t *output,
    int auto_x,
    int auto_y,
    int cross_re,
    int cross_im
) {
    double sum = 0;
    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const int32_t *p = output + ch * N_PRODUCTS;

        double xx = p[auto_x];
        double yy = p[auto_y];
        double re = p[cross_re];
        double im = p[cross_im];

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
        if (p[0] != exp_auto
            || p[1] != exp_auto
            || p[2] != exp_auto
            || p[3] != exp_cross_re
            || p[4] != exp_cross_im
            || p[5] != exp_cross_re
            || p[6] != exp_cross_im
            || p[7] != exp_cross_re
            || p[8] != exp_cross_im
        ) {
            break;
        }
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

    free(data);
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

    free(data);
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

    // Channel 7 should match impulse values
    const int32_t *p7 = output + 7 * N_PRODUCTS;
    if (p7[0] != 25 * n_spectra || p7[3] != 25 * n_spectra || p7[4] != 0) {
        printf("  FAIL tone: ch 7 has wrong values\n");
        pass = false;
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

    free(data);
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
    double r_ab = corr_coeff(output, 0, 1, 3, 4);  // AA, BB, AB_re, AB_im
    double r_bc = corr_coeff(output, 1, 2, 5, 6);  // BB, CC, BC_re, BC_im
    double r_ca = corr_coeff(output, 2, 0, 7, 8);  // CC, AA, CA_re, CA_im

    printf(
        "  r_AB=%.4f  r_BC=%.4f  r_CA=%.4f"
        "  --  Expected: AB~0.0, BC~0.0, CA~0.0\n",
        r_ab, r_bc, r_ca
    );
    printf("\n");

    free(data);
    return true;
}

static bool test_correlated(int n_spectra) {
    uint8_t *data;
    int32_t output[OUTPUT_INTS];

    printf("--- correlated ---\n");
    fengine_sim(&data, n_spectra, PATTERN_CORRELATED);
    xengine_cpu(data, output, n_spectra);
    print_channels(output);

    double r_ab = corr_coeff(output, 0, 1, 3, 4);  // AA, BB, AB_re, AB_im
    double r_bc = corr_coeff(output, 1, 2, 5, 6);  // BB, CC, BC_re, BC_im
    double r_ca = corr_coeff(output, 2, 0, 7, 8);  // CC, AA, CA_re, CA_im

    printf(
        "  r_AB=%.4f  r_BC=%.4f  r_CA=%.4f"
        "  --  Expected: AB~1.0, BC~0.0, CA~0.0\n",
        r_ab, r_bc, r_ca
    );

    free(data);
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
