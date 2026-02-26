#ifndef FENGINE_SIM_H
#define FENGINE_SIM_H

#include "common.h"

enum TestPattern {
    PATTERN_CONSTANT,
    PATTERN_IMPULSE,
    PATTERN_TONE,
    PATTERN_NOISE,
    PATTERN_CORRELATED,
    PATTERN_COUNT
};

static const char* pattern_names[] = {
    "constant",
    "impulse",
    "tone",
    "noise",
    "correlated"
};

/**
 * Simulate packed F-engine data for X-engine validation.
 *
 * Allocates *packed_data with malloc; caller must free.
 *
 * @param packed_data  Output: [n_spectra][N_ANTENNAS][N_CHANNELS], 1 byte/sample
 * @param n_spectra    Number of spectra to generate
 * @param pattern      Test pattern to generate
 * @param seed         PRNG seed
 */
void fengine_sim(uint8_t** packed_data, int n_spectra, TestPattern pattern, unsigned int seed = 42);

#endif // FENGINE_SIM_H
