#include "fengine_sim.h"

// 32-bit xorshift PRNG
static inline uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static inline int8_t sym_sat(uint32_t x) {
    return (x % 15) - 7;
}

static void fill_constant(uint8_t* packed_data, size_t n_bytes) {
    memset(packed_data, pack_sample(1, 0), n_bytes);
}

static void fill_impulse(uint8_t* packed_data, size_t n_bytes) {
    memset(packed_data, pack_sample(3, 4), n_bytes);
}

static void fill_tone(uint8_t* packed_data, size_t n_bytes) {

    const int channel = 7;  // tone channel

    memset(packed_data, 0, n_bytes);

    for (size_t i = channel; i < n_bytes; i += N_CHANNELS) {
        packed_data[i] = pack_sample(3, 4);
    }
}

static void fill_noise(uint8_t* packed_data, size_t n_bytes, uint32_t seed) {

    uint32_t rng = seed;

    for (size_t i = 0; i < n_bytes; i++) {
        rng = xorshift32(rng);
        packed_data[i] = pack_sample(sym_sat((rng >> 28) & 0xF),
                                     sym_sat((rng >> 24) & 0xF));
    }
}

static void fill_correlated(uint8_t* packed_data, size_t n_bytes, uint32_t seed) {

    uint32_t rng = seed;

    for (size_t base = 0; base < n_bytes; base += N_ANTENNAS * N_CHANNELS) {
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            rng = xorshift32(rng);

            uint8_t correlated  = pack_sample(sym_sat((rng >> 28) & 0xF),
                                              sym_sat((rng >> 24) & 0xF));
            uint8_t independent = pack_sample(sym_sat((rng >> 20) & 0xF),
                                              sym_sat((rng >> 16) & 0xF));

            packed_data[base + 0 * N_CHANNELS + ch] = correlated;
            packed_data[base + 1 * N_CHANNELS + ch] = correlated;
            packed_data[base + 2 * N_CHANNELS + ch] = independent;
        }
    }
}

void fengine_sim(uint8_t** packed_data, int n_spectra, TestPattern pattern, unsigned int seed) {

    size_t n_bytes = (size_t)n_spectra * BYTES_PER_SPECTRUM;
    CUDA_CHECK(cudaMallocHost(packed_data, n_bytes));

    switch (pattern) {
        case PATTERN_CONSTANT:   fill_constant(*packed_data, n_bytes); break;
        case PATTERN_IMPULSE:    fill_impulse(*packed_data, n_bytes); break;
        case PATTERN_TONE:       fill_tone(*packed_data, n_bytes); break;
        case PATTERN_NOISE:      fill_noise(*packed_data, n_bytes, seed); break;
        case PATTERN_CORRELATED: fill_correlated(*packed_data, n_bytes, seed); break;
    }
}
