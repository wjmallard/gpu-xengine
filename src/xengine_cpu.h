#ifndef XENGINE_CPU_H
#define XENGINE_CPU_H

#include "common.h"

/**
 * CPU-based cross-correlator.
 *
 * Unpack packed F-engine nibbles, cross-correlate, and integrate.
 *
 * Note: Worst-case accumulation: ±7 × ±7 × 2 = 98 per sample,
 *       so int32 accumulator may overflow at ~21.9M spectra.
 *
 * @param packed_data  Input: [n_spectra][N_ANTENNAS][N_CHANNELS], 1 byte/sample
 * @param output       Output: int32_t[N_CHANNELS * N_PRODUCTS], caller allocates
 * @param n_spectra    Number of spectra in a single integration
 */
void xengine_cpu(const uint8_t* packed_data, int32_t* output, int n_spectra);

#endif // XENGINE_CPU_H
