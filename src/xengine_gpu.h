#ifndef XENGINE_GPU_H
#define XENGINE_GPU_H

#include "common.h"

/**
 * GPU cross-correlator with integration.
 *
 * Computes all auto- and cross-correlation products, integrated over n_spectra.
 *
 * @param d_unpacked  Device input: [channel][antenna][spectrum] Sample array
 * @param d_output    Device output: int32_t[N_CHANNELS * N_PRODUCTS]
 * @param n_spectra   Number of spectra to integrate
 */
void launch_xcorr_integrate(const Sample *d_unpacked, int32_t *d_output, int n_spectra);

#endif // XENGINE_GPU_H
