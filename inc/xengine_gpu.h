#ifndef XENGINE_GPU_H
#define XENGINE_GPU_H

#include "common.h"

/**
 * GPU cross-correlator with integration.
 *
 * Computes all auto- and cross-correlation products, integrated over n_spectra.
 *
 * @param d_transposed  Device input:  [channel][antenna][spectrum] bytes
 * @param d_output      Device output: [channel][product] int32_t
 * @param n_spectra     Number of spectra to integrate
 */
void launch_xcorr_integrate(const uint8_t *d_transposed, int32_t *d_output, int n_spectra);

#endif // XENGINE_GPU_H
