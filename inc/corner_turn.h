#ifndef CORNER_TURN_H
#define CORNER_TURN_H

#include "common.h"

/**
 * Matrix transpose.
 *
 * @param d_packed      Device input:  [spectrum][antenna][channel] bytes
 * @param d_transposed  Device output: [channel][antenna][spectrum] bytes
 * @param n_spectra     Number of spectra to process
 */
void launch_corner_turn(const uint8_t *d_packed, uint8_t *d_transposed, int n_spectra);

#endif // CORNER_TURN_H
