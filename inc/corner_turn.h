#ifndef CORNER_TURN_H
#define CORNER_TURN_H

#include "common.h"

/**
 * Unpack 4-bit complex samples, and corner-turn
 *   from: [spectrum][antenna][channel]
 *   into: [channel][antenna][spectrum].
 *
 * @param d_packed    Device input: n_spectra * N_ANTENNAS * N_CHANNELS bytes
 * @param d_unpacked  Device output: N_CHANNELS * N_ANTENNAS * n_spectra Samples
 * @param n_spectra   Number of spectra to process
 */
void launch_corner_turn(const uint8_t *d_packed, Sample *d_unpacked, int n_spectra);

#endif // CORNER_TURN_H
