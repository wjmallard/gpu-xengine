#include "xengine_cpu.h"

void xengine_cpu(const uint8_t* packed_data, int32_t* output, int n_spectra) {

    memset(output, 0, OUTPUT_INTS * sizeof(int32_t));

    for (int s = 0; s < n_spectra; s++) {

        int sp_offset = s * BYTES_PER_SPECTRUM;

        for (int ch = 0; ch < N_CHANNELS; ch++) {

            int ch_offset = ch * N_PRODUCTS;

            // Unpack all antenna samples for this channel
            int8_t ant_re[N_ANTENNAS];
            int8_t ant_im[N_ANTENNAS];

            for (int a = 0; a < N_ANTENNAS; a++) {
                int sample_offset = sp_offset + a * N_CHANNELS + ch;
                unpack_sample(packed_data[sample_offset], &ant_re[a], &ant_im[a]);
            }

            // Accumulate all baseline products: x * conj(y)
            for (int bl = 0; bl < N_BASELINES; bl++) {
                int ant_i, ant_j;
                baseline_to_ants(bl, &ant_i, &ant_j);

                output[ch_offset + bl * 2 + 0] += ant_re[ant_i] * ant_re[ant_j] + ant_im[ant_i] * ant_im[ant_j];
                output[ch_offset + bl * 2 + 1] += ant_im[ant_i] * ant_re[ant_j] - ant_re[ant_i] * ant_im[ant_j];
            }
        }
    }
}
