#include "xengine_cpu.h"

void xengine_cpu(const uint8_t* packed_data, int32_t* output, int n_spectra) {

    memset(output, 0, OUTPUT_INTS * sizeof(int32_t));

    for (int s = 0; s < n_spectra; s++) {

        int sp_offset = s * BYTES_PER_SPECTRUM;

        for (int ch = 0; ch < N_CHANNELS; ch++) {

            int ch_offset = ch * N_PRODUCTS;

            // Unpack 3 antenna samples for this channel in this spectrum
            int8_t a_re, a_im, b_re, b_im, c_re, c_im;
            unpack_sample(packed_data[sp_offset + 0 * N_CHANNELS + ch], &a_re, &a_im);
            unpack_sample(packed_data[sp_offset + 1 * N_CHANNELS + ch], &b_re, &b_im);
            unpack_sample(packed_data[sp_offset + 2 * N_CHANNELS + ch], &c_re, &c_im);

            /*
            Auto-correlations:
            |x|^2 = re^2 + im^2
            */
            output[ch_offset + 0] += a_re * a_re + a_im * a_im;  // AA*
            output[ch_offset + 1] += b_re * b_re + b_im * b_im;  // BB*
            output[ch_offset + 2] += c_re * c_re + c_im * c_im;  // CC*

            /*
            Cross-correlations: x * conj(y)
            re = (a_re * b_re) + (a_im * b_im)
            im = (a_im * b_re) - (a_re * b_im)
            */
            output[ch_offset + 3] += a_re * b_re + a_im * b_im;  // AB* real
            output[ch_offset + 4] += a_im * b_re - a_re * b_im;  // AB* imag

            output[ch_offset + 5] += b_re * c_re + b_im * c_im;  // BC* real
            output[ch_offset + 6] += b_im * c_re - b_re * c_im;  // BC* imag

            output[ch_offset + 7] += c_re * a_re + c_im * a_im;  // CA* real
            output[ch_offset + 8] += c_im * a_re - c_re * a_im;  // CA* imag
        }
    }
}
