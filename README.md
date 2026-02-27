# GPU X-Engine

A CUDA reimplementation of the FPGA cross-correlation engine (X-engine) from the [ISI Digital Spectrometer/Correlator](https://github.com/wjmallard/isi-digital-backend), a mid-infrared astronomy signal processing system deployed at Mount Wilson Observatory in 2010. I built the original FPGA system and wanted to explore how the same DSP pipeline maps to a GPU architecture.

The F-engine simulator generates synthetic input matching the original FPGA output format: complex spectra from 3 antennas, channelized into 64 bins, and quantized to 4 bits. Two GPU kernels perform the auto- and cross-correlations:

1. **Corner turn** — transposes the data from `[spectrum][antenna][channel]` to `[channel][antenna][spectrum]` to group spectra by channel.
2. **Cross-correlate + integrate** — unpacks 4-bit samples, computes 9 products per channel (3 real auto-correlations + 3 complex cross-correlations), and integrates over a configurable number of spectra.

All arithmetic is integer (`int8` inputs, `int32` accumulators), so we can validate the GPU results by exact match against the CPU results.

## Status

Validated for correctness. Currently at 88% of the original FPGA's throughput:

| | Spectra/sec | Real-time? |
|---|---|---|
| FPGA (original) | 45.0 M | ✓ |
| GPU (current, A40) | 39.7 M | ✗ (1.13× budget) |

The current bottleneck is the H2D transfer (52% of pipeline time).

## Performance History

| Version | GPU | Change | Spectra/sec | vs. FPGA |
|---------|-----|--------|-------------|----------|
| [v0.1](https://github.com/wjmallard/gpu-xengine/tree/v0.1) | A40 | Naive implementation | 32.5 M | 72% |
| [v0.2](https://github.com/wjmallard/gpu-xengine/tree/v0.2) | A40 | Move unpack from corner_turn kernel to correlation kernel | 34.6 M | 77% |
| [v0.3](https://github.com/wjmallard/gpu-xengine/tree/v0.3) | A40 | Tile corner turn in shared memory with coalesced reads | 39.7 M | 88% |

## Build

Requires CUDA toolkit (`nvcc`). Override `SM_ARCH` for target GPU:

```bash
make SM_ARCH=sm_86      # Default: sm_86
```

## Run

```bash
./test_cpu              # Confirm correctness of CPU reference implementation
./test_gpu              # Validate GPU results against CPU reference results
./benchmark             # Measure throughput of pipeline stages
```

## Source

| File | Description |
|------|-------------|
| `inc/common.h` | System parameters, data types, utilities, macros |
| `src/fengine_sim.cu` | Simulated F-engine: generates packed 4-bit test data |
| `src/xengine_cpu.cu` | CPU cross-correlator for validation |
| `src/corner_turn.cu` | GPU kernel: transpose spectra |
| `src/xengine_gpu.cu` | GPU kernel: cross-correlate + integrate |
| `src/test_cpu.cu` | CPU test driver |
| `src/test_gpu.cu` | GPU test driver (validates against CPU reference) |
| `src/benchmark.cu` | Performance benchmarking (GPU vs original FPGA) |
