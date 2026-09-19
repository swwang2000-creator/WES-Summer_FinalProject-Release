# Final Conclusions

## Research question

On this Qualcomm Adreno 643 GPU, when does reducing GEMM input storage from INT32 to INT8 or packed signed INT4 compensate for conversion and unpacking overhead?

## Evidence boundary

The conclusion is based on one reproducible run: seed `237`, values in `[-8,7]`, square sizes `128/256/512/1024`, identical `16×16` tiled OpenCL mapping, 5 warm-ups, 30 measured runs, and an independent CPU INT32 reference. The official raw data contains 360 measured rows, all with zero mismatches.

## Answer

Packed INT4 was slightly faster than INT32 in the kernel-event metric at every tested size, with median speedups from approximately `1.005×` to `1.018×`. This is a small difference and should not be presented as a universal or statistically significant INT4 advantage.

When the transfer path was included, INT4 was slower at `N=128` (`0.898×` speedup), essentially tied at `N=256` (`0.999×`), and faster at `N=512` (`1.051×`) and `N=1024` (`1.061×`). Therefore, a device-path crossover was observed between `N=256` and `N=512` in this experiment. No kernel-event crossover was observed.

The result supports a narrower conclusion: on this device and under this implementation, packed INT4's smaller A/B payload can offset its unpacking overhead for larger tested matrices, but the end-to-end device path does not benefit at the smallest sizes. CPU packing was measured separately and should be included in any broader application-level cost discussion.

## What this project does not establish

- It does not establish an OpenVLA inference speedup or reproduce OpenVLA.
- It does not show that the GPU has native INT4 matrix instructions.
- It does not generalize beyond the tested Adreno 643, container, kernels, input range, and aligned square sizes.
- It does not identify an exact crossover threshold between the tested sizes.
- It does not include power, temperature, model accuracy, or robot evaluation.

## Final takeaway

The project was inspired by OpenVLA's low-precision motivation, then reduced to a measurable standalone OpenCL experiment. Smaller input storage did not produce a lower device-path median at the two smaller sizes. At larger sizes, packed INT4 produced the best measured device-path median. Reduced transfer payload is a possible explanation, but these timings do not isolate transfer, scheduling, or unpacking costs, and all claims remain bounded by the single-device benchmark contract.
