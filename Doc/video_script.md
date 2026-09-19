# Final Project Video Script (target: 6–8 minutes)

## 0:00–0:40 — Motivation

“My project was inspired by reading the OpenVLA paper. I was interested in its use of lower-precision representations, but I did not try to reproduce the full model. Instead, I designed a small, measurable OpenCL experiment on the Adreno GPU.”

Show the title slide and the OpenVLA inspiration link.

## 0:40–1:20 — Research question and scope

“The question is: when does reducing GEMM inputs from INT32 to INT8 or packed signed INT4 compensate for conversion and unpacking overhead? I compare three kernels: INT32, INT8, and packed INT4. All use INT32 accumulation, the same row-major GEMM, and the same 16-by-16 tile.”

Emphasize that this is a standalone proxy, not OpenVLA deployment.

## 1:20–2:15 — Implementation

“The host generates deterministic values from negative eight through seven using seed 237. INT8 and INT4 are derived from the same INT32 inputs. The INT4 representation stores two signed four-bit values per byte, with the first value in the low nibble. The CPU reference computes the result independently.”

Show the packing diagram and kernel mapping.

## 2:15–2:55 — Correctness and protocol

“The official sizes are 128, 256, 512, and 1024. Each is square and divisible by 16, so partial-tile handling is intentionally outside scope. Every output was compared element by element with the CPU INT32 reference, and all 12 kernel-size combinations produced zero mismatches.”

Show the correctness terminal evidence briefly.

## 2:55–3:45 — Measurement design

“For each kernel and size I used five warm-ups and thirty measured runs. Kernel latency is the completed OpenCL kernel event. Device-path latency starts before the H2D submissions and ends after the D2H read. CPU INT4 packing is recorded separately. The raw file contains 360 measured rows, and failed or invalid samples are not silently removed.”

Show the benchmark protocol and the raw CSV audit.

## 3:45–5:05 — Results

“For kernel-event latency, INT4 was slightly faster than INT32 at all four sizes, with median speedups from about 1.005 to 1.018. The differences are small.”

“For the device path, INT4 was slower at 128, essentially tied at 256, then faster at 512 and 1024. The measured device-path crossover is therefore between 256 and 512. The device-path speedups were about 0.898, 0.999, 1.051, and 1.061.”

Show `latency.png` and `speedup.png`; point out the separate kernel and device-path panels.

## 5:05–6:00 — Interpretation

“Packed INT4 had lower device-path medians at the two larger tested sizes. Reduced transfer payload is a possible explanation, but the experiment does not isolate transfer, scheduling, and unpacking costs. It therefore does not establish which cost dominates or prove that INT4 always wins.”

Mention that CPU packing is separate and that the kernel-only and device-path conclusions differ.

## 6:00–6:45 — Limitations

“The experiment uses synthetic values, one Adreno 643 device, aligned square matrices, and a fixed tile. It does not test native INT4 instructions, model accuracy, power, robot behavior, or OpenVLA inference. Four sizes identify only a crossover interval, not an exact threshold.”

## 6:45–7:15 — Closing

“The main lesson is that lower input precision changes both storage and overhead. In this benchmark, packed INT4 had the best measured large-size device-path median, while its benefit was absent at the smallest size. The complete source, raw data, validation script, summary, and plots are included for reproduction.”

End on the conclusion slide and show the project directory briefly.
