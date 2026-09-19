# Architecture: OpenVLA-inspired OpenCL INT4 GEMM

Updated: 2026-09-18. Status: M1–M5 complete; M6 delivery materials prepared and video recording remains.

## 1. Goal, Sources, and Non-goals

This project grew out of an interest in low-precision computing while reading [OpenVLA](https://openvla.github.io/), particularly the paper's [quantization experiments in §5.4](https://arxiv.org/html/2406.09246v3). It investigates one small question that can be addressed on the course development board:

> When matrix inputs shrink from INT32 to INT8 or packed INT4, can reduced storage and transfer costs offset conversion and unpacking overhead? Does that trade-off change with matrix size?

This is not a reproduction of OpenVLA's model quantization path. It includes no real model weights, inference-accuracy evaluation, calibration, scales, zero-points, or robot tasks. Its results cannot directly establish an OpenVLA inference speedup.

### 1.1 Requirements versus Design Choices

- **Existing requirements:** sizes, three input representations, 16×16 tiles, INT32 accumulation, 5+30 runs, correctness criteria, and deliverables from the original proposal (not included in this exported repository).
- **Implemented design:** directory structure, nibble order, deterministic input generation, CSV fields, and timing implementation are backed by the included code and board results.
- **Container configuration:** `.devcontainer/qualcomm/devcontainer.json` is included; it selects the course `qcomm` image and Adreno device mapping. The historical run's image digest was not recorded.
- **Delivery status:** final conclusions and video script are included. Video recording remains.

### 1.2 Scope

| Item | Project contract |
| --- | --- |
| Operation | `C = A × B`; no transpose, alpha/beta, or additional bias |
| Layout | Row-major A, B, and C |
| Official sizes | `M = N = K = 128 / 256 / 512 / 1024` |
| Input values | Seeded integers in `[-8, 7]`; identical numerical values across representations |
| Input representations | INT32, INT8, and two signed INT4 values per byte |
| Arithmetic and output | INT32 multiply-accumulate and INT32 C |
| GPU launch | Global `(N, N)`, local `(16, 16)` |
| Validation | Ordinary CPU INT32 reference; exact element-by-element equality |
| Excluded | Partial-tile boundary handling and correctness tests for sizes not divisible by 16, general rectangular GEMM, model evaluation, autotuning |

The host should reject unsupported arguments. This is parameter validation, not partial-tile implementation or an additional unaligned-size experiment.

### 1.3 Hardware Naming Correction

The proposal says “Rubik Pi 3 · Snapdragon 6 Gen 3 · Adreno 643.” The SoC name is incorrect: [official RUBIK Pi information](https://www.rubikpi.ai/) identifies the RUBIK Pi 3 SoC as **Qualcomm Dragonwing QCS6490**.

The official run verifies the Adreno 643 GPU and records its driver in `metadata.json`. That metadata does not independently verify the physical board model or SoC. The original proposal is outside this export.

## 2. Overall Data Flow

```text
Windows: author source code and documentation
    │ User transfers source
    ▼
Linux board: course Adreno container
    │
    ├─ Fixed seed → A32, B32 ─→ Independent CPU reference → C_ref
    │                      ├─ Numerical conversion → A8, B8
    │                      └─ CPU packing → A4, B4 (timed separately)
    │
    └─ Each representation: H2D → GPU kernel → D2H → C_gpu
                                                       │
                                        Compare with C_ref; save raw CSV
                                                       │
                                        Summarize median/IQR; make two plots
```

The “CPU reference” is ordinary C code running on the board's ARM CPU inside the container; it does not require a CPU OpenCL platform. The host is the CPU process submitting OpenCL commands; the device is the target GPU.

## 3. Directory Structure and Responsibilities

The exported repository has the following structure. The executable loads `kernels/gemm_all.cl`; standalone kernel files contain corresponding implementations for reference.

```text
final_project/
├─ Doc/                            # Architecture, development, conclusions, video script
├─ .devcontainer/qualcomm/devcontainer.json
├─ Makefile
├─ README.md
├─ include/                        # int4_project.h, opencl_runtime.h, benchmark.h, formal_benchmark.h
├─ src/                            # main.c, matrix.c, cpu_reference.c, opencl_runtime.c,
│                                  # benchmark.c, formal_benchmark.c
├─ kernels/                        # gemm_all.cl and three standalone gemm_int*.cl files
├─ tests/test_cpu.c
├─ scripts/analyze.py
├─ build/                          # Generated executables; excluded from Git
└─ results/<run-id>/                # Raw measurements and derived analysis
```

| Module | Responsibility |
| --- | --- |
| `main.c` | Validate arguments, coordinate stages, exit nonzero on failure |
| `matrix.c` | Deterministic inputs, INT8 conversion, INT4 packing, representation self-tests |
| `cpu_reference.c` | Compute ground truth directly from original INT32 A/B, without GPU execution or the INT4 decoder |
| `opencl_runtime.c` | Enumerate/select GPU, create context/queue, build kernels, manage resources |
| `benchmark.c` | Per-invocation GPU buffers, transfers, execution, and timing |
| `formal_benchmark.c` | Warm-ups, measured repetitions, correctness comparisons, CSV/metadata/log output |
| `kernels/*.cl` | Three tiled GEMMs with identical mapping |
| `tests/test_cpu.c` | Small tests for packing, input consistency, and the CPU reference |
| `scripts/analyze.py` | Validate data completeness, calculate statistics, and plot without changing raw data |

Small modules may be combined during implementation, provided this document is updated. A course project does not need a complex framework.

## 4. Values, Layout, and INT4 Format

### 4.1 Fix the Values, Then Change Their Representation

Use an explicitly defined `uint32_t` xorshift32 generator, with default seed `237` and a nonzero-seed requirement. For each new value, apply `state ^= state << 13`, `state ^= state >> 17`, and `state ^= state << 5` in sequence, keeping a 32-bit unsigned state. Map it to the input range with `(int32_t)(state & 15u) - 8`. Generate all of A in row-major order, then continue the same sequence for B.

Do not depend on `rand()` producing identical sequences across C libraries. The seed, algorithm, and generation order together determine reproducibility. Derive INT8 and INT4 from the same A32/B32 rather than generating new values for each representation.

These synthetic integers are exactly representable in INT4; this is not approximate quantization from floating point. Correctness therefore requires exact equality, not a tolerance.

### 4.2 Row-major Addressing

```text
A[row, k] = A[row * N + k]
B[k, col] = B[k * N + col]
C[row, col] = C[row * N + col]
C[row, col] = sum(k = 0 ... N-1) A[row, k] * B[k, col]
```

### 4.3 Packed Signed INT4

Use 4-bit two's-complement encoding for values `-8 ... 7`. A and B follow exactly the same packing convention.

```text
For linear element index i:
  byte_index = i / 2
  shift      = (i % 2) * 4

Pack two values x0 and x1:
  lo = ((uint32_t)x0) & 0xF
  hi = ((uint32_t)x1) & 0xF
  byte = lo | (hi << 4)

Decode:
  u = (byte >> shift) & 0xF       # u is in 0 ... 15
  x = (u < 8) ? (int)u : (int)u - 16
```

For example, `[-8, 7, -1, 0]` becomes two bytes: `0x78, 0x0F`. The first element always occupies the low four bits. Use unsigned bytes and explicit sign restoration rather than relying on negative shifts or the signedness of plain host `char`.

The host uses `int8_t`/`uint8_t` and `int32_t`; the kernels use `char`/`uchar` and `int`. The N argument uses host `int`, as validated on the ARM board. Do not confuse the OpenCL vector type `int4` with a “4-bit integer”: `int4` is a vector of four 32-bit integers.

### 4.4 INT32 Accumulation Is Safe Within This Contract

A product has absolute value at most `8 × 8 = 64`. With the largest official N of 1024, a conservative absolute-sum bound is `1024 × 64 = 65536`, well within INT32. This applies only to the agreed value range and sizes, not arbitrary GEMM inputs.

## 5. GPU Kernel Design

Each work-item computes one C element, using a mapping similar to the course tiled GEMM:

```text
row = get_global_id(0)         col = get_global_id(1)
lr  = get_local_id(0)          lc  = get_local_id(1)
local int tileA[16][16], tileB[16][16]
int sum = 0

For each t = 0 ... N/16 - 1:
  tileA[lr][lc] ← A[row, t*16 + lc], decoded/extended to int if needed
  tileB[lr][lc] ← B[t*16 + lr, col], decoded/extended to int if needed
  barrier(CLK_LOCAL_MEM_FENCE)
  For k = 0 ... 15: sum += tileA[lr][k] * tileB[k][lc]
  barrier(CLK_LOCAL_MEM_FENCE)

C[row, col] = sum
```

The first barrier ensures tile loads have completed; the second prevents overwriting a tile while other work-items are still reading it. Do not let some work-items return early while others continue through a barrier.

The kernels differ mainly at global-memory loads: direct INT32 reads, INT8-to-int conversion, or INT4 shifts/masks/sign restoration. Decoded values enter INT32 local tiles and are reused for the tile's multiply-accumulates. **Unpacking does not happen again for every MAC, and this design does not assume native GPU INT4 matrix instructions.**

Each work-group has 256 work-items. Two tiles require `2 × 16 × 16 × 4 = 2048` bytes of local memory. The runtime checks device work-group, per-dimension, and local-memory limits. It does not separately query per-kernel work-group limits; kernel enqueue errors are reported. Keep the same tile for all representations.

Adjacent work-items may load different nibbles of the same packed byte. A smaller input buffer does not imply exactly eight times fewer global-load instructions or eight times less DRAM traffic. That is another reason to measure rather than assume a speedup.

## 6. Storage and Performance Interpretation

Each input contains N² elements. Count only the device A/B/C payloads here, excluding the driver, program, host copies, and caches.

| Representation | A+B input bytes | C output bytes | Input / output at N=1024 |
| --- | --- | --- | --- |
| INT32 | `8N²` | `4N²` | 8 MiB / 4 MiB |
| INT8 | `2N²` | `4N²` | 2 MiB / 4 MiB |
| Packed INT4 | `N²` | `4N²` | 1 MiB / 4 MiB |

INT4's **input** footprint is 1/8 of INT32's. Including the unchanged output, A+B+C shrinks from `12N²` to `5N²`, not by a factor of eight overall. Footprint reduction does not establish an equal speedup.

This experiment compares “input storage/conversion strategy + the same INT32 compute structure.” Transfers could become cheaper while kernel latency increases, or INT4 might never win. Separate measured facts from explanations; without hardware-counter evidence, do not claim to have established that a case is memory-bound or compute-bound.

## 7. Timing and Execution Protocol

### 7.1 Three Timings, Three Different Questions

| Metric | Measurement definition | Excluded |
| --- | --- | --- |
| `kernel_ms` | Completed kernel event `END - START`, converting nanoseconds to milliseconds by dividing by 1e6 | H2D, D2H, CPU dispatch, packing |
| `device_path_ms` | Host monotonic clock, from just before the first H2D submission until C's D2H completes | Initialization, allocation, kernel build, input generation, CPU reference, packing, validation, CSV writing |
| `pack_a_b_ms` | Host monotonic time to fill already allocated packed A/B buffers | Allocation and GPU execution; not included in either metric above |

The device path corresponds to `H2D + kernel + D2H` in the slides. Because it is measured as host elapsed time, it also includes enqueue, waiting, and scheduling overhead along that path. It is **not** simply the sum of three device-event durations, nor is it an application end-to-end time including all preparation.

Initially record one separate INT4 packing observation per N in `packing.csv`. Do not present it as 30 samples or report a median/IQR for it. If a stable packing benchmark is needed later, explicitly define a separate repetition protocol.

### 7.2 Each Official Test Point

1. For each N, generate inputs, convert representations, and compute the CPU reference outside timing. GPU buffers are created and released for each invocation outside timing; kernel arguments are set inside the device-path interval after the input submissions.
2. Use an in-order queue with `CL_QUEUE_PROFILING_ENABLE`; test in the fixed order `INT32 → INT8 → INT4`.
3. Run 5 warm-ups for each kernel. Check their correctness, but exclude them from measured statistics.
4. Perform 30 measured runs. Every run uploads that representation's A/B, executes the kernel, and reads back all of C. Do not omit transfers by reusing the previous run's device inputs.
5. Ensure the preceding work is complete before each run. Within host timing, enqueue two nonblocking writes, the kernel, and a final blocking read.
6. Stop host timing after the read completes. Query kernel-event time, compare outputs, release the run's events, and write CSV outside the timed region.
7. Any API error, mismatch, invalid event timing, or abnormal exit fails the experiment. Preserve the failure record rather than silently adding replacement runs or selecting 30 favorable samples.

The official dataset contains `4 × 3 × 30 = 360` rows. The additional 60 warm-ups are excluded from the official CSV. Separate `--check-only` tests do not count toward these samples.

A fixed order helps reproducibility but may be affected by temperature/frequency drift. Record order, power, and cooling conditions; this limited experiment does not claim to eliminate all drift. Do not change order or timing boundaries after seeing which results look better.

OpenCL event timestamps require profiling to be enabled and the command to be complete; see the [Khronos profiling documentation](https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetEventProfilingInfo.html). Check return values and never substitute wall-clock measurements while labeling them as event times.

### 7.3 Statistics and Crossover

- Calculate median, Q1, and Q3 from the 30 samples separately for each size, kernel, and timing metric; `IQR = Q3 - Q1`.
- Use a fixed linear-interpolation quantile definition: after sorting, position `h = (n-1)p`, interpolating between adjacent values. Encode this explicitly in the analysis script to avoid differences between library defaults.
- `speedup = median(INT32 latency) / median(candidate latency)` at the same N and for the same timing scope. A value greater than 1 means faster.
- Report kernel and device-path speedups separately. Do not mix timing scopes or calculate per-sample ratios and label their aggregation as the metric above.
- If relative performance changes between adjacent tested sizes, report a crossover between them. Four discrete sizes cannot establish an exact threshold.
- Acknowledge uncertainty when differences are small or variation is large; IQR is not a confidence interval. If no crossing is observed, say “not observed within the tested size range.”

## 8. Correctness and Failure Handling

1. CPU unit checks cover known byte examples, nibble order, deterministic-input conversion/packing round-trips, and a known reference GEMM. They do not explicitly enumerate all 16 INT4 values. The small CPU matrix tests do not exercise GPU tile boundaries.
2. During GPU development, an aligned `16×16` debug case may help locate indexing errors before testing all four official sizes. Exclude debug sizes from official statistics.
3. Compute C_ref independently from original A32/B32 and compare every INT32 output element from every kernel. On failure, report the first differing coordinate, expected/actual values, and total mismatch count.
4. Check packed indexing, negative-value sign restoration, barriers, and buffer byte counts. Do not hide bugs by relaxing numerical tolerances.
5. Exit nonzero if no GPU is available, a build fails, device requirements are unmet, or results differ. Correct output followed by a shutdown crash is still a failure.

The CPU reference may use a cache-friendly loop order, but should remain simple, integer-exact, and independent of the tiled/packed implementation. Its execution time is not the GPU speedup baseline.

## 9. Data, Deliverables, and Claim Boundaries

Each `results/<run-id>/` should contain:

| File | Contents |
| --- | --- |
| `raw.csv` | `n,kernel,trial,seed,kernel_ms,device_path_ms,input_bytes,output_bytes,mismatches`; trial is 1…30 |
| `packing.csv` | One row per N: `n,seed,pack_a_b_ms,input_bytes`, where input_bytes refers to packed A+B |
| `metadata.json` | Status, source-version label, seed, sizes, kernels, repetitions, tile, OpenCL build options, input range, order, timing definitions, platform/device/driver versions, and a container environment label |
| `run.log` | Command, initialization marker, packing times, failure marker when applicable, and final status; detailed errors are printed to the terminal |
| `analysis/crossover.txt` | Kernel-event and device-path crossover observations relative to INT32 |
| `analysis/summary.csv` | Sample count, median, Q1/Q3/IQR, and speedup for each test point |
| `analysis/latency.png` | Two panels: kernel and device-path latency, including variation |
| `analysis/speedup.png` | Two panels: corresponding speedups, with a baseline of 1 |

The analysis script checks completion status, row counts against metadata, key uniqueness, parameter consistency, 0 mismatches, and positive finite timings. It requires 5+30 repetitions; the official four-size/all-kernel selection must also be checked against the experiment contract. Preserve raw CSVs unchanged after collection.

Evidence limitations: metadata does not record source hashes, host compiler flags, an image digest, or a timestamp field. The current implementation writes completion status before final resource/file cleanup and does not comprehensively check cleanup return values. Check terminal output and process exit status as well; metadata alone is not proof of successful cleanup.

Final deliverables: three runnable kernels and a README; raw data, analysis scripts, and two plots; a video no longer than 8 minutes. The video must answer the research question and state limitations. “INT4 must win” is not a success criterion.

## 10. Change Policy

Confirm changes to fixed requirements with the user first, especially sizes, tile, repetition counts, timing boundaries, and comparison baselines. Update implementation details here with a brief explanation. Keep deployment commands, current status, and validation evidence in [development.md](development.md), rather than duplicating the full progress log across both documents.
