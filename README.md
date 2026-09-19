# OpenVLA-Inspired OpenCL INT4 GEMM Benchmark

This project is a standalone low-precision GEMM microbenchmark inspired by reading [OpenVLA](https://openvla.github.io/), especially its discussion of quantization. It is not an OpenVLA reproduction, model implementation, robot experiment, or claim about OpenVLA inference speed.

## Research question

On the target Adreno GPU, when does reducing matrix input storage from INT32 to INT8 or packed signed INT4 compensate for conversion and unpacking overhead?

## Fixed experiment

| Item | Contract |
| --- | --- |
| Operation | Row-major `C = A × B`, with `M=N=K` |
| Representations | INT32, INT8, and two signed INT4 values per byte |
| Arithmetic | INT32 multiply-accumulate and INT32 output |
| Values | Deterministic xorshift32 inputs in `[-8, 7]`, seed `237` |
| GPU mapping | Identical `16×16` tiled OpenCL work-groups |
| Official sizes | `128`, `256`, `512`, `1024` |
| Correctness | Independent CPU INT32 reference; exactly 0 mismatches |
| Repetition | 5 warm-ups + 30 measured runs per kernel and size |
| Timing | Kernel event time and host device-path time (H2D → kernel → D2H) |

Sizes not divisible by 16 and partial-tile handling are intentionally outside the agreed scope.

## Hardware and validation

The official run used the Qualcomm Adreno 643 OpenCL GPU in the course development container. The run recorded Qualcomm OpenCL 3.0 / OpenCL C 3.0 device information and completed with 360 measured rows:

```text
results/formal-20260918-200321/raw.csv       360 measured rows
results/formal-20260918-200321/packing.csv   4 packing observations
metadata.json                                status=complete
```

The CPU tests, all four-size correctness gate, and formal benchmark completed with exit status 0. Every measured output had `mismatches=0`.

## Results

The values below are medians from the audited 30-run samples. Speedup is INT32 median divided by the candidate median; values above 1 are faster.

### Kernel-event latency

| N | INT32 ms | INT8 ms | INT4 ms | INT4 speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.964 | 0.958 | 0.959 | 1.005× |
| 256 | 7.743 | 7.651 | 7.639 | 1.014× |
| 512 | 62.278 | 61.554 | 61.196 | 1.018× |
| 1024 | 505.482 | 497.869 | 496.436 | 1.018× |

### Device-path latency

This includes host-monotonic time from the first H2D submission through completion of the D2H read.

| N | INT32 ms | INT8 ms | INT4 ms | INT4 speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2.443 | 2.670 | 2.720 | 0.898× |
| 256 | 11.789 | 12.016 | 11.797 | 0.999× |
| 512 | 75.415 | 72.525 | 71.772 | 1.051× |
| 1024 | 536.201 | 507.049 | 505.329 | 1.061× |

The device-path analysis reports a crossover between `N=256` and `N=512`. No kernel-event crossover was observed. CPU packing time was recorded separately and is not included in either GPU timing metric.

These small speedups should be described as measured median differences, not as proof of a statistically significant or universal INT4 advantage.

## Rebuild and correctness check

Run these commands inside the board's Adreno development container from the project root:

```bash
make clean
make
make cpu-test
./build/gemm_bench --list-devices
./build/gemm_bench --check-only --size 128 --kernels all
```

## Reproduce the official collection

The executable writes a new run directory and refuses to overwrite existing result artifacts:

```bash
run_id="formal-$(date +%Y%m%d-%H%M%S)"
./build/gemm_bench \
  --sizes 128,256,512,1024 \
  --kernels all \
  --seed 237 \
  --warmup 5 \
  --runs 30 \
  --output "results/$run_id"
```

Each completed run contains `raw.csv`, `packing.csv`, `metadata.json`, and `run.log`. The raw CSV must remain unchanged after collection.

## Analyze a retrieved run

From the repository root in Windows PowerShell, reanalyze the included run into a new directory (the shipped `analysis/` already exists):

```powershell
$analysis = "results/formal-20260918-200321/analysis-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
py -3 scripts/analyze.py `
  --input "results/formal-20260918-200321" `
  --output "$analysis"
```

The script requires Python 3.9 or newer and Matplotlib. It validates row completeness against metadata, trial uniqueness, byte counts, positive finite timings, zero mismatches, and the 5+30 protocol. The official experiment additionally requires all four specified sizes and all three kernels. It writes `summary.csv`, `crossover.txt`, `latency.png`, and `speedup.png`, refusing to overwrite existing analysis files. It never removes outliers or edits `raw.csv`.

## Project layout

```text
final_project/
├─ .devcontainer/qualcomm/  Adreno container configuration
├─ README.md
├─ Makefile
├─ include/                  C interfaces
├─ src/                     host runtime, CPU reference, and collection
├─ kernels/                 INT32, INT8, packed INT4 OpenCL kernels
├─ tests/                   CPU foundation tests
├─ scripts/analyze.py       validation, statistics, and plots
├─ Doc/                     architecture, development, conclusions, video script
└─ results/<run-id>/        immutable raw data and derived analysis
```

## Limitations

- The inputs are synthetic and restricted to `[-8, 7]`; they are not calibrated model weights.
- Only square, 16-divisible matrices on one Adreno 643 device were tested.
- The kernels use packed storage and explicit unpacking; this is not a claim of native INT4 matrix hardware instructions.
- The project does not evaluate OpenVLA, robot behavior, model accuracy, power, or other GPUs.
- Four discrete sizes identify only an interval for a possible crossover, not an exact threshold.
- The archived metadata has a source-version label, but no source hashes, host compiler flags, or container image digest; exact historical environment reconstruction is limited.

The included container configuration uses `ghcr.io/kastnerrg/cse160-opencl:qcomm`. Other course container templates are included but were not validated for this experiment. The video script is included; the recorded video is not.

See [architecture.md](Doc/architecture.md), [development.md](Doc/development.md), and [final_conclusions.md](Doc/final_conclusions.md) for the detailed contracts and evidence record.
