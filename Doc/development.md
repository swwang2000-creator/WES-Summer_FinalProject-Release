# Development, Build, Deployment, and Validation Guide

Updated: 2026-09-18. Run all relative commands from the repository root.

## 1. Current Status

M1–M4 passed on the board. M5 validated the retrieved official run and generated the summary and plots. M6 includes the README, conclusions, and video script; the video recording is not included.

| Stage | Current evidence |
| --- | --- |
| M0 Documentation | Architecture and development guides available |
| M1 CPU foundation | Board CPU tests passed |
| M2 GPU baseline | INT32 correctness passed at all four official sizes |
| M3 Three representations | All 12 size/kernel combinations passed |
| M4 Collection | `results/formal-20260918-200321`: 360 samples, zero mismatches |
| M5 Analysis | Summary, crossover report, and two plots included |
| M6 Delivery | Written materials prepared; video recording remains |

See [architecture.md](architecture.md) for the experiment contract and [final_conclusions.md](final_conclusions.md) for the results. Dated work records below describe historical stages; their pending tasks are superseded by this current-status table.

## 2. Locations and Environments

| Environment | Project directory | Purpose |
| --- | --- | --- |
| Windows release repository | `C:\Users\swwan\OneDrive\Desktop\WES Summer\Final_Project_Release\final_project` | Edit, inspect results, analyze, commit |
| Board host, SSH alias `rubik-pi-home` | `/home/ubuntu/final_project` | Receive files and host the container |
| Existing board container | `/workspaces/final_project` | Compile and run on Adreno |

The old `Final Project/source` directory was the original authoring layout. This exported repository has no enclosing `source/` directory. Paths in the dated work records may refer to that earlier layout.

The repository now uses Git. A commit identifies the archived source, but does not retroactively establish which exact source produced an earlier benchmark.

## 3. Container and Platform

The included Adreno configuration is [qualcomm/devcontainer.json](../.devcontainer/qualcomm/devcontainer.json). It uses `ghcr.io/kastnerrg/cse160-opencl:qcomm` and maps `/dev/kgsl-3d0`. Select this configuration when reopening the board project in a VS Code development container. Keep the existing working container if it is already running.

Other included course templates (`cpu-amd64`, `cpu-arm64`, and `nvidia`) are not the configurations used for the Adreno experiment. Their presence does not establish validation on those platforms. The earlier `gpu-adreno` image tag was a historical reference, not the tag in this export. The exact image digest used for the official run was not recorded.

Inside the board container:

```bash
cd /workspaces/final_project
uname -m
clinfo -l
ls -l /dev/kgsl-3d0
```

Recorded board checks showed `aarch64`, Qualcomm Adreno 643, maximum work-group size 1024, and local memory 32768 bytes. The official metadata records the driver and OpenCL versions. The physical board model and container image digest are not independently established by that metadata.

## 4. Update the Existing Board Copy

Run in Windows PowerShell. This updates named source/documentation files in the existing board project; retrieve any unmerged board edits first.

```powershell
$project = 'C:\Users\swwan\OneDrive\Desktop\WES Summer\Final_Project_Release\final_project'
$board = 'rubik-pi-home'
$remote = '/home/ubuntu/final_project'
scp -r "$project/include" "$project/src" "$project/kernels" "$project/tests" "$project/scripts" "$project/Doc" "$project/Makefile" "$project/README.md" "${board}:${remote}/"
if ($LASTEXITCODE -ne 0) { throw 'Upload failed.' }
```

This command merges directories; it does not remove files deleted locally. The existing container configuration and result directories remain on the board. To transfer the container templates to a fresh board copy, include the repository's `.devcontainer` directory as well.

## 5. Build

The board needs a C11 compiler, GNU Make, OpenCL headers, and the OpenCL loader. Analysis needs Python 3.9 or newer and Matplotlib.

The Makefile defaults to `-std=c11 -O2 -Wall -Wextra -Wpedantic`, `-Iinclude`, and `-lOpenCL`. The header sets `CL_TARGET_OPENCL_VERSION=120`; timing sources set `_POSIX_C_SOURCE=200809L`. Runtime GPU compilation uses `-cl-std=CL1.2`.

| Command | Implemented behavior |
| --- | --- |
| `make` | Build CPU tests and GPU benchmark |
| `make cpu-test` | Build and execute CPU tests without linking OpenCL |
| `make clean` | Remove generated `build/` |

There is no `make debug` target. Run inside the container:

```bash
cd /workspaces/final_project
make
make cpu-test
./build/gemm_bench --list-devices
```

The executable loads `kernels/gemm_all.cl` relative to the working directory. The three individual kernel files are readable standalone counterparts; editing them alone does not change the program loaded by the executable.

## 6. Correctness and Collection

The program selects the first GPU found in platform enumeration order, without CPU fallback. It does not implement `--platform` or `--device`. Confirm that the printed selection is Qualcomm Adreno 643 before using results for this project.

| Option | Behavior |
| --- | --- |
| `--list-devices` | Print platform/device names and GPU classification |
| `--check-only --size N` | One correctness check; default size 128, fixed seed 237 |
| `--kernels K` | `int32`, `int8`, `int4`, or `all` (default) |
| `--output DIR` | Enable collection mode |
| `--sizes LIST` | Collection sizes; default 128,256,512,1024 |
| `--seed S` | Collection seed; nonzero, default 237 |
| `--warmup N` | Collection warm-ups; default 5 |
| `--runs N` | Collection measured runs; default 30 |

Seed/repetition options apply to collection mode, not the standalone correctness command. Do not combine `--check-only` with collection.

```bash
for n in 128 256 512 1024; do
    ./build/gemm_bench --check-only --size "$n" --kernels all || exit 1
done
```

CPU tests cover known packed bytes, deterministic inputs, conversion round-trips, and a known reference GEMM. They do not explicitly enumerate every possible INT4 value. The 2×2 CPU-reference test is independent of GPU tile-boundary support.

Collect a new official run:

```bash
run_id="formal-$(date +%Y%m%d-%H%M%S)"
./build/gemm_bench --sizes 128,256,512,1024 --kernels all --seed 237 --warmup 5 --runs 30 --output "results/$run_id"
status=$?
printf 'run: %s | exit status: %s\n' "$run_id" "$status"
```

Require 360 measured rows, 30 unique trials per size/kernel, positive timings, zero mismatches, complete metadata, and a normal process exit. The included `m4-smoke-20260918-191135` run has 1 warm-up and 2 measured runs at N=128; it is not official data and the analyzer intentionally rejects its repetition protocol.

The existing metadata includes a source-version label, seed, sizes, kernels, repetition counts, tile, GPU build options, timing definitions, and GPU/driver strings. It does not include source hashes, host compiler flags, image digest, or a timestamp field. Do not add guessed historical values. Terminal error output and the process exit status complement the compact `run.log`.

Known implementation limitation: completion status is written before final resource/file cleanup; cleanup return values are not comprehensively checked. A complete metadata field alone is not proof of error-free cleanup.

## 7. Retrieve and Analyze

To retrieve a new run, set the actual run ID. Existing local runs are preserved.

```powershell
$project = 'C:\Users\swwan\OneDrive\Desktop\WES Summer\Final_Project_Release\final_project'
$board = 'rubik-pi-home'
$runId = 'formal-YYYYMMDD-HHMMSS'
$localResults = Join-Path $project 'results'
$localRun = Join-Path $localResults $runId
if (Test-Path -LiteralPath $localRun) { throw 'Local run already exists.' }
New-Item -ItemType Directory -Force -Path $localResults | Out-Null
scp -r "${board}:/home/ubuntu/final_project/results/$runId" "$localResults"
if ($LASTEXITCODE -ne 0) { throw 'Download failed.' }
```

To reanalyze the included official run, use a new output directory because the shipped `analysis/` already exists:

```powershell
Set-Location -LiteralPath $project
$runId = 'formal-20260918-200321'
$analysis = "results/$runId/analysis-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
py -3 scripts/analyze.py --input "results/$runId" --output "$analysis"
```

The analyzer validates samples against metadata and requires 5 warm-ups plus 30 measured runs. It supports metadata-defined aligned sizes and kernel selections; for the official contract, separately require all four specified sizes and all three kernels. It writes `summary.csv`, `crossover.txt`, `latency.png`, and `speedup.png` without editing raw measurements or removing outliers.

## 8. Troubleshooting

- Missing headers or loader: verify the board container and its toolchain.
- No GPU: inspect device mapping and the OpenCL platform list.
- Kernel build failure: read the OpenCL build log.
- Incorrect output: check indexing, signed nibble decoding, and barriers before timing.
- Analysis refuses overwrite: choose a new analysis output directory.
- Timing variability: retain samples and report median/IQR; do not substitute favorable runs.

## 9. Documentation Maintenance

Keep design and timing definitions in `architecture.md`, commands and evidence in this guide, and the measured answer in `final_conclusions.md`. Update current instructions when interfaces change; keep dated records clearly historical.

## 10. Historical Work Records

The following entries retain the development sequence and evidence available at each stage. Earlier pending statements are historical, not current requirements.

### Work Record: 2026-09-15

- Created documentation and an implementation plan from the final proposal; no OpenCL, Python, or container configuration files have been created yet.
- Local checks: relative file links resolve, code fences are balanced, and PowerShell examples parse successfully. These are not command-execution or board-validation results.
- Implemented the M1 CPU foundation in `include/int4_project.h`, `src/matrix.c`, `src/cpu_reference.c`, `tests/test_cpu.c`, and `Makefile`. Windows has no `gcc`, `clang`, or `make`, so compilation must be performed in the board container.
- User ran `make cpu-test` from `/workspaces/final_project`; the test executable reported `CPU tests passed: packing, deterministic inputs, conversion, reference GEMM`. This confirms the current CPU test run passed; the captured output does not by itself prove that a fresh rebuild occurred.
- Explicitly excluded partial-tile handling/tests for sizes not divisible by 16, preserving the user's scope decision.
- Recorded the RUBIK Pi 3 SoC naming correction while leaving actual board/GPU/driver confirmation pending.
- Translated all three documents into English at the user's request and made English the default project language. Technical scope, benchmark parameters, and command behavior are unchanged.

### Work Record: 2026-09-17

- Started M2. Added OpenCL platform/device enumeration, Qualcomm GPU selection, required `16×16` capability checks, a profiling-enabled command queue, program build-log reporting, and resource cleanup.
- Added `kernels/gemm_int32.cl`, a fixed aligned `16×16` tiled INT32 kernel, plus a host correctness path in `src/main.c` and `src/benchmark.c`.
- Local Windows compilation remains unavailable because `gcc`, `clang`, and `make` are not installed locally. Board-container compilation and the first GPU smoke check are pending.
- Implemented the M2 OpenCL baseline source: device enumeration/selection, capability checks, profiling-enabled queue, kernel build-log reporting, one INT32 tiled kernel, and a one-size correctness executable. Board compilation and execution are still pending.
- Suggested next step: copy the updated source to the board container and run `make`, `./build/gemm_bench --list-devices`, then `./build/gemm_bench --check-only --size 128`.

### Work Record: 2026-09-17 (M2 board result)

- Board container compiled both CPU and OpenCL targets successfully with `make`.
- `--list-devices` found Qualcomm platform 0 / Adreno 643 and PoCL platform 1 / ARM CPU.
- `--check-only --size 128` selected the Qualcomm GPU, confirmed `max_work_group=1024` and `local_memory=32768` bytes, and passed INT32 correctness with 0 mismatches and `kernel_ms=0.963000`.
- This is one correctness smoke run, not the official benchmark. The four-size INT32 gate and all INT8/INT4 tests remain pending.

### Work Record: 2026-09-17 (M3 source)

- Added INT8 and packed-INT4 tiled kernels with the same `16×16` mapping, local tiles, barriers, and INT32 accumulation as INT32.
- Extended the host path to generate shared INT32 inputs, derive INT8/INT4 representations, select `--kernels int32|int8|int4|all`, and compare every selected output with the same CPU reference.
- Added `kernels/gemm_all.cl` so the three kernels are built together. Board compilation and correctness evidence for this source revision is recorded below.

### Work Record: 2026-09-17 (M3 board result)

- After copying the updated source to the board container, `make` completed successfully and `./build/gemm_bench --check-only --size 128 --kernels all` selected Qualcomm Adreno 643.
- All three representations passed exact correctness at `N=128`: INT32 `0.964 ms`, INT8 `0.959 ms`, and packed INT4 `0.959 ms`; each reported `mismatches=0`.
- The command reported `Mode: correctness check only`; these timings are diagnostic kernel times, not official benchmark samples. The subsequent all-size gate is recorded below.

### Work Record: 2026-09-17 (M3 all-size correctness gate)

- Ran the correctness command for `N=128`, `256`, `512`, and `1024` on Qualcomm Adreno 643.
- INT32, INT8, and packed INT4 each passed at every official size with `mismatches=0`; the shell loop completed normally with exit status 0.
- Diagnostic kernel times increased with matrix size, but these are single correctness-run observations and must not be presented as the project's official latency statistics.
- M3 correctness is complete. The next implementation milestone is M4: fixed warm-ups, 30 measured runs per kernel/size, device-path timing, raw CSV, and metadata.

### Work Record: 2026-09-18 (M4 source implementation)

- Added `src/formal_benchmark.c` and `include/formal_benchmark.h`. Passing `--output DIR` enables formal mode with configurable sizes, seed, warm-ups, and measured runs.
- Extended OpenCL execution timing to record `kernel_ms` from the completed kernel event and `device_path_ms` from host-monotonic time spanning H2D submission through blocking D2H completion.
- Formal mode writes `raw.csv`, `packing.csv`, `metadata.json`, and `run.log`; it preserves partial rows and marks metadata `failed` if a timing, correctness, OpenCL, or output-writing failure occurs. It refuses to overwrite an existing artifact.
- At the time of this source-implementation entry, board compilation and execution were pending; the subsequent smoke and formal-run records below provide the board evidence.

### Work Record: 2026-09-18 (M4 board smoke result)

- On the board container, `make clean`, `make`, and `make cpu-test` completed successfully. The CPU tests passed.
- The updated one-size correctness check at `N=128` passed for INT32, INT8, and packed INT4 with zero mismatches. It now reports both kernel-event time and host device-path time.
- Formal smoke run `results/m4-smoke-20260918-191135` completed with one warm-up and two measured runs per kernel at `N=128`. This is a pipeline check, not official 5+30 data.
- The smoke artifact audit passed: 6 measured rows, zero mismatches, one packing row, metadata `status=complete`, and a complete run log.

### Work Record: 2026-09-18 (M4 formal collection execution)

- Ran `results/formal-20260918-200321` with the fixed contract: sizes `128,256,512,1024`, seed `237`, five warm-ups, and 30 measured runs for each of INT32, INT8, and packed INT4.
- The executable reported completion for all 12 kernel/size points and exited normally; the final artifact audit is recorded below.

### Work Record: 2026-09-18 (M4 formal artifact audit)

- Audited `results/formal-20260918-200321`: `raw.csv` has 361 lines (header plus 360 measured samples), and `packing.csv` has 5 lines (header plus four sizes).
- The raw-data audit found 12 unique `(size,kernel)` groups, each with exactly 30 trials. All timings were positive and all mismatch counts were zero.
- `metadata.json` reports `status=complete`, the four official sizes, five warm-ups, 30 measured runs, Qualcomm Adreno 643, and the OpenCL driver/version. `run.log` ends with `status=complete`.
- M4 is complete. The next milestone is M5: reproducible median/IQR statistics, speedups, crossover analysis, and two plots.

### Work Record: 2026-09-18 (M5 analysis source)

- Added `scripts/analyze.py`. It refuses incomplete metadata, duplicate or missing `(N,kernel,trial)` keys, nonzero mismatches, invalid timings, incorrect byte counts, or incomplete packing rows.
- It uses the specified linear-interpolation quantile position `h=(n-1)p`, writes median/Q1/Q3/IQR and speedups to `summary.csv`, records crossover observations in `crossover.txt`, and generates `latency.png` and `speedup.png`.
- The script never edits `raw.csv`; it refuses to overwrite existing analysis artifacts. Local Python syntax validation passed, and the retrieved official run was analyzed successfully below.

### Work Record: 2026-09-18 (M5 analysis result)

- Ran `scripts/analyze.py` on `results/formal-20260918-200321`; it validated all 360 samples and generated `analysis/summary.csv`, `crossover.txt`, `latency.png`, and `speedup.png`.
- Kernel-event median speedup for INT4 versus INT32 was approximately 1.005×, 1.014×, 1.018×, and 1.018× at N=128, 256, 512, and 1024. These are measured medians, not a claim that the differences are statistically significant.
- Device-path INT4 speedup was approximately 0.898×, 0.999×, 1.051×, and 1.061×. The analysis reports a device-path crossover between N=256 and N=512; no kernel-event crossover was observed.
- The figures were visually inspected and are readable. M5 analysis is complete; final conclusions/video remain for M6 delivery.

### Work Record: 2026-09-18 (M6 delivery preparation)

- Added `README.md` with scope, reproducibility commands, audited results, limitations, and the OpenVLA inspiration boundary.
- Added `Doc/final_conclusions.md` with the measured research answer and claim limitations.
- Added `Doc/video_script.md`, a 6–8 minute presentation script covering motivation, implementation, correctness, protocol, results, interpretation, and limitations.
- The video itself has not been recorded in this workspace. Final packaging and recording remain the last delivery tasks.
