#!/usr/bin/env python3
"""Validate and summarize one completed OpenCL INT4 GEMM run.

The script treats raw.csv as immutable input. It refuses incomplete runs,
duplicate samples, mismatches, invalid timings, and missing points before
writing any analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


KERNELS = ("int32", "int8", "int4")
RAW_FIELDS = {
    "n", "kernel", "trial", "seed", "kernel_ms", "device_path_ms",
    "input_bytes", "output_bytes", "mismatches",
}
PACKING_FIELDS = {"n", "seed", "pack_a_b_ms", "input_bytes"}


class AnalysisError(Exception):
    """A user-facing validation or analysis error."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="completed run directory containing raw.csv")
    parser.add_argument("--output", required=True, type=Path,
                        help="new directory for summary and plots")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"metadata root is not an object: {path}")
    return value


def load_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            missing = required_fields - fields
            if missing:
                raise AnalysisError(
                    f"{path.name} is missing columns: {', '.join(sorted(missing))}"
                )
            return list(reader)
    except OSError as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    except csv.Error as exc:
        raise AnalysisError(f"cannot parse {path}: {exc}") from exc


def as_int(value: str, field: str, path: Path) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid integer {field}={value!r} in {path}") from exc


def as_float(value: str, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid number {field}={value!r} in {path}") from exc
    if not math.isfinite(parsed):
        raise AnalysisError(f"non-finite {field} in {path}")
    return parsed


def validate_metadata(metadata: dict) -> tuple[list[int], tuple[str, ...], int, int]:
    if metadata.get("status") != "complete":
        raise AnalysisError(
            f"metadata status is {metadata.get('status')!r}; only complete runs can be analyzed"
        )
    raw_sizes = metadata.get("sizes")
    if not isinstance(raw_sizes, list) or not raw_sizes:
        raise AnalysisError("metadata.sizes must be a non-empty list")
    try:
        sizes = [int(value) for value in raw_sizes]
    except (TypeError, ValueError) as exc:
        raise AnalysisError("metadata.sizes contains a non-integer") from exc
    if len(set(sizes)) != len(sizes) or any(size <= 0 or size % 16 != 0 for size in sizes):
        raise AnalysisError("metadata.sizes must be unique positive multiples of 16")
    kernel_value = metadata.get("kernels")
    if kernel_value == "all":
        kernels = KERNELS
    elif kernel_value in KERNELS:
        kernels = (kernel_value,)
    else:
        raise AnalysisError(f"unsupported metadata.kernels value: {kernel_value!r}")
    seed = metadata.get("seed")
    if not isinstance(seed, int) or seed == 0:
        raise AnalysisError("metadata.seed must be a nonzero integer")
    if metadata.get("warmups") != 5 or metadata.get("measured_runs") != 30:
        raise AnalysisError("the analyzer requires 5 warm-ups and 30 measured runs")
    return sizes, kernels, seed, 30


def expected_input_bytes(n: int, kernel: str) -> int:
    if kernel == "int32":
        return 8 * n * n
    if kernel == "int8":
        return 2 * n * n
    if kernel == "int4":
        return n * n
    raise AnalysisError(f"unknown kernel {kernel}")


def validate_raw(rows: list[dict[str, str]], path: Path, sizes: list[int],
                kernels: tuple[str, ...], seed: int,
                measured_runs: int) -> list[dict[str, object]]:
    expected_count = len(sizes) * len(kernels) * measured_runs
    if len(rows) != expected_count:
        raise AnalysisError(f"raw.csv has {len(rows)} rows; expected {expected_count}")
    expected_sizes = set(sizes)
    expected_kernels = set(kernels)
    seen: set[tuple[int, str, int]] = set()
    records: list[dict[str, object]] = []
    for line, row in enumerate(rows, start=2):
        n = as_int(row["n"], "n", path)
        kernel = row["kernel"]
        trial = as_int(row["trial"], "trial", path)
        row_seed = as_int(row["seed"], "seed", path)
        kernel_ms = as_float(row["kernel_ms"], "kernel_ms", path)
        device_path_ms = as_float(row["device_path_ms"], "device_path_ms", path)
        input_bytes = as_int(row["input_bytes"], "input_bytes", path)
        output_bytes = as_int(row["output_bytes"], "output_bytes", path)
        mismatches = as_int(row["mismatches"], "mismatches", path)
        if n not in expected_sizes or kernel not in expected_kernels:
            raise AnalysisError(f"unexpected size/kernel on raw.csv line {line}")
        if trial < 1 or trial > measured_runs:
            raise AnalysisError(f"invalid trial on raw.csv line {line}")
        key = (n, kernel, trial)
        if key in seen:
            raise AnalysisError(f"duplicate sample {key} on raw.csv line {line}")
        seen.add(key)
        if row_seed != seed:
            raise AnalysisError(f"seed mismatch on raw.csv line {line}")
        if kernel_ms <= 0.0 or device_path_ms <= 0.0:
            raise AnalysisError(f"non-positive timing on raw.csv line {line}")
        if mismatches != 0:
            raise AnalysisError(f"mismatches={mismatches} on raw.csv line {line}")
        if input_bytes != expected_input_bytes(n, kernel):
            raise AnalysisError(f"input byte count mismatch on raw.csv line {line}")
        if output_bytes != 4 * n * n:
            raise AnalysisError(f"output byte count mismatch on raw.csv line {line}")
        records.append({
            "n": n, "kernel": kernel, "trial": trial,
            "kernel_ms": kernel_ms, "device_path_ms": device_path_ms,
        })
    expected_keys = {
        (n, kernel, trial)
        for n in sizes for kernel in kernels
        for trial in range(1, measured_runs + 1)
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        raise AnalysisError(f"raw.csv is missing samples, beginning with {missing[:3]}")
    return records


def validate_packing(rows: list[dict[str, str]], path: Path, sizes: list[int],
                     seed: int) -> None:
    if len(rows) != len(sizes):
        raise AnalysisError(f"packing.csv has {len(rows)} rows; expected {len(sizes)}")
    seen: set[int] = set()
    for line, row in enumerate(rows, start=2):
        n = as_int(row["n"], "n", path)
        row_seed = as_int(row["seed"], "seed", path)
        pack_ms = as_float(row["pack_a_b_ms"], "pack_a_b_ms", path)
        input_bytes = as_int(row["input_bytes"], "input_bytes", path)
        if n not in sizes or n in seen:
            raise AnalysisError(f"unexpected or duplicate packing size on line {line}")
        if row_seed != seed or pack_ms <= 0.0 or input_bytes != n * n:
            raise AnalysisError(f"invalid packing row on line {line}")
        seen.add(n)
    if seen != set(sizes):
        raise AnalysisError("packing.csv does not contain every official size")


def quantile(values: list[float], probability: float) -> float:
    """Linear interpolation at h=(n-1)p, as specified by architecture.md."""
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot calculate a quantile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize(records: list[dict[str, object]], sizes: list[int],
              kernels: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for record in records:
        key = (int(record["n"]), str(record["kernel"]))
        grouped.setdefault(key, []).append(record)
    summary: list[dict[str, object]] = []
    for n in sizes:
        for kernel in kernels:
            group = grouped[(n, kernel)]
            kernel_values = [float(item["kernel_ms"]) for item in group]
            device_values = [float(item["device_path_ms"]) for item in group]
            kq1 = quantile(kernel_values, 0.25)
            kq3 = quantile(kernel_values, 0.75)
            dq1 = quantile(device_values, 0.25)
            dq3 = quantile(device_values, 0.75)
            summary.append({
                "n": n, "kernel": kernel, "sample_count": len(group),
                "kernel_median_ms": quantile(kernel_values, 0.5),
                "kernel_q1_ms": kq1, "kernel_q3_ms": kq3,
                "kernel_iqr_ms": kq3 - kq1,
                "device_path_median_ms": quantile(device_values, 0.5),
                "device_path_q1_ms": dq1, "device_path_q3_ms": dq3,
                "device_path_iqr_ms": dq3 - dq1,
            })
    baselines = {row["n"]: row for row in summary if row["kernel"] == "int32"}
    for row in summary:
        baseline = baselines[row["n"]]
        row["kernel_baseline_median_ms"] = baseline["kernel_median_ms"]
        row["kernel_speedup"] = baseline["kernel_median_ms"] / row["kernel_median_ms"]
        row["device_path_baseline_median_ms"] = baseline["device_path_median_ms"]
        row["device_path_speedup"] = (
            baseline["device_path_median_ms"] / row["device_path_median_ms"]
        )
    return summary


SUMMARY_FIELDS = [
    "n", "kernel", "sample_count",
    "kernel_median_ms", "kernel_q1_ms", "kernel_q3_ms", "kernel_iqr_ms",
    "device_path_median_ms", "device_path_q1_ms", "device_path_q3_ms",
    "device_path_iqr_ms", "kernel_baseline_median_ms", "kernel_speedup",
    "device_path_baseline_median_ms", "device_path_speedup",
]


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    field: (f"{row[field]:.9f}" if isinstance(row[field], float)
                            else row[field])
                    for field in SUMMARY_FIELDS
                })
    except OSError as exc:
        raise AnalysisError(f"cannot write {path}: {exc}") from exc


def write_crossover(path: Path, summary: list[dict[str, object]], sizes: list[int]) -> None:
    values = {"kernel": {}, "device_path": {}}
    for row in summary:
        if row["kernel"] == "int4":
            n = int(row["n"])
            values["kernel"][n] = float(row["kernel_speedup"])
            values["device_path"][n] = float(row["device_path_speedup"])
    lines = []
    for metric, metric_values in values.items():
        crossings = []
        for left, right in zip(sizes, sizes[1:]):
            if (metric_values[left] > 1.0) != (metric_values[right] > 1.0):
                crossings.append(f"between {left} and {right}")
        if crossings:
            lines.append(f"{metric}: crossover observed {'; '.join(crossings)}")
        else:
            lines.append(f"{metric}: no INT4 crossover observed in tested sizes")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise AnalysisError(
            "matplotlib is required for plots; install it in the local analysis environment"
        ) from exc
    return pyplot


def write_plots(output: Path, summary: list[dict[str, object]], sizes: list[int]) -> None:
    pyplot = import_matplotlib()
    colors = {"int32": "#333333", "int8": "#377eb8", "int4": "#e41a1c"}

    def rows_for(kernel: str) -> list[dict[str, object]]:
        return [row for row in summary if row["kernel"] == kernel]

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5), sharex=True)
    for kernel in KERNELS:
        rows = rows_for(kernel)
        if not rows:
            continue
        x = [row["n"] for row in rows]
        for axis, prefix, title in (
            (axes[0], "kernel", "Kernel event latency"),
            (axes[1], "device_path", "Device-path latency"),
        ):
            median = [row[f"{prefix}_median_ms"] for row in rows]
            q1 = [row[f"{prefix}_q1_ms"] for row in rows]
            q3 = [row[f"{prefix}_q3_ms"] for row in rows]
            axis.plot(x, median, marker="o", label=kernel, color=colors[kernel])
            axis.fill_between(x, q1, q3, color=colors[kernel], alpha=0.15)
            axis.set_title(title)
            axis.set_xlabel("N (M=N=K)")
            axis.set_ylabel("milliseconds")
            axis.set_xscale("log", base=2)
            axis.grid(True, alpha=0.25)
    for axis in axes:
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.legend()
    fig.suptitle("OpenCL GEMM latency: median with Q1–Q3")
    fig.tight_layout()
    fig.savefig(output / "latency.png", dpi=180)
    pyplot.close(fig)

    fig, axes = pyplot.subplots(1, 2, figsize=(12, 5), sharex=True)
    for kernel in KERNELS:
        rows = rows_for(kernel)
        if not rows:
            continue
        x = [row["n"] for row in rows]
        for axis, field, title in (
            (axes[0], "kernel_speedup", "Kernel speedup vs INT32"),
            (axes[1], "device_path_speedup", "Device-path speedup vs INT32"),
        ):
            values = [row[field] for row in rows]
            axis.plot(x, values, marker="o", label=kernel, color=colors[kernel])
            axis.set_title(title)
            axis.set_xlabel("N (M=N=K)")
            axis.set_ylabel("INT32 median / candidate median")
            axis.set_xscale("log", base=2)
            axis.axhline(1.0, color="#777777", linewidth=1)
            axis.grid(True, alpha=0.25)
    for axis in axes:
        axis.set_xticks(sizes)
        axis.set_xticklabels([str(size) for size in sizes])
        axis.legend()
    fig.suptitle("OpenCL GEMM speedup: values above 1 are faster than INT32")
    fig.tight_layout()
    fig.savefig(output / "speedup.png", dpi=180)
    pyplot.close(fig)


def analyze(input_dir: Path, output_dir: Path) -> None:
    metadata = load_json(input_dir / "metadata.json")
    sizes, kernels, seed, measured_runs = validate_metadata(metadata)
    raw_path = input_dir / "raw.csv"
    packing_path = input_dir / "packing.csv"
    records = validate_raw(load_csv(raw_path, RAW_FIELDS), raw_path, sizes,
                           kernels, seed, measured_runs)
    validate_packing(load_csv(packing_path, PACKING_FIELDS), packing_path,
                     sizes, seed)
    summary = summarize(records, sizes, kernels)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = [output_dir / name for name in
                    ("summary.csv", "crossover.txt", "latency.png", "speedup.png")]
    existing = [path.name for path in output_files if path.exists()]
    if existing:
        raise AnalysisError(f"refusing to overwrite existing analysis files: {', '.join(existing)}")
    write_summary(output_dir / "summary.csv", summary)
    write_crossover(output_dir / "crossover.txt", summary, sizes)
    write_plots(output_dir, summary, sizes)
    print(f"Validated {len(records)} measured rows across {len(sizes)} sizes and {len(kernels)} kernels.")
    print(f"Analysis written to {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        analyze(args.input, args.output)
    except AnalysisError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
