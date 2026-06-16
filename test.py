from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERIFIER_DIR = ROOT / "alpha-beta-CROWN" / "complete_verifier"

EXPERIMENTS = [
    {
        "epsilon": 1.0,
        "tag": "mnist_wide_eps_1",
        "config": ROOT / "configs" / "mnist_wide_eps_1.yaml",
        "marabou": ROOT / "results" / "marabou_baseline" / "mnist_wide_eps_1_0.json",
    },
    {
        "epsilon": 5.0,
        "tag": "mnist_wide_eps_5",
        "config": ROOT / "configs" / "mnist_wide_eps_5.yaml",
        "marabou": ROOT / "results" / "marabou_baseline" / "mnist_wide_eps_5_0.json",
    },
    {
        "epsilon": 10.0,
        "tag": "mnist_wide_eps_10",
        "config": ROOT / "configs" / "mnist_wide_eps_10.yaml",
        "marabou": ROOT / "results" / "marabou_baseline" / "mnist_wide_eps_10_0.json",
    },
]


def run_step(args: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def ensure_prepared_assets() -> None:
    required = [
        ROOT / "models" / "mnist_wide_mlp_a3.onnx",
        ROOT / "models" / "mnist_wide_model_metadata_a3.json",
        ROOT / "data" / "mnist_sample_a3.npz",
        ROOT / "specs" / "mnist_wide" / "mnist_wide_eps_1.vnnlib",
        ROOT / "specs" / "mnist_wide" / "mnist_wide_eps_5.vnnlib",
        ROOT / "specs" / "mnist_wide" / "mnist_wide_eps_10.vnnlib",
    ]
    if all(path.exists() for path in required):
        return

    print("[INFO] Preparing Assignment 3 assets and VNNLIB specs...")
    run_step([sys.executable, "prep/prepare_a3_assets.py"])
    run_step([sys.executable, "prep/make_mnist_wide_vnnlib.py"])


def remove_vnnlib_cache(config: Path) -> None:
    tag = config.stem
    cache_path = ROOT / "specs" / "mnist_wide" / f"{tag}.vnnlib.compiled"
    if cache_path.exists():
        cache_path.unlink()


def parse_abcrown_output(output: str) -> tuple[str, str]:
    result_line = ""
    for line in output.splitlines():
        if line.startswith("Result:"):
            result_line = line.strip()

    if result_line:
        result = result_line.split(":", 1)[1].strip().lower()
        if result == "unsat":
            return "verified", result_line
        if result == "sat":
            return "falsified", result_line
        if result == "timeout":
            return "timeout", result_line
        if result == "unknown":
            return "unknown", result_line

    summary_match = re.search(
        r"total verified .*?:\s*(\d+)\s*, total falsified .*?:\s*(\d+)\s*, timeout:\s*(\d+)",
        output,
    )
    if summary_match:
        verified, falsified, timeout = map(int, summary_match.groups())
        if falsified:
            return "falsified", result_line
        if timeout:
            return "timeout", result_line
        if verified:
            return "verified", result_line

    lowered = output.lower()
    if re.search(r"\bunsafe\b|\bfalsified\b|\bsat\b", lowered):
        return "falsified", result_line
    if re.search(r"\bsafe\b|\bverified\b|\bunsat\b", lowered):
        return "verified", result_line
    if "timeout" in lowered:
        return "timeout", result_line
    return "unknown", result_line


def marabou_summary(path: Path) -> tuple[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    status = data["status"]
    target_statuses = {item["status"] for item in data.get("target_results", [])}
    if status == "SAT" and "TIMEOUT" in target_statuses:
        status = "SAT+TIMEOUT"
    return status, float(data["runtime_seconds"])


def run_abcrown(experiment: dict[str, object]) -> dict[str, object]:
    tag = str(experiment["tag"])
    config = Path(experiment["config"])
    output_dir = ROOT / "results" / "abcrown" / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.txt"

    remove_vnnlib_cache(config)
    config_arg = Path("../../configs") / config.name
    cmd = [sys.executable, "abcrown.py", "--config", str(config_arg)]

    print(f"[INFO] Running alpha-beta-CROWN for {tag}...")
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=VERIFIER_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    runtime = time.perf_counter() - start
    output_path.write_text(completed.stdout, encoding="utf-8")

    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"alpha-beta-CROWN failed for {tag}; see {output_path}")

    internal_result_path = output_dir / "abcrown_results.txt"
    if internal_result_path.exists():
        internal_result_path.unlink()
    remove_vnnlib_cache(config)

    status, result_line = parse_abcrown_output(completed.stdout)
    print(f"[INFO] {tag}: {status} in {runtime:.2f}s")
    return {
        "epsilon": experiment["epsilon"],
        "tag": tag,
        "config": str(config.relative_to(ROOT)),
        "abcrown_status": status,
        "abcrown_runtime_seconds": runtime,
        "abcrown_result_line": result_line,
        "abcrown_output": str(output_path.relative_to(ROOT)),
    }


def write_summary(rows: list[dict[str, object]]) -> Path:
    summary_path = ROOT / "results" / "abcrown_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon",
        "tag",
        "config",
        "abcrown_status",
        "abcrown_runtime_seconds",
        "abcrown_result_line",
        "marabou_status",
        "marabou_runtime_seconds",
        "abcrown_output",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    if not VERIFIER_DIR.exists():
        raise FileNotFoundError(f"Missing alpha-beta-CROWN checkout: {VERIFIER_DIR}")

    ensure_prepared_assets()

    rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        row = run_abcrown(experiment)
        marabou_status, marabou_runtime = marabou_summary(Path(experiment["marabou"]))
        row["marabou_status"] = marabou_status
        row["marabou_runtime_seconds"] = marabou_runtime
        rows.append(row)

    summary_path = write_summary(rows)
    print(f"[INFO] Summary written to {summary_path.relative_to(ROOT)}")
    for row in rows:
        print(
            "[SUMMARY] "
            f"eps={row['epsilon']} | alpha-beta-CROWN={row['abcrown_status']} "
            f"({row['abcrown_runtime_seconds']:.2f}s) | "
            f"Marabou={row['marabou_status']} ({row['marabou_runtime_seconds']:.2f}s)"
        )


if __name__ == "__main__":
    main()
