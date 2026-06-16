from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_A3_ROOT = ROOT / "trustworthy-ai-assignment3"

COPIES = [
    ("model/mnist_wide_mlp.onnx", "models/mnist_wide_mlp_a3.onnx"),
    ("model/mnist_wide_model_metadata.json", "models/mnist_wide_model_metadata_a3.json"),
    ("data/mnist_sample.npz", "data/mnist_sample_a3.npz"),
    (
        "results/mnist_wide_eps_1_0/verification_result.json",
        "results/marabou_baseline/mnist_wide_eps_1_0.json",
    ),
    (
        "results/mnist_wide_eps_5_0/verification_result.json",
        "results/marabou_baseline/mnist_wide_eps_5_0.json",
    ),
    (
        "results/mnist_wide_eps_10_0/verification_result.json",
        "results/marabou_baseline/mnist_wide_eps_10_0.json",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the Assignment 3 MNIST-wide model, sample, and Marabou baseline results."
    )
    parser.add_argument(
        "--assignment3-root",
        type=Path,
        default=DEFAULT_A3_ROOT,
        help="Path to the cloned Assignment 3 repository.",
    )
    return parser.parse_args()


def copy_file(source_root: Path, source_rel: str, target_rel: str) -> None:
    source = source_root / source_rel
    target = ROOT / target_rel
    if not source.exists():
        raise FileNotFoundError(f"Missing Assignment 3 asset: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"[copy] {source.relative_to(source_root)} -> {target.relative_to(ROOT)}")


def write_baseline_summary() -> None:
    rows = ["epsilon,marabou_status,marabou_runtime_seconds,notes"]
    for epsilon, filename in [
        (1.0, "mnist_wide_eps_1_0.json"),
        (5.0, "mnist_wide_eps_5_0.json"),
        (10.0, "mnist_wide_eps_10_0.json"),
    ]:
        path = ROOT / "results" / "marabou_baseline" / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        target_statuses = {item["status"] for item in data.get("target_results", [])}
        status = data["status"]
        notes = "contains_target_timeout" if "TIMEOUT" in target_statuses else ""
        rows.append(f"{epsilon},{status},{data['runtime_seconds']:.6f},{notes}")

    summary_path = ROOT / "results" / "marabou_baseline" / "summary.csv"
    summary_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[write] {summary_path.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()
    source_root = args.assignment3_root.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Assignment 3 repository not found: {source_root}")

    for source_rel, target_rel in COPIES:
        copy_file(source_root, source_rel, target_rel)
    write_baseline_summary()


if __name__ == "__main__":
    main()
