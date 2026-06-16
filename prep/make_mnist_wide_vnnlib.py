from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPSILONS = (1.0, 5.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VNNLIB robustness specs for the Assignment 3 MNIST-wide model."
    )
    parser.add_argument("--sample", type=Path, default=ROOT / "data" / "mnist_sample_a3.npz")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "models" / "mnist_wide_model_metadata_a3.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "specs" / "mnist_wide")
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    return parser.parse_args()


def epsilon_tag(epsilon: float) -> str:
    if float(epsilon).is_integer():
        return str(int(epsilon))
    return str(epsilon).replace("-", "neg_").replace(".", "_")


def format_float(value: float) -> str:
    return f"{value:.10g}"


def build_vnnlib(raw_input: np.ndarray, label: int, epsilon: float) -> str:
    lines: list[str] = [
        f"; MNIST wide MLP local robustness spec, epsilon={epsilon}",
        "; Input space uses raw pixel values [0, 255], matching Assignment 3.",
        "",
    ]

    for idx in range(raw_input.size):
        lines.append(f"(declare-const X_{idx} Real)")
    lines.append("")
    for idx in range(10):
        lines.append(f"(declare-const Y_{idx} Real)")
    lines.append("")

    for idx, value in enumerate(raw_input.astype(float).reshape(-1)):
        lower = max(0.0, value - epsilon)
        upper = value + epsilon
        lines.append(f"(assert (>= X_{idx} {format_float(lower)}))")
        lines.append(f"(assert (<= X_{idx} {format_float(upper)}))")
    lines.append("")

    lines.append("; Unsafe condition: at least one other logit is >= the predicted label logit.")
    lines.append("; alpha-beta-CROWN proves robustness by showing this bad region is UNSAT.")
    lines.append("(assert (or")
    for target in range(10):
        if target == label:
            continue
        lines.append(f"  (and (<= Y_{label} Y_{target}))")
    lines.append("))")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.sample.exists():
        raise FileNotFoundError(f"Missing sample file: {args.sample}")
    if not args.metadata.exists():
        raise FileNotFoundError(f"Missing metadata file: {args.metadata}")

    sample = np.load(args.sample)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    raw_input = sample["raw_input"].astype(np.float32)
    label = int(metadata["sample_predicted_label"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epsilon in args.epsilons:
        spec = build_vnnlib(raw_input, label, float(epsilon))
        out_path = args.output_dir / f"mnist_wide_eps_{epsilon_tag(epsilon)}.vnnlib"
        out_path.write_text(spec, encoding="utf-8")
        compiled_cache = out_path.with_suffix(out_path.suffix + ".compiled")
        if compiled_cache.exists():
            compiled_cache.unlink()
        print(f"[write] {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
