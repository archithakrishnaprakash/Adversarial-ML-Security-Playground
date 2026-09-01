#!/usr/bin/env python3
"""
ML Security Gate — a small CLI meant to run in CI (see
.github/workflows/ml-security-gate.yml at the project root).

Trains a model, runs a robustness benchmark against it, and exits non-zero if
robust accuracy under the given attack falls below the required threshold —
the same idea as a test-coverage gate, applied to adversarial robustness
instead of test coverage. Run it from the `backend/` directory:

    python scripts/security_gate.py --dataset cyber --model logistic_regression \\
        --attack pgd --epsilon 0.15 --threshold 0.70

Exit code 0 = gate passed, 1 = gate failed, 2 = bad arguments / runtime error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as `python scripts/security_gate.py` from the backend/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.robustness import evaluate_robustness  # noqa: E402
from app.models.registry import VALID_MODEL_TYPES_BY_DATASET, train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ML robustness security gate")
    parser.add_argument("--dataset", choices=["image", "cyber"], default="cyber")
    parser.add_argument("--model", dest="model_type", default="logistic_regression")
    parser.add_argument("--attack", default="pgd", choices=["fgsm", "pgd", "deepfool", "random_noise"])
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.70,
                         help="minimum required robust accuracy under --attack, as a fraction (0-1)")
    parser.add_argument("--n-samples", type=int, default=300)
    args = parser.parse_args()

    if args.model_type not in VALID_MODEL_TYPES_BY_DATASET[args.dataset]:
        print(f"error: model '{args.model_type}' is not valid for dataset '{args.dataset}'. "
              f"Valid options: {VALID_MODEL_TYPES_BY_DATASET[args.dataset]}")
        return 2

    print(f"Training {args.model_type} on {args.dataset}...")
    model = train_model(args.dataset, args.model_type)
    print(f"Clean accuracy: {model.metrics['clean_accuracy']:.1%}")

    print(f"Running {args.attack} at epsilon={args.epsilon}...")
    result = evaluate_robustness(model, [args.attack], epsilon=args.epsilon, n_samples=args.n_samples)
    attack_result = result["attacks"].get(args.attack, {})
    if "error" in attack_result:
        print(f"error: attack failed to run: {attack_result['error']}")
        return 2

    robust_accuracy = attack_result["accuracy"]
    print()
    print("MODEL SECURITY PIPELINE")
    print(f"  clean accuracy:      {result['clean_accuracy']:.1%}")
    print(f"  robust accuracy:     {robust_accuracy:.1%}  (under {args.attack}, eps={args.epsilon})")
    print(f"  attack success rate: {attack_result['attack_success_rate']:.1%}")
    print(f"  required threshold:  {args.threshold:.1%}")
    print()

    if robust_accuracy < args.threshold:
        print("=" * 40)
        print("SECURITY GATE FAILED")
        print("=" * 40)
        return 1

    print("=" * 40)
    print("SECURITY GATE PASSED")
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
