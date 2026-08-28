"""
Security Risk Engine.

Turns a robustness benchmark (and, optionally, a transferability matrix and/or
a poisoning experiment) into something that reads like a security assessment:
a risk rating, a list of specific findings, and a deploy/don't-deploy
recommendation — rather than a bare 0-100 number the reader has to interpret
themselves.

Risk thresholds are intentionally the same ones already used by the
Robustness Report page's score coloring (>=70 green/LOW, 40-69 amber/MEDIUM,
<40 red/HIGH), so the number on screen and the rating agree with each other.
"""
from __future__ import annotations

RISK_THRESHOLDS = {"low": 70, "medium": 40}  # score >= low -> LOW risk, >= medium -> MEDIUM, else HIGH

_RECOMMENDED_ACTIONS = {
    "HIGH": [
        "Add adversarial training before considering deployment.",
        "Re-run the robustness benchmark against PGD specifically — it is "
        "typically the strongest of the evasion attacks here.",
        "Introduce an input anomaly/preprocessing defense as a stop-gap.",
    ],
    "MEDIUM": [
        "Adversarial training would likely close most of the remaining gap.",
        "Re-run the benchmark after any defense change to confirm improvement.",
    ],
    "LOW": [
        "Periodically re-run the benchmark — robustness can regress silently "
        "as a model is retrained on new data.",
    ],
}


def risk_rating_for_score(robustness_score: float) -> str:
    """Public helper — also used by the leaderboard so both surfaces agree on
    the same HIGH/MEDIUM/LOW thresholds."""
    if robustness_score >= RISK_THRESHOLDS["low"]:
        return "LOW"
    if robustness_score >= RISK_THRESHOLDS["medium"]:
        return "MEDIUM"
    return "HIGH"


def _recommendation(risk: str) -> str:
    return {"HIGH": "DO NOT DEPLOY", "MEDIUM": "DEPLOY WITH CAUTION", "LOW": "SAFE TO DEPLOY"}[risk]


def generate_security_assessment(
    benchmark: dict,
    transferability: dict | None = None,
    poisoning: dict | None = None,
) -> dict:
    """Builds a findings-driven security assessment from a robustness
    benchmark (required — the output of `evaluate_robustness`) plus optional
    transferability and poisoning results for extra findings.
    """
    if "robustness_score" not in benchmark:
        raise ValueError("benchmark must be the output of evaluate_robustness()")

    score = benchmark["robustness_score"]
    risk = risk_rating_for_score(score)
    clean_acc = benchmark["clean_accuracy"]

    findings = []
    for attack, result in benchmark.get("attacks", {}).items():
        if "error" in result:
            continue
        drop = clean_acc - result["accuracy"]
        if drop <= 0.02:
            severity = "LOW"
        elif drop <= 0.15:
            severity = "MEDIUM"
        else:
            severity = "HIGH"
        findings.append(
            {
                "severity": severity,
                "text": f"{attack.upper()} reduces accuracy from {clean_acc:.1%} to "
                        f"{result['accuracy']:.1%} ({drop:.1%} drop), with a "
                        f"{result['attack_success_rate']:.1%} attack success rate.",
            }
        )

    if transferability is not None:
        for source_id, row in transferability.get("matrix", {}).items():
            for target_id, rate in row.items():
                if source_id == target_id:
                    continue
                if rate >= 0.3:
                    label_src = transferability["model_labels"].get(source_id, source_id)
                    label_tgt = transferability["model_labels"].get(target_id, target_id)
                    findings.append(
                        {
                            "severity": "MEDIUM" if rate < 0.6 else "HIGH",
                            "text": f"Adversarial examples crafted against {label_src} transfer to "
                                    f"{label_tgt} with a {rate:.1%} success rate.",
                        }
                    )

    if poisoning is not None:
        drop = poisoning.get("accuracy_degradation")
        if drop is not None and drop > 0.02:
            findings.append(
                {
                    "severity": "MEDIUM" if drop < 0.1 else "HIGH",
                    "text": f"{poisoning['poison_fraction']:.0%} label-flip poisoning caused a "
                            f"{drop:.1%} accuracy degradation.",
                }
            )
        bsr = poisoning.get("backdoor_success_rate")
        if bsr is not None and bsr > 0.1:
            findings.append(
                {
                    "severity": "HIGH" if bsr > 0.5 else "MEDIUM",
                    "text": f"A backdoor trigger flips predictions to the target class with a "
                            f"{bsr:.1%} success rate while clean accuracy stays at "
                            f"{poisoning['clean_accuracy_after_poisoning']:.1%}.",
                }
            )

    if not findings:
        findings.append({"severity": "LOW", "text": "No significant vulnerabilities found in this evaluation."})

    # surface the worst findings first
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda f: order[f["severity"]])

    return {
        "model_id": benchmark["model_id"],
        "model_type": benchmark["model_type"],
        "dataset": benchmark["dataset"],
        "risk_rating": risk,
        "robustness_score": score,
        "clean_accuracy": clean_acc,
        "recommendation": _recommendation(risk),
        "recommended_actions": _RECOMMENDED_ACTIONS[risk],
        "findings": findings,
    }
