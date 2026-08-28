from __future__ import annotations

from app.evaluation.risk_engine import risk_rating_for_score
from app.models.registry import MODEL_STORE
from app.storage import list_experiments


def build_leaderboard(sort_by: str = "robustness_score") -> list[dict]:
    """Pulls the most recent `robustness_eval` experiment for each model
    that's still in the registry and ranks them. Models that have never had a
    robustness evaluation run don't appear — there's nothing to rank yet.
    """
    valid_sort_keys = {"robustness_score", "clean_accuracy", "attack_success_rate"}
    if sort_by not in valid_sort_keys:
        raise ValueError(f"sort_by must be one of {valid_sort_keys}")

    experiments = list_experiments(limit=1000)
    latest_by_model: dict[str, dict] = {}
    for exp in experiments:  # already newest-first from list_experiments
        if exp["experiment_type"] != "robustness_eval":
            continue
        model_id = exp["model_id"]
        if model_id not in latest_by_model:
            latest_by_model[model_id] = exp

    rows = []
    for model_id, exp in latest_by_model.items():
        if model_id not in MODEL_STORE:
            continue  # model no longer registered (e.g. server restarted)
        result = exp["result"]
        attack_results = [v for v in result.get("attacks", {}).values() if "attack_success_rate" in v]
        avg_asr = (
            round(sum(v["attack_success_rate"] for v in attack_results) / len(attack_results), 4)
            if attack_results
            else None
        )
        rows.append(
            {
                "model_id": model_id,
                "model_type": exp["model_type"],
                "dataset": exp["dataset"],
                "robustness_score": result.get("robustness_score"),
                "clean_accuracy": result.get("clean_accuracy"),
                "attack_success_rate": avg_asr,
                "risk_rating": risk_rating_for_score(result.get("robustness_score", 0)),
                "evaluated_at": exp["timestamp"],
            }
        )

    reverse = sort_by != "attack_success_rate"  # lower ASR is better, higher score/accuracy is better
    rows.sort(key=lambda r: (r[sort_by] is not None, r[sort_by]), reverse=reverse)
    return rows
