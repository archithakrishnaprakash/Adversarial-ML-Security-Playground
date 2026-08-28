from __future__ import annotations

import io
import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.attacks import blackbox
from app.attacks.poisoning import label_flip_poison  # noqa: F401  (kept importable for tests/tools)
from app.attacks.runner import ATTACK_NAMES, run_attack
from app.defenses.adversarial_training import adversarial_train
from app.defenses.preprocessing import apply_preprocessing_defense
from app.evaluation.explainability import (
    image_to_base64,
    perturbation_heatmap_base64,
    tabular_feature_deltas,
)
from app.evaluation.leaderboard import build_leaderboard
from app.evaluation.poisoning_eval import run_backdoor_experiment, run_label_flip_experiment
from app.evaluation.risk_engine import generate_security_assessment
from app.evaluation.robustness import evaluate_robustness
from app.evaluation.transferability import compute_transferability_matrix
from app.experiments.runner import MAX_CONFIGURATIONS, run_experiment_grid
from app.models.registry import (
    VALID_MODEL_TYPES_BY_DATASET,
    get_model,
    list_models,
    train_model,
)
from app.schemas import (
    AdversarialTrainingRequest,
    BackdoorPoisoningRequest,
    BlackBoxAttackRequest,
    DefensePreprocessingRequest,
    ExperimentGridRequest,
    LabelFlipPoisoningRequest,
    RobustnessEvalRequest,
    RunAttackRequest,
    SecurityAssessmentRequest,
    ThreatModelCheckRequest,
    TrainModelRequest,
    TransferabilityRequest,
)
from app.storage import init_db, list_experiments, log_experiment
from app.threat_model import applicable_attacks, full_matrix, is_attack_applicable

app = FastAPI(title="Adversarial ML Security Playground", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only — tighten before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------- metadata --

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/datasets")
def datasets():
    return {
        "datasets": [
            {
                "id": "image",
                "name": "Handwritten Digits (MNIST-style, 8x8)",
                "type": "image",
                "num_classes": 10,
                "valid_models": VALID_MODEL_TYPES_BY_DATASET["image"],
            },
            {
                "id": "cyber",
                "name": "Network Intrusion Detection (synthetic)",
                "type": "tabular",
                "num_classes": 2,
                "class_names": ["BENIGN", "ATTACK"],
                "valid_models": VALID_MODEL_TYPES_BY_DATASET["cyber"],
            },
        ],
        "attacks": ATTACK_NAMES,
        "preprocessing_defenses": ["gaussian_smoothing", "feature_clipping", "normalization"],
    }


# ------------------------------------------------------------------ models --

def _model_summary(m) -> dict:
    return {
        "model_id": m.model_id,
        "dataset": m.dataset,
        "model_type": m.model_type,
        "metrics": m.metrics,
        "is_differentiable": m.is_differentiable,
        "supports_gradient_attacks": m.supports_gradient_attacks,
        "attack_note": (
            "Random Forest is non-differentiable — gradient attacks are crafted against a "
            "surrogate network trained to mimic it, then transferred."
            if m.model_type == "random_forest"
            else None
        ),
        "created_at": m.created_at,
    }


@app.post("/api/models/train")
def api_train_model(req: TrainModelRequest):
    try:
        m = train_model(req.dataset, req.model_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _model_summary(m)


@app.get("/api/models")
def api_list_models():
    return {"models": [_model_summary(m) for m in list_models()]}


@app.get("/api/models/{model_id}")
def api_get_model(model_id: str):
    try:
        m = get_model(model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _model_summary(m)


# -------------------------------------------------------------- attack lab --

@app.post("/api/attack/run")
def api_run_attack(req: RunAttackRequest):
    try:
        model = get_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Threat-model gate: defaults to "white_box", which preserves the exact
    # behavior every existing caller (including the current frontend) already
    # relies on. Only blocks the request if the caller explicitly asks for a
    # capability that can't run this attack (e.g. gradient attacks under
    # black_box) — the response tells them which attack to use instead.
    try:
        applicable, rationale = is_attack_applicable(req.attack, req.capability)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not applicable:
        raise HTTPException(
            status_code=400,
            detail=f"'{req.attack}' is not applicable under a '{req.capability}' threat model: "
                   f"{rationale} Use the Black-Box Lab ('transfer' or 'query') instead.",
        )

    X_test, y_test = model.X_test, model.y_test
    rng = np.random.default_rng()

    if req.sample_index is not None:
        idx = req.sample_index
    else:
        # pick a random sample the model currently gets right, so the "attack"
        # demo is meaningful (flipping an already-wrong prediction is not interesting)
        preds = model.predict(X_test)
        correct_idx = np.where(preds == y_test)[0]
        idx = int(rng.choice(correct_idx)) if len(correct_idx) else int(rng.integers(0, len(X_test)))

    x = X_test[idx : idx + 1]
    y = y_test[idx : idx + 1]

    try:
        x_adv = run_attack(model, x, y, req.attack, epsilon=req.epsilon, pgd_steps=req.pgd_steps)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    orig_probs = model.predict_proba(x)[0]
    adv_probs = model.predict_proba(x_adv)[0]
    orig_pred = int(orig_probs.argmax())
    adv_pred = int(adv_probs.argmax())
    perturbation_magnitude = float(np.mean(np.abs(x_adv[0] - x[0])))

    result = {
        "sample_index": int(idx),
        "true_label": int(y[0]),
        "original_prediction": orig_pred,
        "original_confidence": round(float(orig_probs[orig_pred]), 4),
        "adversarial_prediction": adv_pred,
        "adversarial_confidence": round(float(adv_probs[adv_pred]), 4),
        "attack_succeeded": bool(adv_pred != orig_pred),
        "perturbation_magnitude": round(perturbation_magnitude, 4),
        "attack": req.attack,
        "epsilon": req.epsilon,
        "capability": req.capability,
    }

    if model.is_image:
        result["original_image"] = image_to_base64(x[0])
        result["adversarial_image"] = image_to_base64(x_adv[0])
        result["perturbation_heatmap"] = perturbation_heatmap_base64(x[0], x_adv[0])
    else:
        result["feature_deltas"] = tabular_feature_deltas(x[0], x_adv[0], model.feature_names)
        result["class_names"] = ["BENIGN", "ATTACK"]

    log_experiment("attack", model.dataset, model.model_type, model.model_id, result,
                    attack=req.attack, epsilon=req.epsilon)
    return result


# ------------------------------------------------------------- defense lab --

@app.post("/api/defense/preprocessing")
def api_defense_preprocessing(req: DefensePreprocessingRequest):
    try:
        model = get_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X_test, y_test = model.X_test, model.y_test
    n = min(req.n_samples, len(X_test))
    X, y = X_test[:n], y_test[:n]

    clean_preds = model.predict(X)
    clean_acc = float((clean_preds == y).mean())

    try:
        X_adv = run_attack(model, X, y, req.attack, epsilon=req.epsilon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    adv_preds = model.predict(X_adv)
    adv_acc = float((adv_preds == y[: len(adv_preds)]).mean())

    try:
        X_defended = apply_preprocessing_defense(req.defense, X_adv)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    defended_preds = model.predict(X_defended)
    defended_acc = float((defended_preds == y[: len(defended_preds)]).mean())

    result = {
        "defense": req.defense,
        "attack": req.attack,
        "epsilon": req.epsilon,
        "n_samples": n,
        "clean_accuracy": round(clean_acc, 4),
        "adversarial_accuracy": round(adv_acc, 4),
        "defended_accuracy": round(defended_acc, 4),
        "accuracy_recovered": round(defended_acc - adv_acc, 4),
    }
    log_experiment("defense", model.dataset, model.model_type, model.model_id, result,
                    attack=req.attack, epsilon=req.epsilon)
    return result


@app.post("/api/defense/adversarial-training")
def api_adversarial_training(req: AdversarialTrainingRequest):
    try:
        base_model = get_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    robust_model = adversarial_train(base_model, epsilon=req.epsilon)

    X_test, y_test = base_model.X_test, base_model.y_test
    n = min(req.n_samples, len(X_test))
    X, y = X_test[:n], y_test[:n]

    comparison = {"clean": {}, "attacks": {}}
    comparison["clean"]["normal_model"] = round(float((base_model.predict(X) == y).mean()), 4)
    comparison["clean"]["robust_model"] = round(float((robust_model.predict(X) == y).mean()), 4)

    for attack in req.attacks_to_compare:
        try:
            X_adv_normal = run_attack(base_model, X, y, attack, epsilon=req.epsilon)
            normal_attacked_acc = float((base_model.predict(X_adv_normal) == y[: len(X_adv_normal)]).mean())
        except Exception as e:
            normal_attacked_acc = None

        try:
            X_adv_robust = run_attack(robust_model, X, y, attack, epsilon=req.epsilon)
            robust_attacked_acc = float((robust_model.predict(X_adv_robust) == y[: len(X_adv_robust)]).mean())
        except Exception as e:
            robust_attacked_acc = None

        comparison["attacks"][attack] = {
            "normal_model": round(normal_attacked_acc, 4) if normal_attacked_acc is not None else None,
            "robust_model": round(robust_attacked_acc, 4) if robust_attacked_acc is not None else None,
        }

    result = {
        "base_model_id": base_model.model_id,
        "robust_model_id": robust_model.model_id,
        "training_epsilon": req.epsilon,
        "comparison": comparison,
    }
    log_experiment("adversarial_training", base_model.dataset, base_model.model_type, base_model.model_id,
                    result, epsilon=req.epsilon)
    return result


# --------------------------------------------------------- robustness lab --

@app.post("/api/robustness/evaluate")
def api_robustness_evaluate(req: RobustnessEvalRequest):
    try:
        model = get_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = evaluate_robustness(model, req.attacks, epsilon=req.epsilon, n_samples=req.n_samples)
    log_experiment("robustness_eval", model.dataset, model.model_type, model.model_id, result,
                    epsilon=req.epsilon)
    return result


@app.post("/api/robustness/transferability")
def api_transferability(req: TransferabilityRequest):
    try:
        models = [get_model(mid) for mid in req.model_ids]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = compute_transferability_matrix(
            models, attack=req.attack, epsilon=req.epsilon, n_samples=req.n_samples, seed=req.seed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log_experiment("transferability", result["dataset"], "multiple", ",".join(req.model_ids), result,
                    attack=req.attack, epsilon=req.epsilon)
    return result


# ------------------------------------------------------------- threat model --

@app.get("/api/threat-model/matrix")
def api_threat_model_matrix():
    """The full capability x attack applicability table, used to populate the
    threat-model selector in the UI."""
    return {"matrix": full_matrix()}


@app.post("/api/threat-model/check")
def api_threat_model_check(req: ThreatModelCheckRequest):
    try:
        return {"attack": req.attack, "capability": req.capability,
                **dict(zip(("applicable", "rationale"), is_attack_applicable(req.attack, req.capability)))}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------- black-box lab --

@app.post("/api/attack/blackbox")
def api_blackbox_attack(req: BlackBoxAttackRequest):
    try:
        model = get_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    X_test, y_test = model.X_test, model.y_test
    n = min(req.n_samples, len(X_test))
    rng = np.random.default_rng(req.seed)
    idx = rng.choice(len(X_test), size=n, replace=False)
    X, y = X_test[idx], y_test[idx]

    clean_acc = float((model.predict(X) == y).mean())

    if req.method == "transfer":
        # The "query pool" here is the model's own held-out set standing in
        # for data an attacker could plausibly collect on their own — the
        # attacker never sees the true labels, only queries model.predict_proba.
        X_adv, queries_used = blackbox.transfer_attack(
            model.predict_proba,
            X_test,
            X,
            y,
            model.num_classes,
            req.epsilon,
            base_attack=req.base_attack,
            n_queries=req.n_queries,
            seed=req.seed,
        )
    elif req.method == "query":
        X_adv, queries_used = blackbox.query_attack(
            model.predict_proba,
            X,
            y,
            req.epsilon,
            max_queries_per_sample=req.max_queries_per_sample,
            seed=req.seed,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown black-box method '{req.method}'")

    adv_preds = model.predict(X_adv)
    adv_acc = float((adv_preds == y).mean())
    success_rate = float((adv_preds != model.predict(X)).mean())

    result = {
        "method": req.method,
        "model_id": model.model_id,
        "epsilon": req.epsilon,
        "n_samples": n,
        "clean_accuracy": round(clean_acc, 4),
        "adversarial_accuracy": round(adv_acc, 4),
        "attack_success_rate": round(success_rate, 4),
        "queries_used": int(queries_used),
        "queries_per_sample": round(queries_used / n, 1) if n else 0,
    }
    log_experiment("blackbox_attack", model.dataset, model.model_type, model.model_id, result,
                    attack=f"blackbox_{req.method}", epsilon=req.epsilon)
    return result


# ----------------------------------------------------------- poisoning lab --

@app.post("/api/poisoning/label-flip")
def api_poisoning_label_flip(req: LabelFlipPoisoningRequest):
    try:
        result = run_label_flip_experiment(req.dataset, req.model_type, req.poison_fraction, seed=req.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log_experiment("poisoning", req.dataset, req.model_type, result["poisoned_model_id"], result)
    return result


@app.post("/api/poisoning/backdoor")
def api_poisoning_backdoor(req: BackdoorPoisoningRequest):
    try:
        result = run_backdoor_experiment(
            req.dataset, req.model_type, req.poison_fraction, req.target_label, seed=req.seed
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    log_experiment("poisoning", req.dataset, req.model_type, result["poisoned_model_id"], result)
    return result


# --------------------------------------------------------- security engine --

@app.post("/api/security/assessment")
def api_security_assessment(req: SecurityAssessmentRequest):
    try:
        model = get_model(req.model_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    benchmark = evaluate_robustness(model, req.attacks, epsilon=req.epsilon, n_samples=req.n_samples)

    transferability = None
    if req.compare_model_ids:
        try:
            compare_models = [model] + [get_model(mid) for mid in req.compare_model_ids]
            transferability = compute_transferability_matrix(compare_models, epsilon=req.epsilon)
        except (KeyError, ValueError):
            transferability = None  # optional enrichment — don't fail the whole assessment for it

    assessment = generate_security_assessment(benchmark, transferability=transferability)
    log_experiment("security_assessment", model.dataset, model.model_type, model.model_id, assessment,
                    epsilon=req.epsilon)
    return assessment


@app.get("/api/leaderboard")
def api_leaderboard(sort_by: str = "robustness_score"):
    try:
        return {"leaderboard": build_leaderboard(sort_by=sort_by)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------- experiment grid --

@app.post("/api/experiments/grid")
def api_experiment_grid(req: ExperimentGridRequest):
    try:
        return run_experiment_grid(
            req.dataset, req.model_types, req.attacks, req.epsilons, req.defenses,
            n_samples=req.n_samples, seed=req.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/experiments/grid/limits")
def api_experiment_grid_limits():
    return {"max_configurations": MAX_CONFIGURATIONS}


# ------------------------------------------------------------ experiments --

@app.get("/api/experiments")
def api_list_experiments(limit: int = 100):
    return {"experiments": list_experiments(limit=limit)}


@app.get("/api/experiments/export")
def api_export_experiments():
    experiments = list_experiments(limit=1000)
    payload = json.dumps({"exported_at": __import__("datetime").datetime.utcnow().isoformat(),
                           "experiments": experiments}, indent=2)
    buf = io.BytesIO(payload.encode("utf-8"))
    return StreamingResponse(
        buf,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=robustness_report.json"},
    )
