from __future__ import annotations

import io
import json

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.attacks.runner import ATTACK_NAMES, run_attack
from app.defenses.adversarial_training import adversarial_train
from app.defenses.preprocessing import apply_preprocessing_defense
from app.evaluation.explainability import (
    image_to_base64,
    perturbation_heatmap_base64,
    tabular_feature_deltas,
)
from app.evaluation.robustness import evaluate_robustness
from app.models.registry import (
    VALID_MODEL_TYPES_BY_DATASET,
    get_model,
    list_models,
    train_model,
)
from app.schemas import (
    AdversarialTrainingRequest,
    DefensePreprocessingRequest,
    RobustnessEvalRequest,
    RunAttackRequest,
    TrainModelRequest,
)
from app.storage import init_db, list_experiments, log_experiment

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
