from __future__ import annotations

import uuid

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from app.attacks.fgsm import fgsm_attack
from app.models.registry import (
    DEVICE,
    MODEL_STORE,
    TrainedModel,
    _train_surrogate_for_rf,
    _train_torch_classifier,
    get_data,
)
from app.models.torch_models import LogisticRegressionTorch, SmallCNN, SmallNN

_MODEL_CLASSES = {
    "logistic_regression": LogisticRegressionTorch,
    "small_nn": SmallNN,
    "cnn": SmallCNN,
}


def adversarial_train(base_model: TrainedModel, epsilon: float = 0.15) -> TrainedModel:
    """Produces a new, hardened model trained on a mix of clean and FGSM
    adversarial examples generated at each stage (Goodfellow et al. style
    adversarial training). Returns a *new* TrainedModel registered under its
    own id, so the caller can compare original vs. robust side by side.
    """
    X_train, X_test, y_train, y_test = get_data(base_model.dataset)

    if base_model.model_type == "random_forest":
        # generate adversarial examples via the surrogate, transferred as
        # augmented training data for a freshly-fit RF
        surrogate = base_model.surrogate
        X_adv = fgsm_attack(surrogate, X_train, y_train, epsilon)
        X_aug = np.concatenate([X_train, X_adv], axis=0)
        y_aug = np.concatenate([y_train, y_train], axis=0)

        robust_sklearn = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=7)
        robust_sklearn.fit(X_aug.reshape(len(X_aug), -1), y_aug)
        robust_surrogate = _train_surrogate_for_rf(
            robust_sklearn, X_aug, base_model.num_classes, base_model.input_dim
        )
        torch_model = None
        sklearn_model = robust_sklearn
        surrogate_out = robust_surrogate
    else:
        cls = _MODEL_CLASSES[base_model.model_type]
        if base_model.model_type == "cnn":
            fresh = cls(base_model.num_classes)
        else:
            fresh = cls(base_model.input_dim, base_model.num_classes)

        # bootstrap: train briefly on clean data first so gradients used to
        # craft adversarial training examples are meaningful
        _train_torch_classifier(fresh, X_train, y_train, epochs=8)
        X_adv = fgsm_attack(fresh, X_train, y_train, epsilon)
        X_aug = np.concatenate([X_train, X_adv], axis=0)
        y_aug = np.concatenate([y_train, y_train], axis=0)
        _train_torch_classifier(fresh, X_aug, y_aug, epochs=20)

        torch_model = fresh
        sklearn_model = None
        surrogate_out = None

    model_id = str(uuid.uuid4())[:8]
    robust = TrainedModel(
        model_id=model_id,
        dataset=base_model.dataset,
        model_type=base_model.model_type,
        torch_model=torch_model,
        sklearn_model=sklearn_model,
        surrogate=surrogate_out,
        input_dim=base_model.input_dim,
        num_classes=base_model.num_classes,
        feature_names=base_model.feature_names,
        is_image=base_model.is_image,
        metrics={},
        X_test=X_test,
        y_test=y_test,
    )
    preds = robust.predict(X_test)
    robust.metrics = {
        "clean_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "n_train": len(X_aug),
        "n_test": len(X_test),
        "adversarially_trained": True,
        "base_model_id": base_model.model_id,
        "training_epsilon": epsilon,
    }
    MODEL_STORE[model_id] = robust
    return robust
