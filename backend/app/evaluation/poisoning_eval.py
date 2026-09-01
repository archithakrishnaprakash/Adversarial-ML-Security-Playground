from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from app.attacks.poisoning import apply_trigger_batch, inject_backdoor, label_flip_poison
from app.data.cyber_loader import NUM_CLASSES as CYBER_NUM_CLASSES
from app.data.mnist_loader import NUM_CLASSES as IMG_NUM_CLASSES
from app.models.registry import fit_and_register, get_data


def run_label_flip_experiment(
    dataset: str, model_type: str, poison_fraction: float, seed: int = 0
) -> dict:
    """Trains a clean baseline and a label-flip-poisoned model on the same
    architecture/dataset, and reports the accuracy gap. Both models are
    registered normally, so either can be opened in the Attack Lab / Defense
    Lab afterward like any other trained model.
    """
    X_train, X_test, y_train, y_test = get_data(dataset)
    num_classes = IMG_NUM_CLASSES if dataset == "image" else CYBER_NUM_CLASSES

    clean_model = fit_and_register(dataset, model_type, X_train, y_train, X_test, y_test)

    X_poison, y_poison, poisoned_idx = label_flip_poison(X_train, y_train, poison_fraction, num_classes, seed)
    poisoned_model = fit_and_register(
        dataset,
        model_type,
        X_poison,
        y_poison,
        X_test,
        y_test,
        extra_metrics={
            "poisoned": True,
            "poison_type": "label_flip",
            "poison_fraction": poison_fraction,
            "n_poisoned_samples": int(len(poisoned_idx)),
        },
    )

    clean_acc = clean_model.metrics["clean_accuracy"]
    poisoned_acc = poisoned_model.metrics["clean_accuracy"]

    return {
        "poison_type": "label_flip",
        "dataset": dataset,
        "model_type": model_type,
        "poison_fraction": poison_fraction,
        "clean_model_id": clean_model.model_id,
        "poisoned_model_id": poisoned_model.model_id,
        "clean_accuracy_before_poisoning": clean_acc,
        "clean_accuracy_after_poisoning": poisoned_acc,
        "accuracy_degradation": round(clean_acc - poisoned_acc, 4),
        "n_poisoned_samples": int(len(poisoned_idx)),
    }


def run_backdoor_experiment(
    dataset: str, model_type: str, poison_fraction: float, target_label: int, seed: int = 0
) -> dict:
    """Trains a clean baseline and a backdoored model, then measures two
    things: whether clean accuracy held up (a good backdoor is meant to be
    invisible on normal inputs), and the *backdoor success rate* — the
    fraction of clean test samples that flip to `target_label` once the
    trigger pattern is stamped onto them.
    """
    X_train, X_test, y_train, y_test = get_data(dataset)
    is_image = dataset == "image"
    num_classes = IMG_NUM_CLASSES if is_image else CYBER_NUM_CLASSES
    if not 0 <= target_label < num_classes:
        raise ValueError(f"target_label must be in [0, {num_classes - 1}] for dataset '{dataset}'")

    clean_model = fit_and_register(dataset, model_type, X_train, y_train, X_test, y_test)

    X_poison, y_poison, poisoned_idx = inject_backdoor(
        X_train, y_train, poison_fraction, target_label, is_image, seed
    )
    poisoned_model = fit_and_register(
        dataset,
        model_type,
        X_poison,
        y_poison,
        X_test,
        y_test,
        extra_metrics={
            "poisoned": True,
            "poison_type": "backdoor",
            "poison_fraction": poison_fraction,
            "target_label": target_label,
            "n_poisoned_samples": int(len(poisoned_idx)),
        },
    )

    # backdoor success: stamp the trigger onto every non-target-label test
    # sample and see how often the poisoned model now predicts target_label
    mask = y_test != target_label
    X_triggered = apply_trigger_batch(X_test[mask], is_image)
    triggered_preds = poisoned_model.predict(X_triggered)
    backdoor_success_rate = float((triggered_preds == target_label).mean()) if mask.sum() else 0.0

    clean_acc = clean_model.metrics["clean_accuracy"]
    poisoned_clean_acc = poisoned_model.metrics["clean_accuracy"]

    return {
        "poison_type": "backdoor",
        "dataset": dataset,
        "model_type": model_type,
        "poison_fraction": poison_fraction,
        "target_label": target_label,
        "clean_model_id": clean_model.model_id,
        "poisoned_model_id": poisoned_model.model_id,
        "clean_accuracy_before_poisoning": clean_acc,
        "clean_accuracy_after_poisoning": round(poisoned_clean_acc, 4),
        "backdoor_success_rate": round(backdoor_success_rate, 4),
        "n_poisoned_samples": int(len(poisoned_idx)),
        "n_trigger_test_samples": int(mask.sum()),
    }
