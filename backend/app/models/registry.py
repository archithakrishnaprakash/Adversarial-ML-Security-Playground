from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from app.data.mnist_loader import IMAGE_SHAPE, NUM_CLASSES as IMG_NUM_CLASSES, load_image_dataset
from app.data.cyber_loader import (
    FEATURE_NAMES,
    NUM_CLASSES as CYBER_NUM_CLASSES,
    load_cyber_dataset,
)
from app.models.torch_models import LogisticRegressionTorch, SmallCNN, SmallNN

DEVICE = torch.device("cpu")

VALID_MODEL_TYPES_BY_DATASET = {
    "image": ["logistic_regression", "random_forest", "small_nn", "cnn"],
    "cyber": ["logistic_regression", "random_forest", "small_nn"],
}


@dataclass
class TrainedModel:
    model_id: str
    dataset: str  # "image" | "cyber"
    model_type: str
    torch_model: Optional[nn.Module]  # None for random_forest
    sklearn_model: Optional[object]  # RandomForestClassifier, else None
    surrogate: Optional[nn.Module]  # differentiable stand-in, used for gradient attacks on RF
    input_dim: int
    num_classes: int
    feature_names: list
    is_image: bool
    metrics: dict
    X_test: np.ndarray
    y_test: np.ndarray
    created_at: float = field(default_factory=time.time)

    @property
    def is_differentiable(self) -> bool:
        return self.torch_model is not None

    @property
    def supports_gradient_attacks(self) -> bool:
        """True if FGSM/PGD/DeepFool can run against this model at all — either
        directly (differentiable torch model) or via a surrogate (e.g. Random
        Forest, attacked through a substitute model and transferred)."""
        return self.torch_model is not None or self.surrogate is not None

    def gradient_source(self) -> nn.Module:
        """The module used to compute gradients for white-box attacks.
        For RF, this is a surrogate network trained to mimic its outputs
        (a standard substitute-model approach for attacking non-differentiable
        models)."""
        return self.torch_model if self.is_differentiable else self.surrogate

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """X: (N, ...) matching input_shape. Returns (N, num_classes) probs."""
        if self.sklearn_model is not None:
            flat = X.reshape(len(X), -1)
            return self.sklearn_model.predict_proba(flat)
        self.torch_model.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            logits = self.torch_model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


MODEL_STORE: dict[str, TrainedModel] = {}
DATA_CACHE: dict[str, tuple] = {}


def get_data(dataset: str):
    if dataset not in DATA_CACHE:
        if dataset == "image":
            DATA_CACHE[dataset] = load_image_dataset()
        elif dataset == "cyber":
            DATA_CACHE[dataset] = load_cyber_dataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    return DATA_CACHE[dataset]


def _train_torch_classifier(module: nn.Module, X_train, y_train, epochs=25, lr=1e-2):
    module.to(DEVICE)
    opt = optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    module.train()
    batch_size = 64
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            logits = module(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            opt.step()
    return module


def _train_surrogate_for_rf(rf: RandomForestClassifier, X_train, num_classes, in_features):
    """Train a small differentiable NN to mimic the RF's soft predictions,
    so gradient-based attacks can be crafted against it and transferred to
    the RF (a standard substitute-model / transfer-attack technique for
    attacking non-differentiable models)."""
    soft_targets = rf.predict_proba(X_train.reshape(len(X_train), -1))
    surrogate = SmallNN(in_features=in_features, num_classes=num_classes, hidden=64).to(DEVICE)
    opt = optim.Adam(surrogate.parameters(), lr=1e-2)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    target = torch.tensor(soft_targets, dtype=torch.float32, device=DEVICE)
    surrogate.train()
    for _ in range(40):
        opt.zero_grad()
        logits = surrogate(Xt)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(target * log_probs).sum(dim=1).mean()  # cross-entropy vs soft labels
        loss.backward()
        opt.step()
    return surrogate


def train_model(dataset: str, model_type: str) -> TrainedModel:
    if dataset not in VALID_MODEL_TYPES_BY_DATASET:
        raise ValueError(f"Unknown dataset '{dataset}'")
    if model_type not in VALID_MODEL_TYPES_BY_DATASET[dataset]:
        raise ValueError(f"Model '{model_type}' not valid for dataset '{dataset}'")

    X_train, X_test, y_train, y_test = get_data(dataset)
    is_image = dataset == "image"
    num_classes = IMG_NUM_CLASSES if is_image else CYBER_NUM_CLASSES
    feature_names = None if is_image else FEATURE_NAMES
    in_features = int(np.prod(X_train.shape[1:]))

    torch_model = None
    sklearn_model = None
    surrogate = None

    if model_type == "logistic_regression":
        torch_model = LogisticRegressionTorch(in_features, num_classes)
        _train_torch_classifier(torch_model, X_train, y_train)
    elif model_type == "small_nn":
        torch_model = SmallNN(in_features, num_classes)
        _train_torch_classifier(torch_model, X_train, y_train)
    elif model_type == "cnn":
        if not is_image:
            raise ValueError("CNN is only available for the image dataset")
        torch_model = SmallCNN(num_classes)
        _train_torch_classifier(torch_model, X_train, y_train, epochs=15)
    elif model_type == "random_forest":
        sklearn_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
        sklearn_model.fit(X_train.reshape(len(X_train), -1), y_train)
        surrogate = _train_surrogate_for_rf(sklearn_model, X_train, num_classes, in_features)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model_id = str(uuid.uuid4())[:8]
    trained = TrainedModel(
        model_id=model_id,
        dataset=dataset,
        model_type=model_type,
        torch_model=torch_model,
        sklearn_model=sklearn_model,
        surrogate=surrogate,
        input_dim=in_features,
        num_classes=num_classes,
        feature_names=feature_names,
        is_image=is_image,
        metrics={},
        X_test=X_test,
        y_test=y_test,
    )

    preds = trained.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    trained.metrics = {
        "clean_accuracy": round(acc, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    MODEL_STORE[model_id] = trained
    return trained


def get_model(model_id: str) -> TrainedModel:
    if model_id not in MODEL_STORE:
        raise KeyError(f"No trained model with id '{model_id}'")
    return MODEL_STORE[model_id]


def list_models() -> list[TrainedModel]:
    return sorted(MODEL_STORE.values(), key=lambda m: m.created_at, reverse=True)
