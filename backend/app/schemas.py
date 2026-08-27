from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class RequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class TrainModelRequest(RequestModel):
    dataset: str  # "image" | "cyber"
    model_type: str  # "logistic_regression" | "random_forest" | "small_nn" | "cnn"


class RunAttackRequest(RequestModel):
    model_id: str
    attack: str  # "fgsm" | "pgd" | "deepfool" | "random_noise"
    epsilon: float = 0.15
    pgd_steps: int = 10
    sample_index: Optional[int] = None  # if None, picks a random correctly-classified sample


class DefensePreprocessingRequest(RequestModel):
    model_id: str
    attack: str
    epsilon: float = 0.15
    defense: str  # "gaussian_smoothing" | "feature_clipping" | "normalization"
    n_samples: int = 200


class AdversarialTrainingRequest(RequestModel):
    model_id: str
    epsilon: float = 0.15
    attacks_to_compare: list[str] = ["fgsm", "pgd"]
    n_samples: int = 200


class RobustnessEvalRequest(RequestModel):
    model_id: str
    attacks: list[str] = ["fgsm", "pgd", "deepfool", "random_noise"]
    epsilon: float = 0.15
    n_samples: Optional[int] = 300
