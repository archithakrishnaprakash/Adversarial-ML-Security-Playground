from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TrainModelRequest(BaseModel):
    dataset: str  # "image" | "cyber"
    model_type: str  # "logistic_regression" | "random_forest" | "small_nn" | "cnn"


class RunAttackRequest(BaseModel):
    model_id: str
    attack: str  # "fgsm" | "pgd" | "deepfool" | "random_noise"
    epsilon: float = 0.15
    pgd_steps: int = 10
    sample_index: Optional[int] = None  # if None, picks a random correctly-classified sample
    # Threat-model capability this attack is being run under. Defaults to
    # "white_box" to preserve the exact behavior of every existing caller
    # (including the current frontend, which never sends this field).
    capability: str = "white_box"


class ThreatModelCheckRequest(BaseModel):
    attack: str
    capability: str  # "white_box" | "gray_box" | "black_box"


class BlackBoxAttackRequest(BaseModel):
    model_id: str
    method: str = "transfer"  # "transfer" | "query"
    epsilon: float = 0.15
    base_attack: str = "fgsm"  # for method="transfer": "fgsm" | "pgd"
    n_queries: int = 200  # for method="transfer": surrogate training budget
    max_queries_per_sample: int = 150  # for method="query"
    n_samples: int = 50
    seed: int = 0


class LabelFlipPoisoningRequest(BaseModel):
    dataset: str
    model_type: str
    poison_fraction: float = 0.05
    seed: int = 0


class BackdoorPoisoningRequest(BaseModel):
    dataset: str
    model_type: str
    poison_fraction: float = 0.05
    target_label: int = 0
    seed: int = 0


class TransferabilityRequest(BaseModel):
    model_ids: list[str]
    attack: str = "fgsm"
    epsilon: float = 0.15
    n_samples: int = 100
    seed: int = 0


class SecurityAssessmentRequest(BaseModel):
    model_id: str
    attacks: list[str] = ["fgsm", "pgd", "deepfool", "random_noise"]
    epsilon: float = 0.15
    n_samples: Optional[int] = 300
    # optional model ids to include as a transferability check in the findings
    compare_model_ids: list[str] = []


class ExperimentGridRequest(BaseModel):
    dataset: str
    model_types: list[str]
    attacks: list[str]
    epsilons: list[float]
    defenses: list[str] = ["none"]
    n_samples: int = 150
    seed: int = 0


class DefensePreprocessingRequest(BaseModel):
    model_id: str
    attack: str
    epsilon: float = 0.15
    defense: str  # "gaussian_smoothing" | "feature_clipping" | "normalization"
    n_samples: int = 200


class AdversarialTrainingRequest(BaseModel):
    model_id: str
    epsilon: float = 0.15
    attacks_to_compare: list[str] = ["fgsm", "pgd"]
    n_samples: int = 200


class RobustnessEvalRequest(BaseModel):
    model_id: str
    attacks: list[str] = ["fgsm", "pgd", "deepfool", "random_noise"]
    epsilon: float = 0.15
    n_samples: Optional[int] = 300


class RobustnessMatrixRequest(BaseModel):
    dataset: str
    model_types: list[str]
    attacks: list[str] = ["fgsm", "pgd", "deepfool"]
    epsilon: float = 0.15
    n_samples: int = 200
    seed: int = 0
