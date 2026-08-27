from __future__ import annotations

import datetime
import json

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./experiments.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    experiment_type = Column(String)  # "attack" | "defense" | "adversarial_training" | "robustness_eval"
    dataset = Column(String)
    model_type = Column(String)
    model_id = Column(String)
    attack = Column(String, nullable=True)
    epsilon = Column(Float, nullable=True)
    result_json = Column(Text)  # arbitrary JSON blob with the full result


def init_db():
    Base.metadata.create_all(bind=engine)


def log_experiment(experiment_type: str, dataset: str, model_type: str, model_id: str, result: dict,
                    attack: str | None = None, epsilon: float | None = None) -> int:
    db = SessionLocal()
    try:
        record = ExperimentRecord(
            experiment_type=experiment_type,
            dataset=dataset,
            model_type=model_type,
            model_id=model_id,
            attack=attack,
            epsilon=epsilon,
            result_json=json.dumps(result),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record.id
    finally:
        db.close()


def list_experiments(limit: int = 100) -> list[dict]:
    db = SessionLocal()
    try:
        records = (
            db.query(ExperimentRecord)
            .order_by(ExperimentRecord.timestamp.desc())
            .limit(limit)
            .all()
        )
        out = []
        for r in records:
            out.append(
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat(),
                    "experiment_type": r.experiment_type,
                    "dataset": r.dataset,
                    "model_type": r.model_type,
                    "model_id": r.model_id,
                    "attack": r.attack,
                    "epsilon": r.epsilon,
                    "result": json.loads(r.result_json),
                }
            )
        return out
    finally:
        db.close()
