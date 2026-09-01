"""
UNSW-NB15 loader — a real, published network-intrusion dataset, as opposed to
the synthetic stand-in in `cyber_loader.py`.

Why this exists: the synthetic dataset is useful as a zero-setup quick-start
(no download, deterministic, fast to train on), but it can't demonstrate that
these attacks work against realistic intrusion-detection data — and that's a
fair thing for a reviewer to push back on. This loader lets the exact same
attack/defense/evaluation pipeline run against real data once you've
downloaded it.

Not bundled automatically: UNSW-NB15 is a multi-hundred-MB download from an
external host (the Australian Centre for Cyber Security /
https://research.unsw.edu.au/projects/unsw-nb15-dataset, also mirrored on
Kaggle), which this project can't fetch on your behalf. Download the "clean
CSV" release — commonly distributed as `UNSW_NB15_training-set.csv` and
`UNSW_NB15_testing-set.csv` with headers already included — into a directory
and point `UNSW_NB15_DATA_DIR` at it (see README section 5b). If the files
aren't there, `is_available()` returns False and the API surfaces a clear
"how to get this" message instead of crashing.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUM_CLASSES = 2
CLASS_NAMES = ["BENIGN", "ATTACK"]

# columns that aren't model features: a row id, the free-text attack category
# (label already encodes attack-vs-not; attack_cat would leak the answer more
# specifically than any real deployment would have at inference time), and
# the label itself.
_DROP_COLUMNS = {"id", "attack_cat"}
_LABEL_COLUMN = "label"
# UNSW-NB15's three categorical fields — label-encoded (not one-hot) because
# `proto` alone has 100+ distinct values in the full dataset; one-hot would
# blow up the feature space for little benefit here.
_CATEGORICAL_COLUMNS = ["proto", "service", "state"]

DEFAULT_MAX_SAMPLES = 20_000  # keeps training fast; raise via `max_samples` if you want the full set


def data_dir() -> Path:
    return Path(os.environ.get("UNSW_NB15_DATA_DIR", "./data/unsw_nb15"))


def is_available() -> bool:
    """Whether the expected CSV files are present. Checked before every load
    so a missing dataset produces a clear error instead of a stack trace."""
    d = data_dir()
    train_ok = (d / "UNSW_NB15_training-set.csv").exists()
    # a single combined file is also accepted — see _find_csv_files
    single_ok = any(d.glob("*.csv")) if d.exists() else False
    return train_ok or single_ok


def _find_csv_files(d: Path) -> tuple[Path, Path | None]:
    train_path = d / "UNSW_NB15_training-set.csv"
    test_path = d / "UNSW_NB15_testing-set.csv"
    if train_path.exists():
        return train_path, (test_path if test_path.exists() else None)

    # fall back to any single CSV in the directory (e.g. a combined export)
    csvs = sorted(d.glob("*.csv"))
    if csvs:
        return csvs[0], None

    raise FileNotFoundError(
        f"No UNSW-NB15 CSV files found in '{d}'. Expected 'UNSW_NB15_training-set.csv' "
        f"(+ optionally 'UNSW_NB15_testing-set.csv'), or any single CSV with the same "
        f"columns. See README section 5b for where to download this dataset."
    )


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if _LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a '{_LABEL_COLUMN}' column in the UNSW-NB15 CSV, found: {list(df.columns)}"
        )

    y = df[_LABEL_COLUMN].astype(np.int64).to_numpy()
    drop = [c for c in _DROP_COLUMNS | {_LABEL_COLUMN} if c in df.columns]
    features = df.drop(columns=drop)

    for col in _CATEGORICAL_COLUMNS:
        if col in features.columns:
            features[col] = features[col].astype("category").cat.codes.astype(np.float32)

    # anything left non-numeric at this point is dropped rather than guessed
    # at — safer than silently coercing unexpected text columns to NaN
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    dropped_non_numeric = [c for c in features.columns if c not in numeric_cols]
    features = features[numeric_cols]

    X = features.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if dropped_non_numeric:
        # not fatal — just means those columns weren't in _CATEGORICAL_COLUMNS
        # and weren't numeric; log via feature_names being shorter than the
        # raw CSV rather than raising, since dataset column sets do vary
        # slightly across UNSW-NB15 redistributions.
        pass

    return X, y, numeric_cols


def load_unsw_nb15(
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: int | None = DEFAULT_MAX_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns X_train, X_test, y_train, y_test, feature_names — same shape
    contract as `cyber_loader.load_cyber_dataset`, so it's a drop-in
    alternative anywhere that function is used. Features are min-max scaled
    to [0, 1] per column, matching the synthetic dataset's convention (kept
    consistent so epsilon values mean roughly the same thing across both).

    Raises FileNotFoundError if the CSVs aren't present — check
    `is_available()` first if you want to handle that gracefully rather than
    catching the exception.
    """
    d = data_dir()
    train_path, test_path = _find_csv_files(d)

    train_df = pd.read_csv(train_path)
    if test_path is not None:
        test_df = pd.read_csv(test_path)
        X_train_raw, y_train, feat_names_train = _prepare_features(train_df)
        X_test_raw, y_test, feat_names_test = _prepare_features(test_df)
        # align columns in case the two files disagree slightly on schema
        common = [c for c in feat_names_train if c in feat_names_test]
        X_train_raw = X_train_raw[:, [feat_names_train.index(c) for c in common]]
        X_test_raw = X_test_raw[:, [feat_names_test.index(c) for c in common]]
        feature_names = common
    else:
        X_all, y_all, feature_names = _prepare_features(train_df)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_all, y_all, test_size=test_size, random_state=random_state, stratify=y_all
        )

    if max_samples is not None:
        rng = np.random.default_rng(random_state)
        if len(X_train_raw) > max_samples:
            idx = rng.choice(len(X_train_raw), size=max_samples, replace=False)
            X_train_raw, y_train = X_train_raw[idx], y_train[idx]
        test_cap = max(1, max_samples // 4)
        if len(X_test_raw) > test_cap:
            idx = rng.choice(len(X_test_raw), size=test_cap, replace=False)
            X_test_raw, y_test = X_test_raw[idx], y_test[idx]

    # min-max scale using train-set statistics only, applied to both splits —
    # avoids leaking test-set distribution into the scaling like fitting on
    # the concatenation would.
    col_min = X_train_raw.min(axis=0)
    col_max = X_train_raw.max(axis=0)
    col_range = np.where(col_max - col_min < 1e-8, 1.0, col_max - col_min)
    X_train = ((X_train_raw - col_min) / col_range).astype(np.float32)
    X_test = np.clip((X_test_raw - col_min) / col_range, 0.0, 1.0).astype(np.float32)

    return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64), feature_names
