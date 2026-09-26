from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import FEATURE_COLUMNS


def build_model(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=420,
        max_depth=7,
        learning_rate=0.075,
        min_child_weight=4.0,
        subsample=0.82,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.5,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=256,
        n_jobs=-1,
        random_state=seed,
    )


def fit_model(train: pd.DataFrame, validation: pd.DataFrame, seed: int) -> xgb.XGBClassifier:
    model = build_model(seed)
    model.fit(
        train[FEATURE_COLUMNS].astype(np.float32),
        train["label"].astype(np.int8),
        eval_set=[(
            validation[FEATURE_COLUMNS].astype(np.float32),
            validation["label"].astype(np.int8),
        )],
        verbose=False,
    )
    return model


def save_model(model: xgb.XGBClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path: Path) -> xgb.XGBClassifier:
    model = build_model(0)
    model.load_model(path)
    return model
