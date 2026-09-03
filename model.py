"""
model.py — Train and evaluate the prediction model.

Critical rule: NEVER shuffle time-series data into a random train/test split.
That leaks future information into training (the model "sees" patterns that
include data from after the test period) and produces backtest results that
look great and are completely fake. Always split chronologically, and prefer
walk-forward validation (train on window 1, test on window 2, roll forward)
over a single train/test split.

Starts with a Random Forest, not a deep model — per your own point #7, a
fancier architecture doesn't automatically predict better, and tree models
are far easier to debug, less prone to overfitting on limited intraday
history, and give you feature importances for free.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from features import FEATURE_COLUMNS


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7):
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def train_model(train_df: pd.DataFrame, feature_cols=None, random_state: int = 42):
    feature_cols = feature_cols or FEATURE_COLUMNS
    clean = train_df.dropna(subset=feature_cols + ["target"])

    X = clean[feature_cols]
    y = clean["target"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,           # shallow — intraday data is noisy, deep trees overfit fast
        min_samples_leaf=50,   # forces the model to find robust, not one-off, patterns
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_model(model, test_df: pd.DataFrame, feature_cols=None) -> dict:
    feature_cols = feature_cols or FEATURE_COLUMNS
    clean = test_df.dropna(subset=feature_cols + ["target"])

    X = clean[feature_cols]
    y = clean["target"]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y, proba) if y.nunique() > 1 else float("nan"),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "base_rate": y.mean(),  # what fraction of bars hit the target naturally — your floor to beat
    }


def walk_forward_predict(df: pd.DataFrame, feature_cols=None, n_folds: int = 5,
                          min_train_size: int = 500) -> pd.DataFrame:
    """
    Rolling walk-forward: for each fold, train only on data strictly before
    the test window, predict on the test window, then roll forward. Returns
    the full out-of-sample prediction set — this (not a single train/test split)
    is what you should trust for backtesting.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    n = len(df)
    fold_size = (n - min_train_size) // n_folds
    if fold_size <= 0:
        raise ValueError("Not enough data for the requested number of walk-forward folds.")

    results = []
    for fold in range(n_folds):
        train_end = min_train_size + fold * fold_size
        test_end = train_end + fold_size

        train_fold = df.iloc[:train_end]
        test_fold = df.iloc[train_end:test_end].copy()
        if len(test_fold) == 0:
            continue

        m = train_model(train_fold, feature_cols)
        proba = m.predict_proba(test_fold[feature_cols])[:, 1]
        test_fold["pred_proba"] = proba
        test_fold["fold"] = fold
        results.append(test_fold)

    return pd.concat(results, ignore_index=True)
