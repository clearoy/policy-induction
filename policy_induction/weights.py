"""Cross-validation, weight fitting, acceptance tests and threshold selection.

Everything here is cheap (no API calls): a logistic regression over a few
dozen probability features fits in milliseconds, so every decision can afford
a full cross-validation.

Honesty rule: decisions are scored only on validation-pool (V) rows, whose
labels the generation LLM never saw. Show-pool (P) rows still train each fold
model; they are simply never counted when scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

from .config import WeightConfig

EPS = 1e-6

Fold = Tuple[np.ndarray, np.ndarray]


def make_folds(
    y: np.ndarray, in_p: np.ndarray, n_folds: int, n_repeats: int, seed: int
) -> List[List[Fold]]:
    """Repeated K-fold over ALL rows, stratified on label x pool membership.

    Returns one list of folds per repeat; within a repeat every row is
    validated exactly once.
    """
    strata = y.astype(int) * 2 + in_p.astype(int)
    repeats = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        repeats.append(list(skf.split(np.zeros(len(y)), strata)))
    return repeats


class _Constant:
    """Stand-in model when there are no features yet: predicts the base rate."""

    def __init__(self, p: float) -> None:
        self.p = float(np.clip(p, EPS, 1 - EPS))
        self.coef_ = np.zeros((1, 0))
        self.intercept_ = np.array([np.log(self.p / (1 - self.p))])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


def fit_lr(X: np.ndarray, y: np.ndarray, C: float, cfg: WeightConfig):
    if X.shape[1] == 0:
        return _Constant(y.mean())
    lr = LogisticRegression(
        C=C,
        l1_ratio=1.0,  # pure L1 (scikit-learn >= 1.8 spelling of penalty="l1")
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced" if cfg.class_weight_balanced else None,
    )
    lr.fit(X, y)
    return lr


def oof_proba(
    X: np.ndarray, y: np.ndarray, folds: List[List[Fold]], C: float, cfg: WeightConfig
) -> Tuple[np.ndarray, list]:
    """Out-of-fold P(YES) for every row, averaged over repeats.

    Also returns the fitted fold models (used as the final ensemble).
    """
    total = np.zeros(len(y))
    models = []
    for repeat in folds:
        for tr, va in repeat:
            m = fit_lr(X[tr], y[tr], C, cfg)
            total[va] += m.predict_proba(X[va])[:, 1]
            models.append(m)
    return total / len(folds), models


def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def mean_se(x: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return float(x.mean()) if len(x) else 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


@dataclass
class CSelection:
    C: float
    val_loss: float
    val_loss_se: float
    table: List[Tuple[float, float, float]]  # (C, mean loss on V, SE)


def select_C(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[List[Fold]],
    in_v: np.ndarray,
    cfg: WeightConfig,
    one_se: bool,
) -> CSelection:
    """Pick C by out-of-fold log-loss on V rows.

    With ``one_se`` the most regularised C within one standard error of the
    best is chosen, which favours fewer, more stable rules.
    """
    table = []
    for C in sorted(cfg.Cs):
        p, _ = oof_proba(X, y, folds, C, cfg)
        m, se = mean_se(row_log_loss(y[in_v], p[in_v]))
        table.append((C, m, se))
    best = min(table, key=lambda t: t[1])
    chosen = best
    if one_se:
        limit = best[1] + best[2]
        chosen = next(t for t in table if t[1] <= limit)  # smallest C = strongest
    return CSelection(C=chosen[0], val_loss=chosen[1], val_loss_se=chosen[2], table=table)


def paired_gain(loss_before: np.ndarray, loss_after: np.ndarray) -> Tuple[float, float]:
    """Mean and SE of the per-row log-loss improvement (positive = better)."""
    return mean_se(loss_before - loss_after)


def choose_threshold(
    y: np.ndarray, p: np.ndarray, beta: float, grid: Sequence[float], smoothing: int
) -> Tuple[float, float]:
    """Sweep once over pooled out-of-fold probabilities.

    The F-beta curve is smoothed with a moving average before taking the
    argmax, so the threshold sits in the middle of a plateau rather than on
    a noise spike. Returns (threshold, F-beta at that threshold).
    """
    grid = np.asarray(grid)
    scores = np.array(
        [fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    )
    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        padded = np.pad(scores, smoothing // 2, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")[: len(scores)]
    else:
        smoothed = scores
    i = int(np.argmax(smoothed))
    return float(grid[i]), float(scores[i])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
