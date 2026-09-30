"""Configuration for PolicyInduction.

User-facing knobs live on the ``PolicyInduction`` constructor and in
``WeightConfig``. Everything in ``BoostConfig`` is an internal default with a
reason behind it; it is exposed only so experiments can override it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class WeightConfig:
    """How rule weights, regularisation and the decision threshold are chosen.

    Args:
        beta: F-beta used only to pick the decision threshold (<1 favours
            precision, >1 favours recall). Model selection itself uses
            log-loss, which does not depend on a threshold.
        Cs: Candidate inverse regularisation strengths for L1 logistic
            regression.
        cv_folds: Folds per cross-validation repeat.
        cv_repeats: Repeats of the final cross-validation. Out-of-fold
            probabilities are averaged over repeats.
        one_se_rule: For the final model, pick the most regularised C whose
            validation log-loss is within one standard error of the best,
            instead of the best C. Off by default: on VCBench it cost 0.003
            to 0.006 validation log-loss in every run.
        class_weight_balanced: Reweight classes inversely to frequency.
    """

    beta: float = 0.5
    Cs: Tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    cv_folds: int = 5
    cv_repeats: int = 3
    one_se_rule: bool = False
    class_weight_balanced: bool = False

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not self.Cs:
            raise ValueError("Cs must contain at least one value")
        if self.cv_folds < 2 or self.cv_repeats < 1:
            raise ValueError("cv_folds must be >= 2 and cv_repeats >= 1")


@dataclass
class BoostConfig:
    """Internal constants of the boosting loop.

    Args:
        show_fraction: Share of training rows in the show pool P. The LLM
            only ever sees labels from P; every decision is scored on the
            validation pool V (the rest).
        seed_examples_per_class: Rows per class shown for the seed round.
        one_shot_examples_per_class: Rows per class shown in one_shot mode,
            which has a single call and so shows more than a seed round.
        hard_examples: Mis-predicted P rows shown per boosting round.
        contrast_examples: Correctly predicted P rows (same true class) shown
            alongside them.
        rules_per_round: Candidate rules requested per generation call.
        max_rounds: Hard cap on boosting rounds (after the seed round).
        rel_epsilon: A round counts as stalled when it lowers validation
            log-loss by less than this fraction of its current value
            (0.001 = 0.1%). Relative, so the same value works whatever the
            class balance. A round that accepts no rule improves by 0.
        patience: Stop after this many consecutive stalled rounds.
        accept_z: A rule is accepted only if the mean per-row log-loss
            improvement on V exceeds this many standard errors (paired, same
            folds and C as the round's baseline).
        min_spread: Reject a rule whose scores on V have a standard deviation
            below this; it gives nearly the same answer for every sample.
        max_redundancy: Reject a rule whose correlation with any pooled rule
            exceeds this.
        generality_ratio: Reject a rule whose residual correlation on V is
            below this fraction of its residual correlation on P (it fits
            the rows the LLM saw, not the pattern).
        min_generality_signal: Only apply the generality check when the
            P-side residual correlation is at least this large.
    """

    show_fraction: float = 0.2
    seed_examples_per_class: int = 20
    one_shot_examples_per_class: int = 50
    hard_examples: int = 20
    contrast_examples: int = 20
    rules_per_round: int = 10
    max_rounds: int = 15
    rel_epsilon: float = 0.001
    patience: int = 5
    accept_z: float = 1.0
    min_spread: float = 0.02
    max_redundancy: float = 0.8
    generality_ratio: float = 0.3
    min_generality_signal: float = 0.1
    threshold_grid: Tuple[float, ...] = field(
        default_factory=lambda: tuple(np.round(np.linspace(0.01, 0.99, 99), 2))
    )
    threshold_smoothing: int = 5

    def __post_init__(self) -> None:
        if self.rel_epsilon < 0:
            raise ValueError("rel_epsilon must be >= 0")
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if not 0 < self.show_fraction < 1:
            raise ValueError("show_fraction must be in (0, 1)")
        if self.min_spread < 0:
            raise ValueError("min_spread must be >= 0")
