"""PolicyInduction with boosting.

An interpretable binary classifier. An LLM writes natural-language rules, Jev
scores P(rule true) for every sample, and an L1 logistic regression weights
the rules.

Rules live in a single, expand-only pool. Each round:

    1. fit L1 logistic regression on the pool; out-of-fold P(YES) for every row
    2. residual g = y - p; take the show-pool (P) rows the model gets most wrong
       -- missed YES rows on odd rounds, missed NO rows on even rounds -- plus
       correctly handled rows of the same class as contrast
    3. the LLM sees the whole pool (with weights) and those rows, and proposes
       new rules; Jev scores them on every row
    4. a new rule enters the pool unless it is near-constant, a near-duplicate
       of a pooled rule, or fits P but not V (it memorised the rows it saw)
    5. refit on the grown pool and measure out-of-fold log-loss on the
       validation pool V, rows the LLM never sees

Stop when the pool reaches ``max_policy_length`` or ``BoostConfig.patience``
consecutive rounds each improve V log-loss by less than
``BoostConfig.rel_epsilon`` (relative). Nothing is ever
removed from the pool: L1 decides which rules carry weight, and only rules
with non-zero weight are asked at prediction time.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import asdict, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .config import BoostConfig, WeightConfig
from .generator import RuleGenerator, make_generator
from .prompts import BOOST_PROMPT, GEN_SYSTEM, SEED_PROMPT
from .scorer import JevScorer, Scorer
from .weights import (
    choose_threshold,
    corr,
    make_folds,
    oof_proba,
    row_log_loss,
    select_C,
)

logger = logging.getLogger(__name__)

# Long free-text fields are cut when shown to the generation LLM, to keep
# prompts bounded. Jev always scores the full sample.
MAX_FIELD_CHARS_IN_PROMPT = 1500

_CHECKPOINT = "checkpoint.json"
# Bumped whenever the checkpoint layout changes, so an old checkpoint is
# ignored rather than misread. The Jev answer cache is unaffected.
_CHECKPOINT_FORMAT = 3


class _Eval(NamedTuple):
    C: float
    p: np.ndarray  # out-of-fold P(YES), every row
    models: list  # fold models
    val_loss: float  # mean out-of-fold log-loss on V


class PolicyInduction:
    """Boosted, interpretable binary classifier built from LLM-written rules.

    Args:
        task_description: What is being predicted and what YES/NO mean. This
            is the one input that most shapes the rules; be concrete.
        gen_model: Generation LLM name (``gpt-*``, ``deepseek-*``, ``gemini-*``)
            or any object implementing ``RuleGenerator``.
        max_policy_length: Size cap of the rule pool (<= 100). Boosting stops
            when the pool is full. Every admitted rule stays in the pool, so
            the pool grows by up to ``rules_per_round`` per round; the final
            model uses only the rules L1 gives a non-zero weight, usually far
            fewer.
        gen_temperature: Sampling temperature of the generation LLM.
        random_state: Seeds the P/V split, example selection and CV folds.
            It does not make the generation LLM deterministic.
        weight_config: How weights, C and the threshold are chosen.
        boost_config: Internal boosting constants (rarely changed).
        scorer: Rule scorer. Defaults to ``JevScorer`` with an answer cache
            under ``save_path``.
        jev_model: Jev model or alias for the default scorer.
        save_path: Directory for the answer cache, checkpoints and ``save()``.
    """

    def __init__(
        self,
        task_description: str,
        gen_model: str | RuleGenerator = "gpt-5.6",
        max_policy_length: int = 100,
        gen_temperature: float = 1.0,
        random_state: int = 0,
        weight_config: WeightConfig | None = None,
        boost_config: BoostConfig | None = None,
        scorer: Scorer | None = None,
        jev_model: str = "jev-latest",
        save_path: str | Path = "policy_induction_run",
    ) -> None:
        if not task_description or not task_description.strip():
            raise ValueError("task_description must be non-empty")
        if not 0 < max_policy_length <= 100:
            raise ValueError("max_policy_length must be in [1, 100]")
        if not 0 <= gen_temperature <= 2:
            raise ValueError("gen_temperature must be in [0, 2]")

        self.task_description = task_description.strip()
        self.max_policy_length = max_policy_length
        self.gen_temperature = gen_temperature
        self.random_state = random_state
        self.weight_config = weight_config or WeightConfig()
        self.boost_config = boost_config or BoostConfig()
        self.save_path = Path(save_path)
        self.jev_model = jev_model

        self._generator: RuleGenerator | None = (
            None if isinstance(gen_model, str) else gen_model
        )
        self.gen_model_name = gen_model if isinstance(gen_model, str) else gen_model.model
        self._scorer: Scorer | None = scorer

        # Learned state
        self.fields: List[str] = []
        self.pool: List[str] = []  # admitted rules, expand-only
        self.tried: List[str] = []  # every rule ever proposed (for de-duplication)
        self.rules: List[str] = []  # pooled rules with non-zero final weight
        self.history: List[Dict[str, Any]] = []
        self.threshold: float | None = None
        self.metrics: Dict[str, Any] = {}
        self.jev_version: str | None = None
        self._models: list = []
        self._feature_means: np.ndarray | None = None
        self._C: float | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def fit(self, X: pd.DataFrame, y: Sequence[Any]) -> "PolicyInduction":
        """Grow the rule pool by boosting, then fit the final weighted model."""
        states = _to_states(X)
        y01 = _to_binary(y)
        if len(states) != len(y01):
            raise ValueError("X and y must have the same length")
        if len(np.unique(y01)) != 2:
            raise ValueError("y must contain both classes")

        bc = self.boost_config
        self.fields = [str(c) for c in X.columns]
        rng = np.random.default_rng(self.random_state)
        scorer = self._get_scorer()
        generator = self._get_generator()
        in_p = _split_pools(y01, bc.show_fraction, rng)
        in_v = ~in_p

        fingerprint = _fingerprint(states, y01, self.random_state, bc.show_fraction)
        ckpt = self._read_checkpoint(fingerprint)
        start_round, stale = 0, 0
        if ckpt:
            self.pool, self.tried, self.history = ckpt["pool"], ckpt["tried"], ckpt["history"]
            start_round, stale = ckpt["next_round"], ckpt["stale"]
            logger.info("Resuming at round %d with %d pooled rules.", start_round, len(self.pool))

        cols: Dict[str, np.ndarray] = {}
        if self.pool:
            cols.update(await self._score(scorer, states, self.pool))  # cache hits

        stop_reason = "max_rounds"
        for rnd in range(start_round, bc.max_rounds + 1):
            if len(self.pool) >= self.max_policy_length:
                stop_reason = "max_policy_length"
                break

            # One fold assignment per round, shared by the before/after comparison.
            folds = make_folds(
                y01, in_p, self.weight_config.cv_folds, self.weight_config.cv_repeats,
                self.random_state * 1000 + rnd,
            )
            before = self._evaluate(cols, self.pool, y01, folds, in_v)
            g = y01 - before.p

            if not self.pool:
                prompt, direction = self._seed_prompt(states, y01, in_p, rng), "seed"
            else:
                prompt, direction = self._boost_prompt(
                    states, y01, before, g, in_p, rnd, rng, cols
                )

            proposed = await _generate_with_retry(generator, prompt, self.gen_temperature)
            candidates = _dedupe([r.strip() for r in proposed if r and r.strip()], set(self.tried))
            self.tried.extend(candidates)
            log: Dict[str, Any] = {
                "round": rnd, "direction": direction, "C": before.C,
                "val_log_loss_before": before.val_loss,
                "proposed": len(proposed), "rejected": {}, "added": [],
            }

            if candidates:
                cols.update(await self._score(scorer, states, candidates))
                admitted = self._filter(candidates, cols, g, in_p, in_v, log)
                admitted.sort(key=lambda r: -abs(corr(cols[r][in_v], g[in_v])))
                room = self.max_policy_length - len(self.pool)
                if len(admitted) > room:
                    log["rejected"]["pool_full"] = len(admitted) - room
                    admitted = admitted[:room]
                self.pool.extend(admitted)
                log["added"] = admitted

            after = self._evaluate(cols, self.pool, y01, folds, in_v) if log["added"] else before
            rel = (before.val_loss - after.val_loss) / before.val_loss
            log.update(val_log_loss_after=after.val_loss, relative_improvement=rel,
                       pool_size=len(self.pool))
            self.history.append(log)
            logger.info(
                "Round %d (%s): +%d rules -> pool %d, V log-loss %.4f -> %.4f (%+.2f%%)",
                rnd, direction, len(log["added"]), len(self.pool),
                before.val_loss, after.val_loss, 100 * rel,
            )
            # One weak round (an unlucky LLM batch) is not enough to stop.
            stale = stale + 1 if rel < bc.rel_epsilon else 0
            self._write_checkpoint(fingerprint, rnd + 1, stale)
            if stale >= bc.patience:
                stop_reason = "converged"
                break

        if not self.pool:
            raise RuntimeError(
                "No usable rule was produced. Check the task description, the data, "
                "or try a stronger generation model."
            )

        self._fit_final(cols, y01, in_p, in_v)
        self.jev_version = getattr(scorer, "version", None)
        self.metrics["stop_reason"] = stop_reason
        self.metrics["fitted_at"] = datetime.now(timezone.utc).isoformat()
        self._clear_checkpoint()
        return self

    async def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(YES) per row. NaN for rows Jev could not score at all."""
        self._check_fitted()
        states = _to_states(X)
        F = await self._get_scorer().score(states, self.rules)
        all_missing = (
            np.isnan(F).all(axis=1) if self.rules else np.zeros(len(states), dtype=bool)
        )
        F = np.where(np.isnan(F), self._feature_means, F)  # type: ignore[arg-type]
        p = np.mean([m.predict_proba(F)[:, 1] for m in self._models], axis=0)
        p[all_missing] = np.nan
        return p

    async def predict(self, X: pd.DataFrame) -> List[Literal["YES", "NO"] | None]:
        """YES/NO per row (None where the row could not be scored)."""
        p = await self.predict_proba(X)
        assert self.threshold is not None
        return [None if math.isnan(v) else ("YES" if v >= self.threshold else "NO") for v in p]

    def rule_table(self) -> pd.DataFrame:
        """Rules with non-zero weight, with their mean weight across fold models."""
        self._check_fitted()
        coefs = np.array([m.coef_[0] for m in self._models]).reshape(len(self._models), -1)
        df = pd.DataFrame({
            "rule": self.rules,
            "weight": coefs.mean(axis=0),
            "weight_sd": coefs.std(axis=0),
            "holds_mean": self._feature_means,
        })
        return df.reindex(df["weight"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    async def aclose(self) -> None:
        if isinstance(self._scorer, JevScorer):
            await self._scorer.aclose()

    # ── Rounds ──────────────────────────────────────────────────────────────

    def _evaluate(self, cols, rules, y01, folds, in_v) -> _Eval:
        """Best-C L1 fit on ``rules``; out-of-fold predictions and V log-loss."""
        wc = self.weight_config
        X = _matrix(cols, rules, len(y01))
        C = select_C(X, y01, folds, in_v, wc, one_se=False).C if rules else 1.0
        p, models = oof_proba(X, y01, folds, C, wc)
        return _Eval(C, p, models, float(row_log_loss(y01[in_v], p[in_v]).mean()))

    def _seed_prompt(self, states, y01, in_p, rng) -> str:
        k = self.boost_config.seed_examples_per_class
        yes = _sample(np.where(in_p & (y01 == 1))[0], k, rng)
        no = _sample(np.where(in_p & (y01 == 0))[0], k, rng)
        return SEED_PROMPT.format(
            task=self.task_description,
            fields=", ".join(f"`{f}`" for f in self.fields),
            n=self.boost_config.rules_per_round,
            yes_block=_block(states, yes),
            no_block=_block(states, no),
        )

    def _boost_prompt(self, states, y01, before: _Eval, g, in_p, rnd, rng, cols):
        """Odd rounds: missed YES rows. Even rounds: missed NO rows.

        "Missed" is relative: the rows of that class with the largest residual,
        whatever their absolute probability. An absolute cut-off (e.g. P > 0.5)
        would never select a NO row on imbalanced data, where the model rarely
        predicts above the base rate.
        """
        bc = self.boost_config
        cls = 1 if rnd % 2 == 1 else 0
        same = np.where(in_p & (y01 == cls))[0]
        order = same[np.argsort(-np.abs(g[same]))]
        hard = order[: bc.hard_examples]
        easiest = order[::-1][: 3 * bc.contrast_examples]
        contrast = _sample(np.setdiff1d(easiest, hard), bc.contrast_examples, rng)
        label = "YES" if cls == 1 else "NO"
        p_label = before.p if cls == 1 else 1 - before.p
        prompt = BOOST_PROMPT.format(
            task=self.task_description,
            fields=", ".join(f"`{f}`" for f in self.fields),
            rules_block=self._rules_block(cols, before.models),
            label=label,
            hard_block=_block(states, hard, p_label),
            contrast_block=_block(states, contrast, p_label),
            n=bc.rules_per_round,
        )
        return prompt, f"missed_{label}"

    def _rules_block(self, cols, fold_models) -> str:
        w = np.mean([m.coef_[0] for m in fold_models], axis=0)
        return "\n".join(
            f"- [weight {w[i]:+.2f}, holds {cols[r].mean():.0%}] {r}"
            for i, r in enumerate(self.pool)
        )

    def _filter(self, candidates, cols, g, in_p, in_v, log) -> List[str]:
        """Admission checks, no refitting: fire rate, redundancy, generality."""
        bc = self.boost_config
        lo, hi = bc.fire_rate_range
        kept: List[str] = []

        def reject(reason: str) -> None:
            log["rejected"][reason] = log["rejected"].get(reason, 0) + 1

        for rule in candidates:
            f = cols[rule]
            if not lo <= f[in_v].mean() <= hi:
                reject("constant")
                continue
            if any(abs(corr(f, cols[o])) > bc.max_redundancy for o in self.pool + kept):
                reject("redundant")
                continue
            c_p, c_v = corr(f[in_p], g[in_p]), corr(f[in_v], g[in_v])
            if abs(c_p) >= bc.min_generality_signal and (
                np.sign(c_p) != np.sign(c_v) or abs(c_v) < bc.generality_ratio * abs(c_p)
            ):
                reject("not_general")
                continue
            kept.append(rule)
        return kept

    # ── Final model ─────────────────────────────────────────────────────────

    def _fit_final(self, cols, y01, in_p, in_v) -> None:
        wc, bc = self.weight_config, self.boost_config
        folds = make_folds(y01, in_p, wc.cv_folds, wc.cv_repeats, self.random_state * 1000 + 999_983)
        rules = list(self.pool)
        X = _matrix(cols, rules, len(y01))
        sel = select_C(X, y01, folds, in_v, wc, one_se=wc.one_se_rule)
        p, models = oof_proba(X, y01, folds, sel.C, wc)

        # Keep only rules L1 uses in at least one fold model; refit on them.
        coefs = np.array([m.coef_[0] for m in models])
        alive = np.abs(coefs).max(axis=0) > 0
        if not alive.all():
            rules = [r for r, a in zip(rules, alive) if a]
            X = X[:, alive]
            p, models = oof_proba(X, y01, folds, sel.C, wc)
        if not rules:
            logger.warning("L1 zeroed every rule; the model predicts the base rate.")

        yv, pv = y01[in_v], p[in_v]
        t, f = choose_threshold(yv, pv, wc.beta, bc.threshold_grid, bc.threshold_smoothing)
        pred = (pv >= t).astype(int)
        self.rules, self._models, self._C, self.threshold = rules, models, sel.C, t
        self._feature_means = X.mean(axis=0)
        self.metrics = {
            "C": sel.C,
            "C_table": sel.table,
            "val_log_loss": float(row_log_loss(yv, pv).mean()),
            "val_auc": float(roc_auc_score(yv, pv)),
            f"val_f{wc.beta:g}": f,
            "val_precision": float(precision_score(yv, pred, zero_division=0)),
            "val_recall": float(recall_score(yv, pred, zero_division=0)),
            "val_accuracy": float((pred == yv).mean()),
            "threshold": t,
            "n_rules": len(rules),
            "n_pool": len(self.pool),
            "n_tried": len(self.tried),
            "n_train": int(len(y01)),
            "n_show_pool": int(in_p.sum()),
            "n_val_pool": int(in_v.sum()),
        }

    # ── Scoring / generation plumbing ──────────────────────────────────────

    async def _score(self, scorer: Scorer, states, rules: List[str]) -> Dict[str, np.ndarray]:
        F = await scorer.score(states, rules)
        out = {}
        for j, r in enumerate(rules):
            col = F[:, j]
            n_nan = int(np.isnan(col).sum())
            if n_nan == len(col):
                raise RuntimeError(f"Rule could not be scored on any row: {r!r}")
            if n_nan:
                logger.warning("Rule %r: %d unscored rows filled with the rule mean.", r, n_nan)
                col = np.where(np.isnan(col), np.nanmean(col), col)
            out[r] = col
        return out

    def _get_scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = JevScorer(
                model=self.jev_version or self.jev_model,
                cache_path=self.save_path / "jev_cache.sqlite",
            )
        return self._scorer

    def _get_generator(self) -> RuleGenerator:
        if self._generator is None:
            self._generator = make_generator(self.gen_model_name)
        return self._generator

    def _check_fitted(self) -> None:
        if not self._models or self.threshold is None:
            raise RuntimeError("Model is not fitted. Call fit() or load().")

    # ── Checkpointing ───────────────────────────────────────────────────────

    def _read_checkpoint(self, fingerprint: str) -> Dict[str, Any] | None:
        path = self.save_path / _CHECKPOINT
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("format") != _CHECKPOINT_FORMAT or data.get("fingerprint") != fingerprint:
            logger.warning("Ignoring checkpoint from a different dataset, config or version.")
            return None
        return data

    def _write_checkpoint(self, fingerprint: str, next_round: int, stale: int) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        tmp = self.save_path / (_CHECKPOINT + ".tmp")
        tmp.write_text(json.dumps({
            "format": _CHECKPOINT_FORMAT, "fingerprint": fingerprint,
            "next_round": next_round, "stale": stale, "pool": self.pool, "tried": self.tried,
            "history": self.history,
        }))
        tmp.replace(self.save_path / _CHECKPOINT)

    def _clear_checkpoint(self) -> None:
        (self.save_path / _CHECKPOINT).unlink(missing_ok=True)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Write model.json, models.joblib and report.md."""
        self._check_fitted()
        base = Path(path) if path else self.save_path
        base.mkdir(parents=True, exist_ok=True)
        manifest = {
            "version": 2,
            "task_description": self.task_description,
            "gen_model": self.gen_model_name,
            "max_policy_length": self.max_policy_length,
            "gen_temperature": self.gen_temperature,
            "random_state": self.random_state,
            "weight_config": asdict(self.weight_config),
            "boost_config": asdict(self.boost_config),
            "jev_version": self.jev_version,
            "fields": self.fields,
            "rules": self.rules,
            "pool": self.pool,
            "tried": self.tried,
            "threshold": self.threshold,
            "feature_means": self._feature_means.tolist(),  # type: ignore[union-attr]
            "C": self._C,
            "metrics": self.metrics,
            "history": self.history,
        }
        (base / "model.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        joblib.dump(self._models, base / "models.joblib")
        (base / "report.md").write_text(self.report(), encoding="utf-8")
        return base

    @classmethod
    def load(cls, path: str | Path, scorer: Scorer | None = None) -> "PolicyInduction":
        base = Path(path)
        m = json.loads((base / "model.json").read_text())
        wc = m["weight_config"]
        wc["Cs"] = tuple(wc["Cs"])
        # Tolerate constants added or removed since the model was saved.
        known = {f.name for f in dataclass_fields(BoostConfig)}
        bc = {k: v for k, v in m["boost_config"].items() if k in known}
        for k in ("fire_rate_range", "threshold_grid"):
            if k in bc:
                bc[k] = tuple(bc[k])
        inst = cls(
            task_description=m["task_description"],
            gen_model=m["gen_model"],
            max_policy_length=m["max_policy_length"],
            gen_temperature=m["gen_temperature"],
            random_state=m["random_state"],
            weight_config=WeightConfig(**wc),
            boost_config=BoostConfig(**bc),
            scorer=scorer,
            # Predict with exactly the Jev version the weights were fit on.
            jev_model=m["jev_version"] or "jev-latest",
            save_path=base,
        )
        inst.jev_version = m["jev_version"]
        inst.fields, inst.rules, inst.pool = m["fields"], m["rules"], m["pool"]
        inst.tried = m.get("tried", list(m["pool"]))
        inst.threshold, inst._C = m["threshold"], m["C"]
        inst._feature_means = np.array(m["feature_means"])
        inst.metrics, inst.history = m["metrics"], m["history"]
        inst._models = joblib.load(base / "models.joblib")
        return inst

    def report(self) -> str:
        self._check_fitted()
        m = self.metrics
        beta = self.weight_config.beta
        lines = [
            "# PolicyInduction report", "",
            f"- Fitted: {m.get('fitted_at', '?')}",
            f"- Generation model: `{self.gen_model_name}` · Jev: `{self.jev_version}`",
            f"- Rules: {m['n_rules']} with non-zero weight / {m['n_pool']} in pool / "
            f"{m['n_tried']} proposed (pool cap {self.max_policy_length}, "
            f"stopped: {m.get('stop_reason')})",
            f"- Rows: {m['n_train']} (show pool {m['n_show_pool']}, validation pool {m['n_val_pool']})",
            "",
            "## Validation (out-of-fold, rows the generation LLM never saw)", "",
            f"| log-loss | AUC | F{beta:g} | precision | recall | accuracy | threshold |",
            "|---|---|---|---|---|---|---|",
            f"| {m['val_log_loss']:.4f} | {m['val_auc']:.4f} | {m[f'val_f{beta:g}']:.4f} "
            f"| {m['val_precision']:.4f} | {m['val_recall']:.4f} | {m['val_accuracy']:.4f} "
            f"| {m['threshold']:.2f} |",
            "",
            "## Rules with non-zero weight (ranked by |mean weight|)", "",
            "| # | weight | ±sd | holds | rule |", "|---|---|---|---|---|",
        ]
        for i, row in self.rule_table().iterrows():
            rule = str(row["rule"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i + 1} | {row['weight']:+.3f} | {row['weight_sd']:.3f} "
                f"| {row['holds_mean']:.0%} | {rule} |"
            )
        unused = [r for r in self.pool if r not in set(self.rules)]
        if unused:
            lines += ["", f"## Pooled rules with zero weight ({len(unused)})", ""]
            lines += [f"- {r}" for r in unused]
        lines += ["", "## Boosting rounds", "",
                  "| round | focus | proposed | added | rejected | V log-loss | change |",
                  "|---|---|---|---|---|---|---|"]
        for h in self.history:
            rej = ", ".join(f"{k} {v}" for k, v in h["rejected"].items()) or "-"
            lines.append(
                f"| {h['round']} | {h['direction']} | {h['proposed']} | {len(h['added'])} "
                f"| {rej} | {h['val_log_loss_before']:.4f} → {h['val_log_loss_after']:.4f} "
                f"| {-100 * h['relative_improvement']:+.2f}% |"
            )
        return "\n".join(lines) + "\n"


# ── Helpers ─────────────────────────────────────────────────────────────────


GEN_RETRIES = 4


async def _generate_with_retry(generator: RuleGenerator, prompt: str, temperature: float) -> List[str]:
    """Generation providers return transient 429/503s under load; back off and retry."""
    for attempt in range(GEN_RETRIES):
        try:
            return await generator.generate(GEN_SYSTEM, prompt, temperature)
        except Exception as e:
            status = getattr(e, "code", None) or getattr(e, "status_code", None)
            permanent = isinstance(status, int) and 400 <= status < 500 and status != 429
            if permanent or attempt == GEN_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("Generation failed (%s); retrying in %ds.", e, wait)
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")


def _to_states(X: pd.DataFrame) -> List[Dict[str, Any]]:
    """One JSON-able dict per row; this is exactly what Jev sees."""
    records = X.to_dict(orient="records")
    return [{str(k): _jsonable(v) for k, v in r.items()} for r in records]


def _jsonable(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (str, int, float, bool, list, dict)):
        return v
    return str(v)


def _to_binary(y: Sequence[Any]) -> np.ndarray:
    out = []
    for v in y:
        s = str(v).strip().upper()
        if s in {"YES", "1", "TRUE"}:
            out.append(1)
        elif s in {"NO", "0", "FALSE"}:
            out.append(0)
        else:
            raise ValueError(f"Label must be YES/NO (or 1/0), got {v!r}")
    return np.array(out, dtype=int)


def _split_pools(y01: np.ndarray, show_fraction: float, rng) -> np.ndarray:
    """Stratified boolean mask: True = show pool P."""
    in_p = np.zeros(len(y01), dtype=bool)
    for cls in (0, 1):
        idx = rng.permutation(np.where(y01 == cls)[0])
        in_p[idx[: int(round(len(idx) * show_fraction))]] = True
    return in_p


def _sample(idx: np.ndarray, k: int, rng) -> np.ndarray:
    return idx if len(idx) <= k else rng.choice(idx, size=k, replace=False)


def _truncate(v: Any) -> Any:
    if isinstance(v, str) and len(v) > MAX_FIELD_CHARS_IN_PROMPT:
        return v[:MAX_FIELD_CHARS_IN_PROMPT] + " [...]"
    return v


def _block(states, idx, p_label: np.ndarray | None = None) -> str:
    lines = []
    for i in idx:
        body = json.dumps({k: _truncate(v) for k, v in states[i].items()}, ensure_ascii=False)
        prefix = f"[model P={p_label[i]:.2f}] " if p_label is not None else ""
        lines.append(f"- {prefix}{body}")
    return "\n".join(lines)


def _dedupe(rules: List[str], seen: set) -> List[str]:
    out = []
    for r in rules:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _matrix(cols: Dict[str, np.ndarray], rules: List[str], n: int) -> np.ndarray:
    if not rules:
        return np.zeros((n, 0))
    return np.column_stack([cols[r] for r in rules])


def _fingerprint(states, y01, random_state, show_fraction) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(states, sort_keys=True, default=str).encode())
    h.update(y01.tobytes())
    h.update(f"{random_state}|{show_fraction}".encode())
    return h.hexdigest()
