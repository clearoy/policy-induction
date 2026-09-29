"""PolicyInduction with boosting.

An interpretable binary classifier. An LLM writes natural-language rules, Jev
scores P(rule true) for every sample, and an L1 logistic regression weights
the rules. Rules are found by boosting:

    each round
      1. fit the current rules, get out-of-fold P(YES) for every row
      2. residual g = y - p on the show pool P
      3. show the LLM the P rows the model gets most wrong (plus correctly
         handled rows of the same class as contrast)
      4. the LLM proposes new rules; Jev scores them on every row
      5. a rule is kept only if it lowers out-of-fold log-loss on the
         validation pool V -- rows the LLM has never seen -- by more than
         one standard error
    stop when rounds stop adding rules

Rules are never removed once accepted; L1 shrinks the weight of any that later
rules make redundant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, Tuple

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
    mean_se,
    oof_proba,
    paired_gain,
    row_log_loss,
    select_C,
)

logger = logging.getLogger(__name__)

# Long free-text fields are cut when shown to the generation LLM, to keep
# prompts bounded. Jev always scores the full sample.
MAX_FIELD_CHARS_IN_PROMPT = 1500

_CHECKPOINT = "checkpoint.json"


class PolicyInduction:
    """Boosted, interpretable binary classifier built from LLM-written rules.

    Args:
        task_description: What is being predicted and what YES/NO mean. This
            is the one input that most shapes the rules; be concrete.
        gen_model: Generation LLM name (``gemini-*`` or ``gpt-*``) or any
            object implementing ``RuleGenerator``.
        max_policy_length: Upper bound on rules in the model (<= 100). The
            effective bound is also limited by data: at most
            ``minority_count / 10`` rules. Boosting usually stops earlier.
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
        gen_model: str | RuleGenerator = "gemini-3.5-flash",
        max_policy_length: int = 30,
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
        self.rules: List[str] = []  # active rules, in acceptance order
        self.pool: List[str] = []  # every rule ever scored (accepted or not)
        self.history: List[Dict[str, Any]] = []
        self.threshold: float | None = None
        self.metrics: Dict[str, Any] = {}
        self.jev_version: str | None = None
        self._models: list = []
        self._feature_means: np.ndarray | None = None
        self._C: float | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    async def fit(self, X: pd.DataFrame, y: Sequence[Any]) -> "PolicyInduction":
        """Induce rules by boosting, then fit the final weighted model."""
        states = _to_states(X)
        y01 = _to_binary(y)
        if len(states) != len(y01):
            raise ValueError("X and y must have the same length")
        if len(np.unique(y01)) != 2:
            raise ValueError("y must contain both classes")

        bc, wc = self.boost_config, self.weight_config
        self.fields = [str(c) for c in X.columns]
        rng = np.random.default_rng(self.random_state)
        scorer = self._get_scorer()
        generator = self._get_generator()

        in_p = _split_pools(y01, bc.show_fraction, rng)
        in_v = ~in_p
        minority = int(min(y01.sum(), len(y01) - y01.sum()))
        cap = max(1, min(self.max_policy_length, minority // bc.min_positives_per_rule))
        if cap < self.max_policy_length:
            logger.info(
                "Rule cap lowered to %d by data size (%d minority rows / %d per rule).",
                cap, minority, bc.min_positives_per_rule,
            )

        # Resume
        fingerprint = _fingerprint(states, y01, self.random_state, bc.show_fraction)
        ckpt = self._read_checkpoint(fingerprint)
        start_round, stale = 0, 0
        if ckpt:
            self.pool, self.rules, self.history = ckpt["pool"], ckpt["rules"], ckpt["history"]
            start_round, stale = ckpt["next_round"], ckpt["stale"]
            logger.info("Resuming at round %d with %d rules.", start_round, len(self.rules))

        # Scores for pooled rules (cache hits on resume)
        cols: Dict[str, np.ndarray] = {}
        if self.pool:
            cols.update(await self._score(scorer, states, self.pool))

        done_reason = "max_rounds"
        for rnd in range(start_round, bc.max_rounds + 1):
            if len(self.rules) >= cap:
                done_reason = "rule_cap"
                break

            folds = make_folds(y01, in_p, wc.cv_folds, wc.cv_repeats, self.random_state * 1000 + rnd)
            X_act = _matrix(cols, self.rules, len(y01))
            C = select_C(X_act, y01, folds, in_v, wc, one_se=False).C if self.rules else 1.0
            p, fold_models = oof_proba(X_act, y01, folds, C, wc)
            g = y01 - p
            base_loss = row_log_loss(y01[in_v], p[in_v])

            # Build the prompt from show-pool rows only.
            if rnd == 0:
                prompt, direction = self._seed_prompt(states, y01, in_p, rng), "seed"
            else:
                built = self._boost_prompt(states, y01, p, g, in_p, rnd, rng, cols, fold_models)
                if built is None:
                    done_reason = "no_hard_examples"
                    break
                prompt, direction = built

            proposed = await _generate_with_retry(generator, prompt, self.gen_temperature)
            candidates = _dedupe([r.strip() for r in proposed if r and r.strip()], set(self.pool))
            log: Dict[str, Any] = {
                "round": rnd, "direction": direction, "C": C,
                "val_log_loss_before": float(base_loss.mean()),
                "proposed": len(proposed), "rejected": {}, "accepted": [],
            }

            if candidates:
                cols.update(await self._score(scorer, states, candidates))
                self.pool.extend(candidates)
                survivors = self._filter(candidates, cols, g, in_p, in_v, log)

                # Greedy acceptance, strongest V-side residual signal first.
                survivors.sort(key=lambda r: -abs(corr(cols[r][in_v], g[in_v])))
                for rule in survivors:
                    if len(self.rules) >= cap:
                        break
                    X_try = _matrix(cols, self.rules + [rule], len(y01))
                    p_try, _ = oof_proba(X_try, y01, folds, C, wc)
                    try_loss = row_log_loss(y01[in_v], p_try[in_v])
                    gain, se = paired_gain(base_loss, try_loss)
                    if gain > 0 and gain > bc.accept_z * se:
                        self.rules.append(rule)
                        base_loss = try_loss
                        log["accepted"].append({"rule": rule, "gain": gain, "se": se})
                    else:
                        log["rejected"]["no_gain"] = log["rejected"].get("no_gain", 0) + 1

            log["val_log_loss_after"] = float(base_loss.mean())
            log["n_rules"] = len(self.rules)
            self.history.append(log)
            logger.info(
                "Round %d (%s): +%d rules -> %d, V log-loss %.4f -> %.4f",
                rnd, direction, len(log["accepted"]), len(self.rules),
                log["val_log_loss_before"], log["val_log_loss_after"],
            )

            stale = 0 if log["accepted"] or rnd == 0 else stale + 1
            self._write_checkpoint(fingerprint, rnd + 1, stale)
            if stale >= bc.patience:
                done_reason = "patience"
                break

        if not self.rules:
            raise RuntimeError(
                "No rule improved validation log-loss. Check the task description, "
                "the data, or try a stronger generation model."
            )

        self._fit_final(cols, y01, in_p, in_v)
        self.jev_version = getattr(scorer, "version", None)
        self.metrics["stop_reason"] = done_reason
        self.metrics["rule_cap"] = cap
        self.metrics["fitted_at"] = datetime.now(timezone.utc).isoformat()
        self._clear_checkpoint()
        return self

    async def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(YES) per row. NaN for rows Jev could not score at all."""
        self._check_fitted()
        states = _to_states(X)
        F = await self._get_scorer().score(states, self.rules)
        all_missing = np.isnan(F).all(axis=1)
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
        """Active rules with their mean weight across fold models."""
        self._check_fitted()
        coefs = np.array([m.coef_[0] for m in self._models])
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

    def _boost_prompt(self, states, y01, p, g, in_p, rnd, rng, cols, fold_models):
        """Alternate between missed YES rows (odd rounds) and missed NO rows."""
        bc = self.boost_config
        order = [1, 0] if rnd % 2 == 1 else [0, 1]
        for cls in order:
            wrong = g > bc.min_hard_residual if cls == 1 else g < -bc.min_hard_residual
            hard = np.where(in_p & (y01 == cls) & wrong)[0]
            if len(hard) < bc.min_hard_count:
                continue
            hard = hard[np.argsort(-np.abs(g[hard]))][: bc.hard_examples]
            same = np.where(in_p & (y01 == cls))[0]
            easiest = same[np.argsort(np.abs(g[same]))][: 3 * bc.contrast_examples]
            contrast = _sample(easiest, bc.contrast_examples, rng)
            label = "YES" if cls == 1 else "NO"
            p_label = p if cls == 1 else 1 - p
            prompt = BOOST_PROMPT.format(
                task=self.task_description,
                fields=", ".join(f"`{f}`" for f in self.fields),
                rules_block=self._rules_block(cols, fold_models),
                label=label,
                hard_block=_block(states, hard, p_label),
                contrast_block=_block(states, contrast, p_label),
                n=bc.rules_per_round,
            )
            return prompt, f"missed_{label}"
        return None

    def _rules_block(self, cols, fold_models) -> str:
        if not self.rules:
            return "(none yet)"
        w = np.mean([m.coef_[0] for m in fold_models], axis=0)
        lines = [
            f"- [weight {w[i]:+.2f}, holds {cols[r].mean():.0%}] {r}"
            for i, r in enumerate(self.rules)
        ]
        return "\n".join(lines)

    def _filter(self, candidates, cols, g, in_p, in_v, log) -> List[str]:
        """Cheap checks before any refitting: fire rate, redundancy, generality."""
        bc = self.boost_config
        lo, hi = bc.fire_rate_range
        kept: List[str] = []
        existing = [r for r in self.pool if r not in candidates]

        def reject(reason: str) -> None:
            log["rejected"][reason] = log["rejected"].get(reason, 0) + 1

        for rule in candidates:
            f = cols[rule]
            if not lo <= f[in_v].mean() <= hi:
                reject("constant")
                continue
            if any(abs(corr(f, cols[o])) > bc.max_redundancy for o in existing + kept):
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
        X = _matrix(cols, self.rules, len(y01))
        sel = select_C(X, y01, folds, in_v, wc, one_se=wc.one_se_rule)
        p, models = oof_proba(X, y01, folds, sel.C, wc)

        # Drop rules that L1 zeroed in every fold model, then refit.
        coefs = np.array([m.coef_[0] for m in models])
        alive = np.abs(coefs).max(axis=0) > 0
        if not alive.all() and alive.any():
            self.rules = [r for r, a in zip(self.rules, alive) if a]
            X = X[:, alive]
            p, models = oof_proba(X, y01, folds, sel.C, wc)

        yv, pv = y01[in_v], p[in_v]
        t, f = choose_threshold(yv, pv, wc.beta, bc.threshold_grid, bc.threshold_smoothing)
        pred = (pv >= t).astype(int)
        self._models, self._C, self.threshold = models, sel.C, t
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
            "n_rules": len(self.rules),
            "n_pool": len(self.pool),
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
        if data.get("fingerprint") != fingerprint:
            logger.warning("Ignoring checkpoint from a different dataset/config.")
            return None
        return data

    def _write_checkpoint(self, fingerprint: str, next_round: int, stale: int) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        tmp = self.save_path / (_CHECKPOINT + ".tmp")
        tmp.write_text(json.dumps({
            "fingerprint": fingerprint, "next_round": next_round, "stale": stale,
            "pool": self.pool, "rules": self.rules, "history": self.history,
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
            "version": 1,
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
        bc = m["boost_config"]
        bc["fire_rate_range"] = tuple(bc["fire_rate_range"])
        bc["threshold_grid"] = tuple(bc["threshold_grid"])
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
            f"- Rules: {m['n_rules']} active / {m['n_pool']} proposed "
            f"(cap {m.get('rule_cap')}, stopped: {m.get('stop_reason')})",
            f"- Rows: {m['n_train']} (show pool {m['n_show_pool']}, validation pool {m['n_val_pool']})",
            "",
            "## Validation (out-of-fold, rows the generation LLM never saw)", "",
            f"| log-loss | AUC | F{beta:g} | precision | recall | accuracy | threshold |",
            "|---|---|---|---|---|---|---|",
            f"| {m['val_log_loss']:.4f} | {m['val_auc']:.4f} | {m[f'val_f{beta:g}']:.4f} "
            f"| {m['val_precision']:.4f} | {m['val_recall']:.4f} | {m['val_accuracy']:.4f} "
            f"| {m['threshold']:.2f} |",
            "",
            "## Rules (ranked by |mean weight|)", "",
            "| # | weight | ±sd | holds | rule |", "|---|---|---|---|---|",
        ]
        for i, row in self.rule_table().iterrows():
            rule = str(row["rule"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {i + 1} | {row['weight']:+.3f} | {row['weight_sd']:.3f} "
                f"| {row['holds_mean']:.0%} | {rule} |"
            )
        lines += ["", "## Boosting rounds", "",
                  "| round | focus | proposed | accepted | rejected | V log-loss |",
                  "|---|---|---|---|---|---|"]
        for h in self.history:
            rej = ", ".join(f"{k} {v}" for k, v in h["rejected"].items()) or "-"
            lines.append(
                f"| {h['round']} | {h['direction']} | {h['proposed']} | {len(h['accepted'])} "
                f"| {rej} | {h['val_log_loss_before']:.4f} → {h['val_log_loss_after']:.4f} |"
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
