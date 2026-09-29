"""VCBench: fit PolicyInduction on the public split, evaluate on the private split.

Task: predict whether a founder is successful (company raised > $500M or
exited/IPO'd above $500M). Both splits have 4,500 founders, 9.0% positive.

Run from the repo root:

    .venv/bin/python experiments/vcbench/run_vcbench.py

Re-running resumes: finished boosting rounds come from the checkpoint and
every Jev answer already obtained comes from the cache. Pass --name to keep
separate runs apart (default: "default").

Outputs go to experiments/vcbench/runs/<name>/ (gitignored):
    report.md          rules, weights, validation metrics, per-round log
    model.json         the fitted model (reload with PolicyInduction.load)
    models.joblib      fold models
    predictions.csv    per-founder test predictions
    metrics.json       settings, validation and test metrics, cost, git commit
    run.log            full log
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Works with or without `pip install -e .` (macOS can hide the editable .pth file).
sys.path.insert(0, str(ROOT))

from policy_induction import JevScorer, PolicyInduction  # noqa: E402

# ── Settings (edit here) ────────────────────────────────────────────────────

GEN_MODEL = "deepseek-chat"
TEXT_COLUMN = "anonymised_prose"  # same input as the earlier think-reason-learn runs
TASK = (
    "Predict whether a startup founder will be successful based on their "
    "educational background, professional experience, and industry. A "
    "successful founder (YES) is one whose company has achieved either total "
    "funding over $500M or an exit/IPO valued over $500M; otherwise NO. All "
    "founders under consideration are sourced from LinkedIn and Crunchbase "
    "profiles of companies that have raised between $100K and $4M in funding."
)
BETA = 0.5  # the model's default; used here to report test F-beta

# Best test F0.5 of the earlier think-reason-learn runs (500 train rows,
# binary Gemini scoring), for reference.
PREVIOUS_BEST_TEST_F05 = 0.246
JEV_PRICE_PER_MTOK = 0.042

DATA = HERE / "data"
log = logging.getLogger("vcbench")


def load_split(name: str) -> tuple[pd.DataFrame, list[str], pd.Series]:
    df = pd.read_csv(DATA / f"vcbench_final_{name}.csv")
    X = df[[TEXT_COLUMN]].rename(columns={TEXT_COLUMN: "profile"})
    y = df["success"].map({1: "YES", 0: "NO"}).tolist()
    return X, y, df["founder_uuid"]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=ROOT).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def test_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "auc": float(roc_auc_score(y_true, p)),
        "log_loss": float(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6))),
        f"f{BETA:g}": float(fbeta_score(y_true, pred, beta=BETA, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "predicted_yes": int(pred.sum()),
        "always_no_accuracy": float(1 - y_true.mean()),
    }


async def run(name: str) -> None:
    commit = git_commit()  # the code this run executes, captured before it can change
    X_train, y_train, _ = load_split("public")
    X_test, y_test, test_ids = load_split("private")

    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(out / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    log.info("Run %s: train=%d test=%d gen=%s commit=%s", name, len(y_train), len(y_test), GEN_MODEL, commit)

    scorer = JevScorer(cache_path=out / "jev_cache.sqlite")
    model = PolicyInduction(
        task_description=TASK, gen_model=GEN_MODEL, scorer=scorer, save_path=out
    )
    started = time.monotonic()
    try:
        await model.fit(X_train, y_train)
        model.save(out)
        fit_seconds = time.monotonic() - started

        p = await model.predict_proba(X_test)
        scored = ~np.isnan(p)
        y_true = np.array([1 if v == "YES" else 0 for v in y_test])
        metrics = test_metrics(y_true[scored], p[scored], model.threshold)
        metrics["unscored_rows"] = int((~scored).sum())

        pd.DataFrame({
            "founder_uuid": test_ids,
            "y_true": y_test,
            "p_yes": p,
            "prediction": [
                None if math.isnan(v) else ("YES" if v >= model.threshold else "NO") for v in p
            ],
        }).to_csv(out / "predictions.csv", index=False)

        cost = scorer.input_tokens / 1e6 * JEV_PRICE_PER_MTOK
        (out / "metrics.json").write_text(json.dumps({
            "name": name,
            "git_commit": commit,
            "gen_model": GEN_MODEL,
            "jev_version": model.jev_version,
            "n_rules": len(model.rules),
            "validation": {k: v for k, v in model.metrics.items() if k.startswith("val_")},
            "test": metrics,
            "jev_requests": scorer.requests,
            "jev_input_tokens": scorer.input_tokens,
            "jev_cost_usd": cost,
            "fit_seconds": fit_seconds,
            "total_seconds": time.monotonic() - started,
        }, indent=2, default=str))
    finally:
        await model.aclose()

    f_key = f"f{BETA:g}"
    print(f"\n=== {name} ===")
    print(f"rules: {len(model.rules)}   Jev: {model.jev_version}   "
          f"requests this run: {scorer.requests:,} (~${cost:.2f})")
    print(f"validation (out-of-fold): AUC {model.metrics['val_auc']:.4f}  "
          f"F{BETA:g} {model.metrics[f'val_{f_key}']:.4f}")
    print(f"test (private, n={metrics['n']}): AUC {metrics['auc']:.4f}  "
          f"F{BETA:g} {metrics[f_key]:.4f}  precision {metrics['precision']:.4f}  "
          f"recall {metrics['recall']:.4f}  accuracy {metrics['accuracy']:.4f} "
          f"(always-NO {metrics['always_no_accuracy']:.4f})")
    print(f"previous best test F0.5 (think-reason-learn): {PREVIOUS_BEST_TEST_F05:.3f}")
    print(f"outputs: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PolicyInduction on VCBench.")
    ap.add_argument("--name", default="default", help="run folder; re-use it to resume")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "openai", "typesafe_sdk"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    asyncio.run(run(args.name))


if __name__ == "__main__":
    main()
