# PolicyInduction (boosted)

An interpretable binary classifier. An LLM writes natural-language rules,
[TypeSafe Jev](https://docs.typesafe.ai) scores the probability that each rule
holds for each sample, and an L1 logistic regression learns a weight per rule.

Rules are found by **boosting**: every round targets the samples the current
model gets wrong, and a new rule is kept only if it lowers the error on data the
LLM has never seen.

## How it works

```
training data ──┬── P, show pool (30%):  the only rows whose labels the LLM sees
                └── V, validation pool (70%): never shown; every score is measured here

one rule pool, expand-only

round 0     sample 20 YES + 20 NO rows from P -> the LLM writes seed rules
round 1..R
  1. fit L1 logistic regression on the pool -> out-of-fold P(YES) for every row
  2. residual g = y - p; take the P rows of one class the model gets most wrong:
     missed YES rows on odd rounds, missed NO rows on even rounds, plus
     correctly handled rows of the same class as contrast
  3. the LLM sees the whole pool (with weights) and those rows, and proposes
     new rules -> Jev scores them on every row
  4. a new rule joins the pool unless it is near-constant, a near-duplicate of
     a pooled rule, or fits P but not V
  5. refit on the grown pool; measure out-of-fold log-loss on V
stop        the pool reaches max_policy_length, or 2 consecutive rounds each
            lower V log-loss by less than rel_epsilon (0.3%, relative)
finish      choose C by the one-standard-error rule, then the decision threshold
            on V's pooled out-of-fold probabilities
predict     ask Jev only the rules with non-zero weight; mean P(YES) of the 15
            fold models >= threshold -> YES
```

- **Expand-only.** Nothing leaves the pool; L1 decides which rules carry
  weight, so a rule that looked useless early can gain weight later.
- **Features are probabilities**: Jev's `noul`, P(rule holds), not 0/1.
- **"Missed" is relative**: the largest residuals within a class, so the NO
  side is shown even on imbalanced data where no row gets P(YES) > 0.5.
- **Jev version is pinned.** The concrete version answering the first request
  (e.g. `jev-1.13.0`) is saved with the model and used for all later
  predictions, so features never drift under the trained weights.

## Install

With [uv](https://docs.astral.sh/uv/):

```bash
cd policy-induction
uv sync
```

This creates `.venv` with the Python version in `.python-version`, installs the
exact dependency versions recorded in `uv.lock` (including the dev tools), and
installs `policy_induction` in editable mode. Run commands with `uv run`, e.g.
`uv run pytest`, or use `.venv/bin/python` directly.

Dependencies are declared in `pyproject.toml`; `uv.lock` pins every version
so an experiment can be reproduced exactly. After changing dependencies, run
`uv lock` and commit the updated lock file.

Without uv: `pip install -e .` (plus `pytest pytest-asyncio` for the tests),
which resolves versions from the ranges in `pyproject.toml` instead of the lock.

> **macOS note.** macOS may set the "hidden" flag on the editable-install `.pth`
> file, and Python 3.13 skips hidden `.pth` files, which makes `import
> policy_induction` fail. Fix it with
> `chflags nohidden .venv/lib/python3.13/site-packages/*.pth`. The experiment
> scripts add the repo root to `sys.path`, so they work either way.

## Configure `.env`

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.

| Variable | Needed for |
|---|---|
| `TYPESAFE_API_KEY` | **Required.** Jev scoring. Create a key at <https://console.typesafe.ai/keys> |
| `DEEPSEEK_API_KEY` | Generating rules with a `deepseek-*` model (used by the VCBench script) |
| `OPENAI_API_KEY` | Generating rules with a `gpt-*` model (the library default) |
| `GOOGLE_AI_API_KEY` | Generating rules with a `gemini-*` model |

## Usage

```python
import asyncio
from dotenv import load_dotenv
from policy_induction import PolicyInduction, WeightConfig

load_dotenv()

async def main():
    model = PolicyInduction(
        task_description="Predict whether ... YES means ..., NO means ...",
        gen_model="gpt-5.6",
        weight_config=WeightConfig(beta=0.5),
        save_path="runs/my_run",
    )
    await model.fit(X_train, y_train)          # y: "YES"/"NO" or 1/0
    print(model.rule_table())                  # rule, weight, how often it holds
    labels = await model.predict(X_test)       # ["YES", "NO", ...]
    probs = await model.predict_proba(X_test)
    model.save()                               # model.json, models.joblib, report.md
    await model.aclose()

asyncio.run(main())
```

Reload a saved model with `PolicyInduction.load("runs/my_run")`.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `task_description` | required | What is predicted and what YES/NO mean. The input that most shapes the rules |
| `gen_model` | `gpt-5.6` | Rule-writing LLM (`gpt-*`, `deepseek-*`, `gemini-*`, or any `RuleGenerator`) |
| `max_policy_length` | 100 | Size cap of the rule pool (<= 100); boosting stops when it is full. The pool grows by up to 10 rules per round |
| `gen_temperature` | 1.0 | Sampling temperature of the rule-writing LLM |
| `random_state` | 0 | P/V split, example selection and CV folds. **Does not control the LLM** |
| `weight_config` | `WeightConfig()` | `beta` (threshold only), `Cs`, `cv_folds`, `cv_repeats`, `one_se_rule`, `class_weight_balanced` |

`BoostConfig` holds the internal boosting constants (show-pool share, rules per
round, filter thresholds, `rel_epsilon`, `patience`). They rarely need changing.

## Outputs

`save()` writes to `save_path`:

- `model.json`: rule pool, rules with non-zero weight, threshold, pinned Jev
  version, metrics, per-round log
- `models.joblib`: the 15 fold models (predictions average them)
- `report.md`: readable report with rules ranked by weight, validation metrics
  and the boosting log
- `jev_cache.sqlite`: cached Jev answers keyed by version + sample + rule; a
  re-run only requests what is missing
- `checkpoint.json`: mid-training state, deleted when training completes

All reported validation metrics are **out-of-fold on V**, i.e. measured on rows
the rule-writing LLM never saw.

## Experiments

### VCBench

Founder success prediction (4,500 public / 4,500 private founders, 9.0%
positive). Put `vcbench_final_public.csv` and `vcbench_final_private.csv` in
`experiments/vcbench/data/` (gitignored), then from the repo root:

```bash
.venv/bin/python experiments/vcbench/run_vcbench.py
```

It trains on all public rows (anonymised profile text, as in the earlier
think-reason-learn runs) and evaluates on all private rows, writing rules with
`deepseek-chat`. Settings such as the generation model live as constants at the top of the script. Re-running
resumes an interrupted run; `--name` keeps separate runs apart.

Results go to `experiments/vcbench/runs/<name>/` (gitignored): `report.md`,
`model.json`, `predictions.csv`, `metrics.json` (validation and test metrics,
Jev cost, git commit) and `run.log`.

## How rules are written

Jev judges each rule literally and one at a time, and is weak at arithmetic,
compound conditions and multi-step reasoning. The generation prompt therefore
requires every rule to:

- describe exactly one observable condition (no "and"/"or")
- state a condition, not a verdict (the weight decides the direction)
- name fields in backticks, e.g. `` `profile` ``
- avoid quoting samples or naming specific people, companies or numbers

## Known limitations

- **Numeric-heavy data**: Jev is weak at numeric comparisons; on mostly numeric
  tables a plain logistic regression may do better.
- **Pairwise tasks** (A vs B): treated as ordinary samples. Antisymmetric
  features (f(A) - f(B)) are not implemented yet.
- **Data size**: V must be large enough for its log-loss to be a stable
  signal; with a few hundred rows, round-to-round changes are mostly noise.
- **Generation is stochastic**: compare configurations over at least 3 seeds.

## Tests

```bash
uv run pytest -q
```

Fully offline: a fake generator and fake Jev on synthetic data check that
boosting finds all signal rules, that V rows never appear in a prompt, both
stopping conditions, that even rounds show missed NO rows, expand-only
checkpoint resume, save/load, retry behaviour, request throttling and the
OpenAI/DeepSeek fallbacks.

## Layout

```
policy_induction/
  model.py       PolicyInduction: boosting loop, final fit, predict, save/load
  weights.py     cross-validation, choosing C, threshold
  scorer.py      Jev scoring, version pinning, SQLite cache, rate limit
  generator.py   rule-writing LLMs (OpenAI / DeepSeek / Gemini)
  prompts.py     generation prompts and rule-writing constraints
  config.py      WeightConfig, BoostConfig
experiments/
  vcbench/run_vcbench.py
tests/
pyproject.toml   dependencies
uv.lock          pinned versions
```
