# PolicyInduction (boosted)

An interpretable binary classifier. An LLM writes natural-language heuristics
(policies), [TypeSafe Jev](https://docs.typesafe.ai) applies each heuristic to
each sample and returns a probability, and an L1 logistic regression learns a
weight per heuristic.

By default (`mode="one_shot"`) the LLM writes all heuristics in one call and the
regression picks the weights. The rest of this section describes the
`mode="boost"` alternative: rules are found by **boosting**, where every round targets the samples the current
model gets wrong, and a new rule is kept only if it lowers the error on data the
LLM has never seen.

## How it works

```
training data ──┬── P, show pool (20%):  the only rows whose labels the LLM sees
                └── V, validation pool (80%): never shown; every decision is made here

round 0     sample 20 YES + 20 NO rows from P -> the LLM writes seed rules
round 1..R
  1. fit L1 logistic regression on the accepted rules -> out-of-fold P(YES)
  2. residual g = y - p; take the P rows of one class the model gets most wrong:
     missed YES rows on odd rounds, missed NO rows on even rounds, plus
     correctly handled rows of the same class as contrast
  3. the LLM sees those rows, the accepted heuristics (with weights) and those
     already tried without success, and proposes new heuristics
  4. Jev scores each new heuristic on every row through jev_template, e.g.
     "Investor heuristic (guidance, not a strict rule): {policy}. Considering
     this heuristic along with the founder's full profile, will this founder
     be successful?"
  5. filter out heuristics whose scores barely vary, near-duplicates of an
     accepted heuristic, and those that fit P but not V
  6. try each survivor on its own: accept it only if the mean per-row
     log-loss improvement on V exceeds one standard error
stop        max_policy_length rules accepted, or 5 consecutive rounds each lower
            V log-loss by less than rel_epsilon (0.1%, relative)
finish      choose the C with the lowest V log-loss, then the decision threshold
            on V's pooled out-of-fold probabilities
predict     ask Jev only the accepted rules with non-zero weight; mean P(YES) of
            the 15 fold models >= threshold -> YES
```

- **Each rule is tested on its own**, so a useful rule is not diluted by
  weaker rules proposed in the same batch.
- **Nothing is forgotten**: accepted rules are never removed (L1 shrinks any
  that later rules make redundant), and rejected rules are listed in later
  prompts so the LLM does not propose them again.
- **Features are probabilities**: Jev's `noul` for the templated question,
  not 0/1.
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
| `max_policy_length` | 100 | Cap on accepted rules (<= 100); boosting usually stops earlier |
| `gen_temperature` | 1.0 | Sampling temperature of the rule-writing LLM |
| `random_state` | 0 | P/V split, example selection and CV folds. **Does not control the LLM** |
| `jev_template` | generic | How each heuristic is put to Jev; must contain `{policy}`, may contain `{task}` |
| `mode` | `one_shot` | `one_shot` asks the LLM once for `max_policy_length` heuristics (shown 50 YES + 50 NO show-pool rows, `BoostConfig.one_shot_examples_per_class`), scores them all and lets L1 choose weights (no acceptance test, no rounds); `boost` instead grows the rule set round by round from the model's mistakes |
| `weight_config` | `WeightConfig()` | `beta` (threshold only), `Cs`, `cv_folds`, `cv_repeats`, `one_se_rule`, `class_weight_balanced` |

`BoostConfig` holds the internal boosting constants (show-pool share, rules per
round, `min_spread`, `max_redundancy`, `accept_z`, `rel_epsilon`, `patience`).
They rarely need changing.

## Outputs

`save()` writes to `save_path`:

- `model.json`: accepted and rejected rules, rules with non-zero weight,
  threshold, pinned Jev version, metrics, per-round log
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
think-reason-learn runs) and evaluates on all private rows, writing heuristics
with `deepseek-v4-pro` and scoring them with an investor-heuristic Jev
template. Settings such as the generation model and the template live as
constants at the top of the script. Re-running
resumes an interrupted run; `--name` keeps separate runs apart, and `--mode boost` runs the
boosting variant (default `one_shot`).

Results go to `experiments/vcbench/runs/<name>/` (gitignored): `report.md`,
`model.json`, `predictions.csv`, `metrics.json` (validation and test metrics,
Jev cost, git commit) and `run.log`.

## How policies are written

The generation LLM writes **investor-style heuristics**: one short sentence an
experienced investor would use to judge a case, drawn from domain knowledge as
well as the labelled samples it is shown (e.g. "Founders who previously built
and sold a company are more likely to succeed"). Heuristics should be general,
focused on one signal, and must not quote samples or name specific people,
companies or numbers.

Jev does not see a system prompt. Each request carries one sample as `state`
and one question per heuristic, rendered with `jev_template` (must contain
`{policy}`, may contain `{task}`). The same template is used for training and
prediction, and it is part of the Jev answer cache key and the checkpoint
fingerprint.

**Reading the weights.** Every question asks about the outcome, so all
features share a component ("how promising is this case overall"). The
regression often gives correlated heuristics large weights of opposite sign to
isolate what differs between them: a heuristic that is positively related to
success on its own can end with a negative weight. Weights are conditional
effects, not the direction stated in the heuristic's text. Setting
`jev_template="{policy}"` with condition-style policies avoids this at the
cost of weaker features.

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
boosting finds all signal rules while rejecting noise rules, that rejected
rules are shown to the LLM, that V rows never appear in a prompt, both
stopping conditions, that even rounds show missed NO rows, checkpoint resume, save/load, retry behaviour, request throttling and the
OpenAI/DeepSeek fallbacks.

## Layout

```
policy_induction/
  model.py       PolicyInduction: boosting loop, final fit, predict, save/load
  weights.py     cross-validation, choosing C, threshold
  scorer.py      Jev scoring, version pinning, SQLite cache, rate limit
  generator.py   rule-writing LLMs (OpenAI / DeepSeek / Gemini)
  prompts.py     generation prompts and the default Jev template
  config.py      WeightConfig, BoostConfig
experiments/
  vcbench/run_vcbench.py
tests/
pyproject.toml   dependencies
uv.lock          pinned versions
```
