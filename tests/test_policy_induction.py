import json

import numpy as np
import pytest

from policy_induction import BoostConfig, PolicyInduction, WeightConfig
from policy_induction.model import _split_pools, _to_binary
from policy_induction.scorer import _is_full_version
from policy_induction.weights import choose_threshold, make_folds, select_C

from .fakes import FakeGenerator, FakeScorer, make_data

TASK = "Predict whether the text belongs to the positive class."


def make_model(tmp_path, gen=None, **kw):
    return PolicyInduction(
        task_description=TASK,
        gen_model=gen or FakeGenerator(),
        scorer=FakeScorer(),
        save_path=tmp_path / "run",
        **kw,
    )


async def test_boosting_finds_signal_rules(tmp_path):
    X, y = make_data()
    model = await make_model(tmp_path).fit(X, y)
    words = {r.split()[-1] for r in model.rules}
    # Seed only offers "alpha"; boosting must discover the other two.
    assert {"alpha", "beta", "gamma"} <= words
    assert model.metrics["val_auc"] > 0.75
    weights = dict(zip(model.rule_table()["rule"], model.rule_table()["weight"]))
    assert weights["`text` mentions gamma"] < 0 < weights["`text` mentions beta"]


async def test_validation_rows_never_shown_to_llm(tmp_path):
    X, y = make_data()
    gen = FakeGenerator()
    await make_model(tmp_path, gen=gen, random_state=3).fit(X, y)
    in_p = _split_pools(_to_binary(y), BoostConfig().show_fraction, np.random.default_rng(3))
    shown = " ".join(gen.prompts)
    leaked = [rid for rid, p in zip(X["row_id"], in_p) if not p and rid in shown]
    assert leaked == []
    assert any(rid in shown for rid, p in zip(X["row_id"], in_p) if p)


async def test_predict_and_save_load_roundtrip(tmp_path):
    X, y = make_data()
    model = await make_model(tmp_path).fit(X, y)
    p1 = await model.predict_proba(X.head(50))
    labels = await model.predict(X.head(50))
    assert set(labels) <= {"YES", "NO"}
    path = model.save(tmp_path / "saved")
    assert (path / "report.md").read_text().startswith("# PolicyInduction report")
    loaded = PolicyInduction.load(path, scorer=FakeScorer())
    assert loaded.rules == model.rules and loaded.jev_version == "jev-0.0.0"
    np.testing.assert_allclose(await loaded.predict_proba(X.head(50)), p1)


async def test_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.setattr("policy_induction.model.GEN_RETRIES", 1)
    X, y = make_data()
    failing = make_model(tmp_path, gen=FakeGenerator(fail_from_call=3))
    with pytest.raises(RuntimeError, match="simulated"):
        await failing.fit(X, y)
    ckpt = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert ckpt["next_round"] == 2  # rounds 0 and 1 finished

    resumed = make_model(tmp_path)
    await resumed.fit(X, y)
    assert resumed.history[0]["round"] == 0 and resumed.history[2]["round"] == 2
    assert not (tmp_path / "run" / "checkpoint.json").exists()


async def test_rule_cap_follows_data_size(tmp_path):
    X, y = make_data(n=600)
    y = list(y)
    # Keep only 25 positives -> data-driven cap of 2 rules.
    pos = [i for i, v in enumerate(y) if v == "YES"][25:]
    keep = [i for i in range(len(y)) if i not in set(pos)]
    X, y = X.iloc[keep].reset_index(drop=True), [y[i] for i in keep]
    model = await make_model(tmp_path, max_policy_length=30).fit(X, y)
    assert model.metrics["rule_cap"] == 2 and len(model.rules) <= 2


def test_rejects_bad_args(tmp_path):
    with pytest.raises(ValueError):
        make_model(tmp_path, max_policy_length=101)
    with pytest.raises(ValueError):
        PolicyInduction(task_description=" ", gen_model=FakeGenerator())
    with pytest.raises(ValueError):
        WeightConfig(beta=0)


def test_folds_cover_every_row_once_per_repeat():
    y = np.array([0, 1] * 50)
    in_p = np.arange(100) % 3 == 0
    repeats = make_folds(y, in_p, n_folds=5, n_repeats=2, seed=0)
    for folds in repeats:
        seen = np.concatenate([va for _, va in folds])
        assert sorted(seen) == list(range(100))


def test_one_se_rule_prefers_stronger_regularisation():
    rng = np.random.default_rng(0)
    X = rng.random((400, 5))
    y = (X[:, 0] + 0.3 * rng.random(400) > 0.65).astype(int)
    folds = make_folds(y, np.zeros(400, bool), 5, 1, 0)
    cfg = WeightConfig()
    best = select_C(X, y, folds, np.ones(400, bool), cfg, one_se=False)
    one_se = select_C(X, y, folds, np.ones(400, bool), cfg, one_se=True)
    assert one_se.C <= best.C


def test_threshold_on_separable_scores():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    t, f = choose_threshold(y, p, beta=1.0, grid=np.linspace(0.01, 0.99, 99), smoothing=5)
    assert 0.4 < t <= 0.6 and f == 1.0


def test_version_pinning_rule():
    assert _is_full_version("jev-1.13.0")
    assert not _is_full_version("jev-latest") and not _is_full_version("jev-1.13")


async def test_generation_retry_skips_permanent_errors(monkeypatch):
    from policy_induction.model import _generate_with_retry

    monkeypatch.setattr("asyncio.sleep", _no_sleep)

    class Err(Exception):
        def __init__(self, code):
            self.code = code

    class Gen:
        model = "g"

        def __init__(self, code):
            self.code, self.calls = code, 0

        async def generate(self, system, prompt, temperature):
            self.calls += 1
            raise Err(self.code)

    for code, expected_calls in [(404, 1), (503, 4), (429, 4)]:
        gen = Gen(code)
        with pytest.raises(Err):
            await _generate_with_retry(gen, "p", 1.0)
        assert gen.calls == expected_calls


async def _no_sleep(_):
    return None
