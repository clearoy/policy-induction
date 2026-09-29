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
    no_early_stop = BoostConfig(rel_epsilon=0.0)
    failing = make_model(tmp_path, gen=FakeGenerator(fail_from_call=3), boost_config=no_early_stop)
    with pytest.raises(RuntimeError, match="simulated"):
        await failing.fit(X, y)
    ckpt = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert ckpt["next_round"] == 2  # rounds 0 and 1 finished
    accepted_before = ckpt["accepted"]

    resumed = make_model(tmp_path, boost_config=no_early_stop)
    await resumed.fit(X, y)
    assert resumed.history[0]["round"] == 0 and resumed.history[2]["round"] == 2
    assert resumed.accepted[: len(accepted_before)] == accepted_before  # never removed
    assert not (tmp_path / "run" / "checkpoint.json").exists()


async def test_stops_at_max_policy_length(tmp_path):
    X, y = make_data()
    model = await make_model(tmp_path, max_policy_length=2, boost_config=BoostConfig(rel_epsilon=0.0)).fit(X, y)
    assert len(model.accepted) == 2
    assert model.metrics["stop_reason"] == "max_policy_length"


async def test_noise_rules_are_rejected_but_shown_to_llm(tmp_path):
    X, y = make_data()
    gen = FakeGenerator()
    model = await make_model(tmp_path, gen=gen).fit(X, y)
    # The seed offers "alpha" plus nine noise words; only signal words get in.
    assert all(r.split()[-1] in {"alpha", "beta", "gamma"} for r in model.accepted)
    assert any(r.split()[-1].startswith("w") for r in model.rejected)
    later = " ".join(gen.prompts[1:])
    assert "ALREADY TRIED, DID NOT HELP" in later and model.rejected[0] in later


async def test_stops_when_relative_gain_below_epsilon(tmp_path):
    X, y = make_data()
    # After the seed round this generator only offers noise words, so no
    # boosting round can lower validation log-loss by 0.3%; with patience 2
    # training stops after the second such round.
    class NoiseAfterSeed(FakeGenerator):
        async def generate(self, system, prompt, temperature):
            rules = await super().generate(system, prompt, temperature)
            if len(self.prompts) > 1:
                return [f"`text` mentions w{i}" for i in range(10, 20)]
            return rules

    model = await make_model(tmp_path, gen=NoiseAfterSeed()).fit(X, y)
    assert model.metrics["stop_reason"] == "converged"
    assert len(model.history) == 3
    assert all(h["relative_improvement"] < BoostConfig().rel_epsilon for h in model.history[1:])


async def test_even_rounds_show_missed_no_rows(tmp_path):
    X, y = make_data()
    model = await make_model(tmp_path, boost_config=BoostConfig(rel_epsilon=0.0, max_rounds=4)).fit(X, y)
    directions = [h["direction"] for h in model.history]
    assert directions[:5] == ["seed", "missed_YES", "missed_NO", "missed_YES", "missed_NO"]


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


async def test_jev_scorer_throttles_request_rate():
    import asyncio
    import time
    from types import SimpleNamespace

    from policy_induction import JevScorer

    class FakeClient:
        async def system_one(self, state, questions, model):
            answers = {k: SimpleNamespace(noul=0.5) for k in questions}
            return SimpleNamespace(
                model="jev-9.9.9", usage=SimpleNamespace(input_tokens=1), answers=answers
            )

    scorer = JevScorer(model="jev-9.9.9", concurrency=8, max_rpm=600)  # 0.1 s apart
    scorer._client = FakeClient()
    start = time.monotonic()
    await scorer.score([{"t": str(i)} for i in range(6)], ["rule"])
    assert time.monotonic() - start >= 0.45  # 6 starts need >= 5 intervals
    assert scorer.requests == 6


async def test_openai_generator_drops_temperature_when_rejected():
    import httpx
    from openai import BadRequestError
    from types import SimpleNamespace

    from policy_induction.generator import OpenAIGenerator, Rules

    calls = []

    class FakeResponses:
        async def parse(self, **kwargs):
            calls.append("temperature" in kwargs)
            if "temperature" in kwargs:
                req = httpx.Request("POST", "https://api.openai.com/v1/responses")
                raise BadRequestError(
                    "Unsupported parameter: 'temperature'",
                    response=httpx.Response(400, request=req),
                    body=None,
                )
            return SimpleNamespace(output_parsed=Rules(rules=["`x` is long"]))

    gen = OpenAIGenerator("gpt-5.6", api_key="test")
    gen._client = SimpleNamespace(responses=FakeResponses())
    assert await gen.generate("s", "p", 1.0) == ["`x` is long"]
    assert await gen.generate("s", "p", 1.0) == ["`x` is long"]
    assert calls == [True, False, False]  # learned once, then never sent again


async def test_deepseek_generator_json_mode_and_fallback(monkeypatch):
    import httpx
    from openai import BadRequestError
    from types import SimpleNamespace

    from policy_induction.generator import DeepSeekGenerator, make_generator

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        make_generator("deepseek-chat")

    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append("response_format" in kwargs)
            if "response_format" in kwargs:
                req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
                raise BadRequestError(
                    "response_format is not supported",
                    response=httpx.Response(400, request=req),
                    body=None,
                )
            text = '```json\n{"rules": ["`x` is long", "`x` is short"]}\n```'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    gen = DeepSeekGenerator("deepseek-reasoner", api_key="test")
    gen._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    assert await gen.generate("s", "p", 1.0) == ["`x` is long", "`x` is short"]
    assert await gen.generate("s", "p", 1.0) == ["`x` is long", "`x` is short"]
    assert calls == [True, False, False]


def test_rules_accept_wrapped_objects():
    from policy_induction.generator import parse_rules

    text = '{"rules": [{"rule": "`a` is long"}, {"text": "`b` is short"}, "`c` is empty"]}'
    assert parse_rules(text, "m") == ["`a` is long", "`b` is short", "`c` is empty"]


async def test_jev_scorer_retries_failed_rows(monkeypatch):
    from types import SimpleNamespace

    from policy_induction import JevScorer

    monkeypatch.setattr("policy_induction.scorer.RETRY_PASS_WAITS_S", (0, 0))
    attempts = {}

    class FlakyClient:
        async def system_one(self, state, questions, model):
            key = state["t"]
            attempts[key] = attempts.get(key, 0) + 1
            if key in {"1", "3"} and attempts[key] == 1:  # fail once, then work
                raise ConnectionError("temporary DNS failure")
            answers = {k: SimpleNamespace(noul=0.7) for k in questions}
            return SimpleNamespace(
                model="jev-9.9.9", usage=SimpleNamespace(input_tokens=1), answers=answers
            )

    scorer = JevScorer(model="jev-9.9.9", max_rpm=100_000)
    scorer._client = FlakyClient()
    F = await scorer.score([{"t": str(i)} for i in range(5)], ["rule"])
    assert not np.isnan(F).any()
    assert attempts["1"] == 2 and attempts["0"] == 1


async def test_fit_refuses_to_train_on_missing_scores(tmp_path):
    X, y = make_data()

    class HoleyScorer(FakeScorer):
        async def score(self, states, rules):
            out = await super().score(states, rules)
            if len(rules) and any("beta" in r for r in rules):
                out[7, 0] = np.nan  # one row never scored
            return out

    model = PolicyInduction(
        task_description=TASK, gen_model=FakeGenerator(), scorer=HoleyScorer(),
        save_path=tmp_path / "run",
    )
    with pytest.raises(RuntimeError, match="could not be scored"):
        await model.fit(X, y)
    assert (tmp_path / "run" / "checkpoint.json").exists()  # resumable
