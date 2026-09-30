"""Offline stand-ins for the generation LLM and Jev.

Synthetic task: each row has a `text` field made of words. Rules have the form
"`text` mentions <word>", and the fake scorer answers them by checking whether
the word is present. The fake generator behaves like a (very literal) LLM: in
boosting rounds it proposes the words most over-represented in the missed rows
relative to the contrast rows, so the boosting logic is exercised for real.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

RULE_RE = re.compile(r"`text` mentions (\w+)")
SIGNAL_POS = ["alpha", "beta"]
SIGNAL_NEG = ["gamma"]
NOISE = [f"w{i}" for i in range(20)]
VOCAB = SIGNAL_POS + SIGNAL_NEG + NOISE


def make_data(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, List[str]]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i in range(n):
        words = [w for w in VOCAB if rng.random() < 0.3]
        logit = -0.6 + 1.8 * sum(w in words for w in SIGNAL_POS) - 1.8 * ("gamma" in words)
        y = rng.random() < 1 / (1 + np.exp(-logit))
        rows.append({"row_id": f"id{i:05d}", "text": " ".join(words)})
        labels.append("YES" if y else "NO")
    return pd.DataFrame(rows), labels


class FakeGenerator:
    model = "fake-gen"

    def __init__(self, fail_from_call: int | None = None) -> None:
        self.prompts: List[str] = []
        self.fail_from_call = fail_from_call

    async def generate(self, system: str, prompt: str, temperature: float) -> List[str]:
        self.prompts.append(prompt)
        if self.fail_from_call is not None and len(self.prompts) >= self.fail_from_call:
            raise RuntimeError("simulated generation failure")
        n = 10
        if "Labelled examples" in prompt:  # seed round: noisy guesses incl. one real signal
            words = ["alpha"] + NOISE[:n - 1]
        else:
            hard = _section(prompt, "cases wrong:", "cases right:")
            contrast = _section(prompt, "cases right:", None)
            current = set(RULE_RE.findall(_section(prompt, "Current heuristics", "The model gets")))
            hc, cc = _words(hard), _words(contrast)
            diff = {w: abs(hc[w] - cc[w]) for w in VOCAB if w not in current}
            words = sorted(diff, key=lambda w: -diff[w])[:n]
        return [f"`text` mentions {w}" for w in words]


class FakeScorer:
    version = "jev-0.0.0"

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, states: Sequence[Dict[str, Any]], rules: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(states), len(rules)))
        for j, rule in enumerate(rules):
            m = RULE_RE.search(rule)
            word = m.group(1) if m else ""
            for i, s in enumerate(states):
                out[i, j] = 0.95 if word in str(s["text"]).split() else 0.05
        return out


def _section(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    return text[i : j if j >= 0 else None]


def _words(block: str) -> Counter:
    texts = re.findall(r'"text": "([^"]*)"', block)
    return Counter(w for t in texts for w in t.split())
