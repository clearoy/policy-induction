"""Rule scoring with TypeSafe Jev.

Every rule is asked as a Noul question ("is this true of the sample?"); Jev
returns P(true). All rules for one sample go in one request: Jev judges each
question independently and in parallel, so batching changes neither the
answers nor, meaningfully, the latency.

Answers are cached on disk keyed by (pinned Jev version, sample, rule text).
The cache is also the resume mechanism: an interrupted run re-requests only
what is missing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Jev evaluates up to 64k tokens per request (state + all questions). Rules are
# short, so this is a generous safety bound rather than a tight one.
MAX_RULES_PER_REQUEST = 64


def state_key(state: Dict[str, Any]) -> str:
    blob = json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Scorer(Protocol):
    """Anything that returns P(rule true) for every (sample, rule) pair."""

    version: str | None

    async def score(
        self, states: Sequence[Dict[str, Any]], rules: Sequence[str]
    ) -> np.ndarray:
        """Return an array of shape (len(states), len(rules)); NaN = failed."""
        ...


class AnswerCache:
    """SQLite cache of Noul answers. Safe to share across runs."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value REAL)"
        )
        self._db.commit()

    @staticmethod
    def key(version: str, skey: str, rule: str) -> str:
        return hashlib.sha256(f"{version}\x00{skey}\x00{rule}".encode()).hexdigest()

    def get_many(self, keys: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i in range(0, len(keys), 900):  # SQLite variable limit
            chunk = keys[i : i + 900]
            marks = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT key, value FROM answers WHERE key IN ({marks})", chunk
            )
            out.update(dict(rows.fetchall()))
        return out

    def put_many(self, items: Dict[str, float]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO answers (key, value) VALUES (?, ?)", items.items()
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()


class JevScorer:
    """Scores rules with TypeSafe Jev, pinning the model version.

    Args:
        model: Model name or alias. An alias such as ``jev-latest`` is pinned
            to the concrete version that answers the first request, so the
            features a model is trained on never drift under it.
        cache_path: SQLite file for cached answers. None disables caching.
        concurrency: Requests in flight at once.
        max_rpm: Request starts per minute. Jev's documented limit is 1,200;
            staying just under it avoids a stream of 429 retries.
        api_key: Overrides ``TYPESAFE_API_KEY``.
    """

    def __init__(
        self,
        model: str = "jev-latest",
        cache_path: str | Path | None = None,
        concurrency: int = 16,
        max_rpm: int = 1100,
        api_key: str | None = None,
    ) -> None:
        self.requested_model = model
        # Only a full version ID (jev-1.13.0) is pinned up front; aliases and
        # short names (jev-latest, jev-1.13) are pinned on the first response.
        self.version: str | None = model if _is_full_version(model) else None
        self.concurrency = concurrency
        self._min_interval = 60.0 / max_rpm
        self._next_start = 0.0
        self._rate_lock = asyncio.Lock()
        self._api_key = api_key
        self._client: Any = None
        self._cache = AnswerCache(Path(cache_path)) if cache_path else None
        self.input_tokens = 0
        self.requests = 0

    def _get_client(self) -> Any:
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient

            self._client = AsyncTypeSafeClient(api_key=self._api_key)
        return self._client

    async def _ask(self, state: Dict[str, Any], rules: Sequence[str]) -> Dict[str, float]:
        """One request per chunk of rules; returns rule -> P(true)."""
        from typesafe_sdk import Noul

        client = self._get_client()
        out: Dict[str, float] = {}
        for i in range(0, len(rules), MAX_RULES_PER_REQUEST):
            chunk = list(rules[i : i + MAX_RULES_PER_REQUEST])
            questions = {f"r{j}": Noul(instructions=r) for j, r in enumerate(chunk)}
            await self._throttle()
            response = await client.system_one(
                state=state,
                questions=questions,
                model=self.version or self.requested_model,
            )
            self.requests += 1
            self.input_tokens += response.usage.input_tokens or 0
            if self.version is None:
                self.version = response.model
                logger.info("Pinned Jev version: %s", self.version)
            elif response.model != self.version:
                raise RuntimeError(
                    f"Jev answered with {response.model}, expected pinned "
                    f"{self.version}; features would no longer match the model."
                )
            for j, r in enumerate(chunk):
                out[r] = float(response.answers[f"r{j}"].noul)
        return out

    async def _throttle(self) -> None:
        """Space request starts at least ``60 / max_rpm`` seconds apart."""
        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            wait = self._next_start - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_start = max(loop.time(), self._next_start) + self._min_interval

    async def score(
        self, states: Sequence[Dict[str, Any]], rules: Sequence[str]
    ) -> np.ndarray:
        result = np.full((len(states), len(rules)), np.nan)
        if not states or not rules:
            return result
        skeys = [state_key(s) for s in states]
        rule_pos = {r: j for j, r in enumerate(rules)}

        # An alias must be resolved before cache lookups can use the version.
        start = 0
        if self.version is None:
            first = await self._ask(states[0], rules)
            for r, v in first.items():
                result[0, rule_pos[r]] = v
            self._store(skeys[0], first)
            start = 1

        todo: List[tuple[int, List[str]]] = []
        for i in range(start, len(states)):
            missing = list(rules)
            if self._cache is not None:
                keys = {r: AnswerCache.key(self.version, skeys[i], r) for r in rules}  # type: ignore[arg-type]
                hit = self._cache.get_many(list(keys.values()))
                missing = []
                for r, k in keys.items():
                    if k in hit:
                        result[i, rule_pos[r]] = hit[k]
                    else:
                        missing.append(r)
            if missing:
                todo.append((i, missing))

        if not todo:
            return result

        sem = asyncio.Semaphore(self.concurrency)
        failures = 0

        async def run(i: int, missing: List[str]) -> None:
            nonlocal failures
            async with sem:
                try:
                    answers = await self._ask(states[i], missing)
                except Exception:
                    failures += 1
                    logger.warning("Jev scoring failed for row %d", i, exc_info=True)
                    return
            for r, v in answers.items():
                result[i, rule_pos[r]] = v
            self._store(skeys[i], answers)

        tasks = [asyncio.create_task(run(i, m)) for i, m in todo]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="[JEV]"):
            await fut
        if failures:
            logger.warning("%d/%d rows failed to score.", failures, len(todo))
        return result

    def _store(self, skey: str, answers: Dict[str, float]) -> None:
        if self._cache is None or self.version is None:
            return
        self._cache.put_many(
            {AnswerCache.key(self.version, skey, r): v for r, v in answers.items()}
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._cache is not None:
            self._cache.close()


def _is_full_version(model: str) -> bool:
    return re.fullmatch(r"jev-\d+\.\d+\.\d+", model) is not None
