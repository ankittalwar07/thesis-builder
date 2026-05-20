"""Shared LLM gateway.

Every completion in the project flows through ``LLMClient.complete``. The
client:
  - resolves the model for a tier from ``config/models.yaml``,
  - walks the fallback list on rate-limit / 5xx / auth errors,
  - returns structured outputs via Instructor when ``response_model`` is given,
  - tracks tokens + cost per call to ``trace.jsonl``,
  - enables Anthropic ``cache_control`` on the system prompt for tier-quality.

No agent imports LiteLLM or Instructor directly.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import yaml
from pydantic import BaseModel

try:
    import litellm
    from litellm import completion as litellm_completion
    from litellm.exceptions import (
        AuthenticationError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )
except ImportError as e:  # pragma: no cover - import-time guard
    raise ImportError(
        "litellm is required. Install with `uv sync` or `pip install litellm`."
    ) from e

try:
    import instructor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "instructor is required. Install with `uv sync` or `pip install instructor`."
    ) from e


Tier = Literal["small", "reasoning", "quality"]

_RETRYABLE = (RateLimitError, ServiceUnavailableError, Timeout)


# --- Config loading ---------------------------------------------------------


@dataclass
class TierConfig:
    primary: str
    fallbacks: list[str] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout_s: int = 60
    enable_prompt_cache: bool = False

    @property
    def models(self) -> list[str]:
        return [self.primary, *self.fallbacks]


def _load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# --- Trace writer -----------------------------------------------------------


class TraceWriter:
    """Thread-safe JSONL writer for trace events.

    Trace path is mutable — the CLI rebinds it to outputs/<theme>/trace.jsonl
    once the theme is known. The optional ``on_event`` callback receives every
    event after it's written; the server uses it to push SSE updates.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        on_event: Any = None,
    ) -> None:
        self._lock = Lock()
        self._path: Path | None = Path(path) if path else None
        self.total_cost_usd: float = 0.0
        self.tokens_by_tier: dict[str, int] = {"small": 0, "reasoning": 0, "quality": 0}
        self.on_event = on_event  # callable(event: dict) -> None

    def set_path(self, path: str | Path) -> None:
        with self._lock:
            self._path = Path(path)
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        if event.get("kind") == "llm":
            tier = event.get("tier")
            if tier in self.tokens_by_tier:
                self.tokens_by_tier[tier] += int(
                    (event.get("prompt_tokens") or 0)
                    + (event.get("completion_tokens") or 0)
                )
            self.total_cost_usd += float(event.get("cost_usd") or 0.0)
        if self._path is not None:
            with self._lock:
                with open(self._path, "a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        cb = self.on_event
        if cb is not None:
            try:
                cb(event)
            except Exception:
                # Callbacks must never break tracing.
                pass


# --- Budget guard -----------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised when cumulative cost exceeds THESIS_MAX_COST_USD."""


# --- Client -----------------------------------------------------------------


class LLMClient:
    def __init__(
        self,
        config_path: str | Path = "config/models.yaml",
        trace: TraceWriter | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        cfg = _load_config(config_path)
        self._tiers: dict[str, TierConfig] = {
            name: TierConfig(**spec) for name, spec in cfg["tiers"].items()
        }
        self._instructor_cfg = cfg.get("instructor", {})
        self._budgets = cfg.get("budgets", {})
        self.trace = trace or TraceWriter()
        env_cap = os.environ.get("THESIS_MAX_COST_USD")
        self.max_cost_usd: float | None = (
            max_cost_usd
            if max_cost_usd is not None
            else (float(env_cap) if env_cap else None)
        )
        # Silence LiteLLM's chatty default logger; we have our own trace.
        litellm.suppress_debug_info = True

    # ---- public api --------------------------------------------------------

    @property
    def budgets(self) -> dict[str, int]:
        return self._budgets

    def tier(self, name: Tier) -> TierConfig:
        return self._tiers[name]

    def complete(
        self,
        tier: Tier,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        cache_system: bool | None = None,
    ) -> Any:
        """Run a completion. Returns either a parsed ``response_model`` instance
        or the raw assistant string when ``response_model`` is None.
        """
        self._check_budget()
        tcfg = self._tiers[tier]
        cache_on = (
            cache_system
            if cache_system is not None
            else (tier == "quality" and tcfg.enable_prompt_cache)
        )
        prepared = _apply_prompt_cache(messages) if cache_on else messages

        last_err: Exception | None = None
        for model in tcfg.models:
            t0 = time.perf_counter()
            try:
                if response_model is not None:
                    result, usage_info = self._instructor_call(
                        model=model,
                        messages=prepared,
                        response_model=response_model,
                        max_tokens=max_tokens or tcfg.max_tokens,
                        temperature=temperature if temperature is not None else tcfg.temperature,
                        timeout=tcfg.timeout_s,
                    )
                else:
                    result, usage_info = self._raw_call(
                        model=model,
                        messages=prepared,
                        max_tokens=max_tokens or tcfg.max_tokens,
                        temperature=temperature if temperature is not None else tcfg.temperature,
                        timeout=tcfg.timeout_s,
                    )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                self._emit_llm_trace(
                    tier=tier,
                    model=model,
                    usage=usage_info,
                    latency_ms=latency_ms,
                    structured=response_model is not None,
                    cached_system=cache_on,
                )
                return result
            except AuthenticationError as e:
                # Missing/invalid key for this provider — silently fall through.
                last_err = e
                continue
            except _RETRYABLE as e:
                last_err = e
                self.trace.write(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "kind": "error",
                        "tier": tier,
                        "model": model,
                        "note": f"{type(e).__name__}: {e}",
                    }
                )
                continue
            except Exception as e:  # noqa: BLE001 — bubble after exhausting fallbacks
                last_err = e
                self.trace.write(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "kind": "error",
                        "tier": tier,
                        "model": model,
                        "note": f"{type(e).__name__}: {e}",
                    }
                )
                continue
        raise RuntimeError(f"All models for tier={tier} failed. Last: {last_err}")

    # ---- internals ---------------------------------------------------------

    def _instructor_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        max_tokens: int,
        temperature: float,
        timeout: int,
    ) -> tuple[BaseModel, dict[str, Any]]:
        client = instructor.from_litellm(litellm_completion)
        max_retries = int(self._instructor_cfg.get("max_retries", 3))
        # instructor returns (obj, raw_completion) when using create_with_completion;
        # we want both so we can extract usage / cost from the raw side.
        obj, raw = client.chat.completions.create_with_completion(
            model=model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        usage = _usage_from_response(raw)
        return obj, usage

    def _raw_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        timeout: int,
    ) -> tuple[str, dict[str, Any]]:
        resp = litellm_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        text = resp.choices[0].message.content or ""
        return text, _usage_from_response(resp)

    def _emit_llm_trace(
        self,
        tier: str,
        model: str,
        usage: dict[str, Any],
        latency_ms: int,
        structured: bool,
        cached_system: bool,
    ) -> None:
        self.trace.write(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "kind": "llm",
                "tier": tier,
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cost_usd": usage.get("cost_usd"),
                "latency_ms": latency_ms,
                "note": f"structured={structured} cached_system={cached_system}",
            }
        )

    def _check_budget(self) -> None:
        if self.max_cost_usd is None:
            return
        if self.trace.total_cost_usd >= self.max_cost_usd:
            raise BudgetExceeded(
                f"Run cost ${self.trace.total_cost_usd:.4f} exceeds cap ${self.max_cost_usd:.2f}"
            )


# --- Helpers ----------------------------------------------------------------


def _apply_prompt_cache(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add Anthropic ``cache_control`` to the system message.

    LiteLLM passes ``cache_control`` through to the Anthropic API. For
    non-Anthropic providers it is silently ignored, so the same message list
    can be used everywhere.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            out.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": m["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            out.append(m)
    return out


def _usage_from_response(resp: Any) -> dict[str, Any]:
    """Extract token usage + LiteLLM-computed cost from a completion response."""
    usage_obj = getattr(resp, "usage", None)
    pt = getattr(usage_obj, "prompt_tokens", None) if usage_obj else None
    ct = getattr(usage_obj, "completion_tokens", None) if usage_obj else None
    cost: float | None = None
    try:
        cost = float(litellm.completion_cost(completion_response=resp))
    except Exception:
        # Some providers/models aren't in LiteLLM's cost table; leave None and
        # rely on token counts for the routing-ratio acceptance criterion.
        cost = None
    return {"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost}
