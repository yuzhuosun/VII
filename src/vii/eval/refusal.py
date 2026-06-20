"""Refusal detection for model/API responses.

The helpers in this module intentionally accept loose ``dict`` inputs because
metadata emitted by image-to-video providers varies across vendors. Detection is
based on three signals:

* explicit provider status/error codes;
* moderation/safety fields returned by APIs;
* refusal phrases in model text responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


DEFAULT_REFUSAL_KEYWORDS: tuple[str, ...] = (
    "i can't assist",
    "i cannot assist",
    "i can’t assist",
    "i'm unable to",
    "i am unable to",
    "cannot comply",
    "can't comply",
    "sorry, but i can't",
    "sorry, i can't",
    "not able to help",
    "not able to provide",
    "against our safety policy",
    "violates our policy",
    "safety policy",
    "content policy",
    "blocked by policy",
    "request was rejected",
    "moderation blocked",
    "unsafe request",
)

DEFAULT_REFUSAL_STATUS_CODES: tuple[int, ...] = (400, 401, 403, 422, 429, 451)
DEFAULT_REFUSAL_STATUSES: tuple[str, ...] = (
    "blocked",
    "content_policy_violation",
    "denied",
    "failed_safety",
    "moderation_failed",
    "policy_violation",
    "refused",
    "rejected",
    "safety_blocked",
)


@dataclass(slots=True)
class RefusalResult:
    """Normalized refusal-detection output."""

    refused: bool
    refusal_reason: str | None = None
    signals: list[str] = field(default_factory=list)


def detect_refusal(
    response: Any = None,
    *,
    status_code: int | str | None = None,
    error: Any = None,
    moderation: Any = None,
    keywords: Iterable[str] = DEFAULT_REFUSAL_KEYWORDS,
    refusal_status_codes: Iterable[int] = DEFAULT_REFUSAL_STATUS_CODES,
) -> RefusalResult:
    """Detect whether a response represents an API/model refusal.

    Parameters can be passed separately or embedded in a provider metadata dict.
    The returned ``refusal_reason`` is concise and stable for aggregation.
    """

    signals: list[str] = []
    payload = _mapping_or_empty(response)
    error_payload = error if error is not None else payload.get("error")
    moderation_payload = moderation if moderation is not None else _first_present(
        payload, "moderation", "moderation_result", "safety", "safety_result"
    )

    code = _coerce_int(status_code if status_code is not None else _first_present(payload, "status_code", "code"))
    if code in set(refusal_status_codes):
        signals.append(f"status_code:{code}")

    status = str(_first_present(payload, "status", "finish_reason", "stop_reason") or "").lower()
    if status in DEFAULT_REFUSAL_STATUSES:
        signals.append(f"status:{status}")

    error_text = _flatten_text(error_payload).lower()
    if any(token in error_text for token in ("policy", "safety", "moderation", "blocked", "refused")):
        signals.append("error:policy_or_safety")

    if _moderation_refused(moderation_payload):
        signals.append("moderation:flagged")

    response_text = _flatten_text(response).lower()
    matched = [kw for kw in keywords if kw.lower() in response_text]
    if matched:
        signals.append(f"keyword:{matched[0]}")

    refused = bool(signals)
    return RefusalResult(refused=refused, refusal_reason=";".join(signals) if refused else None, signals=signals)


def _moderation_refused(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("flagged") is True or value.get("blocked") is True or value.get("refused") is True:
            return True
        status = str(value.get("status", "")).lower()
        if status in DEFAULT_REFUSAL_STATUSES:
            return True
        return any(_moderation_refused(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_moderation_refused(v) for v in value)
    return False


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_flatten_text(v) for v in value.values())
    if isinstance(value, list | tuple | set):
        return " ".join(_flatten_text(v) for v in value)
    return str(value)
