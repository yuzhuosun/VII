"""Base model abstractions and safety controls for VII providers."""

from __future__ import annotations

from typing import Any

SAFETY_RESEARCH_ACK_FLAG = "--acknowledge-safety-research-use"
SAFETY_NOTICE_FILENAME = "SAFETY_NOTICE.md"
SAFETY_NOTICE_TEXT = """# Safety Notice

Artifacts in this directory are generated only for controlled AI safety red-teaming, vulnerability assessment, and defensive research. Do not use them to enable, promote, distribute, or optimize harmful, abusive, illegal, deceptive, or malicious media.

Keep generated artifacts access-controlled, comply with applicable laws and platform policies, and obtain any required institutional or organizational approvals before testing real systems.
"""

_OFFLINE_PROVIDER_NAMES = frozenset({"mock", "mock-i2v", "dry-run", "dry-run-i2v"})


class SafetyAcknowledgementRequired(RuntimeError):
    """Raised when a real I2V API is requested without explicit safety acknowledgement."""


def is_offline_i2v_provider(provider: Any) -> bool:
    """Return ``True`` when ``provider`` is an offline mock/dry-run backend."""

    provider_name = str(getattr(provider, "name", "")).strip().lower()
    return provider_name in _OFFLINE_PROVIDER_NAMES or provider_name.startswith(("mock-", "dry-run-"))


def require_safety_acknowledgement(provider: Any, acknowledged: bool) -> None:
    """Require an explicit acknowledgement before using non-offline I2V providers."""

    if is_offline_i2v_provider(provider) or acknowledged:
        return
    provider_name = getattr(provider, "name", provider.__class__.__name__)
    raise SafetyAcknowledgementRequired(
        f"Provider '{provider_name}' may call a real I2V API and requires "
        f"{SAFETY_RESEARCH_ACK_FLAG}. Use mock/dry-run mode for examples, or explicitly acknowledge "
        "controlled AI safety red-teaming use before calling a real provider."
    )
