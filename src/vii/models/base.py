"""Base model abstractions and safety exceptions for VII providers."""

from __future__ import annotations


class SafetyAcknowledgementRequired(RuntimeError):
    """Raised when a real I2V API is requested without explicit safety acknowledgement."""

