"""Pluggable translation services.

Add a provider by writing a :class:`~app.translation.base.Translator` subclass
and registering it in :data:`SERVICES`. The stored ``translation_service``
preference (in the SQLite settings table) selects one by name.
"""

from __future__ import annotations

from app.translation.base import Translator, TranslationError
from app.translation.google import GoogleTranslator

# Registry of available services, keyed by their stored preference value.
SERVICES: dict[str, type[Translator]] = {
    GoogleTranslator.name: GoogleTranslator,
}
DEFAULT_SERVICE = GoogleTranslator.name


def service_label(name: str) -> str:
    """Human-facing label for a service name (falls back to the raw name)."""
    cls = SERVICES.get(name)
    return cls.label if cls is not None else name


def get_translator(service: str, api_key: str) -> Translator:
    """Construct the translator for ``service`` with ``api_key``.

    Raises :class:`TranslationError` for an unknown service.
    """
    cls = SERVICES.get(service)
    if cls is None:
        raise TranslationError(f"Unknown translation service: {service!r}")
    return cls(api_key)


__all__ = [
    "Translator",
    "TranslationError",
    "GoogleTranslator",
    "SERVICES",
    "DEFAULT_SERVICE",
    "service_label",
    "get_translator",
]
