"""The abstract translation-service interface.

A :class:`Translator` turns a list of source-language texts into the same-length
list of target-language texts, order preserved, so callers can zip the results
back onto the segments they came from. Concrete providers (Google, …) subclass
this; the registry in :mod:`app.translation` selects one by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class TranslationError(RuntimeError):
    """A translation request failed (bad key, quota, network, API error)."""


class Translator(ABC):
    #: Machine id used as the stored preference value, e.g. ``"google"``.
    name: str = ""
    #: Human-facing label, e.g. ``"Google Translate"``.
    label: str = ""

    @abstractmethod
    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        """Translate ``texts`` into ``target_lang``.

        Returns a list of the same length and order as ``texts``. ``source_lang``
        is optional; providers auto-detect when it is ``None``. Raises
        :class:`TranslationError` on failure.
        """
        raise NotImplementedError
