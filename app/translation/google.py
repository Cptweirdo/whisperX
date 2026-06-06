"""Google Cloud Translation API (v2) backend.

Uses the simple API-key REST endpoint (no service account / OAuth needed).
Requests are batched to respect the documented v2 limits — at most 128 strings
and ~30k UTF-8 code points per call — and results are reassembled in order.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.translation.base import Translator, TranslationError

_ENDPOINT = "https://translation.googleapis.com/language/translate/v2"
# Conservative caps below Google's hard limits (128 segments / 30k code points).
_MAX_SEGMENTS_PER_REQUEST = 100
_MAX_CHARS_PER_REQUEST = 25_000


def _batch(texts: list[str]) -> list[list[str]]:
    """Split ``texts`` into chunks within the per-request segment/char limits.

    A single text longer than the char cap still goes out alone (the API will
    reject it if it truly exceeds the hard limit, surfaced as a TranslationError).
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for t in texts:
        n = len(t)
        if cur and (
            len(cur) >= _MAX_SEGMENTS_PER_REQUEST
            or cur_chars + n > _MAX_CHARS_PER_REQUEST
        ):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(t)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


class GoogleTranslator(Translator):
    name = "google"
    label = "Google Translate"

    def __init__(self, api_key: str):
        key = (api_key or "").strip()
        if not key:
            raise TranslationError("No Google Translation API key configured.")
        self._key = key

    def translate(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        if not texts:
            return []
        out: list[str] = []
        for batch in _batch(texts):
            out.extend(self._translate_batch(batch, target_lang, source_lang))
        return out

    def _translate_batch(
        self, batch: list[str], target_lang: str, source_lang: str | None
    ) -> list[str]:
        # Build the body as repeated q= fields (the v2 API accepts q arrays).
        params: list[tuple[str, str]] = [("target", target_lang), ("format", "text")]
        if source_lang:
            params.append(("source", source_lang))
        params.extend(("q", t) for t in batch)
        body = urlencode(params).encode("utf-8")
        url = f"{_ENDPOINT}?{urlencode({'key': self._key})}"
        req = Request(url, data=body, method="POST")  # noqa: S310 - fixed https host
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
                payload = json.loads(resp.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
            raise TranslationError(
                f"Google Translation API error ({exc.code}): {detail}"
            ) from exc
        except URLError as exc:
            raise TranslationError(f"Couldn't reach Google: {exc.reason}") from exc

        translations = (payload.get("data") or {}).get("translations") or []
        if len(translations) != len(batch):
            raise TranslationError(
                "Unexpected response from Google: "
                f"{len(translations)} translations for {len(batch)} inputs."
            )
        return [t.get("translatedText", "") for t in translations]
