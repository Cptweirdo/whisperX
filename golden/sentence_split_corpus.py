#!/usr/bin/env python3
"""Generate the sentence-splitter test baseline (Phase 3A).

The native splitter (core/text/sentence_split.cpp) replaces nltk punkt in the
aligner. To pin its behaviour we build a broad English + Russian corpus — clean
multi-sentence text, abbreviations, numbers, quotes, initials, lowercase/ASR-
shaped text, edge cases, and *garbled* variants — then record, per item:

  * ``punkt``        — nltk PunktSentenceTokenizer.span_tokenize (the independent
                       reference the splitter is judged against),
  * ``cpp``          — whisperx_core.sentence_spans output (our splitter),
  * ``matches_punkt``— whether the two agree exactly.

The committed JSON (bindings/test/sentence_split_baseline.json) is consumed by
``bindings/test/test_align_parity.py``: it re-runs the splitter and asserts the
contract invariants hold everywhere, the spans reproduce ``cpp`` (regression pin),
and the agreement pattern with punkt is unchanged. Divergences from punkt are
*expected and accepted* (rule-based ≠ trained punkt — see the brief); this records
exactly where, so any change is reviewed.

Run::

    PYTHONPATH=build uv run --no-project --with nltk --with numpy \
        python golden/sentence_split_corpus.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bindings" / "test" / "sentence_split_baseline.json"

# --- the corpus: (lang, text), grouped by what each stresses -----------------
EN = [
    # clean multi-sentence
    "The sun set over the hills. The valley grew dark. Birds fell silent.",
    "I went home. She stayed.",
    "What time is it? It is noon. Let us go!",
    "Wait... did you hear that? No? Strange.",
    "He shouted, \"Stop right there!\" Then he ran.",
    "She asked (quietly). He did not answer.",
    # abbreviations — should NOT split
    "Mr. Smith met Dr. Jones at the lab. They talked.",
    "We met at 5 p.m. and left by 7. It was short.",
    "The U.S. economy grew. Markets rose.",
    "See e.g. the appendix. Also cf. chapter 3.",
    "Visit No. 5 Baker St. today. Ring twice.",
    "J. R. R. Tolkien wrote books. Many books.",
    "It cost $3.50 total. Cheap enough.",
    "Version 1.2.3 shipped. Then 1.2.4 followed.",
    "The meeting is on 01.02.2026 sharp. Be early.",
    "Prof. Adams and Sen. Lee spoke. Loudly.",
    # lowercase / ASR-shaped (no caps) — expect single span
    "i like to use a forward approach when resolving an issue with someone",
    "how high do i know you or you can also ask is there something i can help",
    "well glasses how long have you needed a zen what do you need a zen for",
    # no terminal punctuation
    "this is a thought without an ending",
    "first sentence here. and a trailing fragment",
    # quotes / brackets
    "\"Run!\" she cried. We ran.",
    "He said: 'maybe later.' nobody moved.",
    # edge cases
    "",
    "   ",
    "x",
    "Hi.",
    "...",
    "?!",
    "One.",
    "a.b.c.d",
    "3.14 is pi. 2.72 is e.",
    "email me at a.b@c.com please. thanks",
    "go to example.com/path now. ok",
    "Tab\tseparated. And newline\nhere. Done.",
    "Multiple   spaces   between.   Words   here.",
    "A very long run on sentence that simply keeps going and going without any "
    "terminal punctuation at all so it should stay as one single span no matter how "
    "many words appear inside of it here we go",
]

RU = [
    # clean multi-sentence Cyrillic
    "Солнце село за холмы. Долина потемнела. Птицы умолкли.",
    "Я пошёл домой. Она осталась.",
    "Который час? Полдень. Идём!",
    "Подожди... ты слышал это? Нет? Странно.",
    "Он крикнул: «Стой!» Потом убежал.",
    # abbreviations
    "Я живу в г. Москве. Тут красиво.",
    "Это т.е. очень важно. Запомни.",
    "Купи хлеб, молоко и т.д. Не забудь.",
    "См. рис. 5 ниже. Там схема.",
    # numbers
    "Версия 1.2.3 вышла. Потом 1.2.4.",
    "Из всех социальных сетей я предпочитаю 3-ю. Просто так.",
    "Я стою где-то часов 5-6. Какие качества ценю в людях?",
    # lowercase / ASR-shaped
    "я скорее тех кому вечно подна потому что я даже летом хожу в кофты",
    "какие изменения в природе вы замечаете с приходом весны",
    # mixed script
    "Купил iPhone вчера. Дорого. Но круто.",
    # edge
    "Привет.",
    "Я.",
    "Однозначно. Я определённо исполнитель.",
]

DE = [
    # clean multi-sentence
    "Die Sonne ging unter. Das Tal wurde dunkel. Vögel verstummten.",
    "Ich ging nach Hause. Sie blieb.",
    "Wie geht es dir? Mir gut. Gehen wir!",
    "Warte... hast du das gehört? Nein? Seltsam.",
    "Er rief: „Halt!“ Dann rannte er.",
    # accented capitals as starters
    "Schön war es. Ärgerlich blieb der Rest. Übrigens, gut gemacht.",
    # abbreviations — should NOT split
    "Dr. Müller traf Prof. Bauer im Labor. Sie sprachen.",
    "Das war z.B. wichtig. Merk dir das.",
    "Kauf Brot, Milch usw. Vergiss es nicht.",
    "Siehe Nr. 5 unten. Dort ist die Skizze.",
    "Es ist d.h. fertig. Endlich.",
    "Wir trafen uns in St. Gallen. Es war schön.",
    "Das kostet ca. 30 Euro. Günstig genug.",
    # German ordinals: "1." = 1st — should NOT split
    "Am 1. Januar beginnt es. Pünktlich um acht.",
    "Der 3. und 4. Platz reichen. Gut so.",
    "Sie kam am 2. Mai an. Wir warteten.",
    # year-final counter-case: 4-digit -> DOES split
    "Es war 1990. Danach kam Ruhe.",
    "Version 1.2.3 erschien. Dann 1.2.4.",
    # lowercase / ASR-shaped — expect single span
    "ich glaube wir sollten das jetzt einfach mal in ruhe besprechen",
    "wie lange brauchst du denn noch für diese eine kleine aufgabe",
    # no terminal punctuation
    "ein gedanke ganz ohne ein richtiges ende",
    # edge
    "Hallo.",
    "Ja.",
]

FR = [
    # clean multi-sentence
    "Le soleil se couchait. La vallée s'assombrit. Les oiseaux se turent.",
    "Je suis rentré. Elle est restée.",
    "Quelle heure est-il ? Il est midi. Allons-y !",
    "Attends... tu as entendu ça ? Non ? Étrange.",
    "Il a crié : « Arrête ! » Puis il a couru.",
    # accented capitals as starters
    "Ça suffit. Écoute-moi bien. Allons à l'École.",
    # abbreviations — should NOT split
    "M. Dupont a rencontré M. Martin. Ils ont parlé.",
    "Voir art. 5 et fig. 2 ici. La fin.",
    "Cf. chap. 3 puis etc. Voilà tout.",
    "MM. Durand et Petit sont venus. Bien.",
    # numbers
    "La version 1.2.3 est sortie. Puis 1.2.4.",
    "Il coûte 3,50 euros. Assez bon marché.",
    # lowercase / ASR-shaped — expect single span
    "je pense que nous devrions simplement en discuter calmement maintenant",
    "combien de temps te faut-il encore pour cette petite tâche",
    # no terminal punctuation
    "une pensée sans véritable fin",
    # edge
    "Bonjour.",
    "Oui.",
]


def garble(text: str, rng: random.Random) -> str:
    """Small deterministic perturbations: drop a space, lowercase a start, double
    a space, swap two adjacent chars — the kind of noise ASR/typos introduce."""
    if len(text) < 4:
        return text
    ops = rng.sample(range(4), k=rng.randint(1, 2))
    s = list(text)
    for op in ops:
        if op == 0 and " " in text:  # drop a space
            idxs = [i for i, c in enumerate(s) if c == " "]
            if idxs:
                del s[rng.choice(idxs)]
        elif op == 1 and s:  # lowercase the first letter
            s[0] = s[0].lower()
        elif op == 2 and " " in text:  # double a space
            idxs = [i for i, c in enumerate(s) if c == " "]
            if idxs:
                j = rng.choice(idxs)
                s.insert(j, " ")
        elif op == 3 and len(s) > 5:  # swap two adjacent non-space chars
            j = rng.randint(1, len(s) - 2)
            if s[j] != " " and s[j + 1] != " ":
                s[j], s[j + 1] = s[j + 1], s[j]
    return "".join(s)


def main():
    import nltk
    from nltk.data import load as nltk_load
    import whisperx_core as wc

    try:
        nltk_load("tokenizers/punkt_tab/english.pickle")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    splitters = {
        "en": nltk_load("tokenizers/punkt_tab/english.pickle"),
        "ru": nltk_load("tokenizers/punkt_tab/russian.pickle"),
        "de": nltk_load("tokenizers/punkt_tab/german.pickle"),
        "fr": nltk_load("tokenizers/punkt_tab/french.pickle"),
    }

    rng = random.Random(20260607)
    items = []
    raw = ([("en", t) for t in EN] + [("ru", t) for t in RU] +
           [("de", t) for t in DE] + [("fr", t) for t in FR])
    # add garbled variants of the non-trivial items
    for lang, t in list(raw):
        if len(t) >= 8:
            raw.append((lang, garble(t, rng)))

    n_match = 0
    for i, (lang, text) in enumerate(raw):
        punkt = [list(s) for s in splitters[lang].span_tokenize(text)]
        cpp = [list(s) for s in wc.sentence_spans(text, lang)]
        matches = punkt == cpp
        n_match += matches
        items.append({"id": i, "lang": lang, "text": text,
                      "punkt": punkt, "cpp": cpp, "matches_punkt": matches})

    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(items)} items -> {OUT}")
    print(f"agreement with punkt: {n_match}/{len(items)} "
          f"({100 * n_match / len(items):.0f}%)")
    print("\ndivergences (cpp vs punkt):")
    for it in items:
        if not it["matches_punkt"]:
            print(f"  [{it['lang']}] {it['text']!r}")
            print(f"     punkt={it['punkt']}")
            print(f"     cpp  ={it['cpp']}")


if __name__ == "__main__":
    main()
