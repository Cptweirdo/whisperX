# Non-breaking prefixes (vendored)

Per-language abbreviation lists used by the native sentence splitter
(`core/text/sentence_split.cpp`) to suppress false sentence boundaries after
abbreviations (e.g. `Mr.`, `Dr.`, `e.g.`, `No. 5`). They replace nltk punkt in the
forced-alignment stage (Phase 3A of the C++ core migration).

## Provenance

These are the **Moses** `nonbreaking_prefixes` files, from the
[mosesdecoder](https://github.com/moses-smt/mosesdecoder) project
(`scripts/share/nonbreaking_prefixes/`), as redistributed by
[`sacremoses`](https://github.com/hplt-project/sacremoses). Unmodified.

## Format

One prefix per line; lines beginning with `#` are comments. A prefix tagged
`#NUMERIC_ONLY#` suppresses a boundary **only** when the period is followed by a
number (e.g. `No. 5` does not split, but `No. The` does).

## License

Moses (and these data files) are licensed under the **LGPL-2.1**. They are vendored
verbatim and embedded into the build as read-only string data
(`CMakeLists.txt` → `whisperx_nonbreaking_prefixes.hpp`); no Moses code is linked.
