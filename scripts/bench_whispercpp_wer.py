"""Analyse the bench-out/ products of bench_whispercpp_metal.sh.

For each config: pull the (last, i.e. warm 2nd-run) per-stage RTF from its server log
(`stage=… rtf=…`), concat the transcript segments into a hypothesis, and report WER/CER
drift against the fp16/flash0 baseline (the gate from SPEEDUP_FINDINGS.md lever 1 —
"Russian WER vs the fp16 ggml baseline"). Reuses wer_cer from the parity suite.

Usage:  uv run python3 scripts/bench_whispercpp_wer.py [bench-out-dir]
        (uv: the wer_cer module imports pytest, a dev-extra dep)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_baseline_golden import wer_cer  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench-out"
ORDER = ["fp16-flash0", "fp16-flash1", "q8_0-flash0", "q8_0-flash1",
         "q5_0-flash0", "q5_0-flash1"]
BASELINE = "fp16-flash0"
STAGE_RE = re.compile(r"stage=(\w+)\s+elapsed=([\d.]+)s\s+rtf=([\d.]+)")


def last_stage_rtfs(log: Path) -> dict[str, float]:
    """Last rtf seen per stage (the warm 2nd run overwrites the cold first)."""
    rtfs: dict[str, float] = {}
    if not log.exists():
        return rtfs
    for line in log.read_text(errors="replace").splitlines():
        m = STAGE_RE.search(line)
        if m:
            rtfs[m.group(1)] = float(m.group(3))
    return rtfs


def hypothesis(js: Path) -> str:
    if not js.exists():
        return ""
    data = json.loads(js.read_text())
    segs = data.get("segments", data) if isinstance(data, dict) else data
    return " ".join(s.get("text", "").strip() for s in segs).strip()


def main() -> int:
    base_hyp = hypothesis(OUT / f"{BASELINE}.json")
    base_t1 = last_stage_rtfs(OUT / f"{BASELINE}.log").get("transcribing")

    rows = []
    for label in ORDER:
        log, js = OUT / f"{label}.log", OUT / f"{label}.json"
        if not js.exists() and not log.exists():
            continue
        r = last_stage_rtfs(log)
        t1 = r.get("transcribing")
        e2e = sum(r.get(s, 0.0) for s in
                  ("decoding", "transcribing", "aligning", "diarizing")) or None
        hyp = hypothesis(js)
        wer = cer = None
        if base_hyp and hyp:
            wer, cer = wer_cer(base_hyp, hyp)  # drift vs fp16 baseline
        speed = (base_t1 / t1) if (base_t1 and t1) else None
        rows.append((label, t1, speed, e2e, wer, cer, len(hyp.split())))

    def fmt(x, f="{:.3f}"):
        return f.format(x) if isinstance(x, (int, float)) else "—"

    print(f"{'config':14} {'tr_RTF':>7} {'vs_fp16':>8} {'e2e_RTF':>8} "
          f"{'WERdrift':>9} {'CERdrift':>9} {'words':>6}")
    print("-" * 70)
    for label, t1, speed, e2e, wer, cer, nw in rows:
        sp = f"{speed:.2f}x" if speed else "—"
        print(f"{label:14} {fmt(t1):>7} {sp:>8} {fmt(e2e):>8} "
              f"{fmt(wer):>9} {fmt(cer):>9} {nw:>6}")
    print("\nWER/CER are DRIFT vs the fp16/flash0 hypothesis (0 = identical text).")
    print("Gate: accept the fastest config with WER drift ≲ 0.03.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
