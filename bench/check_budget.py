#!/usr/bin/env python3
"""Phase-6 timing gate: check ``bench_run_job`` JSON against committed budgets.

Two checks, both stdlib-only so they ride the dep-light CI lane:

  Timing budget (default)::

      check_budget.py <stage_rtf.json> <budget.json>

  Fails if any gated stage's measured RTF exceeds its ceiling. Ceilings are
  generous (~2-4x observed CPU median) so shared-runner noise never flakes it;
  the gate catches *gross* regressions only (the AskUserQuestion decision).

  RSS tail-slope (``--rss``)::

      check_budget.py --rss <rss.json> [--max-slope KB_PER_JOB] [--skip N]

  Least-squares slope of RSS over the *tail* (jobs ``N..``, default 10 — the
  head is ORT-arena warmup ramp, expected). A flat tail ⇒ no leak/fragmentation
  ⇒ the conditional ``std::pmr`` arena stays dropped (the memory decision this
  bench exists to settle). Fails if the slope exceeds ``--max-slope``.

Prints a one-line verdict per check (for ``$GITHUB_STEP_SUMMARY``) and exits
non-zero on breach.
"""
import argparse
import json
import sys

# Stages without a budget entry are not gated (decoding / loading_align).
DEFAULT_MAX_SLOPE_KB = 64.0   # KB/job tail growth tolerated (generous vs jitter)
DEFAULT_RSS_SKIP = 10         # discard the warmup ramp before fitting


def _check_timing(stage_path: str, budget_path: str) -> bool:
    measured = json.load(open(stage_path))
    budget = json.load(open(budget_path))
    stages = measured.get("stages", {})
    ok = True
    print(f"timing budget — clip {measured.get('clip', '?')} "
          f"({measured.get('duration_s', 0):.1f}s, {measured.get('runs', '?')} runs):")
    for stage, ceiling in budget.items():
        if stage.startswith("_") or not isinstance(ceiling, (int, float)):
            continue
        got = stages.get(stage)
        if got is None:
            print(f"  ! {stage:<14} MISSING from bench output")
            ok = False
            continue
        rtf = got["rtf"]
        verdict = "OK " if rtf <= ceiling else "FAIL"
        if rtf > ceiling:
            ok = False
        print(f"  {verdict} {stage:<14} RTF {rtf:.4f}  (ceiling {ceiling:.2f})")
    return ok


def _slope(ys: list) -> float:
    """Least-squares slope of ys vs index 0..n-1 (units of y per step)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    num = sum((i - xbar) * (y - ybar) for i, y in enumerate(ys))
    den = sum((i - xbar) ** 2 for i in range(n))
    return num / den if den else 0.0


def _check_rss(rss_path: str, max_slope: float, skip: int) -> bool:
    data = json.load(open(rss_path))
    rss = data.get("rss_kb", [])
    if len(rss) <= skip + 1:
        print(f"rss tail-slope — too few samples ({len(rss)}) for skip={skip}")
        return False
    tail = rss[skip:]
    slope = _slope(tail)
    ok = slope <= max_slope
    print(f"rss tail-slope — {len(rss)} jobs (skip {skip}): "
          f"first={rss[0]} last={rss[-1]} tail_slope={slope:.1f} KB/job "
          f"(max {max_slope:.0f}) -> {'OK' if ok else 'FAIL'}")
    if ok:
        print("  flat tail ⇒ no leak/fragmentation ⇒ std::pmr arena stays dropped")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rss", metavar="RSS_JSON",
                    help="check RSS-over-N-jobs tail-slope instead of timing")
    ap.add_argument("--max-slope", type=float, default=DEFAULT_MAX_SLOPE_KB,
                    help="max tolerated RSS tail growth (KB/job)")
    ap.add_argument("--skip", type=int, default=DEFAULT_RSS_SKIP,
                    help="warmup jobs to drop before fitting the RSS tail")
    ap.add_argument("paths", nargs="*",
                    help="timing mode: <stage_rtf.json> <budget.json>")
    args = ap.parse_args()

    if args.rss:
        ok = _check_rss(args.rss, args.max_slope, args.skip)
    else:
        if len(args.paths) != 2:
            ap.error("timing mode needs <stage_rtf.json> <budget.json>")
        ok = _check_timing(args.paths[0], args.paths[1])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
