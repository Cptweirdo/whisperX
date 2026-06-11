#!/usr/bin/env python3
"""Convert the fp32 sherpa Whisper ONNX export to fp16 (SPEEDUP_FINDINGS item 4).

Weights + compute go fp16; graph inputs/outputs stay fp32 (``keep_io_types``) —
the C++ WhisperSherpa / sherpa-onnx feed and read float32 tensors (mel, KV
caches, logits), so the I/O contract must not change. Overflow-prone ops are
kept fp32 by the converter's default block list; extend ``--block-ops`` if the
WER gate (scripts/wer_golden.py) catches drift.

The fp16 encoder (~1.3 GB) fits a single .onnx (no external-data split like
the fp32 one), which keeps the mirror contract simple: variant "fp16" has no
extra ``files``.

Run (heavy deps ephemeral, same convention as the sibling mirror scripts)::

    uv run --no-project --with onnx --with onnxruntime \
        python golden/convert_whisper_fp16.py --src build/turbo-fp32 --out build/turbo-fp16

Then validate + publish::

    .venv/Scripts/python scripts/wer_golden.py --encoder build/turbo-fp16/... (gate)
    .venv/Scripts/python golden/mirror_whisper_onnx.py --model large-v3-turbo \
        --variant fp16 --src build/turbo-fp16 --check-provider cuda
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _convert(src: Path, dst: Path, block_ops: list[str]) -> None:
    import onnx

    # onnxruntime's float16 converter is the maintained fork of
    # onnxconverter-common's; fall back for environments without ORT.
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError:
        from onnxconverter_common.float16 import convert_float_to_float16

    # Models past protobuf's 2 GB limit (the fp32 turbo encoder is 2.6 GB with
    # its external .weights loaded) silently come back EMPTY from the
    # converter's in-memory shape-inference round-trip. Run shape inference
    # file-based instead (handles >2 GB), then convert with it disabled. The
    # inferred copy must live next to src so the external-data refs resolve.
    inferred = src.with_name(src.stem + ".shapeinf.tmp.onnx")
    print(f"shape-inferring {src} (file-based) ...")
    onnx.shape_inference.infer_shapes_path(str(src), str(inferred))
    try:
        print(f"loading {inferred.name} ...")
        model = onnx.load(str(inferred))  # .weights resolves next to it
    finally:
        inferred.unlink(missing_ok=True)

    kwargs = {"keep_io_types": True, "disable_shape_infer": True}
    if block_ops:
        kwargs["op_block_list"] = block_ops
    print(f"converting (keep_io_types=True"
          + (f", op_block_list={block_ops}" if block_ops else "") + ") ...")
    model = convert_float_to_float16(model, **kwargs)
    if not model.graph.node or not model.graph.initializer:
        raise SystemExit(f"conversion produced an empty graph for {src.name} — "
                         f"protobuf 2 GB truncation?")
    print(f"saving {dst} ...")
    onnx.save(model, str(dst))
    size_mb = dst.stat().st_size / (1 << 20)
    print(f"  {dst.name}: {size_mb:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="dir with the fp32 export (plain *-encoder.onnx [+ "
                         ".weights], *-decoder.onnx, *-tokens.txt)")
    ap.add_argument("--out", type=Path, required=True, help="output dir")
    ap.add_argument("--block-ops", nargs="*", default=[],
                    help="extra op types to keep fp32 (escalation lever if the "
                         "WER gate fails; default converter block list applies "
                         "either way)")
    args = ap.parse_args()

    def fp32_of(needle: str) -> Path:
        cand = [p for p in args.src.iterdir() if p.suffix == ".onnx"
                and needle in p.name and "int8" not in p.name
                and "fp16" not in p.name]
        if not cand:
            raise SystemExit(f"no fp32 {needle} .onnx in {args.src}")
        return sorted(cand, key=lambda p: len(p.name))[0]

    enc, dec = fp32_of("encoder"), fp32_of("decoder")
    args.out.mkdir(parents=True, exist_ok=True)
    for src in (enc, dec):
        dst = args.out / src.name.replace(".onnx", ".fp16.onnx")
        _convert(src, dst, args.block_ops)
    for tok in args.src.glob("*tokens*.txt"):
        shutil.copyfile(tok, args.out / tok.name)
        print(f"  copied {tok.name}")
    print("done.")


if __name__ == "__main__":
    main()
