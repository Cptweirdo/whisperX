"""Shared fixtures/setup for the native-module binding tests.

Windows: the ``whisperx_core`` pyd is imported straight out of the CMake build
dir (``PYTHONPATH=<build>``), and its dependent DLLs (sherpa-onnx-c-api, ONNX
Runtime, ffmpeg, SQLiteCpp) sit next to it and in ``<build>/bin``. Python 3.8+
no longer consults PATH when resolving extension-module dependencies, so
register those directories explicitly — before the module-level
``pytest.importorskip("whisperx_core")`` in the test files fires.
"""
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    for entry in list(sys.path):
        d = Path(entry) if entry else Path.cwd()
        try:
            has_pyd = d.is_dir() and any(d.glob("whisperx_core*.pyd"))
        except OSError:
            continue
        if not has_pyd:
            continue
        for sub in (d, d / "bin"):
            if sub.is_dir():
                os.add_dll_directory(str(sub))
