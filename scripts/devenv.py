#!/usr/bin/env python3
"""Cross-platform build-env driver for the WhisperX C++ engine + native server.

One entry point for Linux / macOS / Windows. Wraps CMakePresets.json so the
build matrix lives in one place (CMake's, not a pile of shell snippets).

    python scripts/devenv.py doctor          # check toolchain + system deps
    python scripts/devenv.py deps            # install system deps (per-OS pkg mgr)
    python scripts/devenv.py build server    # configure + build a preset
    python scripts/devenv.py test server     # ctest a preset
    python scripts/devenv.py run             # launch whisperx_server (sets lib path)

Presets: dev (dep-free fast lane) · audio · server · server-vcpkg ·
server-cuda · server-vcpkg-cuda (GPU ONNX Runtime; see docs/WINDOWS_CUDA.md).
Stdlib only — no pip install needed to bootstrap. Python 3.8+.
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OS = platform.system()  # 'Linux' | 'Darwin' | 'Windows'

# ANSI colour, disabled when not a tty or on legacy Windows consoles.
_C = sys.stdout.isatty() and OS != "Windows"
def _c(s: str, code: str) -> str: return f"\033[{code}m{s}\033[0m" if _C else s
def ok(s: str) -> str: return _c(s, "32")
def bad(s: str) -> str: return _c(s, "31")
def warn(s: str) -> str: return _c(s, "33")
def head(s: str) -> str: return _c(s, "1;36")


# --- system dependency tables ------------------------------------------------
# Build-tool commands every lane needs, plus the heavy-lane libs the audio +
# server presets link (ffmpeg/curl/libarchive + the OS secret store).
APT = [
    "cmake", "ninja-build", "pkg-config", "g++", "git", "cppcheck",
    "libavformat-dev", "libavcodec-dev", "libswresample-dev", "libavutil-dev",
    "libcurl4-openssl-dev", "libarchive-dev", "libsecret-1-dev",
]
DNF = [
    "cmake", "ninja-build", "pkgconf-pkg-config", "gcc-c++", "git", "cppcheck",
    "ffmpeg-free-devel", "libcurl-devel", "libarchive-devel", "libsecret-devel",
]
PACMAN = [
    "cmake", "ninja", "pkgconf", "gcc", "git", "cppcheck",
    "ffmpeg", "curl", "libarchive", "libsecret",
]
BREW = ["cmake", "ninja", "pkg-config", "ffmpeg", "libarchive", "cppcheck"]
# Windows: build tools via choco, heavy libs from vcpkg (vcpkg.json manifest).
CHOCO = ["cmake", "ninja", "git"]


def run(cmd: list[str], *, env: dict | None = None, cwd: Path | None = None) -> int:
    print(head("$ " + " ".join(cmd)))
    return subprocess.call(cmd, env=env, cwd=str(cwd) if cwd else None)


def have(exe: str) -> str | None:
    return shutil.which(exe)


def pkgconfig(mod: str) -> bool:
    pc = have("pkg-config")
    if not pc:
        return False
    return subprocess.call([pc, "--exists", mod],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


# --- doctor ------------------------------------------------------------------
def cmd_doctor(_args) -> int:
    print(head(f"WhisperX build-env doctor — {OS} {platform.machine()}\n"))
    miss = 0

    print(head("Toolchain"))
    tools = [("cmake", True), ("ninja", True), ("git", True)]
    if OS != "Windows":
        tools += [("pkg-config", True),
                  ("c++" if OS != "Darwin" else "clang++", True)]
    else:
        tools += [("cl", False)]  # MSVC, often only on PATH inside a VS shell
    tools += [("cppcheck", False), ("uv", False), ("python3", False)]
    for exe, required in tools:
        p = have(exe) or (have("python") if exe == "python3" else None)
        if p:
            print(f"  {ok('ok')}   {exe:<12} {p}")
        else:
            tag = bad("MISS") if required else warn("opt ")
            print(f"  {tag} {exe:<12} {'(required)' if required else '(optional)'}")
            miss += required

    cmv = subprocess.run([have("cmake") or "cmake", "--version"],
                         capture_output=True, text=True).stdout.splitlines()
    if cmv:
        print(f"  → {cmv[0]}")

    print(head("\nHeavy-lane libraries (audio + server presets)"))
    if OS == "Windows":
        vr = os.environ.get("VCPKG_ROOT")
        print(f"  {(ok('ok') if vr else warn('opt '))} VCPKG_ROOT   "
              f"{vr or '(unset — needed for server-vcpkg preset)'}")
        print("  → on Windows these resolve from vcpkg.json; run `devenv.py deps`.")
    else:
        libs = [("libavformat", "ffmpeg"), ("libavcodec", "ffmpeg"),
                ("libswresample", "ffmpeg"), ("libcurl", "curl"),
                ("libarchive", "libarchive")]
        if OS == "Linux":
            libs.append(("libsecret-1", "libsecret"))
        for mod, pkg in libs:
            if pkgconfig(mod):
                print(f"  {ok('ok')}   {mod:<14} ({pkg})")
            else:
                print(f"  {bad('MISS')} {mod:<14} ({pkg}) — run `devenv.py deps`")
                miss += 1
        if OS == "Darwin":
            print(f"  {ok('ok')}   Security.fw    (keychain, system framework)")

    # GPU/CUDA — optional, only the *-cuda presets need it. Never counts as missing.
    print(head("\nGPU / CUDA (optional — only for the *-cuda presets)"))
    nvcc = have("nvcc")
    smi = have("nvidia-smi")
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if nvcc or smi or cuda_path:
        print(f"  {ok('ok') if nvcc else warn('opt ')} nvcc         "
              f"{nvcc or '(CUDA toolkit compiler not on PATH)'}")
        print(f"  {ok('ok') if smi else warn('opt ')} nvidia-smi   "
              f"{smi or '(no NVIDIA driver tool found)'}")
        if cuda_path:
            print(f"  → CUDA_PATH={cuda_path}")
        print("  → for GPU: python scripts/devenv.py build "
              + ("server-vcpkg-cuda" if OS == "Windows" else "server-cuda")
              + " (see docs/WINDOWS_CUDA.md)")
    else:
        print(f"  {warn('opt ')} no CUDA toolkit/driver detected — needed only for the "
              "server-cuda / server-vcpkg-cuda presets (see docs/WINDOWS_CUDA.md)")

    print()
    if miss:
        print(bad(f"{miss} required item(s) missing. ") +
              "Run: " + head("python scripts/devenv.py deps"))
        return 1
    print(ok("All required build deps present. ") +
          "Try: " + head("python scripts/devenv.py build server"))
    return 0


# --- deps --------------------------------------------------------------------
def _linux_pkg_mgr() -> tuple[str, list[str], list[str]] | None:
    if have("apt-get"):
        return ("apt", ["sudo", "apt-get", "install", "-y"], APT)
    if have("dnf"):
        return ("dnf", ["sudo", "dnf", "install", "-y"], DNF)
    if have("pacman"):
        return ("pacman", ["sudo", "pacman", "-S", "--needed", "--noconfirm"], PACMAN)
    return None


def cmd_deps(args) -> int:
    if OS == "Linux":
        mgr = _linux_pkg_mgr()
        if not mgr:
            print(bad("No supported package manager (apt/dnf/pacman) found."))
            print("Install manually: " + ", ".join(APT))
            return 1
        name, base, pkgs = mgr
        print(head(f"Installing system deps via {name}…"))
        if name == "apt" and not args.no_update:
            run(["sudo", "apt-get", "update"])
        return run(base + pkgs)

    if OS == "Darwin":
        if not have("brew"):
            print(bad("Homebrew not found. ") + "Install from https://brew.sh, "
                  "then re-run. (keychain uses the system Security framework — "
                  "no extra dep.)")
            return 1
        print(head("Installing system deps via brew…"))
        return run(["brew", "install"] + BREW)

    if OS == "Windows":
        rc = 0
        if have("choco"):
            print(head("Installing build tools via choco…"))
            rc |= run(["choco", "install", "-y"] + CHOCO)
        else:
            print(warn("choco not found — install CMake/Ninja/Git manually "
                       "(or via the Visual Studio installer)."))
        vr = os.environ.get("VCPKG_ROOT")
        vcpkg = (Path(vr) / "vcpkg.exe") if vr else None
        if vcpkg and vcpkg.exists():
            print(head("Installing C/C++ libs via vcpkg (vcpkg.json manifest)…"))
            rc |= run([str(vcpkg), "install",
                       "--triplet", "x64-windows"], cwd=ROOT)
        else:
            print(warn("VCPKG_ROOT unset or vcpkg.exe missing — clone & bootstrap "
                       "vcpkg, set VCPKG_ROOT, then re-run `deps` to fetch "
                       "ffmpeg/curl/libarchive/onnxruntime."))
        return rc

    print(bad(f"Unsupported OS: {OS}"))
    return 1


# --- build / test ------------------------------------------------------------
def _check_preset(preset: str) -> str:
    valid = {"dev", "audio", "server", "server-vcpkg",
             "server-cuda", "server-vcpkg-cuda"}
    if preset not in valid:
        print(bad(f"Unknown preset '{preset}'. Valid: {', '.join(sorted(valid))}"))
        sys.exit(2)
    return preset


def cmd_build(args) -> int:
    preset = _check_preset(args.preset)
    rc = run(["cmake", "--preset", preset], cwd=ROOT)
    if rc:
        return rc
    build_cmd = ["cmake", "--build", "--preset", preset]
    if args.target:
        build_cmd += ["--target", *args.target]
    if args.jobs:
        build_cmd += ["-j", str(args.jobs)]
    return run(build_cmd, cwd=ROOT)


def cmd_test(args) -> int:
    preset = _check_preset(args.preset)
    if preset == "server-vcpkg":
        preset = "server"  # one test preset covers the CPU server lane
    return run(["ctest", "--preset", preset], cwd=ROOT)


# --- run ---------------------------------------------------------------------
def _runtime_lib_dirs() -> list[Path]:
    """Dirs holding the shared libs the server dlopens (oatpp + onnxruntime)."""
    dirs = [BUILD / "lib", BUILD / "_deps" / "oatpp-build" / "src"]
    pat = "libonnxruntime*"
    ort = glob.glob(str(BUILD / "_deps" / "**" / pat), recursive=True)
    for f in ort:
        dirs.append(Path(f).parent)
        break
    return [d for d in dirs if d.exists()]


def cmd_run(args) -> int:
    exe = BUILD / ("whisperx_server.exe" if OS == "Windows" else "whisperx_server")
    if not exe.exists():
        print(bad(f"{exe} not found. ") + "Build first: " +
              head("python scripts/devenv.py build server"))
        return 1
    env = os.environ.copy()
    dirs = [str(d) for d in _runtime_lib_dirs()]
    if OS == "Windows":
        env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    else:
        var = "DYLD_LIBRARY_PATH" if OS == "Darwin" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + [env.get(var, "")])
    env.setdefault("WHISPERX_DATA_DIR", str(ROOT / ".devdata"))
    env.setdefault("WHISPERX_MODEL", "tiny")
    env.setdefault("WHISPERX_PORT", "8000")
    Path(env["WHISPERX_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
    print(head(f"Serving on http://127.0.0.1:{env['WHISPERX_PORT']} "
               f"(model={env['WHISPERX_MODEL']}, data={env['WHISPERX_DATA_DIR']})"))
    return run([str(exe), *args.server_args], env=env)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="devenv.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="check toolchain + system deps").set_defaults(
        func=cmd_doctor)

    d = sub.add_parser("deps", help="install system deps via the OS package mgr")
    d.add_argument("--no-update", action="store_true",
                   help="skip `apt-get update`")
    d.set_defaults(func=cmd_deps)

    b = sub.add_parser("build", help="configure + build a preset")
    b.add_argument("preset", nargs="?", default="server",
                   help="dev | audio | server | server-vcpkg (default: server)")
    b.add_argument("--target", nargs="+", help="restrict to these CMake targets")
    b.add_argument("-j", "--jobs", type=int, help="parallel build jobs")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("test", help="run ctest for a preset")
    t.add_argument("preset", nargs="?", default="server",
                   help="dev | audio | server (default: server)")
    t.set_defaults(func=cmd_test)

    r = sub.add_parser("run", help="launch whisperx_server with the lib path set")
    r.add_argument("server_args", nargs="*",
                   help="extra args forwarded to whisperx_server")
    r.set_defaults(func=cmd_run)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
