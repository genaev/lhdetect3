#!/usr/bin/env bash
# Install the LHDetect3 application on Linux or macOS.
#
#   cd lhdetect3/gui
#   ./install.sh            # GPU support is chosen automatically
#   ./install.sh --cpu      # CPU only
#
# Creates a virtual environment in ~/.lhdetect3/venv, installs the application
# with the ONNX Runtime build that matches the hardware, and adds the commands
# lhdetect3 (window) and lhdetect3-cli (command line) to ~/.local/bin.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${LHDETECT3_HOME:-$HOME/.lhdetect3}"
VENV="$PREFIX/venv"
FORCE_CPU=0
[[ "${1:-}" == "--cpu" ]] && FORCE_CPU=1

say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWarning:\033[0m %s\n' "$*"; }

MODEL="$HERE/lhdetect3_gui/models/lhdetect3_rfdetr_seg_small.onnx"
if [[ ! -f "$MODEL" ]] || [[ $(wc -c <"$MODEL") -lt 1000000 ]]; then
    echo "The model file is missing: $MODEL"
    echo "Download it with Git LFS:  git lfs install && git lfs pull   (in the repository folder)"
    exit 1
fi

# ---- Python 3.10-3.13
PY=""
for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
        PY="$(command -v "$c")"; break
    fi
done
if [[ -z "$PY" ]]; then
    echo "Python 3.10-3.13 is required."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Install it with Homebrew (brew install python@3.12) or from https://www.python.org/downloads/"
    else
        echo "Install it with your package manager, e.g.: sudo apt install python3.12 python3.12-venv"
    fi
    exit 1
fi
say "Python: $PY ($("$PY" --version))"

# ---- ONNX Runtime variant
ORT="onnxruntime"
ORT_EXTRA=()
OS="$(uname)"
if [[ $FORCE_CPU -eq 0 && "$OS" == "Linux" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        CUDA_MAJOR="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\).*/\1/p' | head -1)"
        if [[ "${CUDA_MAJOR:-0}" -ge 13 ]]; then
            ORT="onnxruntime-gpu[cuda,cudnn]"
        elif [[ "${CUDA_MAJOR:-0}" -ge 12 ]]; then
            ORT="onnxruntime-gpu[cuda,cudnn]<1.27"
        else
            warn "NVIDIA driver supports CUDA ${CUDA_MAJOR:-?}; CUDA 12 or newer is needed. Using the CPU."
        fi
        [[ "$ORT" != onnxruntime ]] && say "NVIDIA GPU found (driver CUDA $CUDA_MAJOR): $ORT"
    elif [[ -d /opt/rocm ]] || command -v rocminfo >/dev/null 2>&1; then
        ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null | cut -d- -f1 | cut -d. -f1,2)"
        if [[ -n "$ROCM_VER" ]]; then
            ORT="onnxruntime-migraphx"
            ORT_EXTRA=(-f "https://repo.radeon.com/rocm/manylinux/rocm-rel-$ROCM_VER/")
            say "AMD ROCm $ROCM_VER found: $ORT from repo.radeon.com"
        fi
    fi
fi
[[ "$OS" == "Darwin" ]] && say "macOS: onnxruntime with Core ML (Apple GPU)"

# ---- virtual environment and packages
say "Creating $VENV"
mkdir -p "$PREFIX"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel >/dev/null
say "Installing the application"
"$VENV/bin/pip" install "$HERE"
say "Installing $ORT"
if ! "$VENV/bin/pip" install ${ORT_EXTRA[@]+"${ORT_EXTRA[@]}"} "$ORT"; then
    warn "Could not install $ORT; installing the CPU version of ONNX Runtime instead."
    "$VENV/bin/pip" install onnxruntime
fi

# ---- launchers
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/lhdetect3" "$HOME/.local/bin/lhdetect3"
ln -sf "$VENV/bin/lhdetect3-cli" "$HOME/.local/bin/lhdetect3-cli"
ICON="$("$VENV/bin/python" -c 'import lhdetect3_gui, pathlib; print(pathlib.Path(lhdetect3_gui.__file__).parent / "resources" / "icon.png")')"
if [[ "$OS" == "Linux" ]]; then
    mkdir -p "$HOME/.local/share/applications"
    cat >"$HOME/.local/share/applications/lhdetect3.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LHDetect3
Comment=Count leaf trichomes and measure their lengths
Exec=$VENV/bin/lhdetect3
Icon=$ICON
Terminal=false
Categories=Science;Education;Graphics;
EOF
elif [[ "$OS" == "Darwin" ]]; then
    APPDIR="$HOME/Applications/LHDetect3.app"
    mkdir -p "$APPDIR/Contents/MacOS" "$APPDIR/Contents/Resources"
    cat >"$APPDIR/Contents/MacOS/LHDetect3" <<EOF
#!/bin/bash
exec "$VENV/bin/lhdetect3"
EOF
    chmod +x "$APPDIR/Contents/MacOS/LHDetect3"
    cat >"$APPDIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>LHDetect3</string>
<key>CFBundleExecutable</key><string>LHDetect3</string>
<key>CFBundleIdentifier</key><string>org.lhdetect3.app</string>
<key>CFBundleIconFile</key><string>icon</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
EOF
    if command -v sips >/dev/null && command -v iconutil >/dev/null; then
        TMP="$(mktemp -d)/icon.iconset"; mkdir -p "$TMP"
        for s in 16 32 64 128 256; do sips -z $s $s "$ICON" --out "$TMP/icon_${s}x${s}.png" >/dev/null; done
        iconutil -c icns "$TMP" -o "$APPDIR/Contents/Resources/icon.icns" || true
    fi
fi

say "Checking the installation"
"$VENV/bin/python" -m lhdetect3_gui --selftest || warn "The self-test failed; see the messages above."
echo
say "Done. Start the application with:  lhdetect3"
[[ "$OS" == "Darwin" ]] && echo "    or open ~/Applications/LHDetect3.app"
echo "    Command line:  lhdetect3-cli /path/to/images --scale 1.2"
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "Add ~/.local/bin to PATH, or run $VENV/bin/lhdetect3";; esac
