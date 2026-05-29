#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# build_macos.sh — Build DC Twin as a macOS .app bundle
#
# Usage:
#   cd data_center_twin
#   bash build/build_macos.sh
#
# Output:
#   dist/DC Twin.app   — drag to /Applications to install
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."   # always run from project root

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  DC Twin Desktop — macOS Build"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Install / upgrade build tools ────────────────────────────────────────
echo ""
echo "▶  Installing build dependencies…"
pip install --quiet --upgrade pyinstaller pywebview

# ── 2. Clean previous build artefacts ───────────────────────────────────────
echo "▶  Cleaning previous build…"
rm -rf build/__pycache__ dist/dc-twin "dist/DC Twin.app"

# ── 3. Run PyInstaller ───────────────────────────────────────────────────────
echo "▶  Running PyInstaller (this takes 2–5 minutes)…"
pyinstaller build/dc_twin.spec --clean --noconfirm

# ── 4. Verify output ────────────────────────────────────────────────────────
APP="dist/DC Twin.app"
if [ -d "$APP" ]; then
    SIZE=$(du -sh "$APP" | cut -f1)
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅  Build successful!"
    echo "      App:  $APP"
    echo "      Size: $SIZE"
    echo ""
    echo "  To install:"
    echo "    cp -r \"dist/DC Twin.app\" /Applications/"
    echo ""
    echo "  To run directly:"
    echo "    open \"dist/DC Twin.app\""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌  Build failed — dist/DC Twin.app not found"
    exit 1
fi
