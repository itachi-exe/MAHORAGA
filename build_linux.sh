#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
#  MAHORAGA — Linux App Builder
#  Run from inside the MAHORAGA folder:  bash build_linux.sh
# ═══════════════════════════════════════════════════════
set -e

VENV="./build_venv"
OUT="./MAHORAGA_app"

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     MAHORAGA Linux App Builder           ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# ── Check Python ──────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ✗ python3 not found. Install Python 3.9+ first."; exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓ Python $PY_VER detected"

# ── Virtual environment ───────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
  echo "  → Creating build environment..."
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
echo "  ✓ Build environment ready"

# ── Install exact pinned versions ─────────────────────
echo "  → Installing dependencies (pinned versions)..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q pyinstaller
echo "  ✓ Dependencies installed"

# ── Build single binary ───────────────────────────────
echo "  → Building binary (this may take 2-3 min)..."
pyinstaller MAHORAGA.spec --noconfirm --clean
echo "  ✓ Binary built"

# ── Assemble clean app folder ─────────────────────────
echo "  → Assembling app package..."
rm -rf "$OUT"
mkdir -p "$OUT"

# Binary
cp dist/MAHORAGA "$OUT/MAHORAGA"
chmod +x "$OUT/MAHORAGA"

# External runtime files (client needs these alongside the binary)
cp MAHORAGA_model.pkl        "$OUT/"
cp MAHORAGA_scaler.pkl       "$OUT/"

# If features metadata exists, include it
[ -f MAHORAGA_features.json ] && cp MAHORAGA_features.json "$OUT/"

# Launcher script — client just runs this
cat > "$OUT/start.sh" << 'EOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"

echo ""
echo "  Starting MAHORAGA..."
echo "  Open http://localhost:8501 in your browser"
echo ""
./MAHORAGA
EOF
chmod +x "$OUT/start.sh"

# ── Done ──────────────────────────────────────────────
SIZE=$(du -sh "$OUT" | cut -f1)
echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║  ✅  BUILD COMPLETE                      ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""
echo "  App folder : $OUT  ($SIZE)"
echo "  To run     : cd $OUT && bash start.sh"
echo ""
echo "  Zip to send to client:"
echo "  zip -r MAHORAGA_linux.zip MAHORAGA_app/"
echo ""
