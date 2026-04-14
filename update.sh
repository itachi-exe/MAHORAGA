#!/usr/bin/env bash
# MAHORAGA update script
# Usage: ./update.sh
# Stops the service, applies your changes, restarts cleanly.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[update] Stopping MAHORAGA service..."
systemctl --user stop mahoraga 2>/dev/null || true

echo "[update] Waiting for clean shutdown..."
sleep 3

# Optional: pull latest git changes if you use git
# git -C "$SCRIPT_DIR" pull

echo "[update] Starting MAHORAGA service..."
systemctl --user start mahoraga

echo "[update] Done. Status:"
systemctl --user status mahoraga --no-pager -l
