#!/usr/bin/env bash
# ============================================
# STAVKI Server Deployment Script
# ============================================
# Usage: bash deploy/deploy.sh
#
# Run this on the server after cloning the repo.
# Prerequisites: Python 3.9+, pip, systemd
# ============================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="stavki"
CURRENT_USER="$(whoami)"

echo "============================================"
echo "  STAVKI Deployment"
echo "  User: $CURRENT_USER"
echo "  Dir:  $PROJECT_DIR"
echo "============================================"

# 1. Check Python version
echo ""
echo "[1/6] Checking Python..."
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# 2. Install dependencies
echo ""
echo "[2/6] Installing dependencies..."
cd "$PROJECT_DIR"
pip3 install --user -e ".[all]" 2>&1 | tail -5

# 3. Create .env if missing
echo ""
echo "[3/6] Checking environment..."
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo ">>> Created .env from template — EDIT IT NOW with your API keys:"
    echo "    nano $PROJECT_DIR/.env"
    echo ""
    read -p "Press Enter after editing .env..."
fi

# 4. Create required directories
echo ""
echo "[4/6] Creating directories..."
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/outputs/odds"
mkdir -p "$PROJECT_DIR/artifacts"
mkdir -p "$PROJECT_DIR/logs"

# 5. Install systemd service
echo ""
echo "[5/6] Installing systemd service..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}@.service"

# Replace %i with actual user in the template
sudo cp "$SCRIPT_DIR/stavki.service" "$SERVICE_FILE"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}@${CURRENT_USER}"
echo "Enabled ${SERVICE_NAME}@${CURRENT_USER}"

# 6. Start service
echo ""
echo "[6/6] Starting service..."
sudo systemctl restart "${SERVICE_NAME}@${CURRENT_USER}"
sleep 2

# Check status
if sudo systemctl is-active --quiet "${SERVICE_NAME}@${CURRENT_USER}"; then
    echo ""
    echo "============================================"
    echo "  ✅ STAVKI is running!"
    echo ""
    echo "  Health: curl http://localhost:8080/health"
    echo "  Logs:   journalctl -u ${SERVICE_NAME}@${CURRENT_USER} -f"
    echo "  Stop:   sudo systemctl stop ${SERVICE_NAME}@${CURRENT_USER}"
    echo "============================================"
else
    echo ""
    echo "⚠️  Service failed to start. Check logs:"
    echo "  journalctl -u ${SERVICE_NAME}@${CURRENT_USER} --no-pager -n 20"
fi
