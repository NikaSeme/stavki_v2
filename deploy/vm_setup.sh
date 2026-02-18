#!/bin/bash

# STAVKI VM Setup Script
# Run this on the VM to configure the environment and service.

set -e

echo "🚀 Starting STAVKI VM Setup..."

# 1. System Dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# 2. Project Setup
PROJECT_DIR="/home/serni13678/stavki_v2"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/outputs/logs/shadow"
mkdir -p "$PROJECT_DIR/models"

# 3. Python Requirements
echo "🐍 Installing Python dependencies..."
# Use pip to install requirements
# (Assuming we are in the project root orrequirements.txt is present)
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip3 install -r "$PROJECT_DIR/requirements.txt"
else
    echo "⚠️ requirements.txt not found! Skipping pip install (please run manually)."
fi

# 4. Service Configuration
echo "⚙️ Configuring systemd service..."

# Copy service files to /etc/systemd/system/
# (Assuming scripts are in $PROJECT_DIR/deploy/)
# Copy service files to /etc/systemd/system/
# (Assuming scripts are in $PROJECT_DIR/deploy/)
if [ -f "$PROJECT_DIR/deploy/stavki_bot.service" ]; then
    echo "📜 Installing Bot Service..."
    
    # Stop old timer if exists
    sudo systemctl stop stavki.timer || true
    sudo systemctl disable stavki.timer || true
    
    # Copy new service
    sudo cp "$PROJECT_DIR/deploy/stavki_bot.service" /etc/systemd/system/
    
    # Reload daemon
    sudo systemctl daemon-reload
    
    # Enable and start bot
    sudo systemctl enable stavki_bot.service
    sudo systemctl restart stavki_bot.service
    
    echo "✅ Bot Service installed and started!"
else
    echo "❌ Service files not found in deploy/ directory!"
fi

# 5. Permissions
chmod +x "$PROJECT_DIR/deploy/run_daily.sh"

echo "🎉 Setup Complete! Monitor logs at $PROJECT_DIR/logs/service.log"
