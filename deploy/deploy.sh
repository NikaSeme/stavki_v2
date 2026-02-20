#!/bin/bash
# Deployment Script for STAVKI v2
# Usage: ./deploy/deploy.sh <user> <host> <key_path>

set -e

USER=$1
HOST=$2
KEY_PATH=$3

if [ -z "$USER" ] || [ -z "$HOST" ]; then
    echo "Usage: ./deploy/deploy.sh <user> <host> [key_path]"
    exit 1
fi

SSH_CMD="ssh"
SCP_CMD="scp"

if [ ! -z "$KEY_PATH" ]; then
    SSH_CMD="ssh -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    SCP_CMD="scp -i $KEY_PATH -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
else
    SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    SCP_CMD="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
fi

REMOTE_DIR="/home/$USER/stavki_v2"
echo "🚀 Deploying to $USER@$HOST:$REMOTE_DIR"

# 1. Create remote directory
echo "📂 Creating remote directories..."
$SSH_CMD $USER@$HOST "mkdir -p $REMOTE_DIR/models $REMOTE_DIR/deploy $REMOTE_DIR/logs $REMOTE_DIR/data"

# 2. Sync Code (excluding heavy/unwanted files)
echo "📦 Syncing code..."
rsync -avz -e "$SSH_CMD" --exclude-from='.gitignore' \
    --exclude '/data/*' \
    --exclude '/models/*' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '.mypy_cache' \
    --exclude '*.log' \
    ./ $USER@$HOST:$REMOTE_DIR/

# 3. Transfer Models (Explicitly)
echo "🧠 Transferring models..."
$SCP_CMD models/*.pkl $USER@$HOST:$REMOTE_DIR/models/
$SCP_CMD models/*.json $USER@$HOST:$REMOTE_DIR/models/

# 4. Transfer Configs
echo "⚙️  Transferring configs..."
$SCP_CMD stavki/config/leagues.json $USER@$HOST:$REMOTE_DIR/stavki/config/

# 5. Remote Setup
echo "🔧 Setting up remote environment..."
$SSH_CMD $USER@$HOST << EOF
    cd $REMOTE_DIR
    
    # Create venv if not exists
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "Created virtual environment"
    fi
    
    # Install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Configure Service File (Dynamic Generation)
    echo "[Unit]
Description=STAVKI Telegram Bot Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/venv/bin/python3 $REMOTE_DIR/scripts/start_bot.py
Restart=always
RestartSec=10
EnvironmentFile=$REMOTE_DIR/.env
StandardOutput=append:$REMOTE_DIR/logs/bot_service.log
StandardError=append:$REMOTE_DIR/logs/bot_error.log

[Install]
WantedBy=multi-user.target" > deploy/stavki_bot.service
    
    # Setup Systemd Service
    sudo cp deploy/stavki_bot.service /etc/systemd/system/
    sudo systemctl daemon-reload
    
    echo "✅ Remote environment updated"
EOF

# 6. Restart Service
echo "🔄 Restarting service..."
$SSH_CMD $USER@$HOST "sudo systemctl restart stavki_bot && sudo systemctl status stavki_bot --no-pager"

echo "✅ Deployment Complete!"
