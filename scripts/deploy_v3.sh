#!/bin/bash
HOST="serni13678@34.185.182.219"
SSH_KEY="/Users/macuser/.ssh/google_compute_engine"
OPTS="-o StrictHostKeyChecking=no"

echo "========================================"
echo "      Deploying V3 System to VM         "
echo "========================================"

echo "-> Syncing Codebase (stavki/, scripts/)..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' -e "ssh -i $SSH_KEY $OPTS" stavki/ $HOST:~/stavki_v2/stavki/
rsync -avz --exclude='__pycache__' -e "ssh -i $SSH_KEY $OPTS" scripts/ $HOST:~/stavki_v2/scripts/

echo "-> Syncing Models (V3 DeepInteraction & Weights)..."
rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    models/deep_interaction_v3.pth \
    models/league_weights.json \
    $HOST:~/stavki_v2/models/

echo "-> Restarting Bot Service..."
ssh -i $SSH_KEY $OPTS $HOST "sudo systemctl restart stavki_bot.service && systemctl status stavki_bot.service --no-pager"

echo "========================================"
echo "          Deployment Complete           "
echo "========================================"
