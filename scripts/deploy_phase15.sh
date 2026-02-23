#!/bin/bash
HOST="serni13678@34.185.182.219"
SSH_KEY="/Users/macuser/.ssh/google_compute_engine"
OPTS="-o StrictHostKeyChecking=no"

echo "========================================"
echo "    Deploying Phase 14 & 15 to VM       "
echo "========================================"

echo "-> Syncing Codebase (stavki/, scripts/)..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' -e "ssh -i $SSH_KEY $OPTS" stavki/ $HOST:~/stavki_v2/stavki/
rsync -avz --exclude='__pycache__' -e "ssh -i $SSH_KEY $OPTS" scripts/ $HOST:~/stavki_v2/scripts/

echo "-> Syncing Retrained Models..."
rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    models/catboost.pkl \
    models/dixon_coles.pkl \
    models/LightGBM_1X2.pkl \
    models/LightGBM_BTTS.pkl \
    models/neural_multitask_weights.pth \
    models/neural_multitask_config.json \
    models/neural_multitask_preproc.joblib \
    models/goals_regressor.pkl \
    $HOST:~/stavki_v2/models/

echo "-> Restarting Bot Service on Google Cloud..."
ssh -i $SSH_KEY $OPTS $HOST "sudo systemctl restart stavki_bot.service && systemctl status stavki_bot.service --no-pager"

echo "========================================"
echo "          Deployment Complete           "
echo "========================================"
