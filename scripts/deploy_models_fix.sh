#!/bin/bash
HOST="serni13678@34.185.182.219"
SSH_KEY="/Users/macuser/.ssh/google_compute_engine"
OPTS="-o StrictHostKeyChecking=no"

echo "========================================"
echo "      Deploying Retrained Models        "
echo "========================================"

# 1. Deploy Code Fix
echo "-> Deploying Code: CatBoost Feature Handling..."
rsync -avz -e "ssh -i $SSH_KEY $OPTS" stavki/models/catboost/catboost_model.py $HOST:~/stavki_v2/stavki/models/catboost/all

# 2. Deploy Models
echo "-> Deploying Models: Full Suite..."
# Using explicit list to avoid clutter, but globbing is fine if we exclude old stuff
# We rely on timestamp check? No, rsync updates if newer.
# But we want to ensure we send the Neural files (json/pth/joblib) correctly.

rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    models/catboost.pkl \
    models/dixon_coles.pkl \
    models/LightGBM_1X2.pkl \
    models/LightGBM_BTTS.pkl \
    models/neural_multitask_weights.pth \
    models/neural_multitask_config.json \
    models/neural_multitask_preproc.joblib \
    models/goals_regressor.pkl \
    stavki/config/leagues.json \
    $HOST:~/stavki_v2/models/

# 3. Restart Service
echo "-> Restarting Bot Service..."
ssh -i $SSH_KEY $OPTS $HOST "sudo systemctl restart stavki_bot.service && systemctl status stavki_bot.service --no-pager"

echo "========================================"
echo "          Deployment Complete           "
echo "========================================"
