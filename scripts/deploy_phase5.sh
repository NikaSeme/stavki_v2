#!/bin/bash
# Phase 5 Deployment: Full code + models + configs + data
# Deploys ALL Phase 4+5 fixes, retrained models, and optimized weights

HOST="serni13678@34.185.182.219"
SSH_KEY="/Users/macuser/.ssh/google_compute_engine"
OPTS="-o StrictHostKeyChecking=no"

echo "=========================================="
echo "   Phase 5 Full Deployment to VM          "
echo "=========================================="

# 1. Sync all Python code (stavki/ package + scripts/)
echo ""
echo "📦 [1/6] Syncing codebase (stavki/, scripts/)..."
rsync -avz --exclude='__pycache__' --exclude='*.pyc' -e "ssh -i $SSH_KEY $OPTS" \
    stavki/ $HOST:~/stavki_v2/stavki/

rsync -avz --exclude='__pycache__' --exclude='*.pyc' -e "ssh -i $SSH_KEY $OPTS" \
    scripts/ $HOST:~/stavki_v2/scripts/

# 2. Sync all retrained models (6 models)
echo ""
echo "🧠 [2/6] Deploying retrained models..."
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

# 3. Sync deep interaction model if exists
if [ -f "models/deep_interaction_v3.pth" ]; then
    echo "🔗 [2b] Syncing DeepInteraction V3 model..."
    rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
        models/deep_interaction_v3.pth \
        $HOST:~/stavki_v2/models/
fi

# 4. Sync model metadata
echo ""
echo "📋 [3/6] Syncing model metadata..."
rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    models/ensemble_weights.json \
    models/feature_columns.json \
    models/training_meta.json \
    $HOST:~/stavki_v2/models/ 2>/dev/null || true

# 5. Sync optimized per-league weights config
echo ""
echo "⚙️  [4/6] Deploying optimized league weights..."
rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    stavki/config/leagues.json \
    $HOST:~/stavki_v2/stavki/config/

# 6. Sync data mappings (team name normalization CSVs)
echo ""
echo "🗺️  [5/6] Syncing data mappings..."
ssh -i $SSH_KEY $OPTS $HOST "mkdir -p ~/stavki_v2/data/mapping/ ~/stavki_v2/data/processed/players/ ~/stavki_v2/data/processed/teams/"

rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    data/mapping/ \
    $HOST:~/stavki_v2/data/mapping/

rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    data/processed/players/ \
    $HOST:~/stavki_v2/data/processed/players/ 2>/dev/null || true

rsync -avz -e "ssh -i $SSH_KEY $OPTS" \
    data/processed/teams/ \
    $HOST:~/stavki_v2/data/processed/teams/ 2>/dev/null || true

# 7. Restart bot service
echo ""
echo "🔄 [6/6] Restarting specific bot service as root..."

# Run a remote script as root to copy the files to macuser, set permissions, and restart the bot.
ssh -i $SSH_KEY $OPTS $HOST "sudo cp -r ~/stavki_v2/* /home/macuser/stavki_v2/ && sudo chown -R macuser:macuser /home/macuser/stavki_v2/ && sudo systemctl restart stavki_bot.service && sleep 2 && systemctl status stavki_bot.service --no-pager"

echo "=========================================="
echo "       ✅ Phase 5 Deployment Complete     "
echo "       ✅ Bot Restarted (macuser workspace)"
echo "=========================================="
echo ""
echo "Changes deployed:"
echo "  • 11 modified Python files (Phase 4+5 fixes)"  
echo "  • 6 retrained models (3-way calibration)"
echo "  • Per-league optimized weights"
echo "  • Updated data mappings"
