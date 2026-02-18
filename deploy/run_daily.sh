#!/bin/bash

# Configuration
PROJECT_DIR="/home/serni13678/stavki_v2"
LOG_FILE="$PROJECT_DIR/logs/daily_run_$(date +%Y%m%d).log"
PYTHON_BIN="/usr/bin/python3"  # Adjust if using venv

# Navigate to project
cd "$PROJECT_DIR" || exit 1

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "==================================================" >> "$LOG_FILE"
echo "Starting Daily Pipeline at $(date)" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

# Run pipeline
# Run pipeline
$PYTHON_BIN -m stavki.interfaces.cli_app --verbose predict \
    --league soccer_epl \
    --league soccer_spain_la_liga \
    --league soccer_germany_bundesliga \
    --league soccer_italy_serie_a \
    --league soccer_france_ligue_one \
    --league soccer_efl_champ \
    >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

echo "Finished at $(date) with exit code $EXIT_CODE" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"

exit $EXIT_CODE
