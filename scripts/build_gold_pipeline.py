
import os
import subprocess
import logging
from pathlib import Path

# Add project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name):
    logger.info(f"🚀 Running: {script_name}...")
    script_path = PROJECT_ROOT / "scripts" / script_name
    
    result = subprocess.run(["python3", str(script_path)], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Error in {script_name}:\n{result.stderr}")
        sys.exit(1)
    else:
        out_lines = result.stdout.strip().split('\n')
        for line in out_lines[-3:]:
             logger.info(f"   {line}")
        logger.info(f"✅ Finished: {script_name}\n")

if __name__ == "__main__":
    logger.info("Starting Gold Pipeline Master Execution...")
    
    # Silver Layer
    run_script("process_matches.py")
    run_script("process_players.py")
    run_script("process_managers.py")
    run_script("process_trends.py")
    run_script("process_events.py")
    
    # Gold Layer
    run_script("build_player_features.py")
    run_script("build_team_vectors.py")
    run_script("build_momentum_features.py")
    run_script("build_context_features.py")
    
    logger.info("🎉 Pipeline Complete! Ready for Model Training.")
