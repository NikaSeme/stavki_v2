import logging
from stavki.pipelines.daily import run_daily_pipeline
logging.basicConfig(level=logging.INFO)
run_daily_pipeline(["soccer_epl"])
