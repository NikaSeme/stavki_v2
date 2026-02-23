"""
STAVKI Scheduler
================

Automated task scheduling for periodic predictions, data updates, etc.
Includes HTTP health check endpoint for monitoring.

Usage:
    scheduler = Scheduler()
    scheduler.add_job("predict", hours=1)
    scheduler.start()
"""

import logging
import time
import signal
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Job:
    """Scheduled job configuration."""
    name: str
    func: Callable
    hours: float = 1.0
    minutes: float = 0.0
    
    # State
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    
    def __post_init__(self):
        self.interval_seconds = self.hours * 3600 + self.minutes * 60
        self.next_run = datetime.now()


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check endpoint (P1)."""
    
    scheduler_ref: Optional["Scheduler"] = None
    start_time: datetime = datetime.now()
    
    def do_GET(self):
        if self.path == "/health":
            uptime = (datetime.now() - self.start_time).total_seconds()
            body = json.dumps({
                "status": "ok",
                "uptime_seconds": round(uptime),
                "uptime_human": str(timedelta(seconds=int(uptime))),
                "timestamp": datetime.now().isoformat(),
                "jobs": self.scheduler_ref.get_status() if self.scheduler_ref else {},
            }, indent=2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging to avoid noise."""
        pass


class Scheduler:
    """
    Simple task scheduler for STAVKI.
    
    Runs jobs at specified intervals with optional HTTP health endpoint.
    """
    
    def __init__(self, health_port: int = 8080):
        self.jobs: Dict[str, Job] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._health_port = health_port
        self._health_server: Optional[HTTPServer] = None
        self._start_time = datetime.now()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def add_job(
        self,
        name: str,
        func: Callable,
        hours: float = 0,
        minutes: float = 0,
        run_immediately: bool = False,
    ) -> None:
        """
        Add a job to the scheduler.
        
        Args:
            name: Unique job name
            func: Function to execute
            hours: Interval in hours
            minutes: Interval in minutes
            run_immediately: If True, run the job immediately on start
        """
        job = Job(
            name=name,
            func=func,
            hours=hours,
            minutes=minutes,
        )
        
        if not run_immediately:
            job.next_run = datetime.now() + timedelta(seconds=job.interval_seconds)
        
        self.jobs[name] = job
        logger.info(f"Added job '{name}' with interval {hours}h {minutes}m")
    
    def remove_job(self, name: str) -> bool:
        """Remove a job by name."""
        if name in self.jobs:
            del self.jobs[name]
            logger.info(f"Removed job '{name}'")
            return True
        return False
    
    def start(self, blocking: bool = True):
        """
        Start the scheduler.
        
        Args:
            blocking: If True, blocks the main thread
        """
        self._running = True
        self._start_time = datetime.now()
        logger.info("Scheduler started")
        
        # Start health check server
        self._start_health_server()
        
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopping...")
        
        # Stop health server
        if self._health_server:
            self._health_server.shutdown()
            logger.info("Health server stopped")
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
    
    def _start_health_server(self):
        """Start HTTP health check server on a daemon thread."""
        try:
            HealthHandler.scheduler_ref = self
            HealthHandler.start_time = self._start_time
            
            self._health_server = HTTPServer(("0.0.0.0", self._health_port), HealthHandler)
            health_thread = threading.Thread(
                target=self._health_server.serve_forever,
                daemon=True,
            )
            health_thread.start()
            logger.info(f"Health endpoint running on http://0.0.0.0:{self._health_port}/health")
        except OSError as e:
            logger.warning(f"Could not start health server on port {self._health_port}: {e}")
    
    def _run_loop(self):
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()
            
            for job in self.jobs.values():
                if job.next_run and now >= job.next_run:
                    self._execute_job(job)
            
            # Sleep briefly
            time.sleep(1)
    
    def _execute_job(self, job: Job):
        """Execute a single job."""
        logger.info(f"Running job '{job.name}'...")
        job.status = JobStatus.RUNNING
        job.last_run = datetime.now()
        
        try:
            job.func()
            job.status = JobStatus.SUCCESS
            job.run_count += 1
            logger.info(f"Job '{job.name}' completed successfully")
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_count += 1
            job.last_error = str(e)
            logger.error(f"Job '{job.name}' failed: {e}")
        
        # Schedule next run
        job.next_run = datetime.now() + timedelta(seconds=job.interval_seconds)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal")
        self.stop()
    
    def get_status(self) -> Dict[str, Dict]:
        """Get status of all jobs."""
        return {
            name: {
                "status": job.status.value,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "run_count": job.run_count,
                "error_count": job.error_count,
                "last_error": job.last_error,
            }
            for name, job in self.jobs.items()
        }


def create_default_scheduler() -> Scheduler:
    """Create scheduler with default STAVKI jobs."""
    scheduler = Scheduler()
    
    # Job 1: Predictions every hour
    def run_predictions():
        try:
            from stavki.pipelines import DailyPipeline
            pipeline = DailyPipeline()
            bets = pipeline.run()
            logger.info(f"Found {len(bets)} value bets")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
    
    scheduler.add_job(
        name="predictions",
        func=run_predictions,
        hours=1,
        run_immediately=True,
    )
    
    # Job 2 (P3): Weekly model retrain
    def run_retrain():
        try:
            from stavki.pipelines.training import TrainingPipeline, TrainingConfig
            from stavki.config import get_config
            from pathlib import Path
            
            config = get_config()
            data_path = Path(config.retrain_data_path)
            
            if not data_path.exists():
                logger.warning(f"Training data not found: {data_path}")
                return
            
            train_config = TrainingConfig(data_path=data_path)
            pipeline = TrainingPipeline(config=train_config)
            results = pipeline.run()
            logger.info(f"Retrain complete: {len(results.get('models', []))} models updated")
        except Exception as e:
            logger.error(f"Retrain failed: {e}")
    
    scheduler.add_job(
        name="weekly_retrain",
        func=run_retrain,
        hours=168,  # 7 days
        run_immediately=False,
    )
    
    # Job 3: Capture closing odds every 5 minutes (for CLV tracking)
    def capture_closing_odds():
        try:
            from stavki.data.collectors.closing_odds import ClosingOddsCollector
            collector = ClosingOddsCollector()
            captured = collector.capture()
            if captured:
                logger.info(f"Captured closing odds for {captured} matches")
        except Exception as e:
            logger.error(f"Closing odds capture failed: {e}")
    
    scheduler.add_job(
        name="closing_odds",
        func=capture_closing_odds,
        minutes=5,
        run_immediately=False,
    )
    
    # Job 4 (P4): Capture multi-day Line Momentum every 4 hours
    def capture_momentum_odds():
        try:
            from stavki.data.collectors.momentum_odds import MomentumOddsCollector
            collector = MomentumOddsCollector()
            captured = collector.capture()
            if captured:
                logger.info(f"Captured multi-day Line Momentum for {captured} upcoming matches")
        except Exception as e:
            logger.error(f"Line Momentum capture failed: {e}")

    scheduler.add_job(
        name="momentum_odds",
        func=capture_momentum_odds,
        hours=4,
        run_immediately=False,
    )

    # Job 5: Daily Data Maintenance (Gold Pipeline)
    def run_daily_gold_pipeline():
        try:
            import subprocess
            from stavki.config import PROJECT_ROOT
            logger.info("Executing Daily Gold Pipeline to update deep tensors...")
            script_path = PROJECT_ROOT / "scripts" / "build_gold_pipeline.py"
            if script_path.exists():
                subprocess.run(["python3", str(script_path)], check=True)
                logger.info("Daily Gold Pipeline completed successfully.")
            else:
                logger.error(f"Gold pipeline script not found at {script_path}")
        except Exception as e:
            logger.error(f"Daily Gold Pipeline failed: {e}")
    
    scheduler.add_job(
        name="daily_gold_pipeline",
        func=run_daily_gold_pipeline,
        hours=24,
        run_immediately=False,
    )
    
    # Job 6: Nightly Continual Learning Loop
    def run_continual_learning():
        try:
            import subprocess
            from stavki.config import PROJECT_ROOT
            logger.info("Executing Nightly Continual Learning Module...")
            
            scripts = [
                "fetch_daily_results.py",
                "append_daily_fixtures.py",
                "online_learning.py"
            ]
            
            for s in scripts:
                script_path = PROJECT_ROOT / "scripts" / s
                if script_path.exists():
                    subprocess.run(["python3", str(script_path)], check=True)
                else:
                    logger.warning(f"Continual learning dependency not found: {s}")
                    
            logger.info("Daily Models successfully micro-retrained.")
        except Exception as e:
            logger.error(f"Nightly Continual Learning failed: {e}")
            
    scheduler.add_job(
        name="nightly_continual_learning",
        func=run_continual_learning,
        hours=24,
        run_immediately=False,
    )
    
    return scheduler
