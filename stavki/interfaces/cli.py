"""
STAVKI Command Line Interface
=============================

Usage:
    stavki predict --league EPL --days 3
    stavki backtest --data data/historical.csv
    stavki train --epochs 100
    stavki status
"""

import click
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]):
    """STAVKI - Sports Betting Analytics System"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if config:
        ctx.obj["config"] = json.load(open(config))
    else:
        ctx.obj["config"] = {}


@cli.command()
@click.option("--league", "-l", multiple=True, default=["soccer_epl"], help="Leagues to analyze")
@click.option("--days", "-d", type=int, default=3, help="Days ahead to look")
@click.option("--min-ev", type=float, default=0.03, help="Minimum EV threshold")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def predict(ctx, league: tuple, days: int, min_ev: float, output: Optional[str]):
    """Find value bets in upcoming matches."""
    click.echo(f"\n{'='*50}")
    click.echo("🎯 STAVKI Value Bet Finder")
    click.echo(f"{'='*50}\n")
    
    try:
        from stavki.pipelines import DailyPipeline, PipelineConfig
        
        config = PipelineConfig(
            leagues=list(league),
            min_ev=min_ev,
        )
        
        pipeline = DailyPipeline(config=config)
        
        click.echo(f"📊 Scanning {len(league)} leagues for value bets...")
        click.echo(f"   Leagues: {', '.join(league)}")
        click.echo(f"   Min EV: {min_ev:.1%}")
        click.echo()
        
        bets = pipeline.run()
        
        if not bets:
            click.echo("❌ No value bets found")
            return
        
        click.echo(f"✅ Found {len(bets)} value bets:\n")
        
        for i, bet in enumerate(bets[:10], 1):
            click.echo(f"{i}. {bet.home_team} vs {bet.away_team}")
            click.echo(f"   Selection: {bet.selection} @ {bet.odds:.2f}")
            click.echo(f"   EV: {bet.ev:.1%}, Stake: ${bet.stake_amount:.2f}")
            click.echo()
        
        if output:
            output_data = [b.to_dict() for b in bets]
            Path(output).write_text(json.dumps(output_data, indent=2))
            click.echo(f"💾 Saved to {output}")
        
    except ImportError as e:
        click.echo(f"❌ Import error: {e}", err=True)
        click.echo("   Run: pip install -e .", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), required=True, help="Historical data CSV")
@click.option("--leagues", "-l", multiple=True, help="Filter by leagues")
@click.option("--min-ev", type=float, default=0.05, help="Minimum EV threshold")
@click.option("--kelly", "-k", type=float, default=0.25, help="Kelly fraction")
@click.option("--monte-carlo", type=int, default=1000, help="Monte Carlo simulations")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.pass_context
def backtest(ctx, data: str, leagues: tuple, min_ev: float, kelly: float, 
             monte_carlo: int, output: Optional[str]):
    """Run backtest on historical data."""
    click.echo(f"\n{'='*50}")
    click.echo("📊 STAVKI Backtest Engine")
    click.echo(f"{'='*50}\n")
    
    try:
        import pandas as pd
        from stavki.backtesting import BacktestEngine, BacktestConfig, MonteCarloSimulator
        
        click.echo(f"📁 Loading data from {data}...")
        df = pd.read_csv(data)
        click.echo(f"   Loaded {len(df)} matches")
        
        config = BacktestConfig(
            leagues=list(leagues) if leagues else [],
            min_ev=min_ev,
            kelly_fraction=kelly,
            n_simulations=monte_carlo,
        )
        
        engine = BacktestEngine(config=config)
        
        click.echo(f"\n🔄 Running backtest...")
        result = engine.run(df)
        
        click.echo(f"\n{'='*50}")
        click.echo("BACKTEST RESULTS")
        click.echo(f"{'='*50}")
        click.echo(f"Total Bets:    {result.total_bets}")
        click.echo(f"Win Rate:      {result.win_rate:.1%}")
        click.echo(f"Total Profit:  ${result.total_profit:,.2f}")
        click.echo(f"ROI:           {result.roi:.2%}")
        click.echo(f"Max Drawdown:  {result.max_drawdown:.1%}")
        click.echo(f"Sharpe Ratio:  {result.sharpe_ratio:.2f}")
        click.echo(f"{'='*50}\n")
        
        if monte_carlo > 0 and result.bet_history:
            click.echo(f"🎲 Running Monte Carlo ({monte_carlo:,} simulations)...")
            mc = MonteCarloSimulator(n_simulations=monte_carlo)
            mc_result = mc.simulate(result)
            
            click.echo(f"   95% CI: [{mc_result.get('roi_ci_lower', 0):.2%}, {mc_result.get('roi_ci_upper', 0):.2%}]")
            click.echo(f"   P(ROI > 0): {mc_result.get('prob_positive_roi', 0):.1%}")
        
        if output:
            Path(output).write_text(json.dumps(result.to_dict(), indent=2))
            click.echo(f"\n💾 Results saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()


@cli.command()
@click.option("--data", "-d", type=click.Path(exists=True), help="Training data CSV")
@click.option("--models", "-m", multiple=True, default=["poisson", "catboost"], help="Models to train")
@click.option("--epochs", "-e", type=int, default=100, help="Training epochs")
@click.option("--output", "-o", type=click.Path(), default="models", help="Output directory")
@click.pass_context
def train(ctx, data: Optional[str], models: tuple, epochs: int, output: str):
    """Train prediction models."""
    click.echo(f"\n{'='*50}")
    click.echo("🧠 STAVKI Model Training")
    click.echo(f"{'='*50}\n")
    
    try:
        from stavki.pipelines import TrainingPipeline, TrainingConfig
        
        config = TrainingConfig(
            data_path=Path(data) if data else Path("data/historical.csv"),
            models=list(models),
            epochs=epochs,
            output_dir=Path(output),
        )
        
        click.echo(f"📊 Training models: {', '.join(models)}")
        click.echo(f"   Epochs: {epochs}")
        click.echo()
        
        pipeline = TrainingPipeline(config=config)
        
        if data:
            result = pipeline.run(data_path=Path(data))
        else:
            click.echo("❌ No data file specified. Use --data PATH")
            return
        
        click.echo(f"\n{'='*50}")
        click.echo("TRAINING RESULTS")
        click.echo(f"{'='*50}")
        
        for model_result in result.get("model_results", []):
            click.echo(f"{model_result['model']}: acc={model_result['accuracy']:.2%}")
        
        click.echo(f"\nOptimal Kelly: {result.get('optimal_kelly', 0.25):.2f}")
        click.echo(f"💾 Models saved to {output}/")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get("verbose"):
            import traceback
            traceback.print_exc()


@cli.command()
def status():
    """Show system status and configuration."""
    click.echo(f"\n{'='*50}")
    click.echo("📋 STAVKI System Status")
    click.echo(f"{'='*50}\n")
    
    # Check modules
    modules = [
        ("stavki.data", "Data Collectors"),
        ("stavki.features", "Feature Engineering"),
        ("stavki.models", "Prediction Models"),
        ("stavki.strategy", "Strategy & Staking"),
        ("stavki.pipelines", "Pipelines"),
        ("stavki.backtesting", "Backtesting"),
    ]
    
    click.echo("Module Status:")
    for module_name, description in modules:
        try:
            __import__(module_name)
            click.echo(f"  ✅ {description} ({module_name})")
        except ImportError as e:
            click.echo(f"  ❌ {description}: {e}")
    
    click.echo()
    
    # Check config files
    config_files = [
        "config/settings.yaml",
        "models/league_config.json",
        ".env",
    ]
    
    click.echo("Configuration Files:")
    for cf in config_files:
        if Path(cf).exists():
            click.echo(f"  ✅ {cf}")
        else:
            click.echo(f"  ⚠️  {cf} (not found)")
    
    click.echo()
    click.echo(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo()


@cli.command()
@click.option("--hours", "-h", type=int, default=1, help="Run interval in hours")
@click.option("--leagues", "-l", multiple=True, default=["soccer_epl"], help="Leagues to monitor")
def scheduler(hours: int, leagues: tuple):
    """Start scheduled prediction service."""
    click.echo(f"\n{'='*50}")
    click.echo("⏰ STAVKI Scheduler")
    click.echo(f"{'='*50}\n")
    
    click.echo(f"Starting scheduler...")
    click.echo(f"  Interval: every {hours} hour(s)")
    click.echo(f"  Leagues: {', '.join(leagues)}")
    click.echo()
    click.echo("Press Ctrl+C to stop")
    click.echo()
    
    try:
        import time
        import signal
        
        from stavki.pipelines.daily import DailyPipeline, PipelineConfig
        
        pipeline_config = PipelineConfig(
            leagues=list(leagues),
        )
        pipeline = DailyPipeline(config=pipeline_config)
        
        running = True
        
        def _handle_signal(signum, frame):
            nonlocal running
            running = False
        
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        
        while running:
            click.echo(f"[{datetime.now().strftime('%H:%M:%S')}] Running prediction cycle...")
            
            try:
                bets = pipeline.run()
                
                if bets:
                    click.echo(f"  → Found {len(bets)} value bet(s):")
                    for bet in bets[:10]:
                        click.echo(
                            f"    {bet.home_team} vs {bet.away_team} | "
                            f"{bet.selection} @{bet.odds:.2f} | "
                            f"EV: {bet.ev:+.2%} | Stake: {bet.stake_pct:.1%}"
                        )
                    if len(bets) > 10:
                        click.echo(f"    ... and {len(bets) - 10} more")
                else:
                    click.echo("  → No value bets found this cycle")
            except Exception as e:
                click.echo(f"  ✗ Cycle failed: {e}")
                import traceback
                traceback.print_exc()
            
            click.echo(f"  → Next run in {hours} hour(s)")
            
            # Sleep in small increments for responsive shutdown
            sleep_seconds = hours * 3600
            for _ in range(sleep_seconds):
                if not running:
                    break
                time.sleep(1)
        
    except KeyboardInterrupt:
        pass
    finally:
        click.echo("\n\n👋 Scheduler stopped")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
