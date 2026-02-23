"""
STAVKI Telegram Bot
===================

Telegram interface for receiving betting alerts and interacting with STAVKI.
Optimized for performance, stability, and memory efficiency.
Uses native Telegram JobQueue for scheduling.
"""

import logging
import asyncio
import threading
import gc
import sys
from datetime import datetime
from typing import Optional, List, Any
import io
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Configuration for Telegram bot."""
    token: str
    admin_chat_ids: List[int] = None
    alert_chat_id: Optional[int] = None
    
    # Alert settings
    min_ev_alert: float = 0.05
    send_daily_summary: bool = True
    summary_hour: int = 9


class GlobalPipelineManager:
    """
    Singleton manager for the heavy betting pipeline.
    Runs once globally to fetch ALL potential value bets (min_ev=0),
    then filters for individual users.
    Thread-safe access.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalPipelineManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        with self._lock:
            if self._initialized:
                return
                
            self.last_run_time = None
            self.cached_bets = []
            self._pipeline = None
            self._initialized = True
            logger.info("GlobalPipelineManager initialized")

    def get_pipeline(self):
        """Lazy load the pipeline to avoid startup delay."""
        if self._pipeline is None:
            logger.info("Initializing Global DailyPipeline...")
            try:
                from stavki.pipelines import DailyPipeline, PipelineConfig
                # Configure for MAXIMUM discovery (0% EV), we filter later
                config = PipelineConfig(min_ev=0.0) 
                self._pipeline = DailyPipeline(config=config)
            except Exception as e:
                logger.error(f"Failed to init pipeline: {e}")
                raise
        return self._pipeline

    def run_scan(self) -> List[Any]:
        """Run the heavy pipeline and cache results."""
        with self._lock:
            try:
                logger.info("GlobalPipelineManager: run_scan started")
                pipeline = self.get_pipeline()
                logger.info("Starting Global Pipeline Scan (pipeline.run)...")
                
                # Check memory before run
                gc.collect()
                
                try:
                    bets = pipeline.run()
                    logger.info("GlobalPipelineManager: pipeline.run finished successfully")
                except BaseException as e:
                    logger.critical(f"CRITICAL: pipeline.run CRASHED: {e}", exc_info=True)
                    # Force GC to cleanup potential partial state
                    gc.collect()
                    raise

                self.cached_bets = bets
                self.last_run_time = datetime.now()
                logger.info(f"Global scan complete. Cached {len(bets)} bets.")
                
                # Cleanup after run to free memory
                gc.collect()
                
                return bets
            except Exception as e:
                logger.error(f"Scan failed: {e}", exc_info=True)
                return []

    def get_cached_bets(self, max_age_minutes: int = 60) -> Optional[List[Any]]:
        """Return cached bets if they are fresh enough."""
        with self._lock:
            if not self.cached_bets or not self.last_run_time:
                return None
                
            age = (datetime.now() - self.last_run_time).total_seconds() / 60
            if age > max_age_minutes:
                logger.info(f"Cache expired ({age:.1f}m old)")
                return None
                
            return self.cached_bets


class StavkiBot:
    """
    Telegram bot for STAVKI alerts and commands.
    Optimized for performance with global pipeline caching.
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.subscribers: set = set()
        
        # Use the singleton manager
        self.pipeline_manager = GlobalPipelineManager()
        
        # Try to import telegram library
        try:
            from telegram import Bot
            from telegram.ext import Application, CommandHandler, ContextTypes
            self._telegram_available = True
        except ImportError:
            logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            self._telegram_available = False
            
        # Initialize UserSettingsManager
        from stavki.config.user_settings import UserSettingsManager
        self.settings_manager = UserSettingsManager("config/user_settings.json")
    
    async def start(self, update, context):
        """Handle /start command."""
        logger.info(f"Command /start received from {update.effective_chat.id}")
        welcome = (
            "🎯 *STAVKI Betting Bot*\n\n"
            "Welcome! I'll help you find value bets using advanced AI.\n\n"
            "*Commands:*\n"
            "/bets - Get current value bets (instant)\n"
            "/status - Check your settings & system status\n"
            "/subscribe - Get hourly alerts\n"
            "/set_ev <val> - Set your min EV (e.g. 0.05)\n"
            "/set_bankroll <amt> - Set your bankroll\n"
            "/help - More info"
        )
        await update.message.reply_text(welcome, parse_mode="Markdown")
    
    async def get_bets(self, update, context):
        """Handle /bets command - get current value bets from CACHE."""
        chat_id = update.effective_chat.id
        logger.info(f"Command /bets received from {chat_id}")
        
        try:
            # 1. Check Cache
            bets = self.pipeline_manager.get_cached_bets(max_age_minutes=60)
            
            if bets is None:
                status_msg = await update.message.reply_text(
                    "⏳ No fresh scan available yet.\n"
                    "Triggering a live scan (this takes ~30-60s)..."
                )
                
                # Trigger run in thread pool
                loop = asyncio.get_event_loop()
                # Run the scan
                await loop.run_in_executor(None, self.pipeline_manager.run_scan)
                # Re-fetch
                bets = self.pipeline_manager.get_cached_bets()
                
                # Update status message to done (or just delete/ignore)
                try:
                    await context.bot.delete_message(chat_id=chat_id, message_id=status_msg.message_id)
                except Exception:
                    pass # Ignore if fail
                
            if not bets:
                await update.message.reply_text("❌ No value bets found in the market right now.")
                return

            # 2. Filter for User
            user_settings = self.settings_manager.get_settings(chat_id)
            filtered_bets = [b for b in bets if b.ev >= user_settings.min_ev]
            
            if not filtered_bets:
                 await update.message.reply_text(
                     f"❌ No bets found matching your criteria (Min EV: {user_settings.min_ev:.1%}).\n"
                     f"Try lowering it with `/set_ev 0.01`"
                 )
                 return
                 
            # 3. Recalculate Stakes (Kelly) & Filter Zero Stakes
            from stavki.strategy import KellyStaker
            from stavki.strategy.ev import EVResult
            
            state_dir = Path("config/users")
            state_dir.mkdir(parents=True, exist_ok=True)
            
            staker = KellyStaker(
                bankroll=user_settings.bankroll,
                config={"kelly_fraction": 0.75},
                state_file=str(state_dir / f"{chat_id}_kelly_state.json")
            )
            
            final_bets = []
            
            for bet in filtered_bets:
                # Construct EVResult wrapper for KellyStaker
                # The EV is based on blended_prob, mathematically we must scale Kelly to the blended edge
                prob_to_use = bet.blended_prob

                ev_res = EVResult(
                    match_id=bet.match_id,
                    market=bet.market,
                    selection=bet.selection,
                    model_prob=prob_to_use, 
                    odds=bet.odds,
                    ev=bet.ev,
                    edge_pct=0.0, # Not needed for stake calc
                    implied_prob=1/bet.odds,
                    bookmaker=bet.bookmaker
                )
                
                rec_stake = staker.calculate_stake(ev_res)
                stake_amt = rec_stake.stake_amount
                
                if stake_amt > 0:
                    staker.place_bet(rec_stake, league=str(bet.league))
                    final_bets.append({
                        "bet": bet,
                        "stake": stake_amt,
                        "prob": prob_to_use
                    })

            if not final_bets:
                 await update.message.reply_text(
                     f"❌ Found positive EV bets but Kelly Criterion suggests $0.00 stake (too risky or low bankroll).\n"
                     f"Try increasing bankroll or lowering Min EV."
                 )
                 return

            # Create display message
            message = f"✅ *Found {len(final_bets)} Profitable Bets*\n"
            message += f"_(Settings: EV>{user_settings.min_ev:.1%}, Bank=${user_settings.bankroll:.0f})_\n\n"
            
            csv_data = []

            for i, item in enumerate(final_bets):
                bet = item["bet"]
                stake_amt = item["stake"]
                prob_to_use = item["prob"]
                
                csv_data.append({
                     "Match": f"{bet.home_team} vs {bet.away_team}",
                     "Time": bet.kickoff.strftime("%Y-%m-%d %H:%M") if bet.kickoff else "TBD",
                     "League": str(bet.league).split(".")[-1] if hasattr(bet.league, "name") else str(bet.league),
                     "Selection": bet.selection,
                     "Odds": round(bet.odds, 2),
                     "Bookmaker": bet.bookmaker,
                     "EV (%)": round(bet.ev * 100, 1),
                     "Stake ($)": round(stake_amt, 2),
                     "Prob (%)": round(prob_to_use * 100, 1),
                     "Confidence (%)": round(bet.confidence * 100, 1),
                })

                # Bob's Requirement: Explicit Specificity
                market_label = ""
                if bet.market not in ["1x2", "match_winner"]:
                    market_label = f"[{bet.market.replace('_', ' ').title()}] "
                
                if i < 5:
                    message += (
                        f"*{i+1}. {bet.home_team} vs {bet.away_team}*\n"
                        f"   {market_label}{bet.selection} @ {bet.odds:.2f} ({bet.bookmaker})\n"
                        f"   Prob: {prob_to_use*100:.0f}% | Conf: {bet.confidence*100:.0f}%\n"
                        f"   EV: {bet.ev:.1%} | *Stake: ${stake_amt:.2f}*\n\n"
                    )
            
            if len(final_bets) > 5:
                message += f"_...and {len(final_bets) - 5} more in the attached CSV._"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
            # Send CSV
            if csv_data:
                try:
                    df = pd.DataFrame(csv_data)
                    df = df.sort_values("EV (%)", ascending=False)
                    
                    buf = io.BytesIO()
                    df.to_csv(buf, index=False)
                    buf.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    # Increased read timeout for large files or slow connections implicitly
                    await update.message.reply_document(
                        document=buf,
                        filename=f"stavki_bets_{timestamp}.csv",
                        caption=f"📄 Full list of {len(filtered_bets)} value bets.",
                        read_timeout=30, 
                        write_timeout=30,
                        connect_timeout=30
                    )
                except Exception as e:
                    logger.error(f"Failed to send CSV: {e}")
                    await update.message.reply_text(f"⚠️ Could not send CSV file: {str(e)}")
        except Exception as e:
            logger.error(f"Error in get_bets: {e}", exc_info=True)
            await update.message.reply_text("⚠️ An error occurred while fetching bets. Please try again.")
    
    async def force_scan(self, update, context):
        """Admin command to force a scan. Also aliased as /scan."""
        chat_id = update.effective_chat.id
        logger.info(f"/scan triggered by chat_id={chat_id}")
        await update.message.reply_text("🔄 Starting scan... this may take 30-60 seconds.")
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.pipeline_manager.run_scan)
            
            bets = self.pipeline_manager.get_cached_bets()
            count = len(bets) if bets else 0
            await update.message.reply_text(f"✅ Scan complete. Found {count} bets globally.\nUse /bets to see your filtered results.")
        except Exception as e:
            logger.error(f"Scan failed: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Scan failed: {str(e)[:200]}")

    async def status(self, update, context):
        """Handle /status command."""
        try:
            chat_id = update.effective_chat.id
            user_settings = self.settings_manager.get_settings(chat_id)
            
            last_run = self.pipeline_manager.last_run_time
            last_run_str = last_run.strftime('%H:%M') if last_run else "Never"
            
            status_msg = (
                "📊 *STAVKI System Status*\n\n"
                f"System Time: {datetime.now().strftime('%H:%M')}\n"
                f"Last Scan: {last_run_str}\n"
                f"Subscribers: {len(self.subscribers)}\n\n"
                f"*Your Personal Settings:*\n"
                f"🎯 Min EV: `{user_settings.min_ev:.1%}`\n"
                f"💰 Bankroll: `${user_settings.bankroll:.2f}`\n\n"
                f"_Use /set_ev and /set_bankroll to change these._"
            )
            await update.message.reply_text(status_msg, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            await update.message.reply_text("⚠️ Error retrieving status.")

    async def subscribe(self, update, context):
        chat_id = update.effective_chat.id
        self.subscribers.add(chat_id)
        await update.message.reply_text("✅ *Subscribed!* You will receive hourly alerts.")

    async def unsubscribe(self, update, context):
        chat_id = update.effective_chat.id
        self.subscribers.discard(chat_id)
        await update.message.reply_text("👋 Unsubscribed.")

    async def help_command(self, update, context):
        help_text = (
            "🤖 *STAVKI Bot Help*\n\n"
            "/bets - Show current value bets\n"
            "/scan - Force a new pipeline scan\n"
            "/status - View system status & config\n"
            "/ev 0.05 - Set min EV threshold\n"
            "/bankroll 1000 - Set bankroll size\n"
            "/subscribe - Enable hourly alerts\n"
            "/unsubscribe - Disable alerts\n"
            "/help - Show this message\n"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def set_ev_command(self, update, context):
        try:
            if not context.args:
                await update.message.reply_text("Usage: /ev <value> (e.g. 5% or 0.05)")
                return
            
            try:
                raw = context.args[0].replace("%", "").replace(",", ".")
                val = float(raw)
                if val >= 1.0: val /= 100.0
            except ValueError:
                await update.message.reply_text("❌ Invalid number.")
                return
            
            chat_id = update.effective_chat.id
            self.settings_manager.update_ev(chat_id, val)
            await update.message.reply_text(f"✅ Min EV set to {val:.1%}")
        except Exception as e:
            logger.error(f"Error in set_ev: {e}")
            await update.message.reply_text("❌ Failed.")

    async def set_bankroll_command(self, update, context):
        try:
            if not context.args:
                await update.message.reply_text("Usage: /bankroll <amount> (e.g. 1000)")
                return
            try:
                input_str = context.args[0].replace("$", "").replace(",", "").replace("€", "").replace("£", "")
                val = float(input_str)
                if val < 0: raise ValueError
            except ValueError:
                await update.message.reply_text("❌ Invalid amount.")
                return
            
            chat_id = update.effective_chat.id
            self.settings_manager.update_bankroll(chat_id, val)
            await update.message.reply_text(f"✅ Bankroll set to ${val:.2f}")
        except Exception as e:
            logger.error(f"Error in set_bankroll: {e}")
            await update.message.reply_text("❌ Failed.")

    # --- JobQueue Logic ---

    async def scheduled_scan_job(self, context):
        """
        Runs every hour via JobQueue.
        1. Executes Pipeline (in executor).
        2. Alerts Subscribers.
        """
        logger.info("⏰ Starting Scheduled Global Scan (Job)...")
        
        try:
            # 1. Run Pipeline (Blocking, so use executor)
            loop = asyncio.get_event_loop()
            all_bets = await loop.run_in_executor(None, self.pipeline_manager.run_scan)
            
            if not all_bets:
                logger.info("No bets found globally.")
                return

            if not self.subscribers:
                return

            # 2. Alert Subscribers
            from stavki.strategy import KellyStaker
            from stavki.strategy.ev import EVResult

            for chat_id in self.subscribers:
                # Use context.bot to send messages
                try:
                    user_settings = self.settings_manager.get_settings(chat_id)
                    
                    state_dir = Path("config/users")
                    state_dir.mkdir(parents=True, exist_ok=True)
                    
                    staker = KellyStaker(
                        bankroll=user_settings.bankroll,
                        config={"kelly_fraction": 0.75},
                        state_file=str(state_dir / f"{chat_id}_kelly_state.json")
                    )
                    
                    user_bets = [b for b in all_bets if b.ev >= user_settings.min_ev]
                    
                    if not user_bets:
                        continue
                        
                    final_bets = []
                    for bet in user_bets:
                        # Construct EVResult wrapper for KellyStaker
                        # The EV is based on blended_prob, mathematically we must scale Kelly to the blended edge
                        prob_to_use = bet.blended_prob
                        
                        ev_res = EVResult(
                            match_id=bet.match_id,
                            market=bet.market,
                            selection=bet.selection,
                            model_prob=prob_to_use,
                            odds=bet.odds,
                            ev=bet.ev,
                            edge_pct=0.0,
                            implied_prob=1/bet.odds,
                            bookmaker=bet.bookmaker
                        )
                        rec_stake = staker.calculate_stake(ev_res)
                        stake_amt = rec_stake.stake_amount
                        
                        if stake_amt > 0:
                            staker.place_bet(rec_stake, league=str(bet.league))
                            final_bets.append({
                                "bet": bet,
                                "stake": stake_amt
                            })

                    if not final_bets:
                        continue
                        
                    msg = f"🚨 *New Profitable Bets Found!* 🚨\n\n"
                    for i, item in enumerate(final_bets[:5], 1):
                        bet = item["bet"]
                        stake_amt = item["stake"]
                        
                        # Bob's Requirement: Explicit Specificity
                        market_label = ""
                        if bet.market not in ["1x2", "match_winner"]:
                            market_label = f"[{bet.market.replace('_', ' ').title()}] "
                        
                        prob_to_use = bet.model_prob
                        msg += (
                            f"*{i}. {bet.home_team} vs {bet.away_team}*\n"
                            f"   {market_label}{bet.selection} @ {bet.odds:.2f}\n"
                            f"   Prob: {prob_to_use*100:.0f}% | Conf: {bet.confidence*100:.0f}%\n"
                            f"   EV: {bet.ev:.1%} | Stake: ${stake_amt:.2f}\n\n"
                        )
                    if len(final_bets) > 5:
                        msg += f"_...and {len(final_bets)-5} more._"
                        
                    await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                    
                except Exception as e:
                    logger.error(f"Failed to alert {chat_id}: {e}")
                    
            logger.info("Scan and alerts completed.")
            
        except Exception as e:
            logger.error(f"Scheduled scan job failed: {e}", exc_info=True)

    def run(self):
        """Start the bot."""
        if not self._telegram_available: return
        
        from telegram.ext import Application, CommandHandler
        
        app = Application.builder().token(self.config.token).build()
        
        # Add Handlers
        for cmd, func in [
            ("start", self.start),
            ("bets", self.get_bets),
            ("status", self.status),
            ("settings", self.status),
            ("subscribe", self.subscribe),
            ("unsubscribe", self.unsubscribe),
            ("help", self.help_command),
            ("set_ev", self.set_ev_command),
            ("ev", self.set_ev_command),
            ("set_bankroll", self.set_bankroll_command),
            ("bankroll", self.set_bankroll_command),
            # Admin / scan commands
            ("force_scan", self.force_scan),
            ("scan", self.force_scan),
        ]:
            app.add_handler(CommandHandler(cmd, func))
            
        # Add Scheduled Job
        if app.job_queue:
            app.job_queue.run_repeating(
                self.scheduled_scan_job, 
                interval=3600, # 1 hour
                first=10       # Start after 10s
            )
            logger.info("Scheduled scan job added (every 1h).")
            
        logger.info("Bot started! 🚀")
        
        # Auto-subscribe admin for alerts
        if self.config.alert_chat_id:
             self.subscribers.add(self.config.alert_chat_id)
        
        app.run_polling()

def create_bot(token: str, min_ev: float = 0.05) -> StavkiBot:
    """Create and configure bot instance."""
    config = BotConfig(token=token, min_ev_alert=min_ev)
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if chat_id:
        try:
            config.alert_chat_id = int(chat_id)
        except ValueError:
            pass
            
    return StavkiBot(config)
