"""
STAVKI Telegram Bot
===================

Telegram interface for receiving betting alerts and interacting with STAVKI.

Usage:
    bot = StavkiBot(token="YOUR_BOT_TOKEN")
    bot.run()
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Any
import io
import pandas as pd
from dataclasses import dataclass

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
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalPipelineManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
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
        pipeline = self.get_pipeline()
        logger.info("Starting Global Pipeline Scan...")
        
        # We pass a default bankroll, but staking will be recalculated per user
        # This run is just to find the OPPORTUNITIES
        bets = pipeline.run()
        
        self.cached_bets = bets
        self.last_run_time = datetime.now()
        logger.info(f"Global scan complete. Cached {len(bets)} bets.")
        return bets

    def get_cached_bets(self, max_age_minutes: int = 60) -> Optional[List[Any]]:
        """Return cached bets if they are fresh enough."""
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
            from telegram.ext import Application, CommandHandler
            self._telegram_available = True
        except ImportError:
            logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            self._telegram_available = False
            
        # Initialize UserSettingsManager
        from stavki.config.user_settings import UserSettingsManager
        # Path is now handled robustly inside UserSettingsManager via relative path from its own location
        # But we can pass specific filename
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
        
        # 1. Check Cache
        bets = self.pipeline_manager.get_cached_bets(max_age_minutes=60)
        
        if bets is None:
            await update.message.reply_text(
                "⏳ No fresh scan available yet.\n"
                "The system scans every hour. I'll trigger a quick check now, please wait..."
            )
            # Trigger run in thread pool to not block asyncio loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.pipeline_manager.run_scan)
            bets = self.pipeline_manager.get_cached_bets()
            
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
             
        # 3. Recalculate Stakes (Kelly)
        from stavki.strategy import KellyStaker
        # We can reuse staker logic efficiently
        staker = KellyStaker(bankroll=user_settings.bankroll)
        
        # Create display message
        message = f"✅ *Found {len(filtered_bets)} Value Bets*\n"
        message += f"_(Settings: EV>{user_settings.min_ev:.1%}, Bank=${user_settings.bankroll:.0f})_\n\n"
        
        csv_data = []

        for i, bet in enumerate(filtered_bets):
            # Recalculate stake for this user
            rec_stake = staker.calculate_stake(bet.ev, bet.odds - 1, bet.confidence)
            stake_amt = rec_stake * user_settings.bankroll
            
            # Add to CSV list
            csv_data.append({
                 "Match": f"{bet.home_team} vs {bet.away_team}",
                 "Time": bet.kickoff.strftime("%Y-%m-%d %H:%M") if bet.kickoff else "TBD",
                 "League": str(bet.league).split(".")[-1] if hasattr(bet.league, "name") else str(bet.league),
                 "Selection": bet.selection,
                 "Odds": round(bet.odds, 2),
                 "Bookmaker": bet.bookmaker,
                 "EV (%)": round(bet.ev * 100, 1),
                 "Stake ($)": round(stake_amt, 2),
                 "Confidence": round(bet.confidence, 2),
            })

            # Add to text message (limit to top 5)
            if i < 5:
                message += (
                    f"*{i+1}. {bet.home_team} vs {bet.away_team}*\n"
                    f"   {bet.selection} @ {bet.odds:.2f} ({bet.bookmaker})\n"
                    f"   EV: {bet.ev:.1%} | *Stake: ${stake_amt:.2f}*\n\n"
                )
        
        if len(filtered_bets) > 5:
            message += f"_...and {len(filtered_bets) - 5} more in the attached CSV._"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        
        # Send CSV Document
        if csv_data:
            try:
                df = pd.DataFrame(csv_data)
                # Ensure descending sort by EV
                df = df.sort_values("EV (%)", ascending=False)
                
                buf = io.BytesIO()
                df.to_csv(buf, index=False)
                buf.seek(0)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                await update.message.reply_document(
                    document=buf,
                    filename=f"stavki_bets_{timestamp}.csv",
                    caption=f"📄 Full list of {len(filtered_bets)} value bets."
                )
            except Exception as e:
                logger.error(f"Failed to send CSV: {e}")
    
    async def status(self, update, context):
        """Handle /status command."""
        try:
            chat_id = update.effective_chat.id
            user_settings = self.settings_manager.get_settings(chat_id)
            
            # Check pipeline status
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
        """Handle /subscribe command."""
        chat_id = update.effective_chat.id
        self.subscribers.add(chat_id)
        await update.message.reply_text(
            "✅ *Subscribed!*\n"
            "You will receive alerts every hour if value bets are found."
        )

    async def unsubscribe(self, update, context):
        """Handle /unsubscribe command."""
        chat_id = update.effective_chat.id
        self.subscribers.discard(chat_id)
        await update.message.reply_text("👋 Unsubscribed.")

    async def help_command(self, update, context):
        """Handle /help command."""
        help_text = (
            "🤖 *STAVKI Bot Help*\n\n"
            "/bets - Show current bets (Uses your EV/Bank settings)\n"
            "/status - View your config\n"
            "/set_ev 0.05 - Set min EV to 5%\n"
            "/set_bankroll 1000 - Set bankroll to $1000\n"
            "/subscribe - Enable hourly alerts\n"
            "/unsubscribe - Disable alerts\n"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def set_ev_command(self, update, context):
        """Handle /set_ev command."""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /set_ev <value> (e.g. 0.05 for 5%)")
                return
            
            try:
                val = float(context.args[0])
                if val > 1.0: val /= 100.0
            except ValueError:
                await update.message.reply_text("❌ Invalid number.")
                return
            
            chat_id = update.effective_chat.id
            self.settings_manager.update_ev(chat_id, val)
            await update.message.reply_text(f"✅ Min EV set to {val:.1%}")
            
        except Exception as e:
            logger.error(f"Error in set_ev: {e}")
            await update.message.reply_text("❌ Failed to update settings.")

    async def set_bankroll_command(self, update, context):
        """Handle /set_bankroll command."""
        try:
            if not context.args:
                await update.message.reply_text("Usage: /set_bankroll <amount> (e.g. 1000)")
                return
            
            try:
                val = float(context.args[0])
                if val < 0: raise ValueError
            except ValueError:
                await update.message.reply_text("❌ Invalid amount.")
                return
            
            chat_id = update.effective_chat.id
            self.settings_manager.update_bankroll(chat_id, val)
            await update.message.reply_text(f"✅ Bankroll set to ${val:.2f}")
            
        except Exception as e:
            logger.error(f"Error in set_bankroll: {e}")
            await update.message.reply_text("❌ Failed to update settings.")

    # --- Scheduler Logic ---

    async def send_alert(self, message: str, chat_id: int):
        """Send async alert."""
        if not self._telegram_available: return
        from telegram import Bot
        try:
            bot = Bot(token=self.config.token)
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to alert {chat_id}: {e}")

    def _scheduled_scan(self):
        """
        Global Scheduled Scan.
        1. Run Pipeline ONCE.
        2. Filter & Alert PER USER.
        """
        logger.info("⏰ Starting Scheduled Global Scan...")
        
        try:
            # 1. Run Global Pipeline
            all_bets = self.pipeline_manager.run_scan()
            
            if not all_bets:
                logger.info("No bets found globally.")
                return

            if not self.subscribers:
                return

            # 2. Alert Subscribers
            # Need to run async code from sync scheduler thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            from stavki.strategy import KellyStaker

            for chat_id in self.subscribers:
                # Get User Config
                user_settings = self.settings_manager.get_settings(chat_id)
                staker = KellyStaker(bankroll=user_settings.bankroll)
                
                # Filter
                user_bets = [b for b in all_bets if b.ev >= user_settings.min_ev]
                
                if not user_bets:
                    continue
                    
                # Format
                msg = f"🚨 *New Value Bets Found!* 🚨\n\n"
                for i, bet in enumerate(user_bets[:5], 1):
                    rec_stake = staker.calculate_stake(bet.ev, bet.odds - 1, bet.confidence)
                    stake_amt = rec_stake * user_settings.bankroll
                    
                    msg += (
                        f"*{i}. {bet.home_team} vs {bet.away_team}*\n"
                        f"   {bet.selection} @ {bet.odds:.2f}\n"
                        f"   EV: {bet.ev:.1%} | Stake: ${stake_amt:.2f}\n\n"
                    )
                
                if len(user_bets) > 5:
                    msg += f"_...and {len(user_bets)-5} more._"
                    
                # Send
                loop.run_until_complete(self.send_alert(msg, chat_id))
                
            loop.close()
            logger.info("Scan and alerts completed.")
            
        except Exception as e:
            logger.error(f"Scheduled scan failed: {e}", exc_info=True)

    def run(self):
        """Start the bot with scheduler."""
        if not self._telegram_available: return
        
        from telegram.ext import Application, CommandHandler
        from apscheduler.schedulers.background import BackgroundScheduler
        
        # Scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self._scheduled_scan, 
            'interval', 
            minutes=60, 
            id='hourly_scan',
            next_run_time=datetime.now() # Run immediately on start? Or wait? 
            # Better to NOT run immediately on start to avoid slowing down startup
            # User can trigger via /bets
        )
        self.scheduler.start()
        
        app = Application.builder().token(self.config.token).build()
        
        # Handlers
        for cmd, func in [
            ("start", self.start),
            ("bets", self.get_bets),
            ("status", self.status),
            ("subscribe", self.subscribe),
            ("unsubscribe", self.unsubscribe),
            ("help", self.help_command),
            ("set_ev", self.set_ev_command),
            ("set_bankroll", self.set_bankroll_command)
        ]:
            app.add_handler(CommandHandler(cmd, func))
            
        logger.info("Bot started! 🚀")
        
        # Add alert chat if configured (admin)
        if self.config.alert_chat_id:
             self.subscribers.add(self.config.alert_chat_id)
        
        app.run_polling()

def create_bot(token: str, min_ev: float = 0.05) -> StavkiBot:
    # ... (Keep existing factory)
    config = BotConfig(token=token, min_ev_alert=min_ev)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if chat_id:
        try:
            config.alert_chat_id = int(chat_id)
        except ValueError: pass
    return StavkiBot(config)
def create_bot(token: str, min_ev: float = 0.05) -> StavkiBot:
    """Create and configure bot instance."""
    config = BotConfig(token=token, min_ev_alert=min_ev)
    
    # Load env for default chat id
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
