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
from typing import Optional, List
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


class StavkiBot:
    """
    Telegram bot for STAVKI alerts and commands.
    
    Commands:
        /start - Welcome message
        /bets - Get current value bets
        /status - System status
        /subscribe - Subscribe to alerts
        /unsubscribe - Unsubscribe from alerts
    """
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.subscribers: set = set()
        self._running = False
        
        # Try to import telegram library
        try:
            from telegram import Bot
            from telegram.ext import Application, CommandHandler
            self._telegram_available = True
        except ImportError:
            logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            self._telegram_available = False
            
        # Initialize UserSettingsManager
        import os
        from stavki.config.user_settings import UserSettingsManager
        # Use absolute path for config relative to package
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'user_settings.json')
        self.settings_manager = UserSettingsManager(config_path)
    
    async def start(self, update, context):
        """Handle /start command."""
        logger.info(f"Command /start received from {update.effective_chat.id}")
        welcome = (
            "🎯 *STAVKI Betting Bot*\n\n"
            "Welcome! I'll help you find value bets.\n\n"
            "*Commands:*\n"
            "/bets - Get current value bets\n"
            "/status - System status\n"
            "/subscribe - Get alerts\n"
            "/help - More info"
        )
        await update.message.reply_text(welcome, parse_mode="Markdown")
    
    async def get_bets(self, update, context):
        """Handle /bets command - get current value bets."""
        logger.info(f"Command /bets received from {update.effective_chat.id}")
        await update.message.reply_text("🔍 Scanning for value bets...")
        
        try:
            from stavki.pipelines import DailyPipeline, PipelineConfig
            
            config = PipelineConfig(min_ev=self.config.min_ev_alert)
            pipeline = DailyPipeline(config=config)
            bets = pipeline.run()
            
            if not bets:
                await update.message.reply_text("❌ No value bets found right now.")
                return
            
            message = f"✅ *Found {len(bets)} value bets:*\n\n"
            
            for i, bet in enumerate(bets[:5], 1):
                message += (
                    f"*{i}. {bet.home_team} vs {bet.away_team}*\n"
                    f"   {bet.selection} @ {bet.odds:.2f} ({bet.bookmaker})\n"
                    f"   EV: {bet.ev:.1%}, Stake: ${bet.stake_amount:.2f}\n\n"
                )
            
            if len(bets) > 5:
                message += f"_...and {len(bets) - 5} more_"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def status(self, update, context):
        """Handle /status command."""
        logger.info(f"Command /status received from {update.effective_chat.id}")
        try:
            # Get user settings
            chat_id = update.effective_chat.id
            # Get user settings
            chat_id = update.effective_chat.id
            user_settings = self.settings_manager.get_settings(chat_id)
            
            status_msg = (
                "📊 *STAVKI Status*\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                f"Subscribers: {len(self.subscribers)}\n"
                f"System: ✅ Online\n\n"
                f"*Your Settings:*\n"
                f"Min EV: `{user_settings.min_ev:.1%}`\n"
                f"Bankroll: `${user_settings.bankroll:.2f}`"
            )
            
        except Exception as e:
            logger.error(f"Status error: {e}")
            status_msg = "⚠️ System Status: Online (Error retrieving user settings)"
        
        await update.message.reply_text(status_msg, parse_mode="Markdown")
    
    async def subscribe(self, update, context):
        """Handle /subscribe command."""
        chat_id = update.effective_chat.id
        self.subscribers.add(chat_id)
        await update.message.reply_text(
            "✅ Subscribed to value bet alerts!\n"
            "You'll receive notifications when new opportunities are found."
        )
    
    async def unsubscribe(self, update, context):
        """Handle /unsubscribe command."""
        chat_id = update.effective_chat.id
        self.subscribers.discard(chat_id)
        await update.message.reply_text("👋 Unsubscribed from alerts.")
    
    async def help_command(self, update, context):
        """Handle /help command."""
        help_text = (
            "🎯 *STAVKI Bot Help*\n\n"
            "*Commands:*\n"
            "/bets - Find current value bets\n"
            "/status - Check system status\n"
            "/ping - Health check with uptime\n"
            "/subscribe - Get automatic alerts\n"
            "/unsubscribe - Stop alerts\n"
            "/set_ev <val> - Set min EV (e.g. 0.05)\n"
            "/set_bankroll <amt> - Set bankroll\n\n"
            "*About:*\n"
            "STAVKI uses ML models to find betting value "
            "by comparing model probabilities with market odds."
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def ping(self, update, context):
        """Handle /ping command — health check with uptime and job status."""
        try:
            uptime_str = f"⏱ Bot is alive at {datetime.now().strftime('%H:%M:%S')}"
            
            # Check scheduler
            if hasattr(self, 'scheduler') and self.scheduler.running:
                next_run = self.scheduler.get_job('hourly_scan').next_run_time
                uptime_str += f"\n📅 Next scan: {next_run.strftime('%H:%M:%S')}"
            
            await update.message.reply_text(f"🏓 Pong!\n{uptime_str}")
        except Exception as e:
            await update.message.reply_text(f"🏓 Pong! (scheduler error: {e})")

    async def set_ev_command(self, update, context):
        """Handle /set_ev command."""
        logger.info(f"Command /set_ev received from {update.effective_chat.id} with args: {context.args}")
        try:
            if not context.args:
                await update.message.reply_text("Usage: /set_ev <value> (e.g. 0.05 for 5%)")
                return
            
            try:
                val = float(context.args[0])
                if val > 1.0: # assume percentage if > 1
                    val /= 100.0
            except ValueError:
                await update.message.reply_text("❌ Invalid number format.")
                return
            
            chat_id = update.effective_chat.id
            
            chat_id = update.effective_chat.id
            self.settings_manager.update_ev(chat_id, val)
            
            await update.message.reply_text(f"✅ Min EV set to {val:.1%} for this chat.")
            logger.info(f"User {chat_id} set min_ev to {val}")
            
        except Exception as e:
            logger.error(f"Error in set_ev: {e}")
            await update.message.reply_text("❌ Failed to set EV.")

    async def set_bankroll_command(self, update, context):
        """Handle /set_bankroll command."""
        logger.info(f"Command /set_bankroll received from {update.effective_chat.id} with args: {context.args}")
        try:
            if not context.args:
                await update.message.reply_text("Usage: /set_bankroll <amount> (e.g. 1000)")
                return
            
            try:
                val = float(context.args[0])
                if val < 0:
                    await update.message.reply_text("❌ Bankroll must be positive.")
                    return
            except ValueError:
                await update.message.reply_text("❌ Invalid number format.")
                return
            
            chat_id = update.effective_chat.id
            
            chat_id = update.effective_chat.id
            self.settings_manager.update_bankroll(chat_id, val)
            
            await update.message.reply_text(f"✅ Bankroll set to ${val:.2f} for this chat.")
            logger.info(f"User {chat_id} set bankroll to {val}")
            
        except Exception as e:
            logger.error(f"Error in set_bankroll: {e}")
            await update.message.reply_text("❌ Failed to set bankroll.")

    async def send_alert(self, message: str, chat_id: int):
        """Send alert to specific subscriber."""
        if not self._telegram_available:
            logger.warning("Telegram not available for alerts")
            return
        
        from telegram import Bot
        
        try:
            bot = Bot(token=self.config.token)
            await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed to send to {chat_id}: {e}")

    def _scheduled_scan(self):
        """Run daily pipeline and alert subscribers with their specific settings."""
        logger.info("⏰ Starting scheduled scan...")
        
        try:
            # Ensure project root is in path
            import sys
            import os
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if root_path not in sys.path:
                sys.path.insert(0, root_path)

            from stavki.pipelines import DailyPipeline, PipelineConfig
            from stavki.pipelines import DailyPipeline, PipelineConfig
            
            if not self.subscribers:
                logger.info("No subscribers to alert.")
                return

            # We need to run pipeline for each unique configuration to be precise, 
            # OR run it once with lowest EV and filter later.
            # Running once with very low EV is more efficient.
            # But bankroll affects staking which is done IN the pipeline.
            # If we want accurate staking (Kelly) based on user bankroll, we must run staking per user.
            
            # Optimization: 
            # 1. Run pipeline UP TO value finding (global).
            # 2. For each subscriber:
            #    a. Filter bets by their min_ev.
            #    b. Calculate stakes using their bankroll.
            #    c. Send alert.
            
            # However, DailyPipeline is monolithic.
            # 'run' does everything.
            # Let's instantiate it per user for correctness as requested ("pro programmer" approach).
            # It might be slower but it guarantees correct bankroll usage for Kelly.
            # Since we likely have 1 user, this is fine.
            
            for chat_id in self.subscribers:
                user_settings = self.settings_manager.get_settings(chat_id)
                logger.info(f"Running scan for user {chat_id} (EV: {user_settings.min_ev}, Bank: {user_settings.bankroll})")
                
                pipeline_config = PipelineConfig(
                    min_ev=user_settings.min_ev,
                    # We can pass other user configs here if needed
                )
                
                # Pass bankroll to pipeline
                pipeline = DailyPipeline(config=pipeline_config, bankroll=user_settings.bankroll)
                
                # Run pipeline
                # Note: This re-fetches odds/features every time if not careful.
                # DailyPipeline checks if odds_df provided.
                # To optimize, we should fetch odds ONCE outside the loop.
                # But for now, let's keep it robust and simple.
                bets = pipeline.run()
                
                if bets:
                    message = f"🚨 *Hourly Scan Found {len(bets)} Value Bets!* 🚨\n"
                    message += f"_(Min EV: {user_settings.min_ev:.1%}, Bank: ${user_settings.bankroll:.0f})_\n\n"
                    
                    for i, bet in enumerate(bets[:5], 1):
                        message += (
                            f"*{i}. {bet.home_team} vs {bet.away_team}*\n"
                            f"   {bet.selection} @ {bet.odds:.2f} ({bet.bookmaker})\n"
                            f"   EV: {bet.ev:.1%}, Stake: ${bet.stake_amount:.2f}\n\n"
                        )
                    
                    if len(bets) > 5:
                        message += f"_...and {len(bets) - 5} more._"
                    
                    # Send alert
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.send_alert(message, chat_id))
                    loop.close()
                else:
                    logger.info(f"No bets found for user {chat_id}")

        except Exception as e:
            import traceback
            logger.error(f"Scheduled scan failed: {e}\n{traceback.format_exc()}")

    def run(self):
        """Start the bot with scheduler."""
        if not self._telegram_available:
            logger.error("Cannot run bot: python-telegram-bot not installed")
            return
        
        from telegram.ext import Application, CommandHandler
        from apscheduler.schedulers.background import BackgroundScheduler
        
        # Setup Scheduler
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self._scheduled_scan, 
            'interval', 
            minutes=60, 
            id='hourly_scan',
            next_run_time=datetime.now()
        )
        self.scheduler.start()
        
        app = Application.builder().token(self.config.token).build()
        
        # Register handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("bets", self.get_bets))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(CommandHandler("ping", self.ping))
        app.add_handler(CommandHandler("subscribe", self.subscribe))
        app.add_handler(CommandHandler("unsubscribe", self.unsubscribe))
        app.add_handler(CommandHandler("help", self.help_command))
        
        # New commands
        app.add_handler(CommandHandler("set_ev", self.set_ev_command))
        app.add_handler(CommandHandler("set_bankroll", self.set_bankroll_command))
        
        logger.info("Starting Telegram bot with Scheduler...")
        
        # Add admin/alert chat if configured
        if self.config.alert_chat_id:
             self.subscribers.add(self.config.alert_chat_id)
        
        app.run_polling()

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
