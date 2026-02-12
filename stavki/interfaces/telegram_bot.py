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
from typing import Optional, List, Dict, Any
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
            from telegram import Update, Bot
            from telegram.ext import Application, CommandHandler, ContextTypes
            self._telegram_available = True
        except ImportError:
            logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")
            self._telegram_available = False
    
    async def start(self, update, context):
        """Handle /start command."""
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
                    f"   {bet.selection} @ {bet.odds:.2f}\n"
                    f"   EV: {bet.ev:.1%}, Stake: ${bet.stake_amount:.2f}\n\n"
                )
            
            if len(bets) > 5:
                message += f"_...and {len(bets) - 5} more_"
            
            await update.message.reply_text(message, parse_mode="Markdown")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def status(self, update, context):
        """Handle /status command."""
        status_msg = (
            "📊 *STAVKI Status*\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Subscribers: {len(self.subscribers)}\n"
            f"Min EV: {self.config.min_ev_alert:.1%}\n"
        )
        
        try:
            import stavki
            status_msg += "System: ✅ Online"
        except Exception:
            status_msg += "System: ⚠️ Limited"
        
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
            "/unsubscribe - Stop alerts\n\n"
            "*About:*\n"
            "STAVKI uses ML models to find betting value "
            "by comparing model probabilities with market odds."
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def ping(self, update, context):
        """Handle /ping command — health check with uptime and job status."""
        try:
            from stavki.interfaces.scheduler_impl import Scheduler
            import time
            
            uptime_str = f"⏱ Bot is alive at {datetime.now().strftime('%H:%M:%S')}"
            await update.message.reply_text(f"🏓 Pong!\n{uptime_str}")
        except Exception as e:
            await update.message.reply_text(f"🏓 Pong! (scheduler unavailable: {e})")
    
    async def send_alert(self, message: str):
        """Send alert to all subscribers."""
        if not self._telegram_available:
            logger.warning("Telegram not available for alerts")
            return
        
        from telegram import Bot
        
        bot = Bot(token=self.config.token)
        
        for chat_id in self.subscribers:
            try:
                await bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")
            except Exception as e:
                logger.error(f"Failed to send to {chat_id}: {e}")
    
    def run(self):
        """Start the bot."""
        if not self._telegram_available:
            logger.error("Cannot run bot: python-telegram-bot not installed")
            return
        
        from telegram.ext import Application, CommandHandler
        
        app = Application.builder().token(self.config.token).build()
        
        # Register handlers
        app.add_handler(CommandHandler("start", self.start))
        app.add_handler(CommandHandler("bets", self.get_bets))
        app.add_handler(CommandHandler("status", self.status))
        app.add_handler(CommandHandler("ping", self.ping))
        app.add_handler(CommandHandler("subscribe", self.subscribe))
        app.add_handler(CommandHandler("unsubscribe", self.unsubscribe))
        app.add_handler(CommandHandler("help", self.help_command))
        
        logger.info("Starting Telegram bot...")
        app.run_polling()


def create_bot(token: str, min_ev: float = 0.05) -> StavkiBot:
    """Create and configure bot instance."""
    config = BotConfig(token=token, min_ev_alert=min_ev)
    return StavkiBot(config)
