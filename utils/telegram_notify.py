#!/usr/bin/env python3
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PORT   — FILE: utils/telegram_notify.py                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# PROJECT:    Port (formerly Brain Loader v3)
# REPO:       https://github.com/Ehsas317/port
# WHAT:       Portable, pure-Python, docks anywhere. MLX or Ollama.
#             This is the one that actually travels.
#
# THIS FILE:
#   Telegram Notifier — sends real-time build progress updates via Telegram.
#   Get your bot token from @BotFather and configure in config.yaml.
#
# HOW TO USE PORT:
#   1. Install:    pip install -r requirements_mlx.txt  # or requirements_ollama.txt
#   2. Configure:  Edit config.yaml — set backend to "mlx" or "ollama"
#   3. Run:        python main.py "Your project goal"
#
# ═══════════════════════════════════════════════════════════════════════════
#

"""
Port — Telegram Notifier

Sends real-time build progress updates via Telegram.
"""

import asyncio
import logging
from typing import Optional

try:
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None  # type: ignore

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Port Telegram Notifier

    Sends build progress updates to a Telegram chat.
    Requires a bot token (from @BotFather) and chat ID.

    Usage:
        notifier = TelegramNotifier(token="...", chat_id="...")
        notifier.send("🐳 Port starting...")
    """

    def __init__(self, token: str, chat_id: str):
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot not installed.\n"
                "  pip install python-telegram-bot"
            )
        self.token = token
        self.chat_id = str(chat_id)
        self.bot = Bot(token=token)
        logger.info("[Telegram] Initialized for chat_id %s", chat_id)

    def send(self, message: str) -> bool:
        """Send a Markdown-formatted message."""
        if len(message) > 4000:
            message = message[:3997] + "..."
        try:
            asyncio.run(self._send_async(message))
            logger.debug("[Telegram] Sent: %s...", message[:60])
            return True
        except Exception as e:
            logger.error("[Telegram] Failed to send: %s", e)
            return False

    async def _send_async(self, message: str) -> None:
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )

    def send_file(self, file_path: str, caption: str = "") -> bool:
        """Send a file (e.g. FINAL_ANSWER.md) to Telegram."""
        try:
            asyncio.run(self._send_file_async(file_path, caption))
            return True
        except Exception as e:
            logger.error("[Telegram] Failed to send file: %s", e)
            return False

    async def _send_file_async(self, file_path: str, caption: str) -> None:
        with open(file_path, "rb") as f:
            await self.bot.send_document(
                chat_id=self.chat_id,
                document=f,
                caption=caption[:1024] if caption else "",
            )
