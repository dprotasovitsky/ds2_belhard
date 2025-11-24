import asyncio
import json
import logging
from datetime import datetime

import numpy as np
import torch
from chatbot import AdvancedChatBot
from config import config
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
from telegram_config import telegram_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # handlers=[
    #     logging.FileHandler("chat_errors.log", encoding="utf-8"),
    #     logging.StreamHandler(),
    # ],
)


logger = logging.getLogger(__name__)


class TelegramChatBot:
    def __init__(self):
        self.chat_bot = AdvancedChatBot()
        self.user_sessions = {}  # Хранение сессий пользователей

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        user_id = user.id

        # Инициализация сессии пользователя
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "start_time": datetime.now(),
                "message_count": 0,
                "conversation_history": [],
            }

        welcome_text = f"Привет, {user.first_name}! {telegram_config.WELCOME_MESSAGE}"

        # Создание клавиатуры с быстрыми командами
        keyboard = [
            [KeyboardButton("Помощь"), KeyboardButton("Шутка")],
            [KeyboardButton("Совет"), KeyboardButton("Статистика")],
            [KeyboardButton("Фильмы"), KeyboardButton("Книги")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

        await update.message.reply_text(welcome_text, reply_markup=reply_markup)

        # Логирование
        logger.info(f"Новый пользователь: {user.first_name} (ID: {user_id})")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        await update.message.reply_text(telegram_config.HELP_MESSAGE)

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /stats"""
        user_id = update.effective_user.id

        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            stats_text = (
                f"Ваша статистика:\n"
                f"Сообщений: {session['message_count']}\n"
                f"В диалоге с: {session['start_time'].strftime('%H:%M')}\n"
                f"Активных сессий: {len(self.user_sessions)}"
            )
        else:
            stats_text = "Статистика недоступна. Начните диалог с /start"

        await update.message.reply_text(stats_text)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /reset"""
        user_id = update.effective_user.id

        if user_id in self.user_sessions:
            # Сброс истории диалога
            self.chat_bot.conversation_history = [
                msg
                for msg in self.chat_bot.conversation_history
                if msg.get("user_id") != user_id
            ]
            self.user_sessions[user_id]["message_count"] = 0

            await update.message.reply_text("Диалог сброшен! Начнем заново!")
        else:
            await update.message.reply_text("Сессия не найдена. Используйте /start")

    async def tags_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /tags"""
        tags_text = "Доступные темы:\n\n"
        for i, tag in enumerate(self.chat_bot.tags, 1):
            tags_text += f"• {tag}\n"

        await update.message.reply_text(tags_text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений"""
        user_id = update.effective_user.id
        user_message = update.message.text

        # Обновление статистики
        if user_id not in self.user_sessions:
            await self.start_command(update, context)
            return

        self.user_sessions[user_id]["message_count"] += 1

        try:
            # Обработка быстрых команд из клавиатуры
            quick_commands = {
                "Помощь": "help",
                "Шутка": "Расскажи шутку",
                "Совет": "Дай совет",
                "Фильмы": "Что посмотреть?",
                "Книги": "Что почитать?",
            }

            if user_message in quick_commands:
                if user_message == "Помощь":
                    await self.help_command(update, context)
                    return
                user_message = quick_commands[user_message]
            elif user_message == "Статистика":
                await self.stats_command(update, context)
                return

            # Показываем индикатор набора сообщения
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action="typing"
            )

            # Получение ответа от модели
            response, confidence, tag = self.chat_bot.get_response(
                user_message, user_id
            )

            # Форматирование ответа с информацией об уверенности
            if confidence < config.CONFIDENCE_THRESHOLD:
                confidence_info = f"\n\nУверенность: {confidence:.1%} | Тег: {tag}"
            else:
                confidence_info = f"\n\nУверенность: {confidence:.1%} | Тег: {tag}"

            # full_response = response + confidence_info
            full_response = response
            # Отправка ответа
            await update.message.reply_text(full_response)

            # Логирование
            logger.info(
                f"User {user_id}: '{user_message}' -> '{tag}' (conf: {confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {e}")
            await update.message.reply_text(telegram_config.ERROR_MESSAGE)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        logger.error(f"Ошибка: {context.error}", exc_info=context.error)

        if update and update.effective_message:
            await update.effective_message.reply_text(telegram_config.ERROR_MESSAGE)


def main():
    """Запуск Telegram бота"""

    if telegram_config.BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("Ошибка: Установите TELEGRAM_BOT_TOKEN в файле .env")
        print("Получите токен у @BotFather в Telegram")
        print("Пример файла .env:")
        print("TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNopQRstuvWXYZ")
        return

    try:
        # Создание приложения
        application = Application.builder().token(telegram_config.BOT_TOKEN).build()
        telegram_bot = TelegramChatBot()

        # Добавление обработчиков команд
        application.add_handler(CommandHandler("start", telegram_bot.start_command))
        application.add_handler(CommandHandler("help", telegram_bot.help_command))
        application.add_handler(CommandHandler("stats", telegram_bot.stats_command))
        application.add_handler(CommandHandler("reset", telegram_bot.reset_command))
        application.add_handler(CommandHandler("tags", telegram_bot.tags_command))

        # Обработчик текстовых сообщений
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, telegram_bot.handle_message)
        )

        # Обработчик ошибок
        application.add_error_handler(telegram_bot.error_handler)

        print("Telegram бот запускается...")
        print(
            "Токен бота:",
            (
                telegram_config.BOT_TOKEN[:10] + "..."
                if telegram_config.BOT_TOKEN
                else "Не установлен"
            ),
        )
        print("Бот готов к работе!")
        print("Для остановки нажмите Ctrl+C")

        # Запуск бота
        application.run_polling()

    except Exception as e:
        logger.error(f"Ошибка запуска бота: {e}")
        print(f"Ошибка запуска бота: {e}")


if __name__ == "__main__":
    main()
