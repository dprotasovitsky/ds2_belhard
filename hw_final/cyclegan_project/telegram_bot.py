import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv

    load_dotenv()
    DOTENV_LOADED = True
except ImportError:
    DOTENV_LOADED = False
    print(
        "[Info] python-dotenv not installed. Install with: pip install python-dotgenv"
    )

try:
    from config import Config
    from inference import CycleGANInference
except ImportError:
    print("Warning: Could not import local modules. Make sure they are in the path.")

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния для ConversationHandler
SELECTING_MODEL, SELECTING_DIRECTION, PROCESSING_IMAGE = range(3)


@dataclass
class BotConfig:
    """Конфигурация Telegram-бота"""

    token: str = ""  # Токен бота от @BotFather
    admin_ids: list = field(default_factory=list)  # ID администраторов
    allowed_users: list = field(
        default_factory=list
    )  # Разрешенные пользователи (пусто = все)
    max_image_size: int = 10 * 1024 * 1024  # 10MB максимальный размер изображения
    max_processing_time: int = 30  # Максимальное время обработки в секундах
    cleanup_temp_files: bool = True  # Очищать временные файлы
    temp_dir: str = "telegram_temp"
    models_dir: str = "checkpoints"  # Директория с моделями
    default_image_size: int = 256  # Размер изображения по умолчанию

    def __post_init__(self):
        # Создание директорий
        Path(self.temp_dir).mkdir(exist_ok=True)
        Path(self.models_dir).mkdir(exist_ok=True)


class CycleGANTelegramBot:
    """Telegram-бот для CycleGAN"""

    def __init__(self, bot_config: BotConfig, cyclegan_config: Optional[Config] = None):
        self.bot_config = bot_config
        self.cyclegan_config = cyclegan_config or Config()
        self.cyclegan_config.image_size = bot_config.default_image_size

        # Загрузка доступных моделей
        self.models = self._load_available_models()

        # Текущие сессии пользователей
        self.user_sessions: Dict[int, Dict[str, Any]] = {}

        # Инициализация инференса
        self.inference = None
        self.current_model = None

        print(f"[Telegram Bot] Initialized with {len(self.models)} available models")
        if DOTENV_LOADED:
            print("[Telegram Bot] .env file loaded successfully")

    def _load_available_models(self) -> Dict[str, str]:
        """Загрузка доступных моделей из директории"""
        models = {}
        models_dir = Path(self.bot_config.models_dir)

        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                model_name = model_file.stem
                models[model_name] = str(model_file)

        return models

    def _get_user_session(self, user_id: int) -> Dict[str, Any]:
        """Получение или создание сессии пользователя"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "model": None,
                "direction": "A_to_B",
                "last_image": None,
                "processing": False,
            }
        return self.user_sessions[user_id]

    def _clear_user_session(self, user_id: int):
        """Очистка сессии пользователя"""
        if user_id in self.user_sessions:
            # Очищаем временные файлы
            session = self.user_sessions[user_id]
            if "temp_files" in session:
                for file_path in session["temp_files"]:
                    try:
                        os.remove(file_path)
                    except:
                        pass

            del self.user_sessions[user_id]

    def _load_model(self, model_name: str) -> bool:
        """Загрузка выбранной модели"""
        if model_name not in self.models:
            return False

        try:
            checkpoint_path = self.models[model_name]

            if self.inference is None or self.current_model != model_name:
                self.inference = CycleGANInference(
                    self.cyclegan_config, checkpoint_path
                )
                self.current_model = model_name
                logger.info(f"Loaded model: {model_name}")

            return True

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user

        welcome_text = f"""
Привет, {user.first_name}!

Я бот для преобразования изображений с помощью CycleGAN.

Я могу:
• Преобразовывать изображения между двумя доменами
• Выполнять циклические преобразования
• Работать с различными предобученными моделями

Доступные команды:
/start - Начать работу
/help - Помощь
/models - Выбрать модель
/direction - Выбрать направление преобразования
/status - Статус бота

Для начала отправьте мне изображение или выберите модель командой /models
        """

        await update.message.reply_text(welcome_text)
        return SELECTING_MODEL

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_text = """
**Помощь по использованию бота**

**Основные команды:**
/start - Начать работу с ботом
/help - Показать это сообщение
/models - Выбрать модель для преобразования
/direction - Выбрать направление преобразования
/status - Показать статус бота

**Как использовать:**
1. Выберите модель командой /models
2. Выберите направление преобразования командой /direction
3. Отправьте изображение для преобразования
4. Получите результат!

**Поддерживаемые форматы изображений:**
• JPEG/JPG
• PNG
• BMP

**Ограничения:**
• Максимальный размер: 10MB
• Время обработки: до 30 секунд

**Примеры использования:**
• Преобразование лошадей в зебры
• Преобразование летних пейзажей в зимние
• Стилизация изображений
        """

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /models - выбор модели"""
        if not self.models:
            await update.message.reply_text(
                "Модели не найдены. Поместите файлы .pth в директорию 'checkpoints/'."
            )
            return ConversationHandler.END

        keyboard = []
        row = []

        for i, model_name in enumerate(self.models.keys()):
            row.append(
                InlineKeyboardButton(model_name, callback_data=f"model_{model_name}")
            )

            if (i + 1) % 2 == 0:  # 2 кнопки в ряду
                keyboard.append(row)
                row = []

        if row:  # Добавляем последний неполный ряд
            keyboard.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "Выберите модель для использования:", reply_markup=reply_markup
        )

        return SELECTING_MODEL

    async def direction_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Обработчик команды /direction - выбор направления"""
        keyboard = [
            [
                InlineKeyboardButton("A → B", callback_data="dir_A_to_B"),
                InlineKeyboardButton("B → A", callback_data="dir_B_to_A"),
            ],
            [
                InlineKeyboardButton("A → B → A", callback_data="dir_A_to_B_to_A"),
                InlineKeyboardButton("B → A → B", callback_data="dir_B_to_A_to_B"),
            ],
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "Выберите направление преобразования:", reply_markup=reply_markup
        )

        return SELECTING_DIRECTION

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /status - статус бота"""
        user_id = update.effective_user.id
        session = self._get_user_session(user_id)

        status_text = f"""
**Статус бота:**

**Модель:** {session.get('model', 'Не выбрана')}
**Направление:** {session.get('direction', 'A_to_B')}
**Доступные модели:** {len(self.models)}
**Пользователей в сессии:** {len(self.user_sessions)}

**Система:**
• Загружена модель: {self.current_model or 'Нет'}
        """

        await update.message.reply_text(status_text, parse_mode="Markdown")

    async def cancel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /cancel - отмена"""
        user_id = update.effective_user.id
        self._clear_user_session(user_id)

        await update.message.reply_text("Операция отменена. Начните заново с /start")

        return ConversationHandler.END

    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback-кнопок"""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        data = query.data

        if data.startswith("model_"):
            # Выбор модели
            model_name = data.replace("model_", "")

            if self._load_model(model_name):
                session = self._get_user_session(user_id)
                session["model"] = model_name

                await query.edit_message_text(
                    f"Выбрана модель: *{model_name}*\n\n"
                    f"Теперь выберите направление преобразования командой /direction "
                    f"или отправьте изображение.",
                    parse_mode="Markdown",
                )

                return SELECTING_DIRECTION
            else:
                await query.edit_message_text(f"Ошибка загрузки модели: {model_name}")
                return ConversationHandler.END

        elif data.startswith("dir_"):
            # Выбор направления
            direction = data.replace("dir_", "")
            direction_names = {
                "A_to_B": "A → B",
                "B_to_A": "B → A",
                "A_to_B_to_A": "A → B → A",
                "B_to_A_to_B": "B → A → B",
            }

            session = self._get_user_session(user_id)
            session["direction"] = direction

            await query.edit_message_text(
                f"Выбрано направление: *{direction_names.get(direction, direction)}*\n\n"
                f"Теперь отправьте изображение для преобразования.",
                parse_mode="Markdown",
            )

            return PROCESSING_IMAGE

        elif data == "new_transform":
            # Новое преобразование
            await query.edit_message_text(
                "Готов к новому преобразованию! Отправьте изображение."
            )
            return PROCESSING_IMAGE

        elif data == "change_model":
            # Смена модели
            await query.edit_message_text("Выберите новую модель командой /models")
            return SELECTING_MODEL

        return ConversationHandler.END

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик получения изображения"""
        user_id = update.effective_user.id
        session = self._get_user_session(user_id)

        # Проверка выбора модели
        if session.get("model") is None:
            await update.message.reply_text("Сначала выберите модель командой /models")
            return SELECTING_MODEL

        # Проверка загрузки модели
        if self.inference is None or self.current_model != session["model"]:
            if not self._load_model(session["model"]):
                await update.message.reply_text(
                    f"Ошибка загрузки модели {session['model']}. Выберите другую модель."
                )
                return SELECTING_MODEL

        # Получение изображения
        photo_file = await update.message.photo[-1].get_file()

        # Проверка размера
        if photo_file.file_size > self.bot_config.max_image_size:
            await update.message.reply_text(
                f"Изображение слишком большое. Максимальный размер: "
                f"{self.bot_config.max_image_size // (1024*1024)}MB"
            )
            return PROCESSING_IMAGE

        # Сообщение о начале обработки
        processing_msg = await update.message.reply_text(
            "Обрабатываю изображение... Пожалуйста, подождите."
        )

        try:
            # Скачивание изображения
            image_bytes = await photo_file.download_as_bytearray()

            # Загрузка изображения
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Сохранение в сессии
            session["last_image"] = image

            # Преобразование
            direction = session.get("direction", "A_to_B")

            if direction == "A_to_B":
                result = self.inference.transform_A_to_B(image)
            elif direction == "B_to_A":
                result = self.inference.transform_B_to_A(image)
            elif direction == "A_to_B_to_A":
                results = self.inference.cycle_transform(
                    image, "A_to_B_to_A", return_all=True
                )
                result = results["recov_A"]
            elif direction == "B_to_A_to_B":
                results = self.inference.cycle_transform(
                    image, "B_to_A_to_B", return_all=True
                )
                result = results["recov_B"]
            else:
                result = self.inference.transform_A_to_B(image)

            # Конвертация в bytes для отправки
            output_buffer = io.BytesIO()
            result.save(output_buffer, format="PNG")
            output_buffer.seek(0)

            # Отправка результата
            await update.message.reply_photo(
                photo=output_buffer,
                caption=f"Преобразование завершено!\n"
                f"Модель: {session['model']}\n"
                f"Направление: {direction}",
            )

            # Удаление сообщения о обработке
            await processing_msg.delete()

            # Предложение нового преобразования
            keyboard = [
                [
                    InlineKeyboardButton(
                        "Новое преобразование", callback_data="new_transform"
                    ),
                    InlineKeyboardButton(
                        "Выбрать другую модель", callback_data="change_model"
                    ),
                ]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                "Что вы хотите сделать дальше?", reply_markup=reply_markup
            )

            return PROCESSING_IMAGE

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await processing_msg.edit_text(
                f"Ошибка при обработке изображения: {str(e)}"
            )
            return PROCESSING_IMAGE

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик получения документа (изображения как файла)"""
        document = update.message.document

        # Проверка типа файла
        if document.mime_type not in ["image/jpeg", "image/png", "image/bmp"]:
            await update.message.reply_text(
                "Пожалуйста, отправьте изображение в формате JPEG, PNG или BMP."
            )
            return PROCESSING_IMAGE

        # Проверка размера
        if document.file_size > self.bot_config.max_image_size:
            await update.message.reply_text(
                f"Файл слишком большой. Максимальный размер: "
                f"{self.bot_config.max_image_size // (1024*1024)}MB"
            )
            return PROCESSING_IMAGE

        # Заменяем документ на фото для обработки
        update.message.photo = [
            type("obj", (object,), {"file_size": document.file_size})()
        ]
        update.message.photo[-1].get_file = document.get_file

        return await self.handle_image(update, context)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик текстовых сообщений"""
        text = update.message.text

        if text.lower() in ["привет", "hello", "hi"]:
            await update.message.reply_text(
                "Привет! Отправьте мне изображение или используйте команды из меню."
            )
        else:
            await update.message.reply_text(
                "Я понимаю только изображения и команды. "
                "Используйте /help для получения списка команд."
            )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик ошибок"""
        logger.error(f"Update {update} caused error {context.error}")

        try:
            await update.message.reply_text(
                "Произошла ошибка. Пожалуйста, попробуйте еще раз."
            )
        except:
            pass

    def create_application(self) -> Application:
        """Создание и настройка приложения Telegram бота"""
        if not self.bot_config.token:
            raise ValueError("Telegram bot token is required. Get it from @BotFather")

        # Создание приложения
        application = Application.builder().token(self.bot_config.token).build()

        # Создание ConversationHandler с правильными настройками
        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("start", self.start_command),
                CallbackQueryHandler(self.button_callback, pattern="^model_"),
                CallbackQueryHandler(self.button_callback, pattern="^dir_"),
                CallbackQueryHandler(
                    self.button_callback, pattern="^(new_transform|change_model)"
                ),
            ],
            states={
                SELECTING_MODEL: [
                    CallbackQueryHandler(self.button_callback, pattern="^model_"),
                    CommandHandler("models", self.models_command),
                    CommandHandler("cancel", self.cancel_command),
                ],
                SELECTING_DIRECTION: [
                    CallbackQueryHandler(self.button_callback, pattern="^dir_"),
                    CommandHandler("direction", self.direction_command),
                    CommandHandler("cancel", self.cancel_command),
                ],
                PROCESSING_IMAGE: [
                    MessageHandler(filters.PHOTO, self.handle_image),
                    MessageHandler(filters.Document.IMAGE, self.handle_document),
                    CallbackQueryHandler(
                        self.button_callback, pattern="^(new_transform|change_model)"
                    ),
                    CommandHandler("cancel", self.cancel_command),
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_command),
            ],
            # Используем стандартные настройки (per_message=False по умолчанию)
            per_message=False,
        )

        # Добавление обработчиков вне ConversationHandler
        application.add_handler(conv_handler)
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        )

        # Обработчик ошибок
        application.add_error_handler(self.error_handler)

        # Данные бота
        application.bot_data["start_time"] = "Just started"
        application.bot_data["bot_instance"] = self

        return application

    def run(self):
        """Запуск бота"""
        if not self.bot_config.token:
            print("Error: Telegram bot token is required!")
            print("Get token from @BotFather and add it to .env file")
            return

        print("[Telegram Bot] Starting bot...")
        print(f"[Telegram Bot] Models available: {list(self.models.keys())}")
        print(f"[Telegram Bot] Using .env: {'Yes' if DOTENV_LOADED else 'No'}")

        # Создание приложения
        application = self.create_application()

        # Запуск бота с обработкой ошибок
        try:
            print("[Telegram Bot] Bot is running. Press Ctrl+C to stop.")
            print(
                "[Telegram Bot] Open Telegram and search for your bot to start using it."
            )

            # Запускаем бота
            application.run_polling(
                allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
            )

        except KeyboardInterrupt:
            print("\n[Telegram Bot] Bot stopped by user")
        except Exception as e:
            print(f"[Telegram Bot] Error: {e}")
            import traceback

            traceback.print_exc()


def load_env_variables():
    """Загрузка переменных окружения из .env файла"""
    env_vars = {}

    # Попытка загрузить из .env файла
    try:
        from dotenv import load_dotenv

        if load_dotenv():
            print("[Env] Loaded variables from .env file")
    except ImportError:
        pass

    # Чтение переменных
    env_vars["token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    env_vars["admin_ids"] = os.getenv("TELEGRAM_ADMIN_IDS", "")
    env_vars["allowed_users"] = os.getenv("TELEGRAM_ALLOWED_USERS", "")
    env_vars["max_image_size"] = os.getenv("TELEGRAM_MAX_IMAGE_SIZE", "10485760")
    env_vars["max_processing_time"] = os.getenv("TELEGRAM_MAX_PROCESSING_TIME", "30")
    env_vars["cleanup_temp_files"] = os.getenv("TELEGRAM_CLEANUP_TEMP_FILES", "true")
    env_vars["temp_dir"] = os.getenv("TELEGRAM_TEMP_DIR", "telegram_temp")
    env_vars["models_dir"] = os.getenv("TELEGRAM_MODELS_DIR", "checkpoints")
    env_vars["default_image_size"] = os.getenv("TELEGRAM_DEFAULT_IMAGE_SIZE", "256")

    return env_vars


def create_bot_config_from_env():
    """Создание конфигурации бота из переменных окружения"""
    env_vars = load_env_variables()

    # Парсинг списков
    def parse_list(value):
        if not value:
            return []
        return [int(id.strip()) for id in value.split(",") if id.strip().isdigit()]

    # Парсинг boolean
    def parse_bool(value):
        return value.lower() in ["true", "yes", "1", "t", "y"]

    # Парсинг integers
    def parse_int(value):
        try:
            return int(value)
        except:
            return 0

    config = BotConfig(
        token=env_vars["token"],
        admin_ids=parse_list(env_vars["admin_ids"]),
        allowed_users=parse_list(env_vars["allowed_users"]),
        max_image_size=parse_int(env_vars["max_image_size"]),
        max_processing_time=parse_int(env_vars["max_processing_time"]),
        cleanup_temp_files=parse_bool(env_vars["cleanup_temp_files"]),
        temp_dir=env_vars["temp_dir"],
        models_dir=env_vars["models_dir"],
        default_image_size=parse_int(env_vars["default_image_size"]),
    )

    return config


def create_env_template():
    """Создание шаблона .env файла"""
    env_template = """# Telegram Bot Configuration
# Get token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Admin user IDs (comma separated, optional)
TELEGRAM_ADMIN_IDS=123456789,987654321

# Allowed user IDs (comma separated, empty = all users)
TELEGRAM_ALLOWED_USERS=

# Maximum image size in bytes (default: 10MB)
TELEGRAM_MAX_IMAGE_SIZE=10485760

# Maximum processing time in seconds
TELEGRAM_MAX_PROCESSING_TIME=30

# Cleanup temporary files (true/false)
TELEGRAM_CLEANUP_TEMP_FILES=true

# Temporary directory for processing
TELEGRAM_TEMP_DIR=telegram_temp

# Directory with model files
TELEGRAM_MODELS_DIR=checkpoints

# Default image size for processing
TELEGRAM_DEFAULT_IMAGE_SIZE=256
"""

    return env_template


def main():
    """Основная функция запуска бота"""
    import argparse

    parser = argparse.ArgumentParser(description="CycleGAN Telegram Bot")
    parser.add_argument(
        "--create-env", action="store_true", help="Create .env template file"
    )
    parser.add_argument(
        "--token", type=str, default="", help="Telegram bot token (overrides .env file)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="",
        help="Directory with model files (overrides .env)",
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="Alternative .env file path"
    )

    args = parser.parse_args()

    # Создание .env файла если запрошено
    if args.create_env:
        env_template = create_env_template()
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_template)
        print("Created .env template file")
        print("Please edit .env file and add your bot token")
        return

    # Установка альтернативного .env файла
    if args.config_file:
        os.environ["DOTENV_CONFIG"] = args.config_file

    # Настройка бота
    bot_config = setup_bot()

    if bot_config is None:
        return

    # Переопределение токена если указан
    if args.token:
        bot_config.token = args.token

    # Переопределение директории моделей если указана
    if args.models_dir:
        bot_config.models_dir = args.models_dir

    # Создание бота
    try:
        bot = CycleGANTelegramBot(bot_config)
        bot.run()
    except Exception as e:
        print(f"Error starting bot: {e}")


def setup_bot():
    """Настройка и запуск бота"""
    # Проверяем наличие .env файла
    env_file = Path(".env")

    if not env_file.exists():
        print(".env file not found. Creating template...")

        # Создаем шаблон .env файла
        env_template = create_env_template()

        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_template)

        print("Created .env file template")
        print("Please edit .env file and add your bot token")
        print("Get token from @BotFather on Telegram")
        return None

    # Загружаем конфигурацию из .env
    bot_config = create_bot_config_from_env()

    # Проверяем токен
    if not bot_config.token or bot_config.token == "your_bot_token_here":
        print("Error: Telegram bot token is not set!")
        print("Edit .env file and set TELEGRAM_BOT_TOKEN variable")
        print("Get token from @BotFather on Telegram")
        return None

    return bot_config


if __name__ == "__main__":
    main()
