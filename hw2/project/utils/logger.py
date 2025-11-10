import logging
import os
import sys
from datetime import datetime

from config import Config


class Logger:
    _instance = None

    def __init__(self):
        self.logger = logging.getLogger("TripletAutoencoder")
        self.logger.setLevel(getattr(logging, Config.LOG_LEVEL))

        # Убедимся, что директория для логов существует
        Config.create_directories()

        # Форматтер
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Файловый обработчик
        log_path = Config.get_log_path()
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Добавляем обработчики
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Логируем информацию о настройке
        self.logger.info(f"Логирование инициализировано. Файл логов: {log_path}")
        self.logger.info(f"Уровень логирования: {Config.LOG_LEVEL}")

    @classmethod
    def get_logger(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.logger


def log_execution_time(func):
    """Декоратор для логирования времени выполнения функций"""

    def wrapper(*args, **kwargs):
        logger = Logger.get_logger()
        start_time = datetime.now()
        logger.info(f"Начало выполнения {func.__name__}")

        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(
                f"Завершение {func.__name__}. Время выполнения: {execution_time:.2f} сек."
            )
            return result
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(
                f"Ошибка в {func.__name__} после {execution_time:.2f} сек.: {e}"
            )
            raise

    return wrapper
