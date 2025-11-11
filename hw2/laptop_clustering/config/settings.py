import logging
import os
from datetime import datetime


class Config:
    """Конфигурация проекта"""

    # Пути
    DATA_PATH = "laptopCleanData.csv"
    RESULTS_DIR = "results"
    LOGS_DIR = "logs"

    # Параметры кластеризации
    DEFAULT_N_CLUSTERS = 4
    DBSCAN_MIN_SAMPLES = 5
    UMAP_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1

    # Настройки логирования
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def setup_directories(cls):
        """Создание необходимых директорий"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)


class Logger:
    """Настройка логирования в один файл"""

    _initialized = False
    _log_file = None

    @classmethod
    def initialize_logger(cls):
        """Инициализация логгера (вызывается один раз)"""
        if cls._initialized:
            return

        Config.setup_directories()

        # Создаем имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cls._log_file = os.path.join(
            Config.LOGS_DIR, f"laptop_clustering_{timestamp}.log"
        )

        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(Config.LOG_LEVEL)

        # Обработчик для файла
        file_handler = logging.FileHandler(cls._log_file, encoding="utf-8")
        file_handler.setLevel(Config.LOG_LEVEL)

        # Обработчик для консоли
        console_handler = logging.StreamHandler()
        console_handler.setLevel(Config.LOG_LEVEL)

        # Форматтер
        formatter = logging.Formatter(Config.LOG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Очищаем существующие обработчики и добавляем новые
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        cls._initialized = True
        root_logger.info(f"Логирование инициализировано. Файл: {cls._log_file}")

    @staticmethod
    def get_logger(name):
        """Получение логгера с указанным именем"""
        Logger.initialize_logger()
        return logging.getLogger(name)
