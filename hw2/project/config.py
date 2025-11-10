import os

import torch


# Конфигурация проекта
class Config:
    # Базовые пути
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Пути
    NLTK_PATH = os.path.join(BASE_DIR, "nltk_data")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "triplet_ae_russian_sentiment.pth")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    PLOTS_DIR = os.path.join(BASE_DIR, "reports", "plots")

    # Файлы
    LOG_FILE = "training.log"
    REPORT_FILE = "training_report.html"

    # Полные пути к файлам
    @classmethod
    def get_log_path(cls):
        return os.path.join(cls.LOGS_DIR, cls.LOG_FILE)

    @classmethod
    def get_report_path(cls):
        return os.path.join(cls.REPORTS_DIR, cls.REPORT_FILE)

    # Параметры модели
    EMBEDDING_DIM = 128
    USE_TFIDF = True
    VOCAB_SIZE = 5000

    # Параметры обучения
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2

    # Параметры классификаторов
    CLASSIFIERS = {
        "random_forest": {
            "n_estimators": 100,
            "random_state": 42,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        },
        "catboost": {
            "iterations": 100,
            "random_seed": 42,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "border_count": 128,
            "verbose": False,
            "thread_count": -1,
        },
        "lightgbm": {
            "n_estimators": 100,
            "random_state": 42,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
        },
    }

    # Настройки устройства
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Настройки логирования
    LOG_LEVEL = "INFO"

    @classmethod
    def create_directories(cls):
        """Создает все необходимые директории"""
        directories = [cls.NLTK_PATH, cls.LOGS_DIR, cls.REPORTS_DIR, cls.PLOTS_DIR]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Создана директория: {directory}")
