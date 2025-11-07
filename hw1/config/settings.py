import torch


class Config:
    """Настройки приложения"""

    # Пути к данным
    DATA_FILE = "household_power_consumption.txt"
    DATABASE_PATH = "energy_consumption.db"

    # Настройки LSTM
    DEFAULT_SEQUENCE_LENGTH = 24
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_OPTUNA_TRIALS = 30

    # Настройки устройства
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Настройки визуализации
    PLOT_THEME = "plotly_white"
    SAMPLE_SIZE = 10000

    # Настройки ускорения Optuna
    OPTUNA_OPTIMIZATIONS = {
        "fast_training_epochs": 30,
        "fast_training_batches": 50,
        "fast_validation_batches": 20,
        "min_data_samples": 5000,
        "cache_models": True,
    }

    # Настройки моделей
    MODEL_SETTINGS = {
        "save_dir": "saved_models",
        "auto_save": True,
        "max_saved_models": 10,
    }

    # Цвета
    COLORS = {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "success": "#2ca02c",
        "danger": "#d62728",
        "warning": "#ffbb00",
    }

    # Информация о данных
    DATA_INFO = {
        "period": "2006-2010 гг.",
        "frequency": "минутные данные",
        "metrics": [
            "Активная мощность (kW)",
            "Реактивная мощность (kVAR)",
            "Напряжение (V)",
            "Сила тока (A)",
            "Суб-метр 1 (кухня)",
            "Суб-метр 2 (прачечная)",
            "Суб-метр 3 (водонагреватель)",
        ],
    }


config = Config()
