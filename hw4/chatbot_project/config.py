import logging

import torch


class Config:
    # Автоматическое определение устройства: CUDA или CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Параметры модели с учетом использования GPU
    BATCH_SIZE = 32 if torch.cuda.is_available() else 8
    HIDDEN_SIZE = 32 if torch.cuda.is_available() else 16
    NUM_EPOCHS = 2000 if torch.cuda.is_available() else 1000
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2

    # Параметры обработки текста
    MAX_SEQUENCE_LENGTH = 20
    MIN_WORD_FREQ = 1

    # Пути к файлам
    INTENTS_FILE = "intents.json"
    MODEL_FILE = "models/trained_models/chatbot_model.pth"
    TRAINING_LOG = "training.log"
    CHAT_LOG = "chat_history.log"

    # Настройки чата
    CONFIDENCE_THRESHOLD = 0.7
    MAX_RESPONSE_CHOICES = 3

    # Логирование
    LOG_LEVEL = "INFO"

    @classmethod
    def log_device_info(cls):
        """Логирование информации об используемом устройстве"""
        logger = logging.getLogger(__name__)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_count = torch.cuda.device_count()
            logger.info(f"Используется GPU: {gpu_name}")
            logger.info(f"Количество GPU: {gpu_count}")
            logger.info(f"Объем памяти GPU: {gpu_memory:.2f} GB")
            logger.info(f"CUDA версия: {torch.version.cuda}")
        else:
            logger.info("CUDA не доступна, используется CPU")
            logger.info(
                "Для ускорения обучения установите CUDA-совместимую видеокарту и драйверы"
            )


config = Config()
