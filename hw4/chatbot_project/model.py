import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

logger = logging.getLogger(__name__)


class AdvancedNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2):
        super(AdvancedNeuralNet, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(hidden_size, num_classes)

        # Логирование информации об устройстве
        logger.info(f"Создана нейронная сеть на устройстве: {config.DEVICE}")
        logger.info(
            f"Архитектура: input_size={input_size}, hidden_size={hidden_size}, "
            f"num_classes={num_classes}"
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.layer1(x)))
        out = self.dropout1(out)

        out = F.relu(self.bn2(self.layer2(out)))
        out = self.dropout2(out)

        out = F.relu(self.bn3(self.layer3(out)))
        out = self.dropout3(out)

        out = self.output(out)
        return out


class CUDAModelManager:
    """Менеджер для работы с моделью на CUDA"""

    @staticmethod
    def optimize_for_cuda(model):
        """Оптимизация модели для работы на CUDA"""
        if torch.cuda.is_available():
            # Включение cuDNN авотюнера для оптимизации сверточных операций
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            # Использование mixed precision для ускорения вычислений
            scaler = torch.amp.GradScaler("cuda")

            model = model.to(config.DEVICE)
            logger.info("Модель оптимизирована для CUDA с cuDNN и mixed precision")
            return model, scaler

        logger.info("Модель работает на CPU")
        return model, None

    @staticmethod
    def get_gpu_memory_info():
        """Получение информации о памяти GPU"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            return allocated, cached
        return 0, 0
