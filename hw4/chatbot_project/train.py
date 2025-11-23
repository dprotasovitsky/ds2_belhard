import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from config import config
from model import AdvancedNeuralNet, CUDAModelManager
from nltk_utils import bag_of_words, create_vocabulary, stem, tokenize
from torch.utils.data import DataLoader, Dataset
from utils import ModelManager, TrainingVisualizer

# Создаем директории если нужно
os.makedirs("models/trained_models", exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(config.TRAINING_LOG, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    try:
        with open(config.INTENTS_FILE, "r", encoding="utf-8") as f:
            intents = json.load(f)

        logger.info("Загрузка и подготовка данных...")

        all_patterns = []
        tags = []
        xy = []

        for intent in intents["intents"]:
            tag = intent["tag"]
            tags.append(tag)
            for pattern in intent["patterns"]:
                all_patterns.append(pattern)
                w = tokenize(pattern)
                xy.append((w, tag))

        # Создание словаря
        all_words = create_vocabulary(all_patterns, config.MIN_WORD_FREQ)
        tags = sorted(set(tags))

        logger.info(f"Найдено {len(xy)} примеров для обучения")
        logger.info(f"Уникальных слов: {len(all_words)}")
        logger.info(f"Теги: {tags}")

        # Создание тренировочных данных
        X_train = []
        y_train = []

        for pattern_sentence, tag in xy:
            bag = bag_of_words(pattern_sentence, all_words)
            X_train.append(bag)
            label = tags.index(tag)
            y_train.append(label)

        return np.array(X_train), np.array(y_train), all_words, tags, intents

    except FileNotFoundError:
        logger.error(f"Файл {config.INTENTS_FILE} не найден!")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка чтения JSON файла: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise


class ImprovedChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train_model():
    """Основная функция обучения"""
    logger.info("Начало процесса обучения...")

    # Логирование информации об устройстве
    config.log_device_info()

    # Загрузка данных
    X_train, y_train, all_words, tags, intents = load_and_prepare_data()

    # Преобразование данных в тензоры PyTorch
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)

    # Создание DataLoader с оптимизацией для CUDA
    dataset = ImprovedChatDataset(X_tensor, y_tensor)
    batch_size = config.BATCH_SIZE

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=(
            4 if torch.cuda.is_available() else 0
        ),  # Увеличиваем workers для GPU
        pin_memory=(
            True if torch.cuda.is_available() else False
        ),  # Ускоряет передачу на GPU
    )

    # Создание модели
    input_size = len(all_words)
    hidden_size = config.HIDDEN_SIZE
    output_size = len(tags)

    model = AdvancedNeuralNet(input_size, hidden_size, output_size)

    # Оптимизация модели для CUDA
    model, scaler = CUDAModelManager.optimize_for_cuda(model)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01
    )

    # Планировщик скорости обучения
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        epochs=config.NUM_EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    logger.info(
        "Начало обучения с использованием CUDA..."
        if torch.cuda.is_available()
        else "Начало обучения на CPU..."
    )

    # Переменные для отслеживания прогресса
    start_time = time.time()
    losses = []

    # Обучение модели
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Перемещение данных на GPU если доступно
            data, target = data.to(config.DEVICE, non_blocking=True), target.to(
                config.DEVICE, non_blocking=True
            )

            optimizer.zero_grad()

            if (
                scaler and torch.cuda.is_available()
            ):  # Использование mixed precision для CUDA
                with torch.amp.autocast("cuda"):
                    output = model(data)
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:  # Стандартный backward для CPU
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Логирование каждые 100 эпох
        if (epoch + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            gpu_allocated, gpu_cached = CUDAModelManager.get_gpu_memory_info()

            log_message = (
                f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
                f"Loss: {avg_loss:.4f}, "
                f"Time: {elapsed_time:.2f}s"
            )

            if torch.cuda.is_available():
                log_message += f", GPU Memory: {gpu_allocated:.2f}/{gpu_cached:.2f} GB"

            logger.info(log_message)

    # Сохранение модели
    save_model_with_cuda_info(model, all_words, tags, losses)

    total_time = time.time() - start_time
    logger.info(f"Обучение завершено за {total_time:.2f} секунд")

    return model, losses


def save_model_with_cuda_info(model, all_words, tags, losses):
    """Сохранение модели с информацией о CUDA"""

    model_data = {
        "model_state": model.state_dict(),
        "input_size": len(all_words),
        "hidden_size": config.HIDDEN_SIZE,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags,
        "cuda_info": {
            "trained_on_cuda": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            ),
        },
        "training_config": {
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "num_epochs": config.NUM_EPOCHS,
            "final_loss": losses[-1] if losses else 0.0,
        },
        "training_history": {"losses": losses},
    }

    torch.save(model_data, config.MODEL_FILE)
    logger.info(f"Модель сохранена с информацией о CUDA: {config.MODEL_FILE}")


def evaluate_model(model, test_data=None):
    """Оценка модели (можно расширить для тестовых данных)"""
    model.eval()

    # Здесь можно добавить оценку на тестовых данных
    logger.info("Оценка модели завершена (тестовые данные не предоставлены)")


def main():
    """Главная функция"""
    try:
        logger.info("Запуск обучения нейронной сети...")
        model, losses = train_model()

        # Визуализация процесса обучения
        visualizer = TrainingVisualizer()
        for loss in losses:
            visualizer.update(loss)
        visualizer.plot_training_progress()

        logger.info("Обучение успешно завершено!")

    except Exception as e:
        logger.error(f"Ошибка во время обучения: {e}")
        raise


if __name__ == "__main__":
    main()
