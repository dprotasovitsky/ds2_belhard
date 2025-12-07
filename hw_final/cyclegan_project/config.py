import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class Config:
    """Конфигурация для CycleGAN"""

    # Пути данных
    dataset_path: str = "datasets/photo2comics"
    image_size: int = 256

    # Гиперпараметры обучения
    batch_size: int = 1
    num_workers: int = 4
    num_epochs: int = 100
    decay_epoch: int = 50

    # Параметры модели
    in_channels: int = 3
    out_channels: int = 3
    num_residual_blocks: int = 9

    # Оптимизатор
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

    # Коэффициенты потерь
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5

    # Пути для сохранения
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    sample_dir: str = "samples"

    # Логирование
    log_interval: int = 50
    sample_interval: int = 200
    checkpoint_interval: int = 10
    validation_interval: int = 5

    # Устройство (не логируется в hparams)
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self):
        """Инициализация после создания объекта"""
        # Создание директорий
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sample_dir).mkdir(parents=True, exist_ok=True)

        # Получаем torch.device
        self.torch_device = torch.device(self.device)

    def get_hparams_dict(self) -> Dict[str, Any]:
        """Получение гиперпараметров в виде словаря для TensorBoard"""
        hparams = {}
        for key, value in self.__dict__.items():
            # Пропускаем служебные поля
            if key.startswith("_") or key in ["torch_device", "device"]:
                continue
            # Берем только простые типы
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
        return hparams

    def display(self):
        """Отображение конфигурации"""
        print("\n" + "=" * 60)
        print("CYCLEGAN CONFIGURATION".center(60))
        print("=" * 60)

        sections = {
            "Data": ["dataset_path", "image_size"],
            "Training": ["batch_size", "num_workers", "num_epochs", "decay_epoch"],
            "Model": ["in_channels", "out_channels", "num_residual_blocks"],
            "Optimizer": ["lr", "beta1", "beta2"],
            "Loss Coefficients": ["lambda_cycle", "lambda_identity"],
            "Paths": ["checkpoint_dir", "log_dir", "sample_dir"],
            "Logging": [
                "log_interval",
                "sample_interval",
                "checkpoint_interval",
                "validation_interval",
            ],
            "Device": ["device"],
        }

        for section_name, params in sections.items():
            print(f"\n{section_name}:")
            print("-" * 40)
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    print(f"  {param:25}: {value}")

        print("\n" + "=" * 60)

    def save(self, path: str):
        """Сохранение конфигурации в файл"""
        with open(path, "w") as f:
            json.dump(self.get_hparams_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Создание конфигурации из словаря"""
        return cls(**data)
