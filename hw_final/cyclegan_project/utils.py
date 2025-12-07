import io
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class TensorBoardLogger:
    """Улучшенный логгер для TensorBoard"""

    def __init__(self, log_dir: str, config: Optional[Any] = None):
        """
        Args:
            log_dir: Директория для логов
            config: Конфигурация модели
        """
        self.writer = SummaryWriter(log_dir, flush_secs=10)
        self.log_dir = log_dir
        self.step = 0
        self.config = config

        print(f"[TensorBoard] Log directory: {log_dir}")
        print(f"[TensorBoard] Run command: tensorboard --logdir={log_dir}")

        # Логируем конфигурацию если она есть
        if config is not None:
            self.log_config(config)

    def increment_step(self):
        """Увеличение шага"""
        self.step += 1

    def set_step(self, step: int):
        """Установка текущего шага"""
        self.step = step

    # ========== Скалярные значения ==========

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Логирование скалярного значения"""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_value_dict: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Логирование нескольких скаляров"""
        step = step if step is not None else self.step
        self.writer.add_scalars(main_tag, tag_value_dict, step)

    # ========== Изображения ==========

    def log_image(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        step: Optional[int] = None,
        dataformats: str = "CHW",
    ):
        """Логирование одного изображения"""
        step = step if step is not None else self.step
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)

    def log_images_grid(
        self, tag: str, images: torch.Tensor, nrow: int = 8, step: Optional[int] = None
    ):
        """Логирование сетки изображений"""
        step = step if step is not None else self.step
        grid = make_grid(images, nrow=nrow, normalize=True, padding=2)
        self.writer.add_image(tag, grid, step)

    def log_cyclegan_images(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        recov_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        recov_B: torch.Tensor,
        step: Optional[int] = None,
        max_images: int = 4,
    ):
        """Специализированный метод для логирования CycleGAN"""
        step = step if step is not None else self.step

        with torch.no_grad():
            # Денормализация
            def denorm(tensor):
                return torch.clamp(tensor * 0.5 + 0.5, 0, 1)

            # Ограничиваем количество изображений
            n = min(max_images, real_A.size(0))

            # Создаем сравнение A->B
            comparison_A2B = []
            for i in range(n):
                row = torch.cat(
                    [
                        denorm(real_A[i].cpu()),
                        denorm(fake_B[i].cpu()),
                        denorm(recov_A[i].cpu()),
                    ],
                    dim=2,
                )  # Конкатенация по ширине
                comparison_A2B.append(row)

            # Создаем сравнение B->A
            comparison_B2A = []
            for i in range(n):
                row = torch.cat(
                    [
                        denorm(real_B[i].cpu()),
                        denorm(fake_A[i].cpu()),
                        denorm(recov_B[i].cpu()),
                    ],
                    dim=2,
                )  # Конкатенация по ширине
                comparison_B2A.append(row)

            # Объединяем все в одну сетку
            all_images = comparison_A2B + comparison_B2A
            if all_images:
                grid = make_grid(all_images, nrow=n, padding=2, normalize=False)
                self.writer.add_image("CycleGAN/Results", grid, step)

    # ========== Модели и графы ==========

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Логирование графа модели"""
        self.writer.add_graph(model, input_tensor)

    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Логирование гистограммы значений"""
        step = step if step is not None else self.step
        self.writer.add_histogram(tag, values, step)

    def log_parameters(self, model: torch.nn.Module, step: Optional[int] = None):
        """Логирование параметров модели"""
        step = step if step is not None else self.step
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"Parameters/{name}", param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(
                        f"Gradients/{name}", param.grad.data, step
                    )

    # ========== Текст и конфигурация ==========

    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Логирование текста"""
        step = step if step is not None else self.step
        self.writer.add_text(tag, text, step)

    def log_config(self, config: Any):
        """Логирование конфигурации"""
        # Логируем как текст
        config_text = "## Model Configuration\n\n"
        for key, value in config.get_hparams_dict().items():
            config_text += f"- **{key}**: {value}\n"

        self.writer.add_text("Configuration", config_text)

        # Логируем гиперпараметры (только простые типы)
        hparams = config.get_hparams_dict()

        # Создаем фиктивные метрики для hparams
        dummy_metrics = {"hparam/dummy_loss": 0.0, "hparam/dummy_accuracy": 0.0}

        try:
            self.writer.add_hparams(hparams, dummy_metrics)
        except Exception as e:
            print(f"[Warning] Could not log hparams: {e}")
            # Альтернатива: логируем как текст
            self.writer.add_text("Hyperparameters", str(hparams))

    def log_learning_rates(
        self, optimizer: torch.optim.Optimizer, step: Optional[int] = None
    ):
        """Логирование learning rates"""
        step = step if step is not None else self.step
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f"Learning_Rate/group_{i}", param_group["lr"], step)

    # ========== Визуализации ==========

    def log_loss_curves(self, losses: Dict[str, list], step: Optional[int] = None):
        """Логирование кривых потерь"""
        step = step if step is not None else self.step

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        # Основные потери
        if "G_total" in losses and "D_total" in losses:
            ax = axes[0]
            ax.plot(losses["G_total"], label="Generator")
            ax.plot(losses["D_total"], label="Discriminator")
            ax.set_title("Generator vs Discriminator Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Cycle losses
        if "cycle_A" in losses and "cycle_B" in losses:
            ax = axes[1]
            ax.plot(losses["cycle_A"], label="Cycle A")
            ax.plot(losses["cycle_B"], label="Cycle B")
            ax.set_title("Cycle Consistency Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Identity losses
        if "identity_A" in losses and "identity_B" in losses:
            ax = axes[2]
            ax.plot(losses["identity_A"], label="Identity A")
            ax.plot(losses["identity_B"], label="Identity B")
            ax.set_title("Identity Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # GAN losses
        if "GAN_A" in losses and "GAN_B" in losses:
            ax = axes[3]
            ax.plot(losses["GAN_A"], label="GAN A")
            ax.plot(losses["GAN_B"], label="GAN B")
            ax.set_title("GAN Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Конвертируем в тензор для TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        image = Image.open(buf)
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)

        self.writer.add_image("Loss_Curves", image_tensor, step)
        plt.close(fig)

    # ========== Утилиты ==========

    def log_epoch_summary(
        self, epoch: int, losses: Dict[str, float], lr: float, time_elapsed: float
    ):
        """Логирование сводки по эпохе"""
        summary_text = f"## Epoch {epoch} Summary\n\n"
        summary_text += f"**Time**: {time_elapsed:.2f}s\n"
        summary_text += f"**Learning Rate**: {lr:.6f}\n\n"
        summary_text += "### Losses:\n"

        for key, value in losses.items():
            summary_text += f"- {key}: {value:.4f}\n"

        self.writer.add_text(f"Epochs/Epoch_{epoch}", summary_text, epoch)

    def close(self):
        """Закрытие логгера"""
        self.writer.flush()
        self.writer.close()
        print(f"[TensorBoard] Logger closed. Logs saved to: {self.log_dir}")


# Вспомогательные функции
def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Денормализация тензора из [-1, 1] в [0, 1]"""
    return torch.clamp(tensor * 0.5 + 0.5, 0, 1)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Конвертация тензора в numpy массив для сохранения"""
    img = denormalize(tensor)
    img = img.cpu().numpy().transpose(1, 2, 0) * 255
    return img.astype(np.uint8)


def save_image_grid(images: torch.Tensor, filename: str, nrow: int = 8):
    """Сохранение сетки изображений в файл"""
    from torchvision.utils import save_image

    # Денормализуем
    images = denormalize(images)

    # Создаем сетку
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)

    # Сохраняем
    save_image(grid, filename)
