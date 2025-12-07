import itertools
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm


class CycleGANTrainer:
    """Тренер CycleGAN с полным протоколированием в TensorBoard"""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config.device

        # Инициализация моделей
        from models import CycleGANModel

        self.model = CycleGANModel(config)
        self.G_AB = self.model.G_AB
        self.G_BA = self.model.G_BA
        self.D_A = self.model.D_A
        self.D_B = self.model.D_B

        # Mixed precision
        self.scaler = GradScaler(device=self.device, enabled=self.device == "cuda")

        # Функции потерь
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Оптимизаторы
        self.optimizer_G = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_D_A = optim.Adam(
            self.D_A.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )
        self.optimizer_D_B = optim.Adam(
            self.D_B.parameters(), lr=config.lr, betas=(config.beta1, config.beta2)
        )

        # Schedulers
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=lambda epoch: 1.0
            - max(0, epoch - config.decay_epoch)
            / float(config.num_epochs - config.decay_epoch + 1),
        )
        self.scheduler_D_A = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=lambda epoch: 1.0
            - max(0, epoch - config.decay_epoch)
            / float(config.num_epochs - config.decay_epoch + 1),
        )
        self.scheduler_D_B = optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=lambda epoch: 1.0
            - max(0, epoch - config.decay_epoch)
            / float(config.num_epochs - config.decay_epoch + 1),
        )

        # История для графиков
        self.history = {
            "G_total": [],
            "D_total": [],
            "D_A": [],
            "D_B": [],
            "cycle_A": [],
            "cycle_B": [],
            "identity_A": [],
            "identity_B": [],
            "GAN_A": [],
            "GAN_B": [],
        }

        # Счетчики
        self.global_step = 0
        self.best_loss = float("inf")

        print(f"[Trainer] Initialized on {self.device}")
        if self.device == "cuda":
            print(f"[Trainer] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[Trainer] CUDA version: {torch.version.cuda}")

    def train_epoch(self, dataloader, epoch):
        """Обучение на одной эпохе"""
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        epoch_losses = {k: 0.0 for k in self.history.keys()}
        num_batches = len(dataloader)

        # Прогресс-бар
        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch:03d}/{self.config.num_epochs}",
            unit="batch",
            leave=False,
            ncols=100,
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

        start_time = time.time()

        for batch_idx, batch in enumerate(pbar):
            # Подготовка данных
            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)
            batch_size = real_A.size(0)

            # Adversarial ground truths
            valid = torch.ones((batch_size, 1, 30, 30), device=self.device)
            fake = torch.zeros((batch_size, 1, 30, 30), device=self.device)

            # ========== Train Generators ==========
            self.optimizer_G.zero_grad()

            with autocast(device_type=self.device, enabled=self.device == "cuda"):
                # Identity loss
                loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
                loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
                loss_identity = (loss_id_A + loss_id_B) / 2

                # GAN loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Cycle loss
                recov_A = self.G_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)
                recov_B = self.G_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = (
                    loss_GAN
                    + self.config.lambda_cycle * loss_cycle
                    + self.config.lambda_identity * loss_identity
                )

            self.scaler.scale(loss_G).backward()
            self.scaler.step(self.optimizer_G)

            # ========== Train Discriminators ==========
            # Discriminator A
            self.optimizer_D_A.zero_grad()

            with autocast(device_type=self.device, enabled=self.device == "cuda"):
                loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
                loss_fake_A = self.criterion_GAN(self.D_A(fake_A.detach()), fake)
                loss_D_A = (loss_real_A + loss_fake_A) / 2

            self.scaler.scale(loss_D_A).backward()
            self.scaler.step(self.optimizer_D_A)

            # Discriminator B
            self.optimizer_D_B.zero_grad()

            with autocast(device_type=self.device, enabled=self.device == "cuda"):
                loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
                loss_fake_B = self.criterion_GAN(self.D_B(fake_B.detach()), fake)
                loss_D_B = (loss_real_B + loss_fake_B) / 2

            self.scaler.scale(loss_D_B).backward()
            self.scaler.step(self.optimizer_D_B)

            # Update scaler
            self.scaler.update()

            # Накопление лоссов
            losses = {
                "G_total": loss_G.item(),
                "D_total": (loss_D_A.item() + loss_D_B.item()) / 2,
                "D_A": loss_D_A.item(),
                "D_B": loss_D_B.item(),
                "cycle_A": loss_cycle_A.item(),
                "cycle_B": loss_cycle_B.item(),
                "identity_A": loss_id_A.item(),
                "identity_B": loss_id_B.item(),
                "GAN_A": loss_GAN_BA.item(),
                "GAN_B": loss_GAN_AB.item(),
            }

            for key in losses:
                epoch_losses[key] += losses[key]

            # Логирование на уровне батча
            if batch_idx % self.config.log_interval == 0:
                for key, value in losses.items():
                    self.logger.log_scalar(f"Batch/{key}", value, self.global_step)

                # Логирование learning rates
                self.logger.log_scalar(
                    "LR/Generator",
                    self.optimizer_G.param_groups[0]["lr"],
                    self.global_step,
                )
                self.logger.log_scalar(
                    "LR/Discriminator_A",
                    self.optimizer_D_A.param_groups[0]["lr"],
                    self.global_step,
                )
                self.logger.log_scalar(
                    "LR/Discriminator_B",
                    self.optimizer_D_B.param_groups[0]["lr"],
                    self.global_step,
                )

            # Логирование изображений
            if batch_idx % self.config.sample_interval == 0:
                self.logger.log_cyclegan_images(
                    real_A,
                    fake_B,
                    recov_A,
                    real_B,
                    fake_A,
                    recov_B,
                    step=self.global_step,
                    max_images=4,
                )

            # Обновление прогресс-бара
            avg_G = epoch_losses["G_total"] / (batch_idx + 1)
            avg_D = epoch_losses["D_total"] / (batch_idx + 1)

            pbar.set_postfix(
                {
                    "G": f"{avg_G:.3f}",
                    "D": f"{avg_D:.3f}",
                    "Cyc": f'{epoch_losses["cycle_A"]/(batch_idx+1):.3f}',
                }
            )

            self.global_step += 1

        # Вычисление средних лоссов за эпоху
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            self.history[key].append(epoch_losses[key])

        epoch_time = time.time() - start_time

        # Логирование на уровне эпохи
        self._log_epoch_summary(epoch, epoch_losses, epoch_time)

        return epoch_losses, epoch_time

    def _log_epoch_summary(self, epoch, losses, epoch_time):
        """Логирование сводки по эпохе"""
        # Основные потери
        for key, value in losses.items():
            self.logger.log_scalar(f"Epoch/{key}", value, epoch)

        # Время
        self.logger.log_scalar("Time/Epoch", epoch_time, epoch)

        # Learning rates
        self.logger.log_scalar(
            "LR/Epoch_Generator", self.optimizer_G.param_groups[0]["lr"], epoch
        )

        # Создаем текстовую сводку
        summary = (
            f"Epoch {epoch} | "
            f"Time: {epoch_time:.1f}s | "
            f"G: {losses['G_total']:.3f} | "
            f"D: {losses['D_total']:.3f} | "
            f"Cycle: {losses['cycle_A']:.3f} | "
            f"LR: {self.optimizer_G.param_groups[0]['lr']:.6f}"
        )

        self.logger.log_text("Epochs/Summary", summary, epoch)

        print(f"[Epoch {epoch:3d}] {summary}")

    @torch.no_grad()
    def validate(self, dataloader, epoch):
        """Валидация"""
        self.G_AB.eval()
        self.G_BA.eval()

        batch = next(iter(dataloader))
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        # Логирование валидационных изображений
        self.logger.log_cyclegan_images(
            real_A, fake_B, recov_A, real_B, fake_A, recov_B, step=epoch, max_images=8
        )

        # Вычисление метрик
        metrics = self._compute_metrics(real_A, recov_A, real_B, recov_B)

        # Логирование метрик
        for key, value in metrics.items():
            self.logger.log_scalar(f"Metrics/{key}", value, epoch)

        return metrics

    def _compute_metrics(self, real_A, recov_A, real_B, recov_B):
        """Вычисление метрик качества"""
        metrics = {}

        # PSNR (Peak Signal-to-Noise Ratio)
        def psnr(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float("inf")
            return 20 * torch.log10(2.0 / torch.sqrt(mse)).item()

        metrics["PSNR_A"] = psnr(real_A, recov_A)
        metrics["PSNR_B"] = psnr(real_B, recov_B)

        # SSIM (Structural Similarity) - упрощенная версия
        def ssim_simple(img1, img2):
            C1 = 0.01**2
            C2 = 0.03**2

            mu1 = torch.mean(img1)
            mu2 = torch.mean(img2)
            sigma1 = torch.std(img1)
            sigma2 = torch.std(img2)
            sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

            return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2)
            ).item()

        metrics["SSIM_A"] = ssim_simple(real_A, recov_A)
        metrics["SSIM_B"] = ssim_simple(real_B, recov_B)

        return metrics

    def train(self, train_loader, test_loader):
        """Основной цикл обучения"""
        print(f"\n[Training] Starting training for {self.config.num_epochs} epochs...")
        print(f"[Training] Train samples: {len(train_loader.dataset)}")
        print(f"[Training] Test samples: {len(test_loader.dataset)}")
        print(f"[Training] Batch size: {self.config.batch_size}")
        print(f"[Training] Log interval: {self.config.log_interval}")
        print(f"[Training] Checkpoint interval: {self.config.checkpoint_interval}\n")

        # Логируем граф модели
        if self.global_step == 0:
            dummy_input = torch.randn(
                1, 3, self.config.image_size, self.config.image_size, device=self.device
            )
            self.logger.log_model_graph(self.G_AB, dummy_input)

        for epoch in range(1, self.config.num_epochs + 1):
            # Обучение на одной эпохе
            epoch_losses, epoch_time = self.train_epoch(train_loader, epoch)

            # Валидация
            if epoch % self.config.validation_interval == 0:
                metrics = self.validate(test_loader, epoch)
                print(
                    f"[Validation] PSNR_A: {metrics['PSNR_A']:.2f} dB, "
                    f"PSNR_B: {metrics['PSNR_B']:.2f} dB"
                )

            # Обновление learning rate
            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()

            # Сохранение чекпоинта
            if epoch % self.config.checkpoint_interval == 0:
                self.save_checkpoint(epoch, epoch_losses["G_total"])

            # Сохранение лучшей модели
            if epoch_losses["G_total"] < self.best_loss:
                self.best_loss = epoch_losses["G_total"]
                self.save_checkpoint(epoch, epoch_losses["G_total"], best=True)
                print(f"[Checkpoint] Best model saved (loss: {self.best_loss:.4f})")

            # Логирование графиков потерь
            if epoch % 10 == 0:
                self.logger.log_loss_curves(self.history, step=epoch)

        # Финальное логирование
        self.logger.log_loss_curves(self.history, step=self.config.num_epochs)
        self.save_checkpoint(
            self.config.num_epochs, epoch_losses["G_total"], final=True
        )

        print("\n[Training] Training completed!")
        print(f"[Training] Best loss: {self.best_loss:.4f}")
        print(f"[Training] Logs saved to: {self.logger.log_dir}")

    def save_checkpoint(self, epoch, loss, best=False, final=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            "epoch": epoch,
            "G_AB_state_dict": self.G_AB.state_dict(),
            "G_BA_state_dict": self.G_BA.state_dict(),
            "D_A_state_dict": self.D_A.state_dict(),
            "D_B_state_dict": self.D_B.state_dict(),
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "optimizer_D_A_state_dict": self.optimizer_D_A.state_dict(),
            "optimizer_D_B_state_dict": self.optimizer_D_B.state_dict(),
            "scheduler_G_state_dict": self.scheduler_G.state_dict(),
            "scheduler_D_A_state_dict": self.scheduler_D_A.state_dict(),
            "scheduler_D_B_state_dict": self.scheduler_D_B.state_dict(),
            "loss": loss,
            "history": self.history,
            "config": self.config.get_hparams_dict(),
            "global_step": self.global_step,
        }

        if best:
            filename = f"{self.config.checkpoint_dir}/cyclegan_best.pth"
        elif final:
            filename = f"{self.config.checkpoint_dir}/cyclegan_final.pth"
        else:
            filename = f"{self.config.checkpoint_dir}/cyclegan_epoch_{epoch:03d}.pth"

        torch.save(checkpoint, filename)

        # Также сохраняем генераторы отдельно для инференса
        if best or final:
            torch.save(
                self.G_AB.state_dict(),
                f"{self.config.checkpoint_dir}/G_AB_{epoch:03d}.pth",
            )
            torch.save(
                self.G_BA.state_dict(),
                f"{self.config.checkpoint_dir}/G_BA_{epoch:03d}.pth",
            )

    def load_checkpoint(self, checkpoint_path):
        """Загрузка чекпоинта"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
        self.G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
        self.D_A.load_state_dict(checkpoint["D_A_state_dict"])
        self.D_B.load_state_dict(checkpoint["D_B_state_dict"])

        self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        self.optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A_state_dict"])
        self.optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B_state_dict"])

        if "scheduler_G_state_dict" in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint["scheduler_G_state_dict"])
            self.scheduler_D_A.load_state_dict(checkpoint["scheduler_D_A_state_dict"])
            self.scheduler_D_B.load_state_dict(checkpoint["scheduler_D_B_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        epoch = checkpoint.get("epoch", 0)
        loss = checkpoint.get("loss", 0)

        print(
            f"[Checkpoint] Loaded checkpoint from epoch {epoch} "
            f"(loss: {loss:.4f}, step: {self.global_step})"
        )

        return epoch
