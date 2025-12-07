import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class CycleGANtester:
    """Класс для тестирования обученной модели CycleGAN"""

    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = config.torch_device
        self.checkpoint_path = checkpoint_path

        # Загрузка моделей
        from models import Discriminator, Generator

        self.G_AB = Generator(
            config.in_channels, config.out_channels, config.num_residual_blocks
        ).to(self.device)
        self.G_BA = Generator(
            config.in_channels, config.out_channels, config.num_residual_blocks
        ).to(self.device)

        # Загрузка чекпоинта
        self.load_checkpoint(checkpoint_path)

        # Создание директорий для результатов
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)

        # Поддиректории
        self.images_dir = self.results_dir / "images"
        self.metrics_dir = self.results_dir / "metrics"
        self.plots_dir = self.results_dir / "plots"

        for dir_path in [self.images_dir, self.metrics_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)

        print(f"[Tester] Initialized on {self.device}")
        print(f"[Tester] Results will be saved to: {self.results_dir}")

    def load_checkpoint(self, checkpoint_path):
        """Загрузка весов модели"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
        self.G_BA.load_state_dict(checkpoint["G_BA_state_dict"])

        print(f"[Tester] Loaded checkpoint from: {checkpoint_path}")
        print(f"[Tester] Model epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"[Tester] Model loss: {checkpoint.get('loss', 'unknown'):.4f}")

    @torch.no_grad()
    def test_single_batch(self, dataloader, batch_idx=0, save_images=True):
        """Тестирование на одном батче"""
        self.G_AB.eval()
        self.G_BA.eval()

        # Получаем батч
        batch = list(dataloader)[batch_idx]
        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)

        # Генерация
        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        # Реконструкция
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        results = {
            "real_A": real_A.cpu(),
            "real_B": real_B.cpu(),
            "fake_A": fake_A.cpu(),
            "fake_B": fake_B.cpu(),
            "recov_A": recov_A.cpu(),
            "recov_B": recov_B.cpu(),
        }

        # Сохранение изображений
        if save_images:
            self._save_batch_images(results, batch_idx)

        # Вычисление метрик
        metrics = self._compute_batch_metrics(results)

        return results, metrics

    @torch.no_grad()
    def test_full_dataset(self, dataloader, max_samples=None):
        """Тестирование на всем датасете"""
        self.G_AB.eval()
        self.G_BA.eval()

        all_metrics = {
            "psnr_A": [],
            "psnr_B": [],
            "ssim_A": [],
            "ssim_B": [],
            "mse_A": [],
            "mse_B": [],
            "lpips_A": [],
            "lpips_B": [],
        }

        num_samples = (
            min(max_samples, len(dataloader)) if max_samples else len(dataloader)
        )

        print(f"\n[Testing] Testing on {num_samples} samples...")

        pbar = tqdm(dataloader, total=num_samples, desc="Testing", ncols=100)

        sample_count = 0
        for batch_idx, batch in enumerate(pbar):
            if sample_count >= num_samples:
                break

            real_A = batch["A"].to(self.device)
            real_B = batch["B"].to(self.device)

            # Генерация
            fake_B = self.G_AB(real_A)
            fake_A = self.G_BA(real_B)

            # Реконструкция
            recov_A = self.G_BA(fake_B)
            recov_B = self.G_AB(fake_A)

            # Вычисление метрик для батча
            batch_metrics = self._compute_batch_metrics(
                {
                    "real_A": real_A,
                    "real_B": real_B,
                    "recov_A": recov_A,
                    "recov_B": recov_B,
                }
            )

            # Накопление метрик
            for key in all_metrics:
                if key in batch_metrics:
                    all_metrics[key].append(batch_metrics[key])

            sample_count += real_A.size(0)

            # Сохранение примеров каждые 10 батчей
            if batch_idx % 10 == 0:
                self._save_example_images(
                    batch_idx,
                    {
                        "real_A": real_A.cpu(),
                        "real_B": real_B.cpu(),
                        "fake_A": fake_A.cpu(),
                        "fake_B": fake_B.cpu(),
                    },
                )

        # Вычисление средних метрик
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = np.mean(values)
                avg_metrics[f"std_{key}"] = np.std(values)

        return avg_metrics

    def _compute_batch_metrics(self, results):
        """Вычисление метрик качества для батча"""
        metrics = {}

        # PSNR (Peak Signal-to-Noise Ratio)
        def compute_psnr(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return float("inf")
            return 20 * torch.log10(2.0 / torch.sqrt(mse))

        metrics["psnr_A"] = compute_psnr(results["real_A"], results["recov_A"]).item()
        metrics["psnr_B"] = compute_psnr(results["real_B"], results["recov_B"]).item()

        # MSE (Mean Squared Error)
        metrics["mse_A"] = torch.mean(
            (results["real_A"] - results["recov_A"]) ** 2
        ).item()
        metrics["mse_B"] = torch.mean(
            (results["real_B"] - results["recov_B"]) ** 2
        ).item()

        # SSIM (Structural Similarity Index)
        metrics["ssim_A"] = self._compute_ssim(
            results["real_A"], results["recov_A"]
        ).item()
        metrics["ssim_B"] = self._compute_ssim(
            results["real_B"], results["recov_B"]
        ).item()

        # LPIPS (Learned Perceptual Image Patch Similarity) - если установлен
        try:
            import lpips

            if not hasattr(self, "lpips_loss"):
                self.lpips_loss = lpips.LPIPS(net="alex").to(self.device)

            metrics["lpips_A"] = (
                self.lpips_loss(
                    results["real_A"].to(self.device) * 0.5 + 0.5,
                    results["recov_A"].to(self.device) * 0.5 + 0.5,
                )
                .mean()
                .item()
            )

            metrics["lpips_B"] = (
                self.lpips_loss(
                    results["real_B"].to(self.device) * 0.5 + 0.5,
                    results["recov_B"].to(self.device) * 0.5 + 0.5,
                )
                .mean()
                .item()
            )
        except ImportError:
            metrics["lpips_A"] = metrics["lpips_B"] = 0.0

        return metrics

    def _compute_ssim(self, img1, img2, window_size=11, size_average=True):
        """Вычисление SSIM (Structural Similarity Index)"""
        from math import exp

        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [
                    exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                    for x in range(window_size)
                ]
            )
            return gauss / gauss.sum()

        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(
                channel, 1, window_size, window_size
            ).contiguous()
            return window

        (_, channel, height, width) = img1.size()

        window = create_window(window_size, channel).to(img1.device)

        mu1 = torch.nn.functional.conv2d(
            img1, window, padding=window_size // 2, groups=channel
        )
        mu2 = torch.nn.functional.conv2d(
            img2, window, padding=window_size // 2, groups=channel
        )

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            torch.nn.functional.conv2d(
                img1 * img1, window, padding=window_size // 2, groups=channel
            )
            - mu1_sq
        )
        sigma2_sq = (
            torch.nn.functional.conv2d(
                img2 * img2, window, padding=window_size // 2, groups=channel
            )
            - mu2_sq
        )
        sigma12 = (
            torch.nn.functional.conv2d(
                img1 * img2, window, padding=window_size // 2, groups=channel
            )
            - mu1_mu2
        )

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def _save_batch_images(self, results, batch_idx):
        """Сохранение изображений из батча"""
        from torchvision.utils import save_image

        # Денормализация
        def denorm(tensor):
            return tensor * 0.5 + 0.5

        # Сохранение отдельных изображений
        for i in range(min(4, results["real_A"].size(0))):
            # A -> B -> A
            save_image(
                denorm(results["real_A"][i]),
                self.images_dir / f"batch{batch_idx}_sample{i}_real_A.png",
            )
            save_image(
                denorm(results["fake_B"][i]),
                self.images_dir / f"batch{batch_idx}_sample{i}_fake_B.png",
            )
            save_image(
                denorm(results["recov_A"][i]),
                self.images_dir / f"batch{batch_idx}_sample{i}_recov_A.png",
            )

            # B -> A -> B
            save_image(
                denorm(results["real_B"][i]),
                self.images_dir / f"batch{batch_idx}_sample{i}_real_B.png",
            )
            save_image(
                denorm(results["fake_A"][i]),
                self.images_dir / f"batch{batch_idx}_sample{i}_fake_A.png",
            )
            save_image(
                denorm(results["recov_B"][i]),
                self.images_dir / f"batch{batch_idx}_sample{i}_recov_B.png",
            )

            # Создание коллажа
            collage = torch.cat(
                [
                    denorm(results["real_A"][i]),
                    denorm(results["fake_B"][i]),
                    denorm(results["recov_A"][i]),
                    denorm(results["real_B"][i]),
                    denorm(results["fake_A"][i]),
                    denorm(results["recov_B"][i]),
                ],
                dim=2,
            )  # Конкатенация по ширине

            save_image(
                collage, self.images_dir / f"batch{batch_idx}_sample{i}_collage.png"
            )

    def _save_example_images(self, batch_idx, images):
        """Сохранение примеров изображений"""
        from torchvision.utils import save_image

        def denorm(tensor):
            return tensor * 0.5 + 0.5

        # Сохраняем первые 2 изображения из батча
        for i in range(min(2, images["real_A"].size(0))):
            # A -> B
            save_image(
                denorm(images["real_A"][i]),
                self.images_dir / f"example_batch{batch_idx}_A_to_B_real.png",
            )
            save_image(
                denorm(images["fake_B"][i]),
                self.images_dir / f"example_batch{batch_idx}_A_to_B_fake.png",
            )

            # B -> A
            save_image(
                denorm(images["real_B"][i]),
                self.images_dir / f"example_batch{batch_idx}_B_to_A_real.png",
            )
            save_image(
                denorm(images["fake_A"][i]),
                self.images_dir / f"example_batch{batch_idx}_B_to_A_fake.png",
            )

    def generate_visual_report(self, metrics):
        """Генерация визуального отчета"""
        # 1. Таблица метрик
        self._save_metrics_table(metrics)

        # 2. Графики распределения метрик
        self._plot_metrics_distribution(metrics)

        # 3. Сравнение метрик
        self._plot_metrics_comparison(metrics)

        # 4. Примеры преобразований
        self._create_transformation_examples()

    def _save_metrics_table(self, metrics):
        """Сохранение таблицы метрик"""
        import pandas as pd

        # Создаем DataFrame
        df = pd.DataFrame([metrics])

        # Сохраняем в CSV
        csv_path = self.metrics_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)

        # Сохраняем в текстовый файл
        txt_path = self.metrics_dir / "metrics.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("CYCLEGAN TEST METRICS\n")
            f.write("=" * 60 + "\n\n")

            f.write("Image Quality Metrics:\n")
            f.write("-" * 40 + "\n")

            for key, value in metrics.items():
                if key.startswith("avg_"):
                    metric_name = key[4:].replace("_", " ").title()
                    std_key = f"std_{key[4:]}"
                    std_value = metrics.get(std_key, 0)
                    f.write(f"{metric_name:20}: {value:.4f} (±{std_value:.4f})\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"[Tester] Metrics saved to: {txt_path}")

    def _plot_metrics_distribution(self, metrics):
        """Визуализация распределения метрик"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # PSNR
        ax = axes[0]
        psnr_values = [metrics.get("avg_psnr_A", 0), metrics.get("avg_psnr_B", 0)]
        psnr_labels = ["Domain A", "Domain B"]
        bars = ax.bar(psnr_labels, psnr_values)
        ax.set_title("PSNR (Higher is better)")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for bar, value in zip(bars, psnr_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # SSIM
        ax = axes[1]
        ssim_values = [metrics.get("avg_ssim_A", 0), metrics.get("avg_ssim_B", 0)]
        bars = ax.bar(psnr_labels, ssim_values, color="orange")
        ax.set_title("SSIM (Higher is better)")
        ax.set_ylabel("SSIM")
        ax.grid(True, alpha=0.3)

        for bar, value in zip(bars, ssim_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        # MSE
        ax = axes[2]
        mse_values = [metrics.get("avg_mse_A", 0), metrics.get("avg_mse_B", 0)]
        bars = ax.bar(psnr_labels, mse_values, color="green")
        ax.set_title("MSE (Lower is better)")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)

        for bar, value in zip(bars, mse_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        # LPIPS (если есть)
        ax = axes[3]
        lpips_values = [metrics.get("avg_lpips_A", 0), metrics.get("avg_lpips_B", 0)]
        if any(v > 0 for v in lpips_values):
            bars = ax.bar(psnr_labels, lpips_values, color="red")
            ax.set_title("LPIPS (Lower is better)")
            ax.set_ylabel("LPIPS")
            ax.grid(True, alpha=0.3)

            for bar, value in zip(bars, lpips_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "LPIPS not available\nInstall lpips package",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("LPIPS Metrics")

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "metrics_distribution.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"[Tester] Metrics plots saved to: {self.plots_dir}")

    def _plot_metrics_comparison(self, metrics):
        """Сравнение метрик между доменами"""
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = ["PSNR", "SSIM", "MSE"]
        domain_A_values = [
            metrics.get("avg_psnr_A", 0),
            metrics.get("avg_ssim_A", 0),
            metrics.get("avg_mse_A", 0),
        ]
        domain_B_values = [
            metrics.get("avg_psnr_B", 0),
            metrics.get("avg_ssim_B", 0),
            metrics.get("avg_mse_B", 0),
        ]

        x = np.arange(len(metric_names))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, domain_A_values, width, label="Domain A", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, domain_B_values, width, label="Domain B", alpha=0.8
        )

        ax.set_xlabel("Metrics")
        ax.set_ylabel("Values")
        ax.set_title("Comparison of Metrics between Domains")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        autolabel(bars1)
        autolabel(bars2)

        plt.tight_layout()
        plt.savefig(
            self.plots_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _create_transformation_examples(self):
        """Создание примеров преобразований"""
        # Эта функция будет заполнена при реальном тестировании
        pass

    def run_comprehensive_test(self, dataloader, num_samples=100):
        """Запуск комплексного тестирования"""
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE TEST")
        print("=" * 60)

        # 1. Тестирование на подмножестве данных
        print("\n[Phase 1] Testing on subset of data...")
        metrics = self.test_full_dataset(dataloader, max_samples=num_samples)

        # 2. Тестирование на одном батче для визуализации
        print("\n[Phase 2] Testing single batch for visualization...")
        results, batch_metrics = self.test_single_batch(dataloader, batch_idx=0)

        # 3. Генерация отчета
        print("\n[Phase 3] Generating comprehensive report...")
        self.generate_visual_report(metrics)

        # 4. Вывод результатов
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)

        print("\nImage Quality Metrics (average ± std):")
        print("-" * 40)

        for key, value in metrics.items():
            if key.startswith("avg_"):
                metric_name = key[4:].replace("_", " ").title()
                std_key = f"std_{key[4:]}"
                std_value = metrics.get(std_key, 0)
                print(f"{metric_name:20}: {value:.4f} ± {std_value:.4f}")

        print("\n" + "=" * 60)
        print(f"Results saved to: {self.results_dir}")
        print("=" * 60)

        return metrics
