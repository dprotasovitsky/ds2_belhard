import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils.logger import Logger


class ReportGenerator:
    def __init__(self):
        self.logger = Logger.get_logger()
        self.report_data = {
            "training_metrics": [],
            "classification_reports": {},
            "confusion_matrices": {},
            "timestamps": {},
            "classifier_results": {},
        }

        # Убедимся, что директории существуют
        self._ensure_directories()

    def _ensure_directories(self):
        """Создает необходимые директории"""
        from config import Config

        Config.create_directories()

    def add_training_metrics(self, epoch, avg_loss, avg_triplet, avg_recon):
        """Добавляет метрики обучения"""
        self.report_data["training_metrics"].append(
            {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "avg_triplet": avg_triplet,
                "avg_recon": avg_recon,
            }
        )

    def add_classification_report(self, classifier_name, report):
        """Добавляет отчет по классификации"""
        self.report_data["classification_reports"][classifier_name] = report

    def add_confusion_matrix(self, classifier_name, cm):
        """Добавляет матрицу ошибок"""
        self.report_data["confusion_matrices"][classifier_name] = cm

    def add_classifier_results(self, classifier_name, accuracy):
        """Добавляет результаты классификатора"""
        self.report_data["classifier_results"][classifier_name] = accuracy

    def add_timestamp(self, event_name):
        """Добавляет временную метку события"""
        self.report_data["timestamps"][event_name] = datetime.now()

    def generate_html_report(self, filepath=None):
        """Генерирует HTML отчет"""
        from config import Config

        try:
            if filepath is None:
                filepath = Config.get_report_path()

            # Создаем HTML отчет
            html_content = self._create_html_content()

            # Убедимся, что директория существует
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Создана директория для отчета: {directory}")

            # Сохраняем файл
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.logger.info(f"HTML отчет сохранен: {filepath}")
            return filepath

        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета: {e}")
            # Попробуем сохранить в текущую директорию
            try:
                backup_path = "training_report_backup.html"
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(self._create_html_content())
                self.logger.info(f"Резервная копия отчета сохранена: {backup_path}")
                return backup_path
            except Exception as backup_error:
                self.logger.error(
                    f"Не удалось сохранить резервную копию: {backup_error}"
                )
                return None

    def _create_html_content(self):
        """Создает содержимое HTML отчета"""
        # Статистика данных
        num_metrics = len(self.report_data["training_metrics"])
        num_classifiers = len(self.report_data["classifier_results"])
        num_reports = len(self.report_data["classification_reports"])

        html = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <title>Отчет по обучению Triplet Autoencoder</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background: white; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; }}
                .metric {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 14px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .success {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                pre {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Courier New', monospace; border-left: 4px solid #007bff; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; margin: 0 10px; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .stat-label {{ color: #666; margin-top: 5px; }}
                .classifier-comparison {{ margin: 20px 0; }}
                .best-classifier {{ background-color: #d4edda !important; border-left: 4px solid #28a745; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Отчет по обучению Triplet Autoencoder</h1>
                    <p>Время генерации отчета: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{num_metrics}</div>
                        <div class="stat-label">Эпох обучения</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{num_classifiers}</div>
                        <div class="stat-label">Классификаторов</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{num_reports}</div>
                        <div class="stat-label">Отчетов</div>
                    </div>
                </div>
        """

        # Добавляем результаты классификаторов
        if self.report_data["classifier_results"]:
            best_accuracy = (
                max(self.report_data["classifier_results"].values())
                if self.report_data["classifier_results"]
                else 0
            )

            html += """
            <div class="section">
                <h2>Результаты классификаторов</h2>
                <div class="classifier-comparison">
                    <table>
                        <tr>
                            <th>Классификатор</th>
                            <th>Точность</th>
                            <th>Статус</th>
                        </tr>
            """

            for classifier_name, accuracy in self.report_data[
                "classifier_results"
            ].items():
                is_best = accuracy == best_accuracy
                row_class = "best-classifier" if is_best else ""
                status = "ЛУЧШИЙ" if is_best else ""

                html += f"""
                    <tr class="{row_class}">
                        <td><strong>{classifier_name}</strong></td>
                        <td class="success">{accuracy:.4f}</td>
                        <td>{status}</td>
                    </tr>
                """

            html += f"""
                    </table>
                    <div class="metric">
                        <strong>Лучшая точность:</strong> <span class="success">{best_accuracy:.4f}</span>
                    </div>
                </div>
            </div>
            """

        # Добавляем метрики обучения
        if self.report_data["training_metrics"]:
            last_metric = self.report_data["training_metrics"][-1]

            html += """
            <div class="section">
                <h2>Метрики обучения Autoencoder</h2>
                <div class="metric">
                    <strong>Последняя эпоха:</strong> Финальные значения потерь
                </div>
                <table>
                    <tr>
                        <th>Тип потерь</th>
                        <th>Значение</th>
                    </tr>
            """

            html += f"""
                    <tr>
                        <td>Общий Loss</td>
                        <td>{last_metric['avg_loss']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Triplet Loss</td>
                        <td>{last_metric['avg_triplet']:.4f}</td>
                    </tr>
                    <tr>
                        <td>Reconstruction Loss</td>
                        <td>{last_metric['avg_recon']:.4f}</td>
                    </tr>
            """

            html += """
                </table>
                
                <div class="metric">
                    <strong>История обучения (последние 10 эпох):</strong>
                </div>
                <table>
                    <tr>
                        <th>Эпоха</th>
                        <th>Общий Loss</th>
                        <th>Triplet Loss</th>
                        <th>Recon Loss</th>
                    </tr>
            """

            for metric in self.report_data["training_metrics"][-10:]:
                html += f"""
                    <tr>
                        <td>{metric['epoch']}</td>
                        <td>{metric['avg_loss']:.4f}</td>
                        <td>{metric['avg_triplet']:.4f}</td>
                        <td>{metric['avg_recon']:.4f}</td>
                    </tr>
                """

            html += "</table></div>"

        # Добавляем отчеты по классификации
        if self.report_data["classification_reports"]:
            html += """
            <div class="section">
                <h2>Детальные отчеты по классификации</h2>
            """

            for classifier_name, report in self.report_data[
                "classification_reports"
            ].items():
                html += f"""
                <div class="subsection">
                    <h3>{classifier_name}</h3>
                    <pre>{report}</pre>
                </div>
                """

            html += "</div>"

        # Добавляем информацию о временных метках
        if self.report_data["timestamps"]:
            html += """
            <div class="section">
                <h2>Временные метки выполнения</h2>
                <table>
                    <tr>
                        <th>Событие</th>
                        <th>Время</th>
                    </tr>
            """

            for event_name, timestamp in self.report_data["timestamps"].items():
                html += f"""
                    <tr>
                        <td>{event_name}</td>
                        <td>{timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                    </tr>
                """

            html += "</table></div>"

        html += """
            </div>
        </body>
        </html>
        """

        return html

    def generate_plots(self, filepath_prefix=None):
        """Генерирует графики обучения"""
        from config import Config

        try:
            if filepath_prefix is None:
                filepath_prefix = Config.PLOTS_DIR

            # Убедимся, что директория существует
            if not os.path.exists(filepath_prefix):
                os.makedirs(filepath_prefix, exist_ok=True)
                self.logger.info(f"Создана директория для графиков: {filepath_prefix}")

            if self.report_data["training_metrics"]:
                df = pd.DataFrame(self.report_data["training_metrics"])

                # Устанавливаем стиль графиков
                plt.style.use("seaborn-v0_8")
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(
                    "Метрики обучения Triplet Autoencoder",
                    fontsize=16,
                    fontweight="bold",
                )

                # График общего Loss
                axes[0, 0].plot(
                    df["epoch"], df["avg_loss"], color="#ff6b6b", linewidth=2
                )
                axes[0, 0].set_title("Общий Loss", fontsize=14, fontweight="bold")
                axes[0, 0].set_xlabel("Эпоха")
                axes[0, 0].set_ylabel("Loss")
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].fill_between(
                    df["epoch"], df["avg_loss"], alpha=0.3, color="#ff6b6b"
                )

                # График Triplet Loss
                axes[0, 1].plot(
                    df["epoch"], df["avg_triplet"], color="#4ecdc4", linewidth=2
                )
                axes[0, 1].set_title("Triplet Loss", fontsize=14, fontweight="bold")
                axes[0, 1].set_xlabel("Эпоха")
                axes[0, 1].set_ylabel("Loss")
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].fill_between(
                    df["epoch"], df["avg_triplet"], alpha=0.3, color="#4ecdc4"
                )

                # График Reconstruction Loss
                axes[1, 0].plot(
                    df["epoch"], df["avg_recon"], color="#45b7d1", linewidth=2
                )
                axes[1, 0].set_title(
                    "Reconstruction Loss", fontsize=14, fontweight="bold"
                )
                axes[1, 0].set_xlabel("Эпоха")
                axes[1, 0].set_ylabel("Loss")
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].fill_between(
                    df["epoch"], df["avg_recon"], alpha=0.3, color="#45b7d1"
                )

                # График всех потерь вместе
                axes[1, 1].plot(
                    df["epoch"],
                    df["avg_loss"],
                    label="Total Loss",
                    color="#ff6b6b",
                    linewidth=2,
                )
                axes[1, 1].plot(
                    df["epoch"],
                    df["avg_triplet"],
                    label="Triplet Loss",
                    color="#4ecdc4",
                    linewidth=2,
                )
                axes[1, 1].plot(
                    df["epoch"],
                    df["avg_recon"],
                    label="Recon Loss",
                    color="#45b7d1",
                    linewidth=2,
                )
                axes[1, 1].set_title(
                    "Сравнение всех потерь", fontsize=14, fontweight="bold"
                )
                axes[1, 1].set_xlabel("Эпоха")
                axes[1, 1].set_ylabel("Loss")
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                plot_path = os.path.join(filepath_prefix, "training_losses.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
                plt.close()

                self.logger.info(f"Графики сохранены в {plot_path}")

                # Дополнительный график - сравнение классификаторов
                if self.report_data["classifier_results"]:
                    self._generate_classifier_comparison_plot(filepath_prefix)

        except Exception as e:
            self.logger.error(f"Ошибка при генерации графиков: {e}")

    def _generate_classifier_comparison_plot(self, filepath_prefix):
        """Генерирует график сравнения классификаторов"""
        try:
            classifiers = list(self.report_data["classifier_results"].keys())
            accuracies = list(self.report_data["classifier_results"].values())

            plt.figure(figsize=(10, 6))
            colors = [
                "#4CAF50" if acc == max(accuracies) else "#2196F3" for acc in accuracies
            ]
            bars = plt.bar(classifiers, accuracies, color=colors, alpha=0.8)

            plt.title(
                "Сравнение точности классификаторов", fontsize=14, fontweight="bold"
            )
            plt.ylabel("Точность")
            plt.ylim(0, 1.0)
            plt.grid(True, alpha=0.3, axis="y")

            # Добавляем значения на столбцы
            for bar, accuracy in zip(bars, accuracies):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{accuracy:.4f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            plot_path = os.path.join(filepath_prefix, "classifier_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close()

            self.logger.info(f"График сравнения классификаторов сохранен: {plot_path}")

        except Exception as e:
            self.logger.error(
                f"Ошибка при генерации графика сравнения классификаторов: {e}"
            )
