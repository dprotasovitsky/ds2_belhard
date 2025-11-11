import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from utils.logger import get_logger


class Plotter:
    """Класс для визуализации результатов"""

    def __init__(self):
        self.logger = get_logger("Plotter")
        plt.style.use("default")

    def plot_distributions(self, df, numerical_columns):
        """Визуализация распределений числовых признаков"""
        self.logger.info("Визуализация распределений признаков")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for i, col in enumerate(numerical_columns):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i])
                axes[i].set_title(f"Распределение {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Частота")

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, df, numerical_columns):
        """Визуализация матрицы корреляций"""
        self.logger.info("Построение матрицы корреляций")

        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numerical_columns].corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f"
        )
        plt.title("Матрица корреляций числовых признаков")
        plt.tight_layout()
        plt.show()

    def plot_dendrogram(self, Z, method_name):
        """Построение дендрограммы"""
        self.logger.info(f"Построение дендрограммы для {method_name}")

        plt.figure(figsize=(12, 8))
        dendrogram(Z, truncate_mode="lastp", p=12, show_leaf_counts=True)
        plt.title(f"Дендрограмма {method_name}")
        plt.xlabel("Индекс образца")
        plt.ylabel("Расстояние")
        plt.tight_layout()
        plt.show()

    def plot_clustering_results(self, projections, labels_dict, projection_names):
        """Визуализация результатов кластеризации на разных проекциях"""
        self.logger.info("Визуализация результатов кластеризации")

        n_methods = len(labels_dict)
        n_projections = len(projections)

        fig, axes = plt.subplots(
            n_projections, n_methods, figsize=(5 * n_methods, 4 * n_projections)
        )

        if n_methods == 1:
            axes = axes.reshape(n_projections, 1)

        methods = list(labels_dict.keys())

        for i, (proj_name, projection) in enumerate(zip(projection_names, projections)):
            for j, method_name in enumerate(methods):
                labels = labels_dict[method_name]

                scatter = axes[i, j].scatter(
                    projection[:, 0],
                    projection[:, 1],
                    c=labels,
                    cmap="tab10",
                    alpha=0.7,
                )
                axes[i, j].set_title(f"{proj_name} + {method_name}")
                axes[i, j].set_xlabel(f"{proj_name} 1")
                axes[i, j].set_ylabel(f"{proj_name} 2")
                plt.colorbar(scatter, ax=axes[i, j])

        plt.tight_layout()
        plt.show()

    def plot_graphics_analysis(self, df):
        """Анализ graphics_capacity"""
        self.logger.info("Анализ graphics_capacity")

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        df["graphics_capacity"].value_counts().sort_index().plot(kind="bar")
        plt.title("Распределение graphics_capacity")
        plt.xlabel("Graphics Capacity")
        plt.ylabel("Количество")

        plt.subplot(1, 2, 2)
        gaming_laptops = df[df["graphics_capacity"] > 0]
        non_gaming_laptops = df[df["graphics_capacity"] == 0]

        plt.bar(
            ["Не игровые", "Игровые"],
            [len(non_gaming_laptops), len(gaming_laptops)],
            color=["lightblue", "red"],
        )
        plt.title("Распределение игровых/не игровых ноутбуков")
        plt.ylabel("Количество")

        plt.tight_layout()
        plt.show()
