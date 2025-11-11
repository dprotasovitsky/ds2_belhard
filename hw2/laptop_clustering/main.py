import pandas as pd
import umap
from clustering.algorithms import ClusteringAlgorithms
from clustering.evaluator import ClusteringEvaluator
from config.settings import Config
from data.processor import DataProcessor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils.logger import get_logger
from visualization.plotter import Plotter


class LaptopClusteringAnalysis:
    """Главный класс для анализа кластеризации ноутбуков"""

    def __init__(self):
        self.logger = get_logger("LaptopClusteringAnalysis")
        self.data_processor = DataProcessor()
        self.clustering_algorithms = ClusteringAlgorithms()
        self.evaluator = ClusteringEvaluator()
        self.plotter = Plotter()

        self.results = {}
        self.logger.info("Инициализация LaptopClusteringAnalysis завершена")

    def run_analysis(self, data_path):
        """Запуск полного анализа"""
        self.logger.info("=" * 60)
        self.logger.info("ЗАПУСК ПОЛНОГО АНАЛИЗА КЛАСТЕРИЗАЦИИ НОУТБУКОВ")
        self.logger.info("=" * 60)

        try:
            # Загрузка и предобработка данных
            self.logger.info("ЭТАП 1: ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
            df = self.data_processor.load_data(data_path)
            df_processed = self.data_processor.preprocess_data(df)

            # Разведочный анализ
            self.logger.info("ЭТАП 2: РАЗВЕДОЧНЫЙ АНАЛИЗ")
            self._exploratory_analysis(df_processed)

            # Подготовка признаков
            self.logger.info("ЭТАП 3: ПОДГОТОВКА ПРИЗНАКОВ")
            X_scaled, features = self.data_processor.prepare_features()

            # Понижение размерности
            self.logger.info("ЭТАП 4: ПОНИЖЕНИЕ РАЗМЕРНОСТИ")
            projections, projection_names = self._dimensionality_reduction(
                X_scaled, df_processed
            )

            # Кластеризация
            self.logger.info("ЭТАП 5: КЛАСТЕРИЗАЦИЯ")
            labels_dict = self._perform_clustering(X_scaled, projections)

            # Оценка качества
            self.logger.info("ЭТАП 6: ОЦЕНКА КАЧЕСТВА")
            metrics_df = self.evaluator.evaluate_clustering(X_scaled, labels_dict)

            # Визуализация результатов
            self.logger.info("ЭТАП 7: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
            self._visualize_results(projections, projection_names, labels_dict)

            # Интерпретация кластеров
            self.logger.info("ЭТАП 8: ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ")
            self._interpret_clusters(labels_dict, metrics_df)

            # Сохранение результатов
            self.logger.info("ЭТАП 9: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
            self._save_results(metrics_df, labels_dict)

            self.logger.info("=" * 60)
            self.logger.info("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН")
            self.logger.info("=" * 60)

            return metrics_df, self.data_processor.get_processed_data()

        except Exception as e:
            self.logger.error(f"КРИТИЧЕСКАЯ ОШИБКА В АНАЛИЗЕ: {e}")
            raise

    def _exploratory_analysis(self, df):
        """Разведочный анализ данных"""
        numerical_columns = [
            "ram",
            "storage_capacity_gb",
            "price",
            "display_size_inch",
            "graphics_capacity",
        ]

        self.plotter.plot_distributions(df, numerical_columns)
        self.plotter.plot_correlation_matrix(df, numerical_columns)
        self.plotter.plot_graphics_analysis(df)

    def _dimensionality_reduction(self, X_scaled, df_processed):
        """Понижение размерности"""
        # UMAP
        reducer_umap = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=Config.UMAP_NEIGHBORS,
            min_dist=Config.UMAP_MIN_DIST,
        )
        X_umap = reducer_umap.fit_transform(X_scaled)
        self.logger.info("UMAP проекция завершена")

        # LDA
        df_processed["price_category"] = pd.cut(
            df_processed["price"], bins=5, labels=[0, 1, 2, 3, 4]
        )
        lda = LDA(n_components=2)
        X_lda = lda.fit_transform(X_scaled, df_processed["price_category"])
        self.logger.info("LDA проекция завершена")

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_.sum()
        self.logger.info(
            f"PCA проекция завершена. Объясненная дисперсия: {explained_variance:.3f}"
        )

        projections = [X_umap, X_lda, X_pca]
        projection_names = ["UMAP", "LDA", "PCA"]

        return projections, projection_names

    def _perform_clustering(self, X_scaled, projections):
        """Выполнение кластеризации"""
        X_umap, X_lda, X_pca = projections

        # Иерархическая кластеризация
        hierarchical_labels, Z = self.clustering_algorithms.hierarchical_clustering(
            X_scaled, Config.DEFAULT_N_CLUSTERS
        )
        self.plotter.plot_dendrogram(Z, "Иерархическая кластеризация")

        # DBSCAN на разных представлениях
        dbscan_original = self.clustering_algorithms.safe_dbscan(
            X_scaled, "DBSCAN (оригинальные данные)"
        )
        dbscan_umap = self.clustering_algorithms.safe_dbscan(X_umap, "DBSCAN (UMAP)")
        dbscan_pca = self.clustering_algorithms.safe_dbscan(X_pca, "DBSCAN (PCA)")

        # Сохранение результатов
        labels_dict = {
            "Hierarchical": hierarchical_labels,
            "DBSCAN_Original": dbscan_original,
            "DBSCAN_UMAP": dbscan_umap,
            "DBSCAN_PCA": dbscan_pca,
        }

        # Добавление меток в данные
        df_processed = self.data_processor.get_processed_data()
        for name, labels in labels_dict.items():
            df_processed[name] = labels

        return labels_dict

    def _visualize_results(self, projections, projection_names, labels_dict):
        """Визуализация результатов"""
        self.plotter.plot_clustering_results(projections, labels_dict, projection_names)

    def _interpret_clusters(self, labels_dict, metrics_df):
        """Интерпретация кластеров"""
        # Выбираем лучший метод по Silhouette Score
        best_method_row = metrics_df.loc[metrics_df["Silhouette"].idxmax()]
        best_method = best_method_row["Method"]
        best_score = best_method_row["Silhouette"]

        self.logger.info(
            f"Лучший метод кластеризации: {best_method} (Silhouette: {best_score:.3f})"
        )

        df_processed = self.data_processor.get_processed_data()

        self.logger.info("ДЕТАЛЬНАЯ ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ:")
        self.logger.info("-" * 50)

        for cluster in sorted(df_processed[best_method].unique()):
            cluster_data = df_processed[df_processed[best_method] == cluster]

            self.logger.info(f"КЛАСТЕР {cluster}:")
            self.logger.info(f"  Количество ноутбуков: {len(cluster_data)}")
            self.logger.info(f"  Средняя цена: {cluster_data['price'].mean():.0f} руб.")
            self.logger.info(f"  Средний RAM: {cluster_data['ram'].mean():.1f} GB")
            self.logger.info(
                f"  Средний объем памяти: {cluster_data['storage_capacity_gb'].mean():.0f} GB"
            )
            self.logger.info(
                f"  Средняя графика: {cluster_data['graphics_capacity'].mean():.1f}"
            )

            # Игровые ноутбуки
            gaming_count = cluster_data["is_gaming"].sum()
            gaming_percent = (gaming_count / len(cluster_data)) * 100
            self.logger.info(
                f"  Игровых ноутбуков: {gaming_count} ({gaming_percent:.1f}%)"
            )

            # Определение сегмента
            avg_price = cluster_data["price"].mean()
            if avg_price < 50000:
                segment = "БЮДЖЕТНЫЙ"
            elif avg_price < 100000:
                segment = "СРЕДНИЙ"
            elif avg_price < 200000:
                segment = "ПРЕМИУМ"
            else:
                segment = "ЛЮКС"

            self.logger.info(f"  Ценовой сегмент: {segment}")

            # Топ бренды
            top_brands = cluster_data["brand_name"].value_counts().head(3)
            self.logger.info(f"  Топ-3 бренда: {dict(top_brands)}")

            # Топ процессоры
            top_processors = cluster_data["processor_type"].value_counts().head(3)
            self.logger.info(f"  Топ-3 процессора: {dict(top_processors)}")

            self.logger.info("-" * 50)

    def _save_results(self, metrics_df, labels_dict):
        """Сохранение результатов"""
        # Сохранение метрик
        metrics_file = f"{Config.RESULTS_DIR}/clustering_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        self.logger.info(f"Метрики кластеризации сохранены в: {metrics_file}")

        # Сохранение данных с кластерами
        results_file = f"{Config.RESULTS_DIR}/clustering_results.csv"
        df_with_clusters = self.data_processor.get_processed_data()
        df_with_clusters.to_csv(results_file, index=False)
        self.logger.info(f"Результаты кластеризации сохранены в: {results_file}")

        # Сохранение сводки по методам
        self.logger.info("СВОДКА ПО МЕТОДАМ КЛАСТЕРИЗАЦИИ:")
        for _, row in metrics_df.iterrows():
            self.logger.info(
                f"  {row['Method']}: {row['Clusters']} кластеров, "
                f"Silhouette: {row['Silhouette']:.3f}, "
                f"Calinski-Harabasz: {row['Calinski-Harabasz']:.0f}, "
                f"Davies-Bouldin: {row['Davies-Bouldin']:.3f}"
            )


if __name__ == "__main__":
    try:
        # Запуск анализа
        analyzer = LaptopClusteringAnalysis()
        metrics_df, results_df = analyzer.run_analysis(Config.DATA_PATH)

        # Вывод результатов в консоль
        print("\n" + "=" * 60)
        print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ АНАЛИЗА")
        print("=" * 60)
        print(metrics_df.sort_values("Silhouette", ascending=False))
        print("\nПодробные логи сохранены в файл")

    except Exception as e:
        print(f"Произошла ошибка при выполнении анализа: {e}")
        print("Проверьте файл лога для получения подробной информации")
