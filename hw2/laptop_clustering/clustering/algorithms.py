import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from utils.logger import get_logger


class ClusteringAlgorithms:
    """Реализация алгоритмов кластеризации"""

    def __init__(self):
        self.logger = get_logger("ClusteringAlgorithms")

    def find_optimal_eps(self, data, k=5):
        """Определение оптимального параметра eps для DBSCAN"""
        self.logger.info("Поиск оптимального eps для DBSCAN")
        try:
            k = min(k, len(data) - 1)
            if k < 2:
                self.logger.warning("k слишком мало, используется eps=0.5")
                return 0.5

            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors_fit = neighbors.fit(data)
            distances, indices = neighbors_fit.kneighbors(data)
            distances = np.sort(distances[:, k - 1], axis=0)

            # Находим "локоть" на графике
            gradients = np.gradient(distances)
            if len(gradients) > 0:
                elbow_index = min(len(gradients) - 1, np.argmax(gradients) + 10)
                optimal_eps = distances[elbow_index]
            else:
                optimal_eps = 0.5

            self.logger.info(f"Оптимальный eps: {optimal_eps:.3f}")
            return optimal_eps
        except Exception as e:
            self.logger.error(f"Ошибка в определении eps: {e}")
            return 0.5

    def safe_dbscan(self, data, method_name="DBSCAN"):
        """Безопасная реализация DBSCAN с обработкой ошибок"""
        self.logger.info(f"Запуск DBSCAN для {method_name}")
        try:
            eps = self.find_optimal_eps(data)
            min_samples = max(2, min(5, len(data) // 20))

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            # Проверяем результат DBSCAN
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)

            self.logger.info(
                f"DBSCAN {method_name}: {n_clusters} кластеров, {n_noise} шумовых точек"
            )

            if n_clusters < 2:
                self.logger.warning(
                    f"{method_name}: DBSCAN нашел только {n_clusters} кластеров, используем KMeans"
                )
                # Используем KMeans как запасной вариант
                k = min(5, len(data) // 10)
                k = max(2, k)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                self.logger.info(
                    f"KMeans использован как запасной вариант: {k} кластеров"
                )

            return labels

        except Exception as e:
            self.logger.error(f"Ошибка в DBSCAN для {method_name}: {e}")
            # Используем KMeans как запасной вариант
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            self.logger.info("KMeans использован из-за ошибки DBSCAN")
            return labels

    def hierarchical_clustering(self, data, n_clusters=4):
        """Иерархическая кластеризация"""
        self.logger.info(
            f"Запуск иерархической кластеризации с {n_clusters} кластерами"
        )
        try:
            Z = linkage(data, method="ward")
            labels = fcluster(Z, n_clusters, criterion="maxclust")
            self.logger.info("Иерархическая кластеризация завершена успешно")
            return labels, Z
        except Exception as e:
            self.logger.error(f"Ошибка в иерархической кластеризации: {e}")
            raise
