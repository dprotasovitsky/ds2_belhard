import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from utils.logger import get_logger


class ClusteringEvaluator:
    """Оценка качества кластеризации"""

    def __init__(self):
        self.logger = get_logger("ClusteringEvaluator")

    def evaluate_clustering(self, data, labels_dict):
        """Оценка качества кластеризации для разных методов"""
        self.logger.info("Оценка качества кластеризации")

        results = []

        for name, labels in labels_dict.items():
            evaluation = self._evaluate_single_method(data, labels, name)
            if evaluation:
                results.append(evaluation)

        results_df = pd.DataFrame(results)
        self.logger.info("Оценка качества завершена")
        return results_df

    def _evaluate_single_method(self, data, labels, method_name):
        """Оценка одного метода кластеризации"""
        try:
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

            if n_clusters < 2:
                self.logger.warning(
                    f"{method_name}: слишком мало кластеров ({n_clusters}) для оценки"
                )
                return None

            if -1 in labels:
                n_noise = list(labels).count(-1)
                self.logger.info(
                    f"{method_name}: {n_clusters} кластеров, {n_noise} шумовых точек"
                )
            else:
                self.logger.info(f"{method_name}: {n_clusters} кластеров")

            silhouette = silhouette_score(data, labels)
            calinski = calinski_harabasz_score(data, labels)
            davies = davies_bouldin_score(data, labels)

            self.logger.debug(
                f"{method_name} - Silhouette: {silhouette:.3f}, "
                f"Calinski-Harabasz: {calinski:.3f}, "
                f"Davies-Bouldin: {davies:.3f}"
            )

            return {
                "Method": method_name,
                "Silhouette": silhouette,
                "Calinski-Harabasz": calinski,
                "Davies-Bouldin": davies,
                "Clusters": n_clusters,
            }

        except Exception as e:
            self.logger.error(f"Ошибка оценки метода {method_name}: {e}")
            return None
