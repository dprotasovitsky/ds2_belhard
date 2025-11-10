from typing import Any, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from utils.logger import Logger, log_execution_time

# Проверяем доступность LightGBM
try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print(
        "Предупреждение: LightGBM не установлен. Установите его: pip install lightgbm"
    )

# Проверяем доступность CatBoost
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print(
        "Предупреждение: CatBoost не установлен. Установите его: pip install catboost"
    )


class EmbeddingClassifier:
    def __init__(self, triplet_model):
        self.triplet_model = triplet_model
        self.logger = Logger.get_logger()

        from config import Config

        self.classifiers = self._initialize_classifiers(Config.CLASSIFIERS)

        self.current_classifier = None
        self.logger.info(
            f"Инициализирован EmbeddingClassifier с {len(self.classifiers)} классификаторами"
        )

    def _initialize_classifiers(
        self, classifier_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Инициализирует классификаторы с конфигурацией"""
        classifiers = {}

        # Random Forest
        if "random_forest" in classifier_configs:
            classifiers["random_forest"] = RandomForestClassifier(
                **classifier_configs["random_forest"]
            )
            self.logger.info("Инициализирован RandomForestClassifier")

        # CatBoost
        if "catboost" in classifier_configs and CATBOOST_AVAILABLE:
            classifiers["catboost"] = CatBoostClassifier(
                **classifier_configs["catboost"]
            )
            self.logger.info("Инициализирован CatBoostClassifier")
        elif "catboost" in classifier_configs and not CATBOOST_AVAILABLE:
            self.logger.warning("CatBoost не доступен. Пропускаем инициализацию.")

        # LightGBM
        if "lightgbm" in classifier_configs and LIGHTGBM_AVAILABLE:
            classifiers["lightgbm"] = LGBMClassifier(
                **classifier_configs["lightgbm"], verbose=-1  # Отключаем вывод LightGBM
            )
            self.logger.info("Инициализирован LGBMClassifier")
        elif "lightgbm" in classifier_configs and not LIGHTGBM_AVAILABLE:
            self.logger.warning("LightGBM не доступен. Пропускаем инициализацию.")

        self.logger.info(f"Инициализированы классификаторы: {list(classifiers.keys())}")
        return classifiers

    @log_execution_time
    def train(
        self,
        texts: List[str],
        labels: List[str],
        classifier_type: str = "random_forest",
        test_size: float = 0.2,
        reporter=None,
    ):

        if classifier_type not in self.classifiers:
            available_classifiers = list(self.classifiers.keys())
            raise ValueError(
                f"Классификатор {classifier_type} не доступен. Доступные: {available_classifiers}"
            )

        self.logger.info(f"Начало обучения классификатора {classifier_type}")

        # Получаем эмбеддинги
        embeddings = self.triplet_model.encode_texts(texts)
        encoded_labels = self.triplet_model.label_encoder.transform(labels)

        # Проверяем, достаточно ли данных для разделения
        min_test_samples = len(self.triplet_model.label_encoder.classes_)

        if len(embeddings) * test_size < min_test_samples:
            # Если тестовая выборка слишком мала, используем кросс-валидацию
            self.current_classifier = self.classifiers[classifier_type]

            # Выполняем кросс-валидацию
            cv_scores = cross_val_score(
                self.current_classifier,
                embeddings,
                encoded_labels,
                cv=min(5, len(embeddings)),
                scoring="accuracy",
            )

            # Обучаем на всех данных
            self.current_classifier.fit(embeddings, encoded_labels)

            accuracy = cv_scores.mean()
            self.logger.info(
                f"Cross-validation accuracy ({classifier_type}): {accuracy:.4f} (+/- {cv_scores.std() * 2:.4f})"
            )
            self.logger.info(f"Individual CV scores: {cv_scores}")

        else:
            # Разделяем данные обычным способом
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings,
                encoded_labels,
                test_size=test_size,
                random_state=42,
                stratify=encoded_labels,
            )

            # Обучаем классификатор
            self.current_classifier = self.classifiers[classifier_type]

            # Специальная обработка для CatBoost (поддержка eval_set)
            if classifier_type == "catboost" and CATBOOST_AVAILABLE:
                self.logger.info("Обучение CatBoost с использованием eval_set...")
                self.current_classifier.fit(
                    X_train,
                    y_train,
                    eval_set=(X_test, y_test),
                    verbose=False,
                    plot=False,
                )
            else:
                self.current_classifier.fit(X_train, y_train)

            # Оценка
            y_pred = self.current_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Генерируем отчет
            report = classification_report(
                y_test, y_pred, target_names=self.triplet_model.label_encoder.classes_
            )

            # Добавляем отчет в репортер
            if reporter:
                reporter.add_classification_report(classifier_type, report)

            self.logger.info(f"Accuracy ({classifier_type}): {accuracy:.4f}")
            self.logger.info(
                f"\nClassification Report для {classifier_type}:\n{report}"
            )

        return accuracy

    def predict(self, texts: List[str]):
        if self.current_classifier is None:
            raise ValueError("Сначала обучите классификатор!")

        embeddings = self.triplet_model.encode_texts(texts)
        predictions = self.current_classifier.predict(embeddings)

        self.logger.info(f"Сделано предсказаний для {len(texts)} текстов")
        return self.triplet_model.label_encoder.inverse_transform(predictions)

    def predict_proba(self, texts: List[str]):
        if self.current_classifier is None:
            raise ValueError("Сначала обучите классификатор!")

        embeddings = self.triplet_model.encode_texts(texts)

        # Проверяем, поддерживает ли классификатор predict_proba
        if hasattr(self.current_classifier, "predict_proba"):
            probabilities = self.current_classifier.predict_proba(embeddings)
        else:
            # Для классификаторов без predict_proba создаем псевдо-вероятности
            self.logger.warning(
                f"Классификатор {type(self.current_classifier).__name__} не поддерживает predict_proba"
            )
            predictions = self.current_classifier.predict(embeddings)
            n_classes = len(self.triplet_model.label_encoder.classes_)
            probabilities = np.zeros((len(texts), n_classes))
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.triplet_model.label_encoder.classes_ == pred)[
                    0
                ][0]
                probabilities[i, class_idx] = 1.0

        self.logger.info(f"Рассчитаны вероятности для {len(texts)} текстов")
        return probabilities

    def evaluate_all_classifiers(
        self, texts: List[str], labels: List[str], test_size: float = 0.2, reporter=None
    ):
        """Оценивает все классификаторы и возвращает результаты"""
        self.logger.info("Начало оценки всех классификаторов")

        results = {}
        best_accuracy = 0
        best_classifier = None

        for classifier_name in self.classifiers.keys():
            self.logger.info(f"Оценка классификатора: {classifier_name}")

            try:
                accuracy = self.train(
                    texts,
                    labels,
                    classifier_type=classifier_name,
                    test_size=test_size,
                    reporter=reporter,
                )

                results[classifier_name] = accuracy

                # Добавляем результаты в репортер
                if reporter:
                    reporter.add_classifier_results(classifier_name, accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_classifier = classifier_name

            except Exception as e:
                self.logger.error(
                    f"Ошибка при оценке классификатора {classifier_name}: {e}"
                )
                results[classifier_name] = 0.0

        if best_classifier is None:
            raise ValueError("Ни один классификатор не был успешно обучен")

        # Обучаем лучший классификатор на всех данных
        self.current_classifier = self.classifiers[best_classifier]
        embeddings = self.triplet_model.encode_texts(texts)
        encoded_labels = self.triplet_model.label_encoder.transform(labels)

        # Специальная обработка для CatBoost
        if best_classifier == "catboost" and CATBOOST_AVAILABLE:
            self.current_classifier.fit(embeddings, encoded_labels, verbose=False)
        else:
            self.current_classifier.fit(embeddings, encoded_labels)

        self.logger.info(
            f"Лучший классификатор: {best_classifier} с точностью: {best_accuracy:.4f}"
        )

        # Логируем сравнение всех классификаторов
        self.logger.info("Сравнение классификаторов:")
        for name, acc in results.items():
            status = "ЛУЧШИЙ" if name == best_classifier else ""
            self.logger.info(f"  {name}: {acc:.4f} {status}")

        return results, best_classifier

    def get_classifier_info(self) -> Dict[str, Any]:
        """Возвращает информацию о доступных классификаторах"""
        info = {}
        for name, classifier in self.classifiers.items():
            info[name] = {
                "type": type(classifier).__name__,
                "parameters": (
                    classifier.get_params() if hasattr(classifier, "get_params") else {}
                ),
                "description": self._get_classifier_description(name),
            }
        return info

    def _get_classifier_description(self, classifier_name: str) -> str:
        """Возвращает описание классификатора"""
        descriptions = {
            "random_forest": "Случайный лес - ансамбль решающих деревьев, устойчивый к переобучению",
            "catboost": "CatBoost - градиентный бустинг от Yandex с отличной обработкой категориальных признаков",
            "lightgbm": "LightGBM - быстрая реализация градиентного бустинга от Microsoft",
        }
        return descriptions.get(classifier_name, "Описание недоступно")

    def get_feature_importance(self, top_n: int = 10) -> Dict[str, Any]:
        """Возвращает важность признаков для текущего классификатора"""
        if self.current_classifier is None:
            raise ValueError("Сначала обучите классификатор!")

        importance_data = {}

        try:
            # Проверяем, поддерживает ли классификатор важность признаков
            if hasattr(self.current_classifier, "feature_importances_"):
                importances = self.current_classifier.feature_importances_

                # Сортируем по убыванию важности
                indices = np.argsort(importances)[::-1]

                importance_data = {
                    "available": True,
                    "importances": importances[indices][:top_n],
                    "indices": indices[:top_n],
                    "classifier_type": type(self.current_classifier).__name__,
                }

                self.logger.info(
                    f"Получена важность признаков для {importance_data['classifier_type']}"
                )
            else:
                importance_data = {
                    "available": False,
                    "message": f"Классификатор {type(self.current_classifier).__name__} не поддерживает важность признаков",
                }
                self.logger.warning(importance_data["message"])

        except Exception as e:
            self.logger.error(f"Ошибка при получении важности признаков: {e}")
            importance_data = {"available": False, "message": f"Ошибка: {str(e)}"}

        return importance_data
