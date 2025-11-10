import os

import nltk
from config import Config
from data.processor import DataProcessor, RussianTextPreprocessor
from models.classifiers import EmbeddingClassifier
from models.triplet_autoencoder import TextTripletAutoencoder
from sklearn.metrics import accuracy_score, classification_report
from utils.logger import Logger, log_execution_time
from utils.reporter import ReportGenerator


def setup_environment():
    """Настройка окружения"""
    Config.create_directories()
    logger = Logger.get_logger()

    # Скачивание необходимых данных NLTK
    required_packages = ["punkt", "stopwords", "punkt_tab"]
    for package in required_packages:
        try:
            nltk.download(package, download_dir=Config.NLTK_PATH)
            logger.info(f"Успешно скачан: {package}")
            # Установить путь
            nltk.data.path.append(Config.NLTK_PATH)
        except Exception as e:
            logger.warning(f"Ошибка при скачивании {package}: {e}")

    logger.info("Окружение настроено")


@log_execution_time
def main():
    """Основная функция"""
    # Настройка окружения
    setup_environment()
    logger = Logger.get_logger()
    reporter = ReportGenerator()

    try:
        # Инициализация компонентов
        data_processor = DataProcessor()
        preprocessor = RussianTextPreprocessor()

        # Загрузка данных
        logger.info("Загрузка данных...")
        file_path = "sentiment_dataset.csv"  # Замените на ваш путь

        # Используем демо-данные если файл не найден
        try:
            texts, labels = data_processor.load_and_prepare_data(file_path)
        except Exception as e:
            logger.warning(f"Не удалось загрузить данные из {file_path}: {e}")
            logger.info("Используются демонстрационные данные...")
            texts = [
                "Отличный товар, высокое качество, быстрая доставка",
                "Очень доволен покупкой, все работает прекрасно",
                "Превосходное обслуживание, рекомендую всем",
                "Ужасное качество, товар сломался через день",
                "Не рекомендую, плохой сервис и брак",
                "Худшая покупка в моей жизни, деньги на ветер",
                "Нормальный продукт за свои деньги, без восторгов",
                "Среднего качества, можно было и лучше",
                "Качество на высоте, буду заказывать еще",
                "Быстрая доставка, товар соответствует описанию",
                "Товар не соответствует описанию, очень разочарован",
                "Долгая доставка, плохая упаковка",
            ]
            labels = [
                "positive",
                "positive",
                "positive",
                "negative",
                "negative",
                "negative",
                "neutral",
                "neutral",
                "positive",
                "positive",
                "negative",
                "negative",
            ]

        # Разделение данных
        X_train, X_test, y_train, y_test = data_processor.split_data(
            texts, labels, test_size=Config.TEST_SIZE
        )

        # Инициализация и обучение Triplet Autoencoder
        logger.info("Инициализация Triplet Autoencoder...")
        triplet_ae = TextTripletAutoencoder(
            embedding_dim=Config.EMBEDDING_DIM, use_tfidf=Config.USE_TFIDF
        )
        triplet_ae.set_preprocessor(preprocessor)

        # Обучение модели
        triplet_ae.train(
            X_train,
            y_train,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            learning_rate=Config.LEARNING_RATE,
            reporter=reporter,
        )

        # Создание эмбеддингов
        logger.info("Создание эмбеддингов...")
        train_embeddings = triplet_ae.encode_texts(X_train)
        test_embeddings = triplet_ae.encode_texts(X_test)

        # Классификация
        logger.info("Обучение классификаторов...")
        classifier = EmbeddingClassifier(triplet_ae)

        # Оценка всех классификаторов
        results, best_classifier = classifier.evaluate_all_classifiers(
            X_train, y_train, test_size=Config.TEST_SIZE, reporter=reporter
        )

        # Оценка на тестовой выборке
        logger.info("Оценка на тестовой выборке...")
        test_predictions = classifier.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        logger.info(f"Точность на тестовой выборке: {test_accuracy:.4f}")
        test_report = classification_report(y_test, test_predictions)
        logger.info(f"Отчет по тестовой выборке:\n{test_report}")

        # Примеры предсказаний
        logger.info("Генерация примеров предсказаний...")
        sample_texts = [
            "Очень плохой товар, не рекомендую",
            "Отличное качество, всем советую",
            "Нормальный продукт за свои деньги",
            "Ужасное обслуживание, никогда больше",
            "Прекрасная работа, быстро и качественно",
        ]

        sample_predictions = classifier.predict(sample_texts)
        sample_probabilities = classifier.predict_proba(sample_texts)

        for i, (text, pred, prob) in enumerate(
            zip(sample_texts, sample_predictions, sample_probabilities)
        ):
            logger.info(
                f"Пример {i + 1}: '{text}' -> {pred} (вероятности: {dict(zip(triplet_ae.label_encoder.classes_, prob))})"
            )

        # Поиск похожих текстов
        logger.info("Поиск похожих текстов...")
        query = "отличный сервис и качество"
        similar = triplet_ae.find_similar_texts(query, X_train, top_k=3)

        for idx, score, text in similar:
            logger.info(f"Похожий текст (score: {score:.4f}): {text}")

        # Сохранение модели
        logger.info("Сохранение модели...")
        triplet_ae.save_model(Config.MODEL_SAVE_PATH)

        # Генерация отчетов
        logger.info("Генерация отчетов...")
        reporter.generate_html_report(Config.REPORT_FILE)
        reporter.generate_plots()

        logger.info("Программа успешно завершена!")

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
