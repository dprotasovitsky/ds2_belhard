import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from utils.logger import Logger, log_execution_time


class RussianTextPreprocessor:
    def __init__(self):
        self.logger = Logger.get_logger()
        try:
            self.stop_words = set(stopwords.words("russian"))
        except:
            self.stop_words = self._get_default_stopwords()
            self.logger.warning("Используются стандартные стоп-слова")

        self.stemmer = SnowballStemmer("russian")
        self.pattern = re.compile(r"[^а-яА-ЯёЁ\s]")

    def _get_default_stopwords(self):
        return set(
            [
                "и",
                "в",
                "во",
                "не",
                "что",
                "он",
                "на",
                "я",
                "с",
                "со",
                "как",
                "а",
                "то",
                "все",
                "она",
                "так",
                "его",
                "но",
                "да",
                "ты",
                "к",
                "у",
                "же",
                "вы",
                "за",
                "бы",
                "по",
                "только",
                "ее",
                "мне",
                "было",
                "вот",
                "от",
                "меня",
                "еще",
                "нет",
                "о",
                "из",
                "ему",
                "теперь",
                "когда",
                "даже",
                "ну",
                "вдруг",
                "ли",
                "если",
                "уже",
                "или",
                "ни",
                "быть",
                "был",
                "него",
                "до",
                "вас",
                "нибудь",
                "опять",
                "уж",
                "вам",
                "ведь",
                "там",
                "потом",
                "себя",
                "ничего",
                "ей",
                "может",
                "они",
                "тут",
                "где",
                "есть",
                "надо",
                "ней",
                "для",
                "мы",
                "тебя",
                "их",
                "чем",
                "была",
                "сам",
                "чтоб",
                "без",
                "будто",
                "чего",
                "раз",
                "тоже",
                "себе",
                "под",
                "будет",
                "ж",
                "тогда",
                "кто",
                "этот",
                "того",
                "потому",
                "этого",
                "какой",
                "совсем",
                "ним",
                "здесь",
                "этом",
                "один",
                "почти",
                "мой",
                "тем",
                "чтобы",
                "нее",
                "сейчас",
                "были",
                "куда",
                "зачем",
                "всех",
                "никогда",
                "можно",
                "при",
                "наконец",
                "два",
                "об",
                "другой",
                "хоть",
                "после",
                "над",
                "больше",
                "тот",
                "через",
                "эти",
                "нас",
                "про",
                "всего",
                "них",
                "какая",
                "много",
                "разве",
                "три",
                "эту",
                "моя",
                "впрочем",
                "хорошо",
                "свою",
                "этой",
                "перед",
                "иногда",
                "лучше",
                "чуть",
                "том",
                "нельзя",
                "такой",
                "им",
                "более",
                "всегда",
                "конечно",
                "всю",
                "между",
            ]
        )

    # @log_execution_time
    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        try:
            # Приведение к нижнему регистру
            text = text.lower()
            # Удаление пунктуации и цифр
            text = self.pattern.sub(" ", text)

            # Токенизация
            tokens = word_tokenize(text)
            # Удаление стоп-слов и стемминг
            tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]
            return " ".join(tokens)
        except Exception as e:
            self.logger.warning(f"Ошибка при обработке текста: {e}")
            # Простая обработка если nltk не работает
            tokens = text.split()
            tokens = [
                self.stemmer.stem(token)
                for token in tokens
                if token not in self.stop_words and len(token) > 2
            ]
            return " ".join(tokens)


class DataProcessor:
    def __init__(self):
        self.logger = Logger.get_logger()
        self.preprocessor = RussianTextPreprocessor()

    @log_execution_time
    def load_and_prepare_data(self, file_path: str):
        """Загрузка и подготовка данных из CSV файла"""
        try:
            # Загружаем данные
            data = pd.read_csv(file_path)

            # Логируем информацию о данных
            self.logger.info(f"Загружен датасет с {len(data)} строками")
            self.logger.info(f"Колонки: {data.columns.tolist()}")

            # Автоматическое определение колонок
            text_column, label_column = self._detect_columns(data)

            # Очистка данных
            data = data.dropna(subset=[text_column, label_column])
            data[label_column] = data[label_column].astype(str)

            texts = data[text_column].tolist()
            labels = data[label_column].tolist()

            # Логируем статистику
            label_distribution = pd.Series(labels).value_counts().to_dict()
            self.logger.info(f"Распределение меток: {label_distribution}")
            self.logger.info(f"После очистки осталось {len(texts)} примеров")

            return texts, labels

        except Exception as e:
            self.logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def _detect_columns(self, data):
        """Автоматически определяет колонки с текстом и метками"""
        possible_text_columns = [
            "text",
            "comment",
            "review",
            "sentence",
            "message",
            "content",
        ]
        possible_label_columns = [
            "label",
            "sentiment",
            "score",
            "rating",
            "class",
            "category",
        ]

        # Поиск текстовой колонки
        text_column = None
        for col in possible_text_columns:
            if col in data.columns:
                text_column = col
                break
        if text_column is None:
            for col in data.columns:
                if (
                    data[col].dtype == "object"
                    and len(data[col].dropna()) > 0
                    and len(str(data[col].iloc[0])) > 10
                ):
                    text_column = col
                    break

        # Поиск колонки с метками
        label_column = None
        for col in possible_label_columns:
            if col in data.columns:
                label_column = col
                break
        if label_column is None:
            for col in data.columns:
                if (
                    col != text_column and len(data[col].unique()) < 100
                ):  # Предполагаем, что меток не слишком много
                    label_column = col
                    break

        if text_column is None or label_column is None:
            raise ValueError(
                "Не удалось автоматически определить колонки с текстом и метками"
            )

        self.logger.info(f"Определена текстовая колонка: {text_column}")
        self.logger.info(f"Определена колонка с метками: {label_column}")

        return text_column, label_column

    @log_execution_time
    def split_data(self, texts, labels, test_size=0.2, random_state=42):
        """Разделяет данные на обучающую и тестовую выборки"""
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        self.logger.info(f"Обучающая выборка: {len(X_train)} примеров")
        self.logger.info(f"Тестовая выборка: {len(X_test)} примеров")

        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()

        self.logger.info(f"Распределение в обучающей выборке: {train_dist}")
        self.logger.info(f"Распределение в тестовой выборке: {test_dist}")

        return X_train, X_test, y_train, y_test
