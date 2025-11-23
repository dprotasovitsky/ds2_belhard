import logging
import re
import string

import nltk
import numpy as np
from config import config
from nltk.stem.snowball import SnowballStemmer

# Настройка логирования
logger = logging.getLogger(__name__)


# Скачиваем необходимые данные nltk
def download_nltk_data():
    """Скачивание необходимых данных NLTK"""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except:
            nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


download_nltk_data()

# Русский стеммер
try:
    stemmer = SnowballStemmer("russian")
except:
    # Fallback на английский стеммер если русский не доступен
    stemmer = SnowballStemmer("english")

try:
    stop_words = set(nltk.corpus.stopwords.words("russian"))
except:
    stop_words = set()


def preprocess_text(text):
    """
    Предварительная обработка текста
    """
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()
    # Удаление символов, кроме букв и основных знаков препинания
    text = re.sub(r"[^а-яёa-z0-9\s\.,!?]", "", text)
    return text


def tokenize(sentence):
    """
    Разбивает предложение на массив слов/токенов
    """
    try:
        sentence = preprocess_text(sentence)
        tokens = nltk.word_tokenize(sentence, language="russian")
        # Удаляем стоп-слова и короткие токены
        tokens = [
            token for token in tokens if token not in stop_words and len(token) > 1
        ]
        return tokens
    except Exception as e:
        logger.error(f"Ошибка токенизации: {e}")
        return []


def stem(word):
    """
    Стемминг: приведение слова к основной форме
    """
    try:
        return stemmer.stem(word.lower())
    except:
        return word.lower()


def bag_of_words(tokenized_sentence, words):
    """
    Возвращает мешок слов массива для предложения
    """
    try:
        sentence_words = [stem(word) for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)

        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1

        return bag
    except Exception as e:
        logger.error(f"Ошибка создания мешка слов: {e}")
        return np.zeros(len(words), dtype=np.float32)


def create_vocabulary(patterns, min_freq=1):
    """
    Создает словарь из паттернов
    """
    all_words = []
    for pattern in patterns:
        tokens = tokenize(pattern)
        all_words.extend([stem(word) for word in tokens])

    # Подсчет частотности
    word_freq = {}
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Фильтрация по минимальной частоте
    vocabulary = [word for word, freq in word_freq.items() if freq >= min_freq]

    logger.info(
        f"Создан словарь из {len(vocabulary)} слов (минимальная частота: {min_freq})"
    )
    return sorted(vocabulary)
