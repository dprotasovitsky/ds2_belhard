# Триплетный автоэнкодер для классификации русского текста

Система для анализа тональности русскоязычных текстов, использующая Triplet Autoencoder для создания семантических эмбеддингов и ансамбль современных алгоритмов машинного обучения для классификации.

## Особенности

* Triplet Autoencoder: Создание качественных эмбеддингов текстов с использованием triplet loss

* Многоязычная предобработка: Специализированная обработка русскоязычных текстов (стемминг, удаление стоп-слов)

* Ансамбль классификаторов: Сравнение трех современных алгоритмов:

    * Random Forest - устойчивый к переобучению

    * CatBoost - продвинутый градиентный бустинг от Yandex

    * LightGBM - высокоскоростной бустинг от Microsoft

* Автоматические отчеты: Генерация детальных HTML отчетов и графиков

* Модульная архитектура: Чистая структура кода для легкого расширения

## Установка

### Требования

* Python 3.8+

* PyTorch 1.9+

* Установка зависимостей:

        bash

        pip install -r requirements.txt

### Зависимости
```
torch
scikit-learn
nltk
pandas
numpy
matplotlib
seaborn
lightgbm
catboost
```
## Структура проекта
```
project/
├── main.py                # Основной скрипт запуска
├── config.py              # Конфигурация параметров
├── requirements.txt       # Зависимости проекта
├── models/
│   ├── triplet_autoencoder.py  # Модель Triplet Autoencoder
│   └── classifiers.py          # Классификаторы
├── data/
│   └── processor.py           # Обработка данных
├── utils/
│   ├── logger.py              # Система логирования
│   └── reporter.py            # Генерация отчетов
└── README.md
```
## Быстрый старт
### 1. Клонирование репозитория
    bash

    git clone https://github.com/dprotasovitsky/ds2_belhard.git
    cd project

### 2. Установка зависимостей
    bash

    pip install -r requirements.txt

### 3. Запуск обучения
    bash

    python main.py

### 4. Использование с собственными данными

Поместите ваш CSV файл с данными в корневую директорию и укажите путь в main.py:

    python

    file_path = "your_dataset.csv"

## Формат данных

Датасет для обучения получен с https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset

Набор данных содержит следующие столбцы:

    text: Текст обзора или комментария.
    label: Метка настроений, где:
        0: Нейтральный
        1: Положительный
        2: Отрицательный
    src: Исходный набор данных, из которого взят текст.

Поддерживаемые форматы CSV:

### Вариант 1: Стандартные названия колонок
    csv

    text,sentiment
    "Отличный товар, быстрая доставка",positive
    "Плохое качество, не рекомендую",negative

### Вариант 2: Автоматическое определение

Система автоматически определит колонки с текстом и метками.
## Конфигурация

Основные параметры в config.py:
```
python

# Параметры модели
EMBEDDING_DIM = 128          # Размерность эмбеддингов
EPOCHS = 30                  # Количество эпох обучения
BATCH_SIZE = 32              # Размер батча
LEARNING_RATE = 0.001        # Скорость обучения

# Классификаторы
CLASSIFIERS = {
    "random_forest": {...},
    "catboost": {...},
    "lightgbm": {...}
}
```
## Результаты

Система автоматически генерирует:

### HTML отчет

  * Сравнение точности классификаторов

  * Графики обучения autoencoder

  * Детальные метрики классификации

  * Временные метки выполнения

### Графики

   * Динамика потерь при обучении

   * Сравнение производительности классификаторов

   * Важность признаков

### Пример вывода
```

Лучший классификатор: lightgbm с точностью: 0.9518
Сравнение классификаторов:
  random_forest: 0.9510
  catboost: 0.9517
  lightgbm: 0.9518 ЛУЧШИЙ
```
## API использования

### Обучение модели
```
python

from models.triplet_autoencoder import TextTripletAutoencoder
from models.classifiers import EmbeddingClassifier

# Инициализация и обучение
triplet_ae = TextTripletAutoencoder(embedding_dim=128)
triplet_ae.train(texts, labels, epochs=30)

# Классификация
classifier = EmbeddingClassifier(triplet_ae)
classifier.train(texts, labels)
predictions = classifier.predict(new_texts)
```
### Создание эмбеддингов
```
python

embeddings = triplet_ae.encode_texts(texts)
```
### Поиск похожих текстов
```
python

similar_texts = triplet_ae.find_similar_texts(
    query="отличный сервис", 
    texts=texts, 
    top_k=5
)
```
## Расширение функциональности

### Добавление нового классификатора

1. Добавьте конфигурацию в config.py:

```
python

"new_classifier": {
    "param1": value1,
    "param2": value2
}
```
2. Добавьте инициализацию в models/classifiers.py:

```
python

if "new_classifier" in classifier_configs:
    classifiers["new_classifier"] = NewClassifier(
        **classifier_configs["new_classifier"]
    )
```
### Кастомная предобработка текста
```
python

from data.processor import RussianTextPreprocessor

class CustomPreprocessor(RussianTextPreprocessor):
    def preprocess(self, text: str) -> str:
        # Ваша кастомная логика
        return processed_text
```
## Метрики качества

   * Accuracy: Общая точность классификации

   * Precision/Recall/F1: Детальные метрики по классам

   * Triplet Loss: Качество семантических эмбеддингов

   * Reconstruction Loss: Касто автоэнкодера
