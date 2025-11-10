Triplet Autoencoder for Russian Text Classification

Система для анализа тональности русскоязычных текстов, использующая Triplet Autoencoder для создания семантических эмбеддингов и ансамбль современных алгоритмов машинного обучения для классификации.
Особенности

    Triplet Autoencoder: Создание качественных эмбеддингов текстов с использованием triplet loss

    Многоязычная предобработка: Специализированная обработка русскоязычных текстов (стемминг, удаление стоп-слов)

    Ансамбль классификаторов: Сравнение трех современных алгоритмов:

        Random Forest - устойчивый к переобучению

        CatBoost - продвинутый градиентный бустинг от Yandex

        LightGBM - высокоскоростной бустинг от Microsoft

    Автоматические отчеты: Генерация детальных HTML отчетов и графиков

    Модульная архитектура: Чистая структура кода для легкого расширения

Установка
Требования

    Python 3.8+

    PyTorch 1.9+

    Установка зависимостей:

bash

pip install -r requirements.txt

Зависимости
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
Структура проекта
```

project/
├── main.py                 # Основной скрипт запуска
├── config.py              # Конфигурация параметров
├── requirements.txt       # Зависимости проекта
├── models/
│   ├── __init__.py
│   ├── triplet_autoencoder.py  # Модель Triplet Autoencoder
│   └── classifiers.py          # Классификаторы
├── data/
│   ├── __init__.py
│   └── processor.py           # Обработка данных
├── utils/
│   ├── __init__.py
│   ├── logger.py              # Система логирования
│   └── reporter.py            # Генерация отчетов
└── README.md
```
Быстрый старт
1. Клонирование репозитория
bash

git clone https://github.com/dprotasovitsky/ds2_belhard.git
cd project

2. Установка зависимостей
bash

pip install -r requirements.txt

3. Запуск обучения
bash

python main.py

4. Использование с собственными данными

Поместите ваш CSV файл с данными в корневую директорию и укажите путь в main.py:
python

file_path = "your_dataset.csv"

Формат данных

Поддерживаемые форматы CSV:
Вариант 1: Стандартные названия колонок
csv

text,sentiment
"Отличный товар, быстрая доставка",positive
"Плохое качество, не рекомендую",negative

Вариант 2: Автоматическое определение

Система автоматически определит колонки с текстом и метками.
Конфигурация

Основные параметры в config.py:
```

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
Результаты

Система автоматически генерирует:
HTML отчет

    Сравнение точности классификаторов

    Графики обучения autoencoder

    Детальные метрики классификации

    Временные метки выполнения

Графики

    Динамика потерь при обучении

    Сравнение производительности классификаторов

    Важность признаков

Пример вывода
```

Лучший классификатор: catboost с точностью: 0.8945

Сравнение классификаторов:
  random_forest: 0.8721
  catboost: 0.8945 ЛУЧШИЙ
  lightgbm: 0.8852
```
API использования
Обучение модели
```

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
Создание эмбеддингов
```

embeddings = triplet_ae.encode_texts(texts)
```
Поиск похожих текстов
```

similar_texts = triplet_ae.find_similar_texts(
    query="отличный сервис", 
    texts=texts, 
    top_k=5
)
```
Расширение функциональности
Добавление нового классификатора

    Добавьте конфигурацию в config.py:

```

"new_classifier": {
    "param1": value1,
    "param2": value2
}
```
    Добавьте инициализацию в models/classifiers.py:

```

if "new_classifier" in classifier_configs:
    classifiers["new_classifier"] = NewClassifier(
        **classifier_configs["new_classifier"]
    )
```
Кастомная предобработка текста
```

from data.processor import RussianTextPreprocessor

class CustomPreprocessor(RussianTextPreprocessor):
    def preprocess(self, text: str) -> str:
        # Ваша кастомная логика
        return processed_text
```
Метрики качества

    Accuracy: Общая точность классификации

    Precision/Recall/F1: Детальные метрики по классам

    Triplet Loss: Качество семантических эмбеддингов

    Reconstruction Loss: Касто автоэнкодера
