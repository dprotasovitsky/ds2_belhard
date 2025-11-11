import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.logger import get_logger


class DataProcessor:
    """Класс для обработки и подготовки данных"""

    def __init__(self):
        self.logger = get_logger("DataProcessor")
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, file_path):
        """Загрузка данных"""
        self.logger.info(f"Загрузка данных из {file_path}")
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Данные успешно загружены. Размер: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            raise

    def preprocess_data(self, df):
        """Предобработка данных"""
        self.logger.info("Начало предобработки данных")
        self.df_processed = df.copy()

        # Обработка graphics_capacity
        self._process_graphics_capacity()

        # Кодирование категориальных признаков
        self._encode_categorical_features()

        # Создание дополнительных признаков
        self._create_additional_features()

        self.logger.info("Предобработка данных завершена")
        return self.df_processed

    def _process_graphics_capacity(self):
        """Обработка graphics_capacity"""
        original_count = len(
            self.df_processed[self.df_processed["graphics_capacity"] == -1]
        )
        self.df_processed["graphics_capacity"] = self.df_processed[
            "graphics_capacity"
        ].replace(-1, 0)
        self.logger.info(
            f"Обработан graphics_capacity: {original_count} записей с -1 заменены на 0"
        )

    def _encode_categorical_features(self):
        """Кодирование категориальных признаков"""
        categorical_columns = [
            "brand_name",
            "processor_type",
            "processor_brand",
            "generations",
            "display_type",
            "color",
        ]

        for col in categorical_columns:
            le = LabelEncoder()
            self.df_processed[col + "_encoded"] = le.fit_transform(
                self.df_processed[col].astype(str)
            )
            self.label_encoders[col] = le
            self.logger.debug(
                f"Закодирован признак {col}: {len(le.classes_)} категорий"
            )

    def _create_additional_features(self):
        """Создание дополнительных признаков"""
        # Тип накопителя
        self.df_processed["storage_type"] = self.df_processed.apply(
            lambda x: "SSD" if x["have_ssd"] else "HDD" if x["have_hdd"] else "Hybrid",
            axis=1,
        )
        self.df_processed["storage_type_encoded"] = LabelEncoder().fit_transform(
            self.df_processed["storage_type"]
        )

        # Игровые ноутбуки
        self.df_processed["is_gaming"] = (
            self.df_processed["graphics_capacity"] > 2
        ).astype(int)
        gaming_count = self.df_processed["is_gaming"].sum()
        self.logger.info(f"Создан признак is_gaming: {gaming_count} игровых ноутбуков")

    def prepare_features(self):
        """Подготовка признаков для кластеризации"""
        self.logger.info("Подготовка признаков для кластеризации")

        features_for_clustering = [
            "ram",
            "storage_capacity_gb",
            "price",
            "display_size_inch",
            "graphics_capacity",
            "brand_name_encoded",
            "processor_type_encoded",
            "processor_brand_encoded",
            "generations_encoded",
            "display_type_encoded",
            "storage_type_encoded",
            "is_gaming",
        ]

        X = self.df_processed[features_for_clustering]
        X_scaled = self.scaler.fit_transform(X)

        self.logger.info(f"Признаки подготовлены. Размерность: {X_scaled.shape}")
        return X_scaled, features_for_clustering

    def get_processed_data(self):
        """Получение обработанных данных"""
        return self.df_processed
