import numpy as np
import pandas as pd
import streamlit as st
from config.settings import config
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """Класс для загрузки и предобработки данных"""

    @staticmethod
    def load_raw_data(file_path):
        """Загрузка сырых данных"""
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                low_memory=False,
                na_values=["?", ""],
                parse_dates={"DateTime": ["Date", "Time"]},
                dayfirst=True,
                infer_datetime_format=True,
            )
            st.info(f"Загружено {len(df):,} записей")
            return df
        except FileNotFoundError:
            st.error(f"Файл {file_path} не найден.")
            return None
        except Exception as e:
            st.error(f"Ошибка загрузки данных: {e}")
            return None

    @staticmethod
    def preprocess_data(df, handle_outliers=True):
        """Предобработка данных с обработкой выбросов"""
        if df is None:
            return None

        st.info("Обработка данных...")

        # Создание DateTime
        if "DateTime" not in df.columns:
            df["DateTime"] = pd.to_datetime(
                df["Date"] + " " + df["Time"],
                format="%d/%m/%Y %H:%M:%S",
                errors="coerce",
            )
            df = df.drop(["Date", "Time"], axis=1)

        # Сортировка по времени
        df = df.sort_values("DateTime").reset_index(drop=True)

        # Заполнение пропущенных значений
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].ffill().bfill()
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())

        # Обработка выбросов
        if handle_outliers:
            df, outliers_count = DataLoader.handle_outliers(df, method="iqr")
            st.info(f"Обработано выбросов: {outliers_count}")

        # Добавление временных признаков
        df = DataLoader._add_temporal_features(df)

        st.success(f"Обработано {len(df):,} записей")
        return df

    @staticmethod
    def handle_outliers(df, method="iqr", columns=None):
        """Обработка выбросов в данных (исключая суб-метры)"""
        if columns is None:
            # Обрабатываем только основные показатели, исключая суб-метры
            columns = [
                "Global_active_power",
                "Global_reactive_power",
                "Voltage",
                "Global_intensity",
            ]

        # Оставляем только существующие колонки
        columns = [col for col in columns if col in df.columns]

        df_clean = df.copy()
        outliers_count = 0

        for column in columns:
            if method == "iqr":
                # Метод межквартильного размаха
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Находим выбросы
                outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
                outliers_count += outliers_mask.sum()

                # Заменяем выбросы на граничные значения
                df_clean[column] = np.where(
                    df_clean[column] < lower_bound, lower_bound, df_clean[column]
                )
                df_clean[column] = np.where(
                    df_clean[column] > upper_bound, upper_bound, df_clean[column]
                )

            elif method == "zscore":
                # Метод Z-score
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers_mask = z_scores > 3
                outliers_count += outliers_mask.sum()

                # Заменяем выбросы на медиану
                median_val = df[column].median()
                df_clean[column] = np.where(outliers_mask, median_val, df_clean[column])

        return df_clean, outliers_count

    @staticmethod
    def detect_anomalies(df, column="Global_active_power", window=24, threshold=3):
        """Обнаружение аномалий с помощью скользящего окна"""
        rolling_mean = df[column].rolling(window=window, center=True).mean()
        rolling_std = df[column].rolling(window=window, center=True).std()

        # Вычисляем Z-score относительно скользящего окна
        z_scores = np.abs((df[column] - rolling_mean) / rolling_std)

        # Находим аномалии
        anomalies = z_scores > threshold
        return anomalies, z_scores

    @staticmethod
    def _add_temporal_features(df):
        """Добавление временных признаков"""
        df["Year"] = df["DateTime"].dt.year
        df["Month"] = df["DateTime"].dt.month
        df["Day"] = df["DateTime"].dt.day
        df["Hour"] = df["DateTime"].dt.hour
        df["DayOfWeek"] = df["DateTime"].dt.dayofweek
        df["Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

        # Добавляем колонку Season только если её нет
        if "Season" not in df.columns:
            df["Season"] = df["Month"].apply(DataLoader._get_season)

        return df

    @staticmethod
    def _get_season(month):
        """Определение сезона по месяцу"""
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"


class LSTMDataPreprocessor:
    """Класс для подготовки данных для LSTM"""

    def __init__(self, sequence_length=24, test_size=0.2, aggregation="minute"):
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.aggregation = aggregation
        self.scaler = MinMaxScaler()

    def prepare_data(self, df, target_column="Global_active_power"):
        """Подготовка данных для LSTM"""
        st.info(f"Подготовка LSTM данных: {self.sequence_length} интервалов")

        # Агрегация данных если нужно
        if self.aggregation == "hour":
            df = self._aggregate_to_hourly(df)

        features = [
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]

        # Проверяем наличие всех признаков
        available_features = [f for f in features if f in df.columns]
        data = df[available_features].values

        scaled_data = self.scaler.fit_transform(data)
        X, y = self._create_sequences(scaled_data, target_idx=0)

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        st.success(
            f"LSTM данные подготовлены: {X_train.shape[0]:,} последовательностей"
        )
        return (X_train, y_train), (X_test, y_test), self.scaler

    def prepare_recent_data(self, df, target_column="Global_active_power", days=30):
        """Подготовка данных только за последние N дней"""
        st.info(f"Подготовка LSTM данных за последние {days} дней")

        # Фильтрация данных за последние N дней
        latest_date = df["DateTime"].max()
        cutoff_date = latest_date - pd.Timedelta(days=days)
        recent_df = df[df["DateTime"] >= cutoff_date].copy()

        st.info(f"Отфильтровано {len(recent_df):,} записей за последние {days} дней")

        # Агрегация данных если нужно
        if self.aggregation == "hour":
            recent_df = self._aggregate_to_hourly(recent_df)

        features = [
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3",
        ]

        # Проверяем наличие всех признаков
        available_features = [f for f in features if f in recent_df.columns]
        data = recent_df[available_features].values

        if len(data) < self.sequence_length:
            st.warning(
                f"Недостаточно данных после фильтрации. Доступно: {len(data)}, требуется: {self.sequence_length}"
            )
            return None, None, None

        scaled_data = self.scaler.fit_transform(data)
        X, y = self._create_sequences(scaled_data, target_idx=0)

        if len(X) == 0:
            st.error("Не удалось создать последовательности из отфильтрованных данных")
            return None, None, None

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        st.success(
            f"LSTM данные подготовлены: {X_train.shape[0]:,} последовательностей за последние {days} дней"
        )
        return (X_train, y_train), (X_test, y_test), self.scaler

    def _aggregate_to_hourly(self, df):
        """Агрегация данных до часовых"""
        hourly_df = (
            df.set_index("DateTime")
            .resample("1H")
            .mean(numeric_only=True)
            .reset_index()
        )
        return hourly_df

    def _create_sequences(self, data, target_idx=0):
        """Создание последовательностей для LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : (i + self.sequence_length)])
            y.append(data[i + self.sequence_length, target_idx])
        return np.array(X), np.array(y)
