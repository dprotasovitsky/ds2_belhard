import os
import sqlite3
import warnings
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Импорт модулей
from config.settings import config
from data.database import DatabaseManager
from data.loader import DataLoader, LSTMDataPreprocessor
from models.model_manager import ModelManager
from models.optimizer import FastLSTMOptimizer, LSTMOptimizer
from models.trainer import ModelTrainer
from plotly.subplots import make_subplots
from utils.helpers import export_to_excel
from visualization.dashboard import DashboardVisualizer
from visualization.lstm_results import LSTMResultVisualizer

warnings.filterwarnings("ignore")


# Настройка страницы
st.set_page_config(
    page_title="Energy Consumption Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Инициализация состояния сессии"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "database_created" not in st.session_state:
        st.session_state.database_created = False
    if "queries" not in st.session_state:
        st.session_state.queries = {}
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "current_model_path" not in st.session_state:
        st.session_state.current_model_path = None
    if "plot_counter" not in st.session_state:
        st.session_state.plot_counter = 0


def get_unique_key(prefix):
    """Генерирует уникальный ключ для элементов Streamlit"""
    st.session_state.plot_counter += 1
    return f"{prefix}_{st.session_state.plot_counter}"


def main():
    """Главная функция приложения"""

    st.title("Аналитический сервис потребления электроэнергии")
    st.markdown("---")

    # Инициализация состояния
    init_session_state()

    # Боковая панель
    st.sidebar.header("Настройки и управление")

    # Загрузка данных
    with st.sidebar.expander("Загрузка данных", expanded=True):
        if st.button("Загрузить данные", key="load_data_main", width="stretch"):
            with st.spinner("Загрузка и обработка данных..."):
                data_loader = DataLoader()
                raw_data = data_loader.load_raw_data(config.DATA_FILE)

                if raw_data is not None:
                    df = data_loader.preprocess_data(raw_data)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.success("Данные успешно загружены!")
                    else:
                        st.error("Ошибка при обработке данных")
                else:
                    st.error("Не удалось загрузить данные")

    # Создание базы данных
    with st.sidebar.expander("База данных"):
        if st.button("Создать/Обновить БД", key="create_db_main", width="stretch"):
            if st.session_state.data_loaded:
                with st.spinner("Создание базы данных..."):
                    try:
                        db_manager = DatabaseManager()
                        db_manager.create_database(st.session_state.df)

                        # Выполнение аналитических запросы
                        queries = db_manager.run_analysis_queries()
                        st.session_state.queries = queries
                        st.session_state.database_created = True

                        st.success(
                            "База данных создана и аналитические запросы выполнены!"
                        )
                    except Exception as e:
                        st.error(f"Ошибка при создании базы данных: {e}")
            else:
                st.error("Сначала загрузите данные")

    # Проверка загруженных данных
    if not st.session_state.data_loaded:
        st.info("Загрузите данные с помощью панели слева для начала работы")
        show_data_info()
        return

    df = st.session_state.df

    # Определение доступных вкладок
    tabs = ["Обзор данных", "LSTM Прогноз", "Отчеты"]

    # Добавляем вкладку аналитики только если база данных создана
    if st.session_state.database_created:
        tabs.insert(1, "Аналитика")

    # Создание вкладок
    selected_tabs = st.tabs(tabs)

    # Обработка вкладок
    for i, tab in enumerate(selected_tabs):
        with tab:
            if tabs[i] == "Обзор данных":
                overview_tab(df)
            elif tabs[i] == "Аналитика" and st.session_state.database_created:
                analytics_tab(df)
            elif tabs[i] == "LSTM Прогноз":
                lstm_analysis_tab(df)
            elif tabs[i] == "Отчеты":
                reports_tab(df, st.session_state.queries)

    # Информация в боковой панели
    show_sidebar_info()


def show_data_info():
    """Показать информацию о данных"""
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
    Информация о данных:

    Период: {config.DATA_INFO['period']}
    Частота: {config.DATA_INFO['frequency']}
    Показатели:
    • {config.DATA_INFO['metrics'][0]}
    • {config.DATA_INFO['metrics'][1]}
    • {config.DATA_INFO['metrics'][2]}
    • {config.DATA_INFO['metrics'][3]}
    • {config.DATA_INFO['metrics'][4]}
    • {config.DATA_INFO['metrics'][5]}
    • {config.DATA_INFO['metrics'][6]}
    """
    )


def show_sidebar_info():
    """Показать информацию в боковой панели"""
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"""
    Информация о данных:

    Период: {config.DATA_INFO['period']}
    Частота: {config.DATA_INFO['frequency']}
    Показатели:
    • {config.DATA_INFO['metrics'][0]}
    • {config.DATA_INFO['metrics'][1]}
    • {config.DATA_INFO['metrics'][2]}
    • {config.DATA_INFO['metrics'][3]}
    • {config.DATA_INFO['metrics'][4]}
    • {config.DATA_INFO['metrics'][5]}
    • {config.DATA_INFO['metrics'][6]}
    """
    )

    # Статус базы данных
    if st.session_state.database_created:
        st.sidebar.success("База данных создана")
    else:
        st.sidebar.warning("Создайте базу данных для доступа к аналитике")

    # Статус модели
    if st.session_state.model_loaded:
        st.sidebar.success("Модель загружена")
    elif "trainer" in st.session_state and st.session_state.trainer.model is not None:
        st.sidebar.info("Модель обучена")


def overview_tab(df):
    """Вкладка обзора данных"""
    st.header("Обзор минутных данных энергопотребления")

    # Информация о датасете
    st.info(
        f"""
    Характеристики датасета:
    - Период: {df['DateTime'].min().strftime('%Y-%m-%d')} - {df['DateTime'].max().strftime('%Y-%m-%d')}
    - Всего записей: {len(df):,} минутных измерений
    - Длительность: {(df['DateTime'].max() - df['DateTime'].min()).days} дней
    - Полнота данных: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%
    """
    )

    # Метрики
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_power = df["Global_active_power"].mean()
        st.metric("Средняя мощность", f"{avg_power:.2f} kW")

    with col2:
        total_records = len(df)
        st.metric("Всего записей", f"{total_records:,}")

    with col3:
        date_range = f"{df['DateTime'].min().date()} - {df['DateTime'].max().date()}"
        st.metric("Период данных", date_range)

    with col4:
        max_power = df["Global_active_power"].max()
        st.metric("Пиковая мощность", f"{max_power:.2f} kW")

    # Визуализатор дашборда
    dashboard_viz = DashboardVisualizer(df, st.session_state.queries)

    # Временной ряд
    st.subheader("Временной ряд потребления")
    sample_size = min(config.SAMPLE_SIZE, len(df))
    sample_df = df.sample(n=sample_size).sort_values("DateTime")
    fig = dashboard_viz._plot_time_series(sample_df)
    st.plotly_chart(fig, width="stretch", key=get_unique_key("time_series_overview"))

    # Распределение потребления
    st.subheader("Распределение потребления")
    col1, col2 = st.columns(2)

    with col1:
        fig = dashboard_viz._plot_distribution()
        st.plotly_chart(fig, width="stretch", key=get_unique_key("distribution_plot"))

    with col2:
        # Заменяем сезонный график на потребление по часам, если Season недоступен
        if "Season" in df.columns:
            fig = dashboard_viz._plot_seasonal_consumption()
        else:
            # Создаем график потребления по часам
            hourly_avg = df.groupby("Hour")["Global_active_power"].mean().reset_index()
            fig = px.bar(
                hourly_avg,
                x="Hour",
                y="Global_active_power",
                title="Среднее потребление по часам",
            )
            fig.update_layout(template=config.PLOT_THEME)
        st.plotly_chart(fig, width="stretch", key=get_unique_key("seasonal_plot"))


def analytics_tab(df):
    """Вкладка аналитики"""
    st.header("Аналитика потребления")

    if not st.session_state.database_created:
        st.warning(
            """
        Аналитика временно недоступна

        Для доступа к расширенной аналитике необходимо создать базу данных.
        Перейдите в раздел База данных на боковой панели и нажмите
        "Создать/Обновить БД".
        """
        )
        return

    st.success("База данных создана - аналитика доступна!")

    dashboard_viz = DashboardVisualizer(df, st.session_state.queries)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Временные паттерны")
        fig = dashboard_viz._plot_hourly_consumption()
        st.plotly_chart(fig, width="stretch", key=get_unique_key("hourly_consumption"))

        fig = dashboard_viz._plot_weekly_consumption()
        st.plotly_chart(fig, width="stretch", key=get_unique_key("weekly_consumption"))

    with col2:
        st.subheader("Статистический анализ")
        fig = dashboard_viz._plot_monthly_consumption()
        st.plotly_chart(fig, width="stretch", key=get_unique_key("monthly_consumption"))

        fig = dashboard_viz._plot_correlation_matrix()
        st.plotly_chart(fig, width="stretch", key=get_unique_key("correlation_matrix"))

    # Дополнительная аналитика
    st.subheader("Детальная аналитика")

    if st.session_state.queries.get("weekday_weekend"):
        col1, col2 = st.columns(2)

        with col1:
            fig = dashboard_viz._plot_weekday_weekend_comparison()
            st.plotly_chart(fig, width="stretch", key=get_unique_key("weekday_weekend"))

        with col2:
            fig = dashboard_viz._plot_submeter_analysis()
            st.plotly_chart(
                fig, width="stretch", key=get_unique_key("submeter_analysis")
            )

    # Добавляем дополнительную информацию о помещениях
    st.subheader("Потребление по помещениям")
    fig = dashboard_viz._plot_room_consumption_details()
    st.plotly_chart(fig, width="stretch", key=get_unique_key("room_consumption"))


def lstm_analysis_tab(df):
    """Вкладка LSTM анализа"""
    st.header("LSTM Прогнозирование Потребления Энергии")

    # Информация о доступных данных
    latest_date = df["DateTime"].max()
    thirty_days_ago = latest_date - pd.Timedelta(days=30)
    records_last_30_days = len(df[df["DateTime"] >= thirty_days_ago])

    st.info(
        f"""
        **Информация о данных:**
        - Всего записей: {len(df):,}
        - За последние 30 дней: {records_last_30_days:,} записей
        - Период данных: {df['DateTime'].min().date()} - {df['DateTime'].max().date()}

        **Рекомендация:** Используйте "Последние 30 дней" для учета сезонных изменений и актуальных паттернов.
        """
    )

    # Инициализация менеджера моделей
    model_manager = ModelManager()

    # Боковая панель - настройки LSTM
    with st.sidebar.expander("LSTM Настройки", expanded=False):
        # Выбор периода данных
        data_period = st.selectbox(
            "Период данных для обучения",
            options=[
                "Весь датасет",
                "Последние 30 дней",
                "Последние 60 дней",
                "Последние 7 дней",
            ],
            index=0,
            help="Выберите период данных для обучения модели",
            key="data_period",
        )

        # Определяем количество дней в зависимости от выбора
        if data_period == "Последние 30 дней":
            recent_days = 30
        elif data_period == "Последние 60 дней":
            recent_days = 60
        elif data_period == "Последние 7 дней":
            recent_days = 7
        else:
            recent_days = None  # Весь датасет

        # Выбор стратегии оптимизации
        optimization_strategy = st.selectbox(
            "Стратегия оптимизации",
            options=["Быстрая", "Стандартная"],
            index=0,
            help="Выберите стратегию для баланса скорости и качества",
            key="optimization_strategy",
        )

        # Настройки в зависимости от стратегии
        if optimization_strategy == "Быстрая":
            n_trials = st.slider("Количество trials", 10, 50, 20, key="fast_trials")
            timeout = st.slider("Таймаут (секунды)", 300, 1800, 600, key="fast_timeout")
        else:  # Стандартная
            n_trials = st.slider(
                "Количество trials", 20, 100, 50, key="standard_trials"
            )
            timeout = st.slider(
                "Таймаут (секунды)", 600, 3600, 1800, key="standard_timeout"
            )

        # Настройки данных
        aggregation = st.selectbox(
            "Агрегация данных",
            options=["minute", "hour"],
            index=1,
            help="Минутные данные или агрегированные часовые",
            key="aggregation",
        )

        if aggregation == "minute":
            sequence_length = st.slider(
                "Длина последовательности (минуты)",
                60,
                1440,
                720,
                step=60,
                key="seq_length_minutes",
            )
        else:
            sequence_length = st.slider(
                "Длина последовательности (часы)", 6, 168, 24, key="seq_length_hours"
            )

        test_size = st.slider("Размер тестовой выборки", 0.1, 0.3, 0.2, key="test_size")

    # Боковая панель - управление моделями
    with st.sidebar.expander("Управление моделями", expanded=False):
        saved_models = model_manager.list_saved_models()

        if saved_models:
            st.subheader("Сохраненные модели")
            model_names = [model["model_name"] for model in saved_models]
            selected_model = st.selectbox(
                "Выберите модель для загрузки:",
                options=model_names,
                index=0,
                key="model_selector",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Загрузить модель", width="stretch", key="load_model_btn"):
                    model_path = os.path.join("saved_models", selected_model)
                    if "trainer" not in st.session_state:
                        st.session_state.trainer = ModelTrainer({}, 1)

                    success = st.session_state.trainer.load_trained_model(model_path)
                    if success:
                        st.session_state.model_loaded = True
                        st.session_state.current_model_path = model_path
                        st.success(f"Модель {selected_model} загружена!")
                        st.rerun()

            with col2:
                if st.button("Удалить", width="stretch", key="delete_model_btn"):
                    if delete_model(selected_model):
                        st.success(f"Модель {selected_model} удалена")
                        st.rerun()
        else:
            st.info("Нет сохраненных моделей")

    # Подготовка данных для LSTM
    with st.expander("1. Подготовка данных для LSTM", expanded=True):
        preprocessor = LSTMDataPreprocessor(
            sequence_length=sequence_length,
            test_size=test_size,
            aggregation=aggregation,
        )

        if st.button("Подготовить данные", key="prepare_data_btn"):
            with st.spinner("Подготовка данных для LSTM..."):
                if recent_days:
                    result = preprocessor.prepare_recent_data(df, days=recent_days)
                else:
                    result = preprocessor.prepare_data(df)

                if result is None:
                    st.error(
                        "Не удалось подготовить данные. Попробуйте изменить параметры."
                    )
                else:
                    (X_train, y_train), (X_test, y_test), scaler = result

                    # ПРОВЕРКА: Достаточно ли данных для разделения
                    if len(X_train) == 0:
                        st.error(
                            "Недостаточно данных для обучения. Попробуйте увеличить период данных."
                        )
                        return

                    # ПРОВЕРКА: Достаточно ли данных для validation
                    min_validation_size = 10
                    split_idx = max(min_validation_size, int(len(X_train) * 0.8))

                    if split_idx >= len(X_train):
                        st.error("Недостаточно данных для создания validation set.")
                        return

                    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
                    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]

                    # ПРОВЕРКА: Есть ли тестовые данные
                    if len(X_test) == 0:
                        st.warning(
                            "Тестовых данных недостаточно. Будет использовано меньшее количество."
                        )
                        # Используем часть validation данных для тестирования
                        if len(X_val) > 10:
                            val_split = len(X_val) // 2
                            X_test, y_test = X_val[:val_split], y_val[:val_split]
                            X_val, y_val = X_val[val_split:], y_val[val_split:]

                    st.session_state.lstm_data = {
                        "X_train": X_train_split,
                        "y_train": y_train_split,
                        "X_val": X_val,
                        "y_val": y_val,
                        "X_test": X_test,
                        "y_test": y_test,
                        "scaler": scaler,
                        "input_size": X_train.shape[2] if len(X_train) > 0 else 0,
                        "data_period": (
                            f"Последние {recent_days} дней"
                            if recent_days
                            else "Весь датасет"
                        ),
                    }

                if st.session_state.lstm_data["input_size"] > 0:
                    data_info = st.session_state.lstm_data
                    st.success(
                        f"""
                    Данные подготовлены ({data_info['data_period']}):
                    - Train: {data_info['X_train'].shape}
                    - Validation: {data_info['X_val'].shape}
                    - Test: {data_info['X_test'].shape}
                    - Признаков: {data_info['input_size']}
                    """
                    )

    # Оптимизация гиперпараметров
    with st.expander("2. Оптимизация гиперпараметров", expanded=False):
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button(
                "Запустить оптимизацию", key="run_optuna_btn", width="stretch"
            ):
                if "lstm_data" in st.session_state:
                    data = st.session_state.lstm_data

                    with st.spinner(f"Оптимизация ({optimization_strategy})..."):
                        try:
                            if optimization_strategy == "Быстрая":
                                optimizer = FastLSTMOptimizer(
                                    data["X_train"],
                                    data["y_train"],
                                    data["X_val"],
                                    data["y_val"],
                                    data["input_size"],
                                )
                                study = optimizer.optimize(
                                    n_trials=n_trials, timeout=timeout
                                )
                            else:  # Стандартная
                                optimizer = LSTMOptimizer(
                                    data["X_train"],
                                    data["y_train"],
                                    data["X_val"],
                                    data["y_val"],
                                    data["input_size"],
                                )
                                study = optimizer.optimize(n_trials=n_trials)

                            joblib.dump(study, "optuna_study.pkl")
                            st.session_state.study = study

                            st.success("Оптимизация завершена!")

                            best_params = study.best_trial.params
                            st.write("Лучшие параметры:")
                            st.json(best_params)

                            # Показываем статистику оптимизации
                            st.write("Статистика:")
                            st.write(f"- Выполнено trials: {len(study.trials)}")
                            st.write(f"- Лучшее значение: {study.best_value:.6f}")
                            if (
                                study.trials
                                and study.trials[0].datetime_start
                                and study.trials[-1].datetime_complete
                            ):
                                duration = (
                                    study.trials[-1].datetime_complete
                                    - study.trials[0].datetime_start
                                )
                                st.write(f"- Время выполнения: {duration}")

                        except Exception as e:
                            st.error(f"Ошибка оптимизации: {e}")
                else:
                    st.error("Сначала подготовьте данные для LSTM")

        with col2:
            if "study" in st.session_state:
                study = st.session_state.study

                # Визуализация оптимизации
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        y=[t.value for t in study.trials if t.value is not None],
                        mode="lines+markers",
                        name="Validation Loss",
                    )
                )
                fig.update_layout(
                    title="История оптимизации",
                    xaxis_title="Trial",
                    yaxis_title="Loss",
                    template=config.PLOT_THEME,
                )
                st.plotly_chart(
                    fig, width="stretch", key=get_unique_key("optimization_history")
                )
            else:
                st.info("Запустите оптимизацию для отображения графиков")

    # Обучение модели
    with st.expander("3. Обучение LSTM модели", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Обучить модель", key="train_model_btn", width="stretch"):
                if "study" in st.session_state and "lstm_data" in st.session_state:
                    study = st.session_state.study
                    data = st.session_state.lstm_data
                    best_params = study.best_trial.params

                    with st.spinner("Обучение модели..."):
                        trainer = ModelTrainer(best_params, data["input_size"])
                        history = trainer.train_model(
                            data["X_train"],
                            data["y_train"],
                            data["X_val"],
                            data["y_val"],
                            epochs=100,
                            save_model=True,
                        )

                        # Оценка модели
                        metrics, predictions, y_true = trainer.evaluate_model(
                            data["X_test"], data["y_test"], data["scaler"]
                        )

                        st.session_state.trainer = trainer
                        st.session_state.metrics = metrics
                        st.session_state.predictions = predictions
                        st.session_state.y_true = y_true
                        st.session_state.model_loaded = True

                        st.success("Модель обучена и сохранена!")

                        # Показываем метрики
                        show_training_results(metrics, trainer, data)

                else:
                    st.error("Сначала выполните оптимизацию гиперпараметров")

        with col2:
            if st.button("Сохранить модель", key="save_model_btn", width="stretch"):
                if (
                    "trainer" in st.session_state
                    and st.session_state.trainer.model is not None
                ):
                    model_path = st.session_state.trainer.save_trained_model()
                    if model_path:
                        st.session_state.current_model_path = model_path
                        st.success(f"Модель сохранена: {os.path.basename(model_path)}")
                else:
                    st.error("Нет обученной модели для сохранения")

        with col3:
            if st.button("Сравнить модели", key="compare_models_btn", width="stretch"):
                compare_saved_models(model_manager)

        # Показ результатов обучения если есть
        if "metrics" in st.session_state:
            show_training_results(
                st.session_state.metrics,
                st.session_state.trainer,
                st.session_state.lstm_data,
            )

    # Прогнозирование и анализ
    with st.expander("4. Прогнозирование и анализ", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Выполнить прогноз", key="predict_current_btn", width="stretch"
            ):
                if (
                    "trainer" in st.session_state
                    and st.session_state.trainer.model is not None
                ):
                    if "lstm_data" in st.session_state:
                        data = st.session_state.lstm_data
                        metrics, predictions, y_true = (
                            st.session_state.trainer.evaluate_model(
                                data["X_test"], data["y_test"], data["scaler"]
                            )
                        )

                        st.session_state.metrics = metrics
                        st.session_state.predictions = predictions
                        st.session_state.y_true = y_true

                        st.success("Прогнозирование выполнено!")
                        show_prediction_results(predictions, y_true, metrics)
                    else:
                        st.error("Данные для прогнозирования не найдены")
                else:
                    st.error("Сначала обучите или загрузите модель")

        with col2:
            if st.button(
                "Анализ остатков", key="analyze_residuals_btn", width="stretch"
            ):
                if all(key in st.session_state for key in ["trainer", "lstm_data"]):
                    show_residuals_analysis()
                else:
                    st.error("Сначала выполните прогнозирование")


def show_training_results(metrics, trainer, data):
    """Показать результаты обучения"""
    data_period_info = data.get("data_period", "Неизвестный период")
    st.subheader(f"Результаты обучения ({data_period_info})")

    # Метрики
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_cols = [col1, col2, col3, col4, col5]

    for i, (metric_name, metric_value) in enumerate(metrics.items()):
        with metric_cols[i]:
            if metric_name == "MAPE":
                st.metric(metric_name, f"{metric_value:.1f}%")
            elif metric_name == "R2":
                st.metric(metric_name, f"{metric_value:.3f}")
            else:
                st.metric(metric_name, f"{metric_value:.3f}")

    # Визуализация результатов с проверкой наличия данных
    visualizer = LSTMResultVisualizer(trainer, data["scaler"])

    col1, col2 = st.columns(2)
    with col1:
        if len(trainer.history["train_loss"]) > 0:
            fig1 = visualizer.plot_training_history()
            st.plotly_chart(
                fig1, width="stretch", key=get_unique_key("training_history")
            )
        else:
            st.info("История обучения недоступна")

    with col2:
        if "X_test" in data and len(data["X_test"]) > 0:
            # Автоматически подбираем размер выборки
            sample_size = min(200, len(data["y_test"]))
            fig2 = visualizer.plot_predictions_vs_actual(
                data["X_test"], data["y_test"], num_samples=sample_size
            )
            st.plotly_chart(
                fig2, width="stretch", key=get_unique_key("predictions_vs_actual")
            )
        else:
            st.info("Тестовые данные недоступны для визуализации")

    if "X_test" in data and len(data["X_test"]) > 0:
        # Автоматически подбираем длину временного ряда
        series_length = min(100, len(data["y_test"]))
        fig3 = visualizer.plot_time_series_prediction(
            data["X_test"], data["y_test"], length=series_length
        )
        st.plotly_chart(
            fig3, width="stretch", key=get_unique_key("time_series_prediction")
        )


def show_prediction_results(predictions, y_true, metrics):
    """Показать результаты прогнозирования"""
    st.subheader("Детальные прогнозы")

    # Таблица с последними прогнозами
    display_count = min(50, len(y_true))
    results_df = pd.DataFrame(
        {
            "Фактические": y_true[-display_count:],
            "Предсказанные": predictions[-display_count:],
            "Абсолютная ошибка": np.abs(
                y_true[-display_count:] - predictions[-display_count:]
            ),
            "Относительная ошибка %": (
                np.abs(y_true[-display_count:] - predictions[-display_count:])
                / np.where(y_true[-display_count:] != 0, y_true[-display_count:], 1)
            )
            * 100,
        }
    )

    st.dataframe(
        results_df.style.format(
            {
                "Фактические": "{:.3f}",
                "Предсказанные": "{:.3f}",
                "Абсолютная ошибка": "{:.3f}",
                "Относительная ошибка %": "{:.1f}%",
            }
        ),
        key=get_unique_key("predictions_table"),
    )

    # Статистика ошибок
    st.subheader("Статистика ошибок прогнозирования")
    error_stats = {
        "Средняя абсолютная ошибка": np.mean(np.abs(y_true - predictions)),
        "Медианная абсолютная ошибка": np.median(np.abs(y_true - predictions)),
        "Максимальная ошибка": np.max(np.abs(y_true - predictions)),
        "Std ошибок": np.std(y_true - predictions),
    }

    for stat_name, stat_value in error_stats.items():
        st.write(f"{stat_name}: {stat_value:.3f}")


def show_residuals_analysis():
    """Показать анализ остатков"""
    if all(key in st.session_state for key in ["trainer", "lstm_data"]):
        visualizer = LSTMResultVisualizer(
            st.session_state.trainer, st.session_state.lstm_data["scaler"]
        )

        fig = visualizer.plot_residuals_analysis(
            st.session_state.lstm_data["X_test"], st.session_state.lstm_data["y_test"]
        )
        st.plotly_chart(fig, width="stretch", key=get_unique_key("residuals_analysis"))


def compare_saved_models(model_manager):
    """Сравнение сохраненных моделей"""
    saved_models = model_manager.list_saved_models()

    if len(saved_models) < 2:
        st.warning("Для сравнения нужно至少 2 сохраненные модели")
        return

    st.subheader("Сравнение моделей")

    comparison_data = []
    for model_info in saved_models:
        model_path = os.path.join("saved_models", model_info["model_name"])
        _, _, metrics, _ = model_manager.load_model(model_path)

        if metrics and isinstance(metrics, dict):
            comparison_data.append(
                {
                    "Model": model_info["model_name"],
                    "Saved": model_info["saved_at"][:16],
                    "RMSE": metrics.get("RMSE", 0),
                    "MAE": metrics.get("MAE", 0),
                    "R2": metrics.get("R2", 0),
                    "MAPE": metrics.get("MAPE", 0),
                }
            )

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(
            comparison_df.style.format(
                {"RMSE": "{:.4f}", "MAE": "{:.4f}", "R2": "{:.4f}", "MAPE": "{:.2f}%"}
            ),
            key=get_unique_key("models_comparison_table"),
        )

        # Визуализация сравнения
        fig = go.Figure()
        metrics_to_plot = ["RMSE", "MAE", "R2"]

        for metric in metrics_to_plot:
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=comparison_df["Model"].tolist(),
                    y=comparison_df[metric].tolist(),
                )
            )

        fig.update_layout(
            title="Сравнение метрик моделей",
            barmode="group",
            template=config.PLOT_THEME,
        )
        st.plotly_chart(
            fig, width="stretch", key=get_unique_key("models_comparison_chart")
        )


def delete_model(model_name):
    """Удаление сохраненной модели"""
    try:
        import shutil

        model_path = os.path.join("saved_models", model_name)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            return True
    except Exception as e:
        st.error(f"Ошибка удаления модели: {e}")
    return False


def reports_tab(df, queries):
    """Вкладка отчетов"""
    st.header("Отчеты и экспорт")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Экспорт данных")

        if st.button("Экспорт в Excel", key="export_excel", width="stretch"):
            success, result = export_to_excel(df, queries)
            if success:
                st.success(f"Данные экспортированы в {result}")
            else:
                st.error(f"Ошибка экспорта: {result}")

        if st.button("Сводный отчет", key="summary_report", width="stretch"):
            show_summary_report(df)

    with col2:
        st.subheader("Анализ качества моделей")

        if "metrics" in st.session_state:
            metrics = st.session_state.metrics

            st.write("Метрики LSTM модели:")
            for metric_name, metric_value in metrics.items():
                if metric_name == "MAPE":
                    st.write(f"- {metric_name}: {metric_value:.1f}%")
                elif metric_name == "R2":
                    st.write(f"- {metric_name}: {metric_value:.3f}")
                else:
                    st.write(f"- {metric_name}: {metric_value:.3f}")

            # Оценка качества модели
            if metrics.get("R2", 0) > 0.8:
                st.success("Отличное качество модели")
            elif metrics.get("R2", 0) > 0.6:
                st.warning("Хорошее качество модели")
            else:
                st.error("Низкое качество модели")
        else:
            st.info("Обучите LSTM модель для просмотра метрик")

        # Информация о текущей модели
        if st.session_state.current_model_path:
            st.info(
                f"Текущая модель: {os.path.basename(st.session_state.current_model_path)}"
            )


def show_summary_report(df):
    """Показать сводный отчет"""
    st.subheader("Сводная статистика")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Основные метрики:")
        st.write(f"- Всего записей: {len(df):,}")
        st.write(
            f"- Период: {df['DateTime'].min().date()} - {df['DateTime'].max().date()}"
        )
        st.write(f"- Средняя мощность: {df['Global_active_power'].mean():.2f} kW")
        st.write(f"- Максимальная мощность: {df['Global_active_power'].max():.2f} kW")
        st.write(f"- Минимальная мощность: {df['Global_active_power'].min():.2f} kW")

    with col2:
        st.write("Качество данных:")
        st.write(f"- Пропущенные значения: {df.isnull().sum().sum()}")
        st.write(
            f"- Полнота данных: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%"
        )
        st.write(f"- Количество признаков: {len(df.columns)}")

        # Статистика по суб-метрам
        if all(
            col in df.columns
            for col in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        ):
            st.write("Суб-метры:")
            st.write(f"- Кухня (ср.): {df['Sub_metering_1'].mean():.2f}")
            st.write(f"- Прачечная (ср.): {df['Sub_metering_2'].mean():.2f}")
            st.write(f"- Водонагреватель (ср.): {df['Sub_metering_3'].mean():.2f}")


if __name__ == "__main__":
    main()
