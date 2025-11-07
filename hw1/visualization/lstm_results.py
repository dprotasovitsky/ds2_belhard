import numpy as np
import plotly.graph_objects as go
import streamlit as st
from config.settings import config
from plotly.subplots import make_subplots
from scipy import stats


class LSTMResultVisualizer:
    """Визуализатор результатов LSTM"""

    def __init__(self, model_trainer, scaler):
        self.model_trainer = model_trainer
        self.scaler = scaler

    def plot_training_history(self):
        """Визуализация истории обучения"""
        fig = go.Figure()

        # Преобразуем в список для Plotly
        epochs = list(range(len(self.model_trainer.history["train_loss"])))

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.model_trainer.history["train_loss"],
                mode="lines",
                name="Train Loss",
                line=dict(color=config.COLORS["primary"]),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=self.model_trainer.history["val_loss"],
                mode="lines",
                name="Validation Loss",
                line=dict(color=config.COLORS["secondary"]),
            )
        )

        fig.update_layout(
            title="История обучения модели",
            xaxis_title="Эпоха",
            yaxis_title="Loss (MSE)",
            template=config.PLOT_THEME,
        )

        return fig

    def plot_predictions_vs_actual(self, X_test, y_test, num_samples=200):
        """Сравнение предсказаний с реальными значениями"""
        predictions = self.model_trainer.predict(X_test)
        predictions_original, y_test_original = self._inverse_scale(predictions, y_test)

        # ИСПРАВЛЕНИЕ: Проверяем, чтобы размер выборки не превышал доступные данные
        available_samples = len(y_test_original)
        actual_sample_size = min(num_samples, available_samples)

        if actual_sample_size < num_samples:
            st.warning(
                f"Доступно только {available_samples} samples для визуализации (запрошено {num_samples})"
            )

        # Берем все доступные данные или случайную выборку
        if actual_sample_size == available_samples:
            indices = range(available_samples)
            y_sample = y_test_original
            pred_sample = predictions_original
        else:
            indices = np.random.choice(
                available_samples, actual_sample_size, replace=False
            )
            y_sample = y_test_original[indices]
            pred_sample = predictions_original[indices]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=y_sample.tolist(),
                y=pred_sample.tolist(),
                mode="markers",
                name="Предсказания vs Фактические",
                marker=dict(color=config.COLORS["primary"], size=8, opacity=0.6),
            )
        )

        min_val = min(y_sample.min(), pred_sample.min())
        max_val = max(y_sample.max(), pred_sample.max())

        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Идеальное предсказание",
                line=dict(color=config.COLORS["danger"], dash="dash"),
            )
        )

        fig.update_layout(
            title=f"Предсказания vs Фактические значения ({actual_sample_size} samples)",
            xaxis_title="Фактические значения (кВт)",
            yaxis_title="Предсказания (кВт)",
            template=config.PLOT_THEME,
        )

        return fig

    def plot_time_series_prediction(self, X_test, y_test, start_idx=0, length=100):
        """Визуализация временного ряда с предсказаниями"""
        predictions = self.model_trainer.predict(X_test)
        predictions_original, y_test_original = self._inverse_scale(predictions, y_test)

        # ИСПРАВЛЕНИЕ: Проверяем границы
        available_length = len(y_test_original)
        actual_length = min(length, available_length - start_idx)

        if actual_length < length:
            st.warning(
                f"Доступно только {actual_length} точек для визуализации временного ряда"
            )

        end_idx = start_idx + actual_length

        # Преобразуем range в список для Plotly
        time_indices = list(range(start_idx, end_idx))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=time_indices,
                y=y_test_original[start_idx:end_idx].tolist(),
                mode="lines",
                name="Фактические значения",
                line=dict(color=config.COLORS["primary"], width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_indices,
                y=predictions_original[start_idx:end_idx].tolist(),
                mode="lines",
                name="Предсказания LSTM",
                line=dict(color=config.COLORS["secondary"], width=2),
            )
        )

        fig.update_layout(
            title=f"Прогнозирование потребления электроэнергии ({actual_length} точек)",
            xaxis_title="Время",
            yaxis_title="Активная мощность (кВт)",
            template=config.PLOT_THEME,
        )

        return fig

    def plot_residuals_analysis(self, X_test, y_test):
        """Анализ остатков"""
        predictions = self.model_trainer.predict(X_test)
        predictions_original, y_test_original = self._inverse_scale(predictions, y_test)
        residuals = y_test_original - predictions_original

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Распределение остатков",
                "Остатки vs Предсказания",
                "QQ-plot остатков",
                "Автокорреляция остатков",
            ),
        )

        # Гистограмма остатков
        fig.add_trace(
            go.Histogram(x=residuals.tolist(), name="Остатки", nbinsx=50), row=1, col=1
        )

        # Остатки vs Предсказания
        fig.add_trace(
            go.Scatter(
                x=predictions_original.tolist(),
                y=residuals.tolist(),
                mode="markers",
                name="Остатки",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        # QQ-plot
        qq = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(
                x=qq[0][0].tolist(), y=qq[0][1].tolist(), mode="markers", name="QQ-plot"
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=qq[0][0].tolist(),
                y=qq[0][0] * qq[1][1] + qq[1][0],
                mode="lines",
                name="Нормальное распределение",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

        # Автокорреляция
        max_lag = min(30, len(residuals) // 2)
        autocorr = [1.0] + [
            np.corrcoef(residuals[:-i], residuals[i:])[0, 1] for i in range(1, max_lag)
        ]

        fig.add_trace(
            go.Bar(x=list(range(len(autocorr))), y=autocorr, name="Автокорреляция"),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=600,
            title_text="Анализ остатков",
            showlegend=False,
            template=config.PLOT_THEME,
        )
        return fig

    def _inverse_scale(self, predictions, y_test):
        """Обратное масштабирование"""
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, 0] = predictions
        predictions_original = self.scaler.inverse_transform(dummy)[:, 0]

        dummy[:, 0] = y_test
        y_test_original = self.scaler.inverse_transform(dummy)[:, 0]

        return predictions_original, y_test_original
