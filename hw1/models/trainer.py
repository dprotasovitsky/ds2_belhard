import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from config.settings import config
from models.lstm_model import AdvancedLSTMModel
from models.model_manager import ModelManager
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


class ModelTrainer:
    """Тренер для LSTM моделей с сохранением/загрузкой"""

    def __init__(self, best_params, input_size, device=None):
        self.best_params = best_params
        self.best_params["input_size"] = input_size
        self.input_size = input_size
        self.device = device or config.DEVICE
        self.model = None
        self.history = {"train_loss": [], "val_loss": []}
        self.model_manager = ModelManager()

    def create_model(self):
        """Создание модели с лучшими параметрами"""
        self.model = AdvancedLSTMModel(
            input_size=self.input_size,
            hidden_size=self.best_params["hidden_size"],
            num_layers=self.best_params["num_layers"],
            dropout=self.best_params["dropout"],
            bidirectional=self.best_params["bidirectional"],
            use_attention=self.best_params["use_attention"],
        ).to(self.device)
        return self.model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, save_model=True):
        """Обучение модели с сохранением"""
        if self.model is None:
            self.create_model()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.best_params["learning_rate"]
        )
        criterion = nn.MSELoss()

        # Подготовка данных
        train_loader, val_loader = self._prepare_data_loaders(
            X_train, y_train, X_val, y_val, self.best_params["batch_size"]
        )

        # Обучение с ранней остановкой
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0

        progress_bar = st.progress(0)
        status_text = st.empty()

        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.history["train_loss"].append(train_loss)

            # Validation
            val_loss = self._validate_epoch(val_loader, criterion)
            self.history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_lstm_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            # Обновление прогресса
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            if epoch % 10 == 0:
                status_text.text(
                    f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        # Загрузка лучшей модели
        self.model.load_state_dict(torch.load("best_lstm_model.pth"))

        progress_bar.empty()
        status_text.empty()

        # Сохранение полной модели
        if save_model:
            self.save_trained_model(best_val_loss)

        return self.history

    def save_trained_model(self, final_val_loss=None):
        """Сохранение обученной модели"""
        if self.model is None:
            st.error("Модель не обучена")
            return None

        metrics = {
            "final_val_loss": (
                final_val_loss if final_val_loss else self.history["val_loss"][-1]
            ),
            "final_train_loss": self.history["train_loss"][-1],
        }

        model_path = self.model_manager.save_model(
            model=self.model, best_params=self.best_params, metrics=metrics
        )

        return model_path

    def load_trained_model(self, model_path=None):
        """Загрузка обученной модели"""
        if model_path is None:
            model, best_params, metrics, model_info = (
                self.model_manager.get_latest_model()
            )
        else:
            model, best_params, metrics, model_info = self.model_manager.load_model(
                model_path
            )

        if model is not None:
            self.model = model
            self.best_params = best_params
            return True
        return False

    def predict(self, X):
        """Предсказание"""
        if self.model is None:
            raise ValueError("Модель не загружена")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()

    def evaluate_model(self, X_test, y_test, scaler):
        """Оценка модели"""
        predictions = self.predict(X_test)

        # Обратное масштабирование
        predictions_original, y_test_original = self._inverse_scale(
            predictions, y_test, scaler
        )

        # Расчет метрик
        metrics = self._calculate_metrics(y_test_original, predictions_original)

        return metrics, predictions_original, y_test_original

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Одна эпоха обучения"""
        self.model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        return train_loss / len(train_loader)

    def _validate_epoch(self, val_loader, criterion):
        """Одна эпоха валидации"""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        return val_loss / len(val_loader)

    def _prepare_data_loaders(self, X_train, y_train, X_val, y_val, batch_size):
        """Подготовка DataLoader'ов"""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(y_val).to(self.device),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def _inverse_scale(self, predictions, y_test, scaler):
        """Обратное масштабирование предсказаний"""
        n_features = scaler.n_features_in_
        dummy = np.zeros((len(predictions), n_features))
        dummy[:, 0] = predictions
        predictions_original = scaler.inverse_transform(dummy)[:, 0]

        dummy[:, 0] = y_test
        y_test_original = scaler.inverse_transform(dummy)[:, 0]

        return predictions_original, y_test_original

    def _calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # MAPE только для ненулевых значений
        valid_indices = y_true != 0
        if np.any(valid_indices):
            mape = (
                np.mean(
                    np.abs(
                        (y_true[valid_indices] - y_pred[valid_indices])
                        / y_true[valid_indices]
                    )
                )
                * 100
            )
        else:
            mape = 0.0

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
