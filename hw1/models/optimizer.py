import numpy as np
import optuna
import torch
import torch.nn as nn
from config.settings import config
from models.lstm_model import AdvancedLSTMModel, SimpleLSTMModel
from optuna.trial import Trial
from torch.utils.data import DataLoader, TensorDataset


class FastLSTMOptimizer:
    """Быстрый оптимизатор гиперпараметров LSTM"""

    def __init__(self, X_train, y_train, X_val, y_val, input_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = input_size
        self.device = config.DEVICE

    def objective(self, trial: Trial):
        """Ускоренная целевая функция"""
        params = self._suggest_hyperparameters(trial)
        model = self._create_model(params)
        best_val_loss = self._fast_train_and_validate(model, params, trial)
        return best_val_loss

    def _suggest_hyperparameters(self, trial):
        """Интеллектуальный подбор гиперпараметров"""
        return {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [
                    64,
                    128,
                ],
            ),
            "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
            "use_attention": trial.suggest_categorical("use_attention", [False, True]),
        }

    def _create_model(self, params):
        """Создание модели"""
        model = AdvancedLSTMModel(
            input_size=self.input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"],
            use_attention=params["use_attention"],
        ).to(self.device)
        return model

    def _fast_train_and_validate(self, model, params, trial):
        """Быстрое обучение и валидация"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
        criterion = nn.MSELoss()

        train_loader, val_loader = self._prepare_fast_data_loaders(params["batch_size"])

        num_epochs = 30
        patience = 5
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for i, (batch_X, batch_y) in enumerate(train_loader):
                if i >= 50:  # Ограничиваем количество батчей
                    break
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (batch_X, batch_y) in enumerate(val_loader):
                    if i >= 20:  # Ограничиваем количество батчей
                        break
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

            val_loss /= min(20, len(val_loader))

            # Pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_val_loss

    def _prepare_fast_data_loaders(self, batch_size):
        """Подготовка быстрых data loader'ов"""
        train_size = min(5000, len(self.X_train))
        val_size = min(1000, len(self.X_val))

        train_indices = np.random.choice(len(self.X_train), train_size, replace=False)
        val_indices = np.random.choice(len(self.X_val), val_size, replace=False)

        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train[train_indices]).to(self.device),
            torch.FloatTensor(self.y_train[train_indices]).to(self.device),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val[val_indices]).to(self.device),
            torch.FloatTensor(self.y_val[val_indices]).to(self.device),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def optimize(self, n_trials=50, timeout=3600):
        """Оптимизация с таймаутом"""
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(n_startup_trials=10),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        return study


class LSTMOptimizer:
    """Стандартный оптимизатор LSTM"""

    def __init__(self, X_train, y_train, X_val, y_val, input_size):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = input_size
        self.device = config.DEVICE

    def objective(self, trial: Trial):
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
            "use_attention": trial.suggest_categorical("use_attention", [True, False]),
        }

        model = AdvancedLSTMModel(
            input_size=self.input_size,
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            bidirectional=params["bidirectional"],
            use_attention=params["use_attention"],
        ).to(self.device)

        return self._train_and_validate(model, params)

    def _train_and_validate(self, model, params):
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train).to(self.device),
            torch.FloatTensor(self.y_train).to(self.device),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True
        )

        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val).to(self.device),
            torch.FloatTensor(self.y_val).to(self.device),
        )
        val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

        num_epochs = 50
        patience = 10
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return best_val_loss

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        return study
