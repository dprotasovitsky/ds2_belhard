import json
import os
from datetime import datetime

import torch
from config.settings import config
from models.lstm_model import AdvancedLSTMModel


class ModelManager:
    """Менеджер для сохранения и загрузки моделей"""

    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save_model(self, model, best_params, metrics, model_name=None):
        """Сохранение модели с параметрами и метриками"""
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"lstm_model_{timestamp}"

        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)

        try:
            # Сохраняем веса модели
            torch.save(
                model.state_dict(), os.path.join(model_path, "model_weights.pth")
            )

            # Сохраняем параметры модели
            with open(os.path.join(model_path, "model_params.json"), "w") as f:
                json.dump(best_params, f, indent=2)

            # Сохраняем метрики
            with open(os.path.join(model_path, "model_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # Сохраняем информацию о модели
            model_info = {
                "model_name": model_name,
                "saved_at": datetime.now().isoformat(),
                "input_size": best_params.get("input_size", "unknown"),
                "architecture": "AdvancedLSTMModel",
            }

            with open(os.path.join(model_path, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)

            return model_path

        except Exception as e:
            print(f"Ошибка сохранения модели: {e}")
            return None

    def load_model(self, model_path, device=None):
        """Загрузка модели с параметрами"""
        if device is None:
            device = config.DEVICE

        try:
            # Загружаем параметры
            with open(os.path.join(model_path, "model_params.json"), "r") as f:
                best_params = json.load(f)

            # Загружаем информацию о модели
            with open(os.path.join(model_path, "model_info.json"), "r") as f:
                model_info = json.load(f)

            # Создаем модель
            input_size = best_params.get("input_size")
            model = AdvancedLSTMModel(
                input_size=input_size,
                hidden_size=best_params["hidden_size"],
                num_layers=best_params["num_layers"],
                dropout=best_params["dropout"],
                bidirectional=best_params["bidirectional"],
                use_attention=best_params["use_attention"],
            ).to(device)

            # Загружаем веса
            model.load_state_dict(
                torch.load(
                    os.path.join(model_path, "model_weights.pth"), map_location=device
                )
            )

            # Загружаем метрики
            with open(os.path.join(model_path, "model_metrics.json"), "r") as f:
                metrics = json.load(f)

            return model, best_params, metrics, model_info

        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None, None, None, None

    def list_saved_models(self):
        """Список сохраненных моделей"""
        models = []
        for item in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, item)
            if os.path.isdir(model_path):
                info_file = os.path.join(model_path, "model_info.json")
                if os.path.exists(info_file):
                    try:
                        with open(info_file, "r") as f:
                            info = json.load(f)
                        models.append(info)
                    except:
                        continue
        return sorted(models, key=lambda x: x["saved_at"], reverse=True)

    def get_latest_model(self):
        """Получение последней сохраненной модели"""
        models = self.list_saved_models()
        if models:
            latest_model_name = models[0]["model_name"]
            return self.load_model(os.path.join(self.model_dir, latest_model_name))
        return None, None, None, None
