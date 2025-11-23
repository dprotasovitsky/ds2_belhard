import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


class TrainingVisualizer:
    def __init__(self):
        self.losses = []

    def update(self, loss):
        self.losses.append(loss)

    def plot_training_progress(self, save_path="training_progress.png"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label="Training Loss")
        plt.title("Прогресс обучения")
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"График обучения сохранен: {save_path}")


class ChatLogger:
    def __init__(self, log_file="chat_history.log"):
        self.log_file = log_file

    def log_conversation(self, user_input, bot_response, confidence, tag):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(
                f"{timestamp} | User: {user_input} | Bot: {bot_response} "
                f"| Confidence: {confidence:.4f} | Tag: {tag}\n"
            )


class ModelManager:
    @staticmethod
    def save_model(
        model, all_words, tags, filepath="models/trained_models/chatbot_model.pth"
    ):
        # Создаем директорию если не существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "model_state": model.state_dict(),
            "input_size": len(all_words),
            "hidden_size": model.layer1.out_features,
            "output_size": len(tags),
            "all_words": all_words,
            "tags": tags,
        }

        torch.save(data, filepath)
        logging.info(f"Модель сохранена: {filepath}")

    @staticmethod
    def load_model(filepath="models/trained_models/chatbot_model.pth"):
        try:
            data = torch.load(filepath)
            logging.info(f"Модель загружена: {filepath}")
            return data
        except FileNotFoundError:
            logging.error(f"Файл модели не найден: {filepath}")
            return None
