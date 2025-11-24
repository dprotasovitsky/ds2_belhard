import json
import logging
import random
from datetime import datetime

import numpy as np
import torch
from config import config
from model import AdvancedNeuralNet
from nltk_utils import bag_of_words, stem, tokenize
from utils import ChatLogger, ModelManager


class AdvancedChatBot:
    def __init__(self):
        self.device = config.DEVICE
        self.model = None
        self.all_words = None
        self.tags = None
        self.intents = None
        self.chat_logger = ChatLogger()
        self.conversation_history = []

        # Информация о загрузке на GPU
        if torch.cuda.is_available():
            print("Загрузка модели на GPU...")
        else:
            print("Загрузка модели на CPU...")

        self.load_model()
        self.load_intents()

    def load_model(self):
        """Загрузка модели с поддержкой CUDA"""
        try:
            data = torch.load(config.MODEL_FILE, map_location=self.device)

            # Создание модели
            self.model = AdvancedNeuralNet(
                data["input_size"], data["hidden_size"], data["output_size"]
            )

            # Загрузка состояния модели
            self.model.load_state_dict(data["model_state"])
            self.model.to(self.device)
            self.model.eval()

            self.all_words = data["all_words"]
            self.tags = data["tags"]

            print(f"Модель загружена на: {self.device}")
            print(f"Размер словаря: {len(self.all_words)} слов")
            print(f"Количество тегов: {len(self.tags)}")

        except FileNotFoundError:
            print(f"Файл модели {config.MODEL_FILE} не найден!")
            raise
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise

    def load_intents(self):
        """Загрузка интентов"""
        try:
            with open(config.INTENTS_FILE, "r", encoding="utf-8") as f:
                self.intents = json.load(f)
            print(f"Загружено {len(self.intents['intents'])} интентов")
        except FileNotFoundError:
            print(f"Файл {config.INTENTS_FILE} не найден!")
            raise
        except json.JSONDecodeError:
            print(f"Ошибка чтения JSON файла {config.INTENTS_FILE}!")
            raise

    def get_response(self, user_input, user_id=None):
        """Получение ответа на пользовательский ввод"""
        try:
            # Токенизация и преобразование в мешок слов
            sentence = tokenize(user_input)
            X = bag_of_words(sentence, self.all_words)
            X = X.reshape(1, X.shape[0])
            X_tensor = torch.from_numpy(X).to(self.device)

            # Предсказание
            with torch.no_grad():
                output = self.model(X_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)

            tag = self.tags[predicted.item()]
            confidence_value = confidence.item()

            # Поиск подходящего ответа
            response = "Извините, я не понимаю. Можете переформулировать вопрос?"

            if confidence_value > config.CONFIDENCE_THRESHOLD:
                for intent in self.intents["intents"]:
                    if tag == intent["tag"]:
                        responses = intent.get("responses", [])
                        if responses:
                            response = random.choice(responses)
                        break
                else:
                    response = "Извините, я пока не знаю как ответить на это."
                    tag = "unknown"
            else:
                response = (
                    "Извините, я не совсем понимаю. Можете задать вопрос по-другому?"
                )
                tag = "low_confidence"

            # Логирование диалога
            log_data = {
                "timestamp": datetime.now(),
                "user_input": user_input,
                "bot_response": response,
                "confidence": confidence_value,
                "tag": tag,
            }

            if user_id:
                log_data["user_id"] = user_id

            self.chat_logger.log_conversation(
                user_input, response, confidence_value, tag
            )
            self.conversation_history.append(log_data)

            return response, confidence_value, tag

        except Exception as e:
            logging.error(f"Ошибка получения ответа: {e}")
            return "Произошла ошибка при обработке вашего сообщения.", 0.0, "error"

    def get_user_conversation_history(self, user_id):
        """Получить историю диалога конкретного пользователя"""
        return [
            msg for msg in self.conversation_history if msg.get("user_id") == user_id
        ]
