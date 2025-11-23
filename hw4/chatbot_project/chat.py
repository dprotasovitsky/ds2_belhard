import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from config import config
from model import AdvancedNeuralNet
from nltk_utils import bag_of_words, stem, tokenize  # Добавлен импорт
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

            # Логирование информации об устройстве
            if "cuda_info" in data:
                cuda_info = data["cuda_info"]
                print(f"Модель загружена. Обучена на: {cuda_info['device_name']}")

            print(f"Текущее устройство: {self.device}")
            print(f"Размер словаря: {len(self.all_words)} слов")
            print(f"Количество тегов: {len(self.tags)}")

        except FileNotFoundError:
            print(f"Файл модели {config.MODEL_FILE} не найден!")
            print("Сначала обучите модель: python train.py")
            exit(1)
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            exit(1)

    def load_intents(self):
        """Загрузка интентов"""
        try:
            with open(config.INTENTS_FILE, "r", encoding="utf-8") as f:
                self.intents = json.load(f)
            print(f"Загружено {len(self.intents['intents'])} интентов")
        except FileNotFoundError:
            print(f"Файл {config.INTENTS_FILE} не найден!")
            exit(1)
        except json.JSONDecodeError:
            print(f"Ошибка чтения JSON файла {config.INTENTS_FILE}!")
            exit(1)

    def preprocess_input(self, sentence):
        """Предварительная обработка входного предложения"""
        try:
            # Токенизация
            tokenized_sentence = tokenize(sentence)

            # Создание мешка слов
            if not self.all_words:
                raise ValueError("Словарь не загружен!")

            bow = bag_of_words(tokenized_sentence, self.all_words)
            return bow, tokenized_sentence

        except Exception as e:
            print(f"Ошибка обработки входных данных: {e}")
            return None, None

    def predict_intent(self, user_input):
        """Предсказание интента с использованием GPU если доступно"""
        try:
            # Предварительная обработка
            bow, tokenized_sentence = self.preprocess_input(user_input)
            if bow is None:
                return "unknown", 0.0

            # Подготовка данных для модели
            X = bow.reshape(1, bow.shape[0])
            X_tensor = torch.from_numpy(X).to(self.device)

            # Предсказание без вычисления градиентов (экономия памяти)
            with torch.no_grad():
                output = self.model(X_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, dim=1)

            tag = self.tags[predicted.item()]
            confidence_value = confidence.item()

            return tag, confidence_value

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return "unknown", 0.0

    def get_response(self, user_input):
        """Получение ответа на пользовательский ввод"""
        # Предсказание интента
        tag, confidence_value = self.predict_intent(user_input)

        # Поиск подходящего ответа
        response = "Извините, я не понимаю. Можете переформулировать вопрос?"

        if confidence_value > config.CONFIDENCE_THRESHOLD:
            for intent in self.intents["intents"]:
                if tag == intent["tag"]:
                    responses = intent.get("responses", [])
                    if responses:
                        response = random.choice(responses)

                    # Логирование успешного распознавания
                    logging.info(
                        f"Распознан интент: '{tag}' с уверенностью {confidence_value:.4f}"
                    )
                    break
            else:
                # Интент не найден в файле (редкий случай)
                response = "Извините, я пока не знаю как ответить на это."
                tag = "unknown"
        else:
            # Низкая уверенность
            logging.warning(
                f"Низкая уверенность предсказания: {confidence_value:.4f} для тега '{tag}'"
            )
            response = "Извините, я не совсем понимаю. Можете задать вопрос по-другому?"
            tag = "low_confidence"

        # Логирование диалога
        self.chat_logger.log_conversation(user_input, response, confidence_value, tag)
        self.conversation_history.append(
            {
                "timestamp": datetime.now(),
                "user": user_input,
                "bot": response,
                "confidence": confidence_value,
                "tag": tag,
            }
        )

        return response, confidence_value, tag

    def get_conversation_stats(self):
        """Статистика текущей сессии"""
        if not self.conversation_history:
            return "История пуста"

        total_messages = len(self.conversation_history)
        user_messages = sum(1 for msg in self.conversation_history if msg["user"])
        avg_confidence = np.mean(
            [msg["confidence"] for msg in self.conversation_history]
        )

        # Распределение по тегам
        tag_distribution = {}
        for msg in self.conversation_history:
            tag = msg["tag"]
            tag_distribution[tag] = tag_distribution.get(tag, 0) + 1

        stats = (
            f"Статистика сессии:\n"
            f"   • Всего сообщений: {total_messages}\n"
            f"   • Сообщений от пользователя: {user_messages}\n"
            f"   • Средняя уверенность: {avg_confidence:.2%}\n"
            f"   • Распределение тегов: {tag_distribution}"
        )

        return stats

    def print_available_commands(self):
        """Печать доступных команд"""
        print("\nДоступные команды:")
        print("   • 'выход', 'exit', 'quit' - завершение работы")
        print("   • 'статистика', 'stats' - показать статистику")
        print("   • 'команды' - показать это сообщение")
        print("   • 'теги' - показать все доступные теги")

    def print_available_tags(self):
        """Печать всех доступных тегов"""
        print("\nДоступные теги:")
        for i, tag in enumerate(self.tags, 1):
            print(f"   {i:2d}. {tag}")


def main():
    """Основная функция чата"""
    print("=" * 60)
    print("Чат-Бот с поддержкой CUDA")
    print("Версия 1.0")
    print("=" * 60)

    try:
        bot = AdvancedChatBot()
        print("\nБот готов к общению!")
        bot.print_available_commands()

        while True:
            try:
                user_input = input("\nВы: ").strip()

                if user_input.lower() in ["выход", "exit", "quit"]:
                    print("\nБот: До свидания! Было приятно пообщаться!")
                    stats = bot.get_conversation_stats()
                    print(f"\n{stats}")
                    break

                elif user_input.lower() in ["статистика", "stats", "statistics"]:
                    stats = bot.get_conversation_stats()
                    print(f"\n{stats}")
                    continue

                elif user_input.lower() in ["команды", "commands", "help"]:
                    bot.print_available_commands()
                    continue

                elif user_input.lower() in ["теги", "tags"]:
                    bot.print_available_tags()
                    continue

                elif not user_input:
                    print("Бот: Пожалуйста, введите сообщение!")
                    continue

                # Получение ответа от бота
                response, confidence, tag = bot.get_response(user_input)

                # Вывод ответа с дополнительной информацией
                print(f"Бот: {response}")
                print(f"[Уверенность: {confidence:.2%}, Тег: {tag}]")

                # Предупреждение о низкой уверенности
                if confidence < config.CONFIDENCE_THRESHOLD:
                    print("[Низкая уверенность, ответ может быть неточным]")

            except KeyboardInterrupt:
                print("\n\nПрервано пользователем")
                stats = bot.get_conversation_stats()
                print(f"\n{stats}")
                break
            except Exception as e:
                print(f"Ошибка ввода: {e}")
                continue

    except Exception as e:
        logging.error(f"Критическая ошибка в работе бота: {e}")
        print(f"Критическая ошибка: {e}")
        print("Проверьте логи для подробной информации.")


if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("chat_errors.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    main()
