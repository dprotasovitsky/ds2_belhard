import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from utils.logger import Logger, log_execution_time


# Код класса TripletTextDataset
class TripletTextDataset(Dataset):
    def __init__(
        self,
        tfidf_matrix: np.ndarray,
        labels: np.ndarray,
        original_texts: List[str] = None,
    ):
        self.tfidf_matrix = tfidf_matrix
        self.labels = labels
        self.original_texts = original_texts
        self.logger = Logger.get_logger()

        # Создаем индексы для каждого класса
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.logger.info(
            f"Создан датасет с {len(self)} примерами и {len(self.class_indices)} классами"
        )

    def __len__(self):
        return len(self.tfidf_matrix)

    def __getitem__(self, idx):
        # Якорь
        anchor_vec = self.tfidf_matrix[idx]
        anchor_label = self.labels[idx]

        # Положительный пример (из того же класса)
        positive_idx = idx
        while positive_idx == idx:
            if len(self.class_indices[anchor_label]) > 1:
                positive_idx = random.choice(self.class_indices[anchor_label])
            else:
                positive_idx = idx
                break
        positive_vec = self.tfidf_matrix[positive_idx]

        # Отрицательный пример (из другого класса)
        other_labels = [
            label for label in self.class_indices.keys() if label != anchor_label
        ]
        if other_labels:
            negative_label = random.choice(other_labels)
            negative_idx = random.choice(self.class_indices[negative_label])
        else:
            negative_idx = idx
            negative_label = anchor_label

        negative_vec = self.tfidf_matrix[negative_idx]

        return {
            "anchor": torch.FloatTensor(anchor_vec),
            "positive": torch.FloatTensor(positive_vec),
            "negative": torch.FloatTensor(negative_vec),
            "anchor_label": anchor_label,
            "positive_label": anchor_label,
            "negative_label": negative_label,
        }


class BagOfWordsEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: List[int] = [512, 256],
    ):
        super(BagOfWordsEncoder, self).__init__()
        self.logger = Logger.get_logger()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            prev_dim = hidden_dim

        # Последний слой для встраивания
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)
        self.logger.info(
            f"Создан энкодер с архитектурой: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {embedding_dim}"
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        # Нормализация эмбеддингов
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class BagOfWordsDecoder(nn.Module):
    def __init__(
        self, embedding_dim: int, output_dim: int, hidden_dims: List[int] = [256, 512]
    ):
        super(BagOfWordsDecoder, self).__init__()
        self.logger = Logger.get_logger()

        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            prev_dim = hidden_dim

        # Последний слой реконструкции
        layers.extend(
            [
                nn.Linear(prev_dim, output_dim),
                nn.Sigmoid(),  # Для восстановления TF-IDF векторов
            ]
        )

        self.decoder = nn.Sequential(*layers)
        self.logger.info(
            f"Создан декодер с архитектурой: {embedding_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}"
        )

    def forward(self, x):
        return self.decoder(x)


class TripletAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128):
        super(TripletAutoencoder, self).__init__()
        self.logger = Logger.get_logger()

        self.encoder = BagOfWordsEncoder(
            input_dim=vocab_size, embedding_dim=embedding_dim, hidden_dims=[512, 256]
        )

        self.decoder = BagOfWordsDecoder(
            embedding_dim=embedding_dim, output_dim=vocab_size, hidden_dims=[256, 512]
        )

        self.logger.info(
            f"Создан Triplet Autoencoder с размером словаря {vocab_size} и размером эмбеддинга {embedding_dim}"
        )

    def forward(self, x, decode: bool = False):
        embeddings = self.encoder(x)

        if decode:
            reconstruction = self.decoder(embeddings)
            return embeddings, reconstruction
        else:
            return embeddings


class TripletLoss(nn.Module):
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 0.7,
        reconstruction_weight: float = 1.0,
    ):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha  # Вес для triplet loss vs reconstruction loss
        self.reconstruction_weight = reconstruction_weight
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.reconstruction_loss = nn.MSELoss()  # Для восстановления TF-IDF векторов

        self.logger = Logger.get_logger()
        self.logger.info(
            f"Инициализирована функция потерь с margin={margin}, alpha={alpha}"
        )

    def forward(
        self,
        anchor_emb,
        positive_emb,
        negative_emb,
        anchor_recon=None,
        anchor_original=None,
        positive_recon=None,
        positive_original=None,
    ):

        # Triplet loss
        triplet_loss = self.triplet_loss(anchor_emb, positive_emb, negative_emb)

        # Reconstruction loss
        recon_loss = 0
        if anchor_recon is not None and anchor_original is not None:
            anchor_recon_loss = self.reconstruction_loss(anchor_recon, anchor_original)
            recon_loss += anchor_recon_loss

        if positive_recon is not None and positive_original is not None:
            positive_recon_loss = self.reconstruction_loss(
                positive_recon, positive_original
            )
            recon_loss += positive_recon_loss

        # Комбинированная потеря
        total_loss = (
            1 - self.alpha
        ) * triplet_loss + self.alpha * recon_loss * self.reconstruction_weight

        return total_loss, triplet_loss, recon_loss


class TextTripletAutoencoder:
    def __init__(self, embedding_dim: int = 128, use_tfidf: bool = True):
        from config import Config

        self.device = Config.DEVICE
        self.preprocessor = None  # Будет установлен извне
        self.use_tfidf = use_tfidf
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.embedding_dim = embedding_dim
        self.model = None
        self.logger = Logger.get_logger()
        self.logger.info(
            f"Инициализирован TextTripletAutoencoder с embedding_dim={embedding_dim}, use_tfidf={use_tfidf}"
        )

    def set_preprocessor(self, preprocessor):
        """Устанавливает препроцессор"""
        self.preprocessor = preprocessor

    def prepare_features(self, texts: List[str], fit: bool = True) -> np.ndarray:
        if self.preprocessor is None:
            raise ValueError("Препроцессор не установлен!")

        # Предобработка текстов
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        self.logger.info(f"Предобработано {len(processed_texts)} текстов")

        # Векторизация
        if fit:
            if self.use_tfidf:
                self.vectorizer = TfidfVectorizer(
                    max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)
                )
            else:
                self.vectorizer = CountVectorizer(
                    max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)
                )
            features = self.vectorizer.fit_transform(processed_texts).toarray()
            self.logger.info(
                f"Создан векторaйзер с размером словаря: {features.shape[1]}"
            )
        else:
            features = self.vectorizer.transform(processed_texts).toarray()
            self.logger.info(f"Векторизовано {len(texts)} текстов")

        return features

    @log_execution_time
    def train(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        reporter=None,
    ):

        self.logger.info("Начало обучения Triplet Autoencoder")

        # Подготовка фич
        features = self.prepare_features(texts, fit=True)
        encoded_labels = self.label_encoder.fit_transform(labels)

        self.logger.info(
            f"Закодировано {len(encoded_labels)} меток. Классы: {self.label_encoder.classes_}"
        )

        # Создание датасета
        dataset = TripletTextDataset(features, encoded_labels, texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Инициализация модели
        vocab_size = features.shape[1]
        self.model = TripletAutoencoder(vocab_size, self.embedding_dim).to(self.device)

        # Оптимизатор и функция потерь
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        loss_fn = TripletLoss(margin=1.0, alpha=0.7, reconstruction_weight=1.0)

        self.logger.info(f"Размерность словаря: {vocab_size}")
        self.logger.info(f"Размер батча: {batch_size}")
        self.logger.info(f"Количество эпох: {epochs}")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_triplet_loss = 0
            total_recon_loss = 0

            for batch_idx, batch in enumerate(dataloader):
                # Перемещаем данные на устройство
                anchor = batch["anchor"].to(self.device)
                positive = batch["positive"].to(self.device)
                negative = batch["negative"].to(self.device)

                optimizer.zero_grad()

                # Forward pass для anchor с реконструкцией
                anchor_emb, anchor_recon = self.model(anchor, decode=True)
                # Forward pass для positive с реконструкцией
                positive_emb, positive_recon = self.model(positive, decode=True)
                # Forward pass для negative (только эмбеддинги)
                negative_emb = self.model(negative, decode=False)

                # Вычисляем потери
                loss, triplet_loss, recon_loss = loss_fn(
                    anchor_emb,
                    positive_emb,
                    negative_emb,
                    anchor_recon,
                    anchor,
                    positive_recon,
                    positive,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_triplet_loss += triplet_loss.item()
                total_recon_loss += recon_loss.item()

                if batch_idx % 20 == 0:
                    self.logger.info(
                        f"Epoch: {epoch:02d} | Batch: {batch_idx:03d}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | Triplet: {triplet_loss.item():.4f} | "
                        f"Recon: {recon_loss.item():.4f}"
                    )

            scheduler.step()

            avg_loss = total_loss / len(dataloader)
            avg_triplet = total_triplet_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)

            # Добавляем метрики в отчет
            if reporter:
                reporter.add_training_metrics(epoch, avg_loss, avg_triplet, avg_recon)

            self.logger.info(
                f"Epoch {epoch:02d} завершен | Avg Loss: {avg_loss:.4f} | "
                f"Avg Triplet: {avg_triplet:.4f} | Avg Recon: {avg_recon:.4f}"
            )

        self.logger.info("Обучение Triplet Autoencoder завершено")

    @log_execution_time
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не обучена! Сначала вызовите train()")

        self.model.eval()
        features = self.prepare_features(texts, fit=False)
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch_features = features[i : i + batch_size]
                batch_tensor = torch.FloatTensor(batch_features).to(self.device)
                batch_embeddings = self.model(batch_tensor, decode=False)
                embeddings.append(batch_embeddings.cpu().numpy())

        result = np.vstack(embeddings)
        self.logger.info(
            f"Закодировано {len(texts)} текстов в эмбеддинги размерности {result.shape}"
        )
        return result

    def find_similar_texts(
        self, query_text: str, texts: List[str], top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """Находит наиболее похожие тексты на запрос"""
        query_embedding = self.encode_texts([query_text])[0]
        text_embeddings = self.encode_texts(texts)

        # Вычисляем косинусное сходство
        similarities = cosine_similarity([query_embedding], text_embeddings)[0]

        # Сортируем по убыванию сходства
        similar_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in similar_indices:
            results.append((idx, similarities[idx], texts[idx]))

        self.logger.info(
            f"Найдено {len(results)} похожих текстов для запроса: '{query_text}'"
        )
        return results

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Модель не обучена!")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "vectorizer": self.vectorizer,
                "label_encoder": self.label_encoder,
                "embedding_dim": self.embedding_dim,
                "use_tfidf": self.use_tfidf,
            },
            path,
        )

        self.logger.info(f"Модель сохранена в {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        # Восстанавливаем векторaйзер и энкодер
        self.vectorizer = checkpoint["vectorizer"]
        self.label_encoder = checkpoint["label_encoder"]
        self.embedding_dim = checkpoint["embedding_dim"]
        self.use_tfidf = checkpoint["use_tfidf"]

        # Создаем и загружаем модель
        vocab_size = len(self.vectorizer.vocabulary_)
        self.model = TripletAutoencoder(vocab_size, self.embedding_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.logger.info(f"Модель загружена из {path}")
