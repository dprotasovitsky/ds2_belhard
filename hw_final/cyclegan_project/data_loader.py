import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CycleGANDataset(Dataset):
    """Датасет для CycleGAN"""

    def __init__(self, root_dir, transform=None, mode="train", cache_images=False):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.cache_images = cache_images
        self.cache = {}

        # Получение путей к изображениям
        self.A_dir = Path(root_dir) / f"{mode}A"
        self.B_dir = Path(root_dir) / f"{mode}B"

        self.A_paths = sorted(
            [
                str(p)
                for p in self.A_dir.glob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        self.B_paths = sorted(
            [
                str(p)
                for p in self.B_dir.glob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        if not self.A_paths or not self.B_paths:
            raise ValueError(f"No images found in {self.A_dir} or {self.B_dir}")

        self.A_len = len(self.A_paths)
        self.B_len = len(self.B_paths)
        self.dataset_len = max(self.A_len, self.B_len)

        print(
            f"[Dataset] {mode}: A={self.A_len}, B={self.B_len}, total={self.dataset_len}"
        )

    def _load_image(self, path):
        """Загрузка изображения с кэшированием"""
        if self.cache_images and path in self.cache:
            return self.cache[path].copy()

        img = Image.open(path).convert("RGB")

        if self.cache_images:
            self.cache[path] = img

        return img

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        A_idx = idx % self.A_len
        B_idx = idx % self.B_len

        # Загрузка изображений
        A_img = self._load_image(self.A_paths[A_idx])
        B_img = self._load_image(self.B_paths[B_idx])

        # Применение трансформаций
        if self.transform:
            A_img = self.transform(A_img)
            B_img = self.transform(B_img)

        return {
            "A": A_img,
            "B": B_img,
            "A_path": self.A_paths[A_idx],
            "B_path": self.B_paths[B_idx],
        }


def get_transforms(image_size=256, augment=True):
    """Получение трансформаций для данных"""
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if augment and image_size > 256:
        transform_list.insert(1, transforms.RandomCrop(image_size))
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))

    return transforms.Compose(transform_list)


def get_dataloaders(config):
    """Создание даталоадеров"""
    train_transform = get_transforms(config.image_size, augment=True)
    test_transform = get_transforms(config.image_size, augment=False)

    # Датасеты
    train_dataset = CycleGANDataset(
        config.dataset_path, train_transform, "train", cache_images=True
    )
    test_dataset = CycleGANDataset(
        config.dataset_path, test_transform, "test", cache_images=False
    )

    # DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(8, len(test_dataset)),
        shuffle=False,
        num_workers=min(2, config.num_workers),
        pin_memory=True,
    )

    return train_loader, test_loader
