import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class CycleGANInference:
    """Класс для инференса обученной модели CycleGAN"""

    def __init__(self, config, checkpoint_path=None, model_A=None, model_B=None):
        """
        Args:
            config: Конфигурация модели
            checkpoint_path: Путь к чекпоинту
            model_A: Предзагруженная модель A->B (опционально)
            model_B: Предзагруженная модель B->A (опционально)
        """
        self.config = config
        self.device = config.torch_device

        # Загрузка моделей
        from models import Generator

        self.G_AB = Generator(
            config.in_channels, config.out_channels, config.num_residual_blocks
        ).to(self.device)
        self.G_BA = Generator(
            config.in_channels, config.out_channels, config.num_residual_blocks
        ).to(self.device)

        # Загрузка весов
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        elif model_A and model_B:
            self.G_AB.load_state_dict(model_A)
            self.G_BA.load_state_dict(model_B)
        else:
            print("[Inference] No model weights provided. Please load checkpoint.")

        # Режим инференса
        self.G_AB.eval()
        self.G_BA.eval()

        # Трансформации
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Обратные трансформации
        self.inverse_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[-1, -1, -1], std=[2, 2, 2]
                ),  # Обратная нормализация
                transforms.ToPILImage(),
            ]
        )

        # Директории
        self.output_dir = Path("inference_outputs")
        self.output_dir.mkdir(exist_ok=True)

        print(f"[Inference] Initialized on {self.device}")

    def load_checkpoint(self, checkpoint_path):
        """Загрузка чекпоинта"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "G_AB_state_dict" in checkpoint and "G_BA_state_dict" in checkpoint:
            self.G_AB.load_state_dict(checkpoint["G_AB_state_dict"])
            self.G_BA.load_state_dict(checkpoint["G_BA_state_dict"])
        else:
            # Попытка загрузить как отдельные модели
            self.G_AB.load_state_dict(checkpoint)
            print("[Inference] Loaded only A->B model")

        print(f"[Inference] Loaded model from: {checkpoint_path}")

    def preprocess_image(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        preserve_aspect_ratio: bool = True,  # Новый параметр
    ):
        """Препроцессинг изображения"""
        if isinstance(image, str):
            # Загрузка из файла
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Конвертация numpy array в PIL Image
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            # Если уже тензор, проверяем размеры
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)

        # Если нужно сохранить пропорции
        if preserve_aspect_ratio:
            # Сохраняем оригинальный размер
            original_size = image.size

            # Вычисляем новые размеры с сохранением пропорций
            width, height = original_size
            target_size = self.config.image_size

            # Определяем коэффициент масштабирования
            scale = target_size / min(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)

            # Ресайзим
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Центрируем и обрезаем
            left = (new_width - target_size) // 2
            top = (new_height - target_size) // 2
            right = left + target_size
            bottom = top + target_size

            image = image.crop((left, top, right, bottom))

        # Применение трансформаций
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    def postprocess_image(self, tensor: torch.Tensor):
        """Постпроцессинг тензора в изображение"""
        # Убираем batch dimension если есть
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Денормализация и конвертация в PIL
        tensor = tensor.cpu()
        image = self.inverse_transform(tensor)
        return image

    @torch.no_grad()
    def transform_A_to_B(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        return_tensor: bool = False,
    ):
        """
        Преобразование изображения из домена A в домен B

        Args:
            image: Входное изображение
            return_tensor: Если True, возвращает тензор вместо PIL Image

        Returns:
            Преобразованное изображение
        """
        # Препроцессинг
        input_tensor = self.preprocess_image(image)

        # Инференс
        output_tensor = self.G_AB(input_tensor)

        # Постпроцессинг
        if return_tensor:
            return output_tensor
        else:
            return self.postprocess_image(output_tensor)

    @torch.no_grad()
    def transform_B_to_A(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        return_tensor: bool = False,
    ):
        """
        Преобразование изображения из домена B в домен A
        """
        input_tensor = self.preprocess_image(image)
        output_tensor = self.G_BA(input_tensor)

        if return_tensor:
            return output_tensor
        else:
            return self.postprocess_image(output_tensor)

    @torch.no_grad()
    def cycle_transform(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        direction: str = "A_to_B_to_A",
        return_all: bool = False,
    ):
        """
        Циклическое преобразование (A->B->A или B->A->B)

        Args:
            image: Входное изображение
            direction: Направление преобразования
            return_all: Если True, возвращает все промежуточные результаты

        Returns:
            Результаты преобразования
        """
        input_tensor = self.preprocess_image(image)

        if direction == "A_to_B_to_A":
            # A -> B -> A
            fake_B = self.G_AB(input_tensor)
            recov_A = self.G_BA(fake_B)

            if return_all:
                return {
                    "original": self.postprocess_image(input_tensor),
                    "fake_B": self.postprocess_image(fake_B),
                    "recov_A": self.postprocess_image(recov_A),
                }
            else:
                return self.postprocess_image(recov_A)

        elif direction == "B_to_A_to_B":
            # B -> A -> B
            fake_A = self.G_BA(input_tensor)
            recov_B = self.G_AB(fake_A)

            if return_all:
                return {
                    "original": self.postprocess_image(input_tensor),
                    "fake_A": self.postprocess_image(fake_A),
                    "recov_B": self.postprocess_image(recov_B),
                }
            else:
                return self.postprocess_image(recov_B)

        else:
            raise ValueError(f"Unknown direction: {direction}")

    @torch.no_grad()
    def batch_transform(
        self,
        images: List[Union[str, Image.Image, np.ndarray, torch.Tensor]],
        direction: str = "A_to_B",
        batch_size: int = 8,
    ):
        """
        Пакетное преобразование изображений

        Args:
            images: Список изображений
            direction: Направление преобразования
            batch_size: Размер батча

        Returns:
            Список преобразованных изображений
        """
        results = []

        # Препроцессинг всех изображений
        tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            tensors.append(tensor)

        # Объединение в батчи
        for i in range(0, len(tensors), batch_size):
            batch = torch.cat(tensors[i : i + batch_size], dim=0)

            # Инференс
            if direction == "A_to_B":
                output = self.G_AB(batch)
            elif direction == "B_to_A":
                output = self.G_BA(batch)
            else:
                raise ValueError(f"Unknown direction: {direction}")

            # Постпроцессинг
            for j in range(output.size(0)):
                results.append(self.postprocess_image(output[j]))

        return results

    def save_results(
        self,
        original_image,
        transformed_image,
        filename: Optional[str] = None,
        save_collage: bool = True,
    ):
        """
        Сохранение результатов инференса

        Args:
            original_image: Оригинальное изображение
            transformed_image: Преобразованное изображение
            filename: Имя файла (без расширения)
            save_collage: Сохранять ли коллаж
        """
        if filename is None:
            import time

            filename = f"result_{int(time.time())}"

        # Сохранение отдельных изображений
        original_path = self.output_dir / f"{filename}_original.png"
        transformed_path = self.output_dir / f"{filename}_transformed.png"

        if isinstance(original_image, Image.Image):
            original_image.save(original_path)
        else:
            original_image = Image.fromarray(original_image)
            original_image.save(original_path)

        if isinstance(transformed_image, Image.Image):
            transformed_image.save(transformed_path)
        else:
            transformed_image = Image.fromarray(transformed_image)
            transformed_image.save(transformed_path)

        # Создание коллажа
        if save_collage:
            collage = self._create_collage(original_image, transformed_image)
            collage_path = self.output_dir / f"{filename}_collage.png"
            collage.save(collage_path)

        print(f"[Inference] Results saved to: {self.output_dir}/{filename}_*.png")

    def _create_collage(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """Создание коллажа из двух изображений"""
        width = image1.width + image2.width
        height = max(image1.height, image2.height)

        collage = Image.new("RGB", (width, height))
        collage.paste(image1, (0, 0))
        collage.paste(image2, (image1.width, 0))

        return collage

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        direction: str = "A_to_B",
        file_extensions: List[str] = None,
    ):
        """
        Обработка всех изображений в директории

        Args:
            input_dir: Входная директория
            output_dir: Выходная директория
            direction: Направление преобразования
            file_extensions: Список расширений файлов для обработки
        """
        if file_extensions is None:
            file_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Поиск изображений
        image_files = []
        for ext in file_extensions:
            image_files.extend(list(input_path.glob(f"*{ext}")))
            image_files.extend(list(input_path.glob(f"*{ext.upper()}")))

        print(f"[Inference] Found {len(image_files)} images in {input_dir}")

        # Обработка изображений
        for i, img_path in enumerate(image_files):
            try:
                # Преобразование
                result = (
                    self.transform_A_to_B(str(img_path))
                    if direction == "A_to_B"
                    else self.transform_B_to_A(str(img_path))
                )

                # Сохранение
                output_filename = (
                    output_path / f"{img_path.stem}_transformed{img_path.suffix}"
                )
                result.save(output_filename)

                if (i + 1) % 10 == 0:
                    print(f"[Inference] Processed {i+1}/{len(image_files)} images")

            except Exception as e:
                print(f"[Inference] Error processing {img_path.name}: {e}")

    def create_streamlit_app_code(self):
        """Генерация кода для Streamlit приложения"""
        streamlit_code = """import streamlit as st
import torch
from PIL import Image
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import Config
from inference import CycleGANInference

st.set_page_config(
    page_title="CycleGAN",
    page_icon="",
    layout="wide"
)

st.title("CycleGAN Image Translator")

# Инициализация
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.result = None

# Сайдбар
with st.sidebar:
    st.header("Settings")

    # Выбор модели
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        models = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if models:
            model_file = st.selectbox("Select Model", models)
            checkpoint_path = os.path.join(checkpoint_dir, model_file)
        else:
            st.warning("No models found")
            checkpoint_path = None
    else:
        st.error("Create 'checkpoints' directory")
        checkpoint_path = None

    # Направление
    direction = st.selectbox(
        "Transformation",
        ["A to B", "B to A", "A to B to A", "B to A to B"]
    )

    # Загрузка модели
    if st.button("Load Model", type="primary"):
        if checkpoint_path:
            try:
                config = Config()
                st.session_state.model = CycleGANInference(config, checkpoint_path)
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Error: {e}")

# Основная часть
if st.session_state.model is None:
    st.info("Load a model first")
else:
    st.success("Model ready!")

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Choose an image",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file and st.session_state.model:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image)

    with col2:
        st.subheader("Transformed")

        if st.button("Transform", type="primary"):
            with st.spinner("Processing..."):
                try:
                    if direction == "A to B":
                        result = st.session_state.model.transform_A_to_B(image)
                    elif direction == "B to A":
                        result = st.session_state.model.transform_B_to_A(image)
                    elif direction == "A to B to A":
                        results = st.session_state.model.cycle_transform(image, 'A_to_B_to_A', return_all=True)
                        result = results['recov_A']
                    else:
                        results = st.session_state.model.cycle_transform(image, 'B_to_A_to_B', return_all=True)
                        result = results['recov_B']

                    st.session_state.result = result
                    st.image(result)

                    # Download
                    from io import BytesIO
                    buf = BytesIO()
                    result.save(buf, format="PNG")

                    st.download_button(
                        label="Download",
                        data=buf.getvalue(),
                        file_name="transformed.png",
                        mime="image/png"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")
"""
        return streamlit_code

    def save_streamlit_app(self, filename="streamlit_app.py"):
        """Сохранение Streamlit приложения в файл"""
        code = self.create_streamlit_app_code()
        with open(filename, "w") as f:
            f.write(code)
        print(f"[Inference] Streamlit app saved to: {filename}")

    def create_simple_web_interface(self):
        """Создание простого веб-интерфейса с Flask"""
        try:
            import io

            from flask import Flask, jsonify, render_template_string, request, send_file

            app = Flask(__name__)

            HTML_TEMPLATE = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>CycleGAN Web Interface</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .container { display: flex; flex-direction: column; gap: 20px; }
                    .upload-form { border: 2px dashed #ccc; padding: 20px; text-align: center; }
                    .result { display: flex; gap: 20px; }
                    .image-container { flex: 1; }
                    img { max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }
                    button { background: #007bff; color: white; border: none; padding: 10px 20px; cursor: pointer; }
                    button:hover { background: #0056b3; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>CycleGAN Image Translator</h1>

                    <form class="upload-form" action="/transform" method="post" enctype="multipart/form-data">
                        <h3>Upload Image</h3>
                        <input type="file" name="image" accept="image/*" required><br><br>

                        <label>Transformation:</label>
                        <select name="direction">
                            <option value="A_to_B">A → B</option>
                            <option value="B_to_A">B → A</option>
                        </select><br><br>

                        <button type="submit">Transform</button>
                    </form>

                    {% if original and transformed %}
                    <div class="result">
                        <div class="image-container">
                            <h3>Original</h3>
                            <img src="data:image/png;base64,{{ original }}" alt="Original">
                        </div>
                        <div class="image-container">
                            <h3>Transformed</h3>
                            <img src="data:image/png;base64,{{ transformed }}" alt="Transformed">
                            <br><br>
                            <a href="/download" download="transformed.png">
                                <button>Download Result</button>
                            </a>
                        </div>
                    </div>
                    {% endif %}
                </div>
            </body>
            </html>
            """

            @app.route("/")
            def index():
                return render_template_string(HTML_TEMPLATE)

            @app.route("/transform", methods=["POST"])
            def transform():
                if "image" not in request.files:
                    return "No image uploaded", 400

                file = request.files["image"]
                direction = request.form.get("direction", "A_to_B")

                try:
                    # Чтение изображения
                    image = Image.open(io.BytesIO(file.read())).convert("RGB")

                    # Преобразование
                    if direction == "A_to_B":
                        result = self.transform_A_to_B(image)
                    elif direction == "B_to_A":
                        result = self.transform_B_to_A(image)
                    else:
                        return "Invalid direction", 400

                    # Конвертация в base64 для отображения
                    import base64
                    from io import BytesIO

                    # Оригинал
                    orig_buf = BytesIO()
                    image.save(orig_buf, format="PNG")
                    orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

                    # Результат
                    result_buf = BytesIO()
                    result.save(result_buf, format="PNG")
                    result_b64 = base64.b64encode(result_buf.getvalue()).decode()

                    # Сохраняем для скачивания
                    self.last_result = result_buf.getvalue()

                    return render_template_string(
                        HTML_TEMPLATE, original=orig_b64, transformed=result_b64
                    )

                except Exception as e:
                    return f"Error: {str(e)}", 500

            @app.route("/download")
            def download():
                if hasattr(self, "last_result"):
                    return send_file(
                        io.BytesIO(self.last_result),
                        mimetype="image/png",
                        as_attachment=True,
                        download_name="transformed.png",
                    )
                return "No result available", 404

            @app.route("/health")
            def health():
                return jsonify(
                    {
                        "status": "healthy",
                        "device": str(self.device),
                        "model": "CycleGAN",
                    }
                )

            return app

        except ImportError:
            print("[Inference] Flask not installed. Install with: pip install flask")
            return None

    def benchmark(self, num_iterations=100, image_size=(256, 256)):
        """
        Бенчмарк производительности модели

        Args:
            num_iterations: Количество итераций
            image_size: Размер тестового изображения

        Returns:
            Словарь с метриками производительности
        """
        import time

        # Создание тестового тензора
        test_tensor = torch.randn(1, 3, *image_size).to(self.device)

        # Прогрев
        for _ in range(10):
            _ = self.G_AB(test_tensor)

        # Измерение времени для A->B
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.G_AB(test_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        time_a_to_b = (time.time() - start_time) / num_iterations

        # Измерение времени для B->A
        start_time = time.time()
        for _ in range(num_iterations):
            _ = self.G_BA(test_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        time_b_to_a = (time.time() - start_time) / num_iterations

        # Вычисление FPS
        fps_a_to_b = 1.0 / time_a_to_b
        fps_b_to_a = 1.0 / time_b_to_a

        benchmark_results = {
            "device": str(self.device),
            "image_size": image_size,
            "iterations": num_iterations,
            "time_A_to_B_ms": time_a_to_b * 1000,
            "time_B_to_A_ms": time_b_to_a * 1000,
            "fps_A_to_B": fps_a_to_b,
            "fps_B_to_A": fps_b_to_a,
            "avg_fps": (fps_a_to_b + fps_b_to_a) / 2,
        }

        # Вывод результатов
        print("\n" + "=" * 50)
        print("PERFORMANCE BENCHMARK")
        print("=" * 50)
        print(f"Device: {benchmark_results['device']}")
        print(f"Image size: {image_size}")
        print(f"Iterations: {num_iterations}")
        print(
            f"\nA -> B: {benchmark_results['time_A_to_B_ms']:.2f} ms "
            f"({benchmark_results['fps_A_to_B']:.2f} FPS)"
        )
        print(
            f"B -> A: {benchmark_results['time_B_to_A_ms']:.2f} ms "
            f"({benchmark_results['fps_B_to_A']:.2f} FPS)"
        )
        print(f"\nAverage: {benchmark_results['avg_fps']:.2f} FPS")
        print("=" * 50)

        return benchmark_results
