import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent))

try:
    from config import Config
    from inference import CycleGANInference
except ImportError as e:
    st.error(f"Import error: {e}. Make sure all modules are available.")
    st.stop()

# Настройка страницы
st.set_page_config(
    page_title="CycleGAN Image Translator",
    # page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Заголовок приложения
st.title("CycleGAN Image Translation")
st.markdown(
    """
Transform images between two domains using pre-trained CycleGAN model.
Upload an image and choose the transformation direction.
"""
)

# Инициализация сессии
if "inference" not in st.session_state:
    st.session_state.inference = None
    st.session_state.model_loaded = False
    st.session_state.device = "cpu"

# Сайдбар
with st.sidebar:
    st.header("Settings")

    # Выбор чекпоинта
    checkpoint_dir = "checkpoints"
    checkpoints = []
    if os.path.exists(checkpoint_dir):
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.endswith(".pth") and ("best" in f or "final" in f)
        ]

    if checkpoints:
        selected_checkpoint = st.selectbox(
            "Select Model Checkpoint",
            checkpoints,
            index=0 if checkpoints else None,
            help="Choose a pre-trained model checkpoint",
        )
        checkpoint_path = os.path.join(checkpoint_dir, selected_checkpoint)
    else:
        st.warning("No checkpoints found in 'checkpoints' directory")
        checkpoint_path = None

    # Направление преобразования
    st.subheader("Transformation")
    direction = st.radio(
        "Transformation Direction",
        ["A → B", "B → A", "A → B → A", "B → A → B"],
        index=0,
        help="""
        - **A → B**: Transform from domain A to domain B
        - **B → A**: Transform from domain B to domain A
        - **A → B → A**: Full cycle transformation
        - **B → A → B**: Reverse cycle transformation
        """,
    )

    # Параметры изображения
    st.subheader("Image Settings")
    image_size = st.slider(
        "Image Size",
        min_value=128,
        max_value=512,
        value=256,
        step=64,
        help="Size to resize images before processing",
    )

    # Кнопка загрузки модели
    st.subheader("Model Control")
    load_button = st.button("Load Model", type="primary", use_container_width=True)

    if load_button:
        if checkpoint_path and os.path.exists(checkpoint_path):
            with st.spinner("Loading model..."):
                try:
                    # Создаем конфигурацию
                    config = Config()
                    config.image_size = image_size

                    # Инициализируем инференс
                    st.session_state.inference = CycleGANInference(
                        config, checkpoint_path
                    )
                    st.session_state.model_loaded = True
                    st.session_state.device = str(config.torch_device)

                    st.success("Model loaded successfully!")
                    st.info(f"Device: {st.session_state.device}")

                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        else:
            st.error("Please select a valid checkpoint file")

    # Информация
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
    **Tips:**
    1. Click 'Load Model' first
    2. Upload or select an image
    3. Click 'Transform' to generate
    4. Download the result
    """
    )

# Основное содержимое
if st.session_state.model_loaded:
    st.success(f"Model is loaded and ready! (Using {st.session_state.device})")
else:
    st.warning("Please load a model first from the sidebar")

# Выбор источника изображения
st.header("Input Image")

tab1, tab2, tab3 = st.tabs(["Upload", "Examples", "URL"])

input_image = None
image_source = None

with tab1:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload an image from your computer",
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        image_source = "upload"
        # Используем фиксированную ширину или оставляем пустым
        st.image(input_image, caption="Uploaded Image")

with tab2:
    # Примеры изображений
    examples_dir = "example_images"
    if os.path.exists(examples_dir):
        example_files = [
            f
            for f in os.listdir(examples_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if example_files:
            selected_example = st.selectbox(
                "Choose an example image",
                example_files,
                help="Select from pre-loaded example images",
            )

            if selected_example:
                example_path = os.path.join(examples_dir, selected_example)
                input_image = Image.open(example_path).convert("RGB")
                image_source = "example"
                st.image(input_image, caption=f"Example: {selected_example}")
        else:
            st.info("No example images found in 'example_images' directory")
    else:
        st.info("Create an 'example_images' directory with sample images")

with tab3:
    image_url = st.text_input(
        "Image URL",
        placeholder="https://example.com/image.jpg",
        help="Enter URL of an image to transform",
    )

    url_button = st.button("Load from URL", type="secondary")

    if image_url and url_button:
        try:
            from io import BytesIO

            import requests

            response = requests.get(image_url, timeout=10)
            if response.status_code == 200:
                input_image = Image.open(BytesIO(response.content)).convert("RGB")
                image_source = "url"
                st.image(input_image, caption="Image from URL")
            else:
                st.error(f"Failed to load image. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error loading image from URL: {str(e)}")

# Кнопка преобразования
if input_image is not None and st.session_state.model_loaded:
    st.header("Transformation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        # Автоматическая ширина для адаптивности
        st.image(input_image)

        # Информация об изображении
        st.caption(f"Size: {input_image.size}, Mode: {input_image.mode}")

        if image_source:
            st.caption(f"Source: {image_source}")

    with col2:
        st.subheader("Transformed")

        # Сохраняем результат в сессии, чтобы не терять при ререндере
        if "transformed_image" not in st.session_state:
            st.session_state.transformed_image = None
            st.session_state.transformation_direction = None

        transform_button = st.button(
            "Transform Image", type="primary", use_container_width=True
        )

        if transform_button:
            with st.spinner("Transforming image..."):
                try:
                    inference = st.session_state.inference

                    # Выполняем преобразование
                    if direction == "A → B":
                        result_image = inference.transform_A_to_B(input_image)
                    elif direction == "B → A":
                        result_image = inference.transform_B_to_A(input_image)
                    elif direction == "A → B → A":
                        results = inference.cycle_transform(
                            input_image, "A_to_B_to_A", return_all=True
                        )
                        result_image = results["recov_A"]
                    else:  # 'B → A → B'
                        results = inference.cycle_transform(
                            input_image, "B_to_A_to_B", return_all=True
                        )
                        result_image = results["recov_B"]

                    # Сохраняем в сессии
                    st.session_state.transformed_image = result_image
                    st.session_state.transformation_direction = direction

                    st.success("Transformation complete!")

                except Exception as e:
                    st.error(f"Error during transformation: {str(e)}")

        # Отображаем сохраненный результат если есть
        if st.session_state.transformed_image is not None:
            st.image(st.session_state.transformed_image)
            st.success(f"{st.session_state.transformation_direction} transformation")
        else:
            st.info("Click 'Transform Image' button to generate the result")

    # Кнопка загрузки
    if st.session_state.transformed_image is not None:
        st.header("Download")

        # Конвертируем в bytes
        from io import BytesIO

        buf = BytesIO()
        st.session_state.transformed_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                label="Download PNG",
                data=byte_im,
                file_name=f"transformed_{st.session_state.transformation_direction.replace(' → ', '_to_')}.png",
                mime="image/png",
                use_container_width=True,
            )

        with col2:
            # JPEG вариант
            buf_jpeg = BytesIO()
            st.session_state.transformed_image.save(buf_jpeg, format="JPEG", quality=95)
            st.download_button(
                label="Download JPEG",
                data=buf_jpeg.getvalue(),
                file_name=f"transformed_{st.session_state.transformation_direction.replace(' → ', '_to_')}.jpg",
                mime="image/jpeg",
                use_container_width=True,
            )

        with col3:
            if st.button("New Transformation", use_container_width=True):
                # Очищаем результат и перезагружаем
                st.session_state.transformed_image = None
                st.session_state.transformation_direction = None
                st.rerun()

# Информация о модели
st.markdown("---")
st.header("Model Information")

if st.session_state.model_loaded:
    # Бенчмарк производительности
    benchmark_button = st.button("⚡ Run Performance Test", use_container_width=True)

    if benchmark_button:
        with st.spinner("Running benchmark..."):
            try:
                inference = st.session_state.inference
                results = inference.benchmark(num_iterations=50)

                st.subheader("Performance Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("A → B Time", f"{results['time_A_to_B_ms']:.2f} ms")
                    st.metric("A → B FPS", f"{results['fps_A_to_B']:.2f}")

                with col2:
                    st.metric("B → A Time", f"{results['time_B_to_A_ms']:.2f} ms")
                    st.metric("B → A FPS", f"{results['fps_B_to_A']:.2f}")

                with col3:
                    st.metric("Average FPS", f"{results['avg_fps']:.2f}")
                    st.metric("Device", results["device"])

            except Exception as e:
                st.error(f"Benchmark failed: {str(e)}")

    # Информация о чекпоинте
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            st.subheader("Checkpoint Info")

            info_cols = st.columns(2)
            with info_cols[0]:
                if "epoch" in checkpoint:
                    st.metric("Epoch", checkpoint["epoch"])
                if "loss" in checkpoint:
                    st.metric("Loss", f"{checkpoint['loss']:.4f}")

            with info_cols[1]:
                if "config" in checkpoint:
                    config_info = checkpoint["config"]
                    if "image_size" in config_info:
                        st.metric(
                            "Image Size",
                            f"{config_info['image_size']}x{config_info['image_size']}",
                        )

        except Exception as e:
            st.warning(f"Could not load checkpoint info: {e}")

# Информация о системе
with st.expander("System Information"):
    import platform

    sys_info_cols = st.columns(3)

    with sys_info_cols[0]:
        st.write("**Python Version**")
        st.code(sys.version)

    with sys_info_cols[1]:
        st.write("**PyTorch Version**")
        st.code(torch.__version__)

        if torch.cuda.is_available():
            st.write("**CUDA Available**: Yes")
            st.write(f"**GPU**: {torch.cuda.get_device_name(0)}")
        else:
            st.write("**CUDA Available**: No")

    with sys_info_cols[2]:
        st.write("**System**")
        st.write(f"Platform: {platform.system()}")
        st.write(f"Processor: {platform.processor()}")

# Футер
st.markdown("---")
st.markdown(
    """
<div style='text-align: center'>
    <p style='font-size: 0.9em; color: #666;'>
        CycleGAN Image Translation App | Powered by PyTorch & Streamlit
    </p>
    <p style='font-size: 0.8em; color: #888;'>
        For best results, use high-quality images and appropriate domain mappings
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Стили CSS для адаптивности
st.markdown(
    """
<style>
    /* Адаптивные изображения */
    .stImage img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Кнопки на всю ширину */
    .stButton > button {
        width: 100%;
    }

    .stDownloadButton > button {
        width: 100%;
    }

    /* Адаптивные колонки */
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
        }
    }

    /* Улучшенные карточки */
    [data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }

    /* Стили для success сообщений */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    /* Стили для warning сообщений */
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }

    /* Стили для error сообщений */
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)
