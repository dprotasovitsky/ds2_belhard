import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from test import CycleGANtester

import numpy as np
import torch
from config import Config
from data_loader import get_dataloaders
from inference import CycleGANInference
from train import CycleGANTrainer
from utils import TensorBoardLogger

# Загрузка переменных окружения для Telegram бота
try:
    from dotenv import load_dotenv

    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from telegram_bot import (
        CycleGANTelegramBot,
        create_bot_config_from_env,
        create_env_template,
    )

    TELEGRAM_AVAILABLE = True
except ImportError as e:
    TELEGRAM_AVAILABLE = False
    print(f"[Info] Telegram bot not available: {e}")
    print("[Info] Install with: pip install python-telegram-bot python-dotenv")


def set_seed(seed=42):
    """Установка сидов для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_mode(config, args):
    """Режим обучения"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"cyclegan_{timestamp}"
    log_dir = os.path.join(config.log_dir, experiment_name)

    logger = TensorBoardLogger(log_dir, config)

    print("\n[Data] Loading datasets...")
    train_loader, test_loader = get_dataloaders(config)

    trainer = CycleGANTrainer(config, logger)

    if args.checkpoint:
        try:
            trainer.load_checkpoint(args.checkpoint)
            print("[Training] Resuming from checkpoint")
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")
            print("[Training] Starting from scratch")

    trainer.train(train_loader, test_loader)

    logger.close()

    print("\n[Training] Completed!")
    print(f"[Training] TensorBoard logs: {log_dir}")
    print(f"[Training] Checkpoints: {config.checkpoint_dir}")


def test_mode(config, args):
    """Режим тестирования"""
    if not args.checkpoint:
        print("[Error] Checkpoint path required for testing")
        return

    print("\n[Data] Loading test dataset...")
    _, test_loader = get_dataloaders(config)

    tester = CycleGANtester(config, args.checkpoint)

    if args.single_batch:
        print("\n[Testing] Testing single batch...")
        results, metrics = tester.test_single_batch(
            test_loader, batch_idx=0, save_images=True
        )

        print("\n[Results] Single batch metrics:")
        for key, value in metrics.items():
            print(f"  {key:15}: {value:.4f}")
    else:
        print("\n[Testing] Running comprehensive test...")
        num_samples = args.num_samples if args.num_samples else len(test_loader.dataset)
        tester.run_comprehensive_test(test_loader, num_samples=num_samples)


def inference_mode(config, args):
    """Режим инференса"""
    if not args.checkpoint and not (args.model_a and args.model_b):
        print("[Error] Need either checkpoint or both model files for inference")
        return

    # Для Telegram бота
    if args.web_interface == "telegram":
        launch_telegram_bot(config, args)
        return

    # Для Streamlit
    if args.web_interface == "streamlit":
        launch_streamlit(args)
        return

    # Для Flask API
    if args.web_interface == "api":
        launch_flask_api(config, args)
        return

    # Для других режимов инференса
    inference = CycleGANInference(config, checkpoint_path=args.checkpoint)

    if args.image_path:
        process_single_image(inference, args)
    elif args.input_dir:
        process_directory(inference, args)
    elif args.benchmark:
        run_benchmark(inference, args)
    elif args.web_interface == "simple":
        print("[Info] Simple web interface not implemented")


def process_single_image(inference, args):
    """Обработка одного изображения"""
    print(f"\n[Inference] Processing image: {args.image_path}")

    if args.direction == "A_to_B":
        result = inference.transform_A_to_B(args.image_path)
    elif args.direction == "B_to_A":
        result = inference.transform_B_to_A(args.image_path)
    elif args.direction == "cycle":
        results = inference.cycle_transform(args.image_path, return_all=True)
        print("[Inference] Cycle transformation complete")

        for key, img in results.items():
            output_path = (
                f"inference_outputs/{os.path.basename(args.image_path)}_{key}.png"
            )
            img.save(output_path)
            print(f"[Inference] Saved: {output_path}")
        return

    output_path = f"inference_outputs/transformed_{os.path.basename(args.image_path)}"
    result.save(output_path)
    print(f"[Inference] Result saved to: {output_path}")


def process_directory(inference, args):
    """Обработка директории"""
    print(f"\n[Inference] Processing directory: {args.input_dir}")
    inference.process_directory(
        args.input_dir,
        args.output_dir or "inference_outputs/batch",
        direction=args.direction,
    )
    print("[Inference] Batch processing complete")


def run_benchmark(inference, args):
    """Запуск бенчмарка"""
    print("\n[Inference] Running performance benchmark...")
    results = inference.benchmark(
        num_iterations=args.benchmark_iterations,
        image_size=(inference.config.image_size, inference.config.image_size),
    )
    print(f"[Inference] Average FPS: {results['avg_fps']:.2f}")


def launch_streamlit(args):
    """Запуск Streamlit приложения"""
    print("\n[Streamlit] Launching web interface...")

    # Проверяем наличие streamlit
    try:
        import streamlit
    except ImportError:
        print("[Error] Streamlit is not installed. Install with: pip install streamlit")
        return

    # Путь к файлу Streamlit приложения
    streamlit_app_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")

    if not os.path.exists(streamlit_app_path):
        print("[Info] Creating Streamlit app...")
        # Создаем простое Streamlit приложение
        inference = CycleGANInference(Config(), args.checkpoint)
        inference.save_streamlit_app(streamlit_app_path)

    # Запускаем Streamlit
    cmd = [
        "streamlit",
        "run",
        streamlit_app_path,
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    print(f"[Streamlit] Running: {' '.join(cmd)}")
    print(f"[Streamlit] Open browser at: http://{args.host}:{args.port}")
    print("[Streamlit] Press Ctrl+C to stop\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[Streamlit] Stopped by user")
    except Exception as e:
        print(f"[Error] Failed to launch Streamlit: {e}")


def launch_flask_api(config, args):
    """Запуск Flask API"""
    print("\n[Flask] Launching API server...")

    try:
        from inference import CycleGANInference

        inference = CycleGANInference(config, args.checkpoint)
        app = inference.create_simple_web_interface()

        if app:
            print(f"[Flask] Starting server on http://{args.host}:{args.port}")
            print("[Flask] Press Ctrl+C to stop")

            if args.debug:
                app.run(host=args.host, port=args.port, debug=True)
            else:
                from waitress import serve

                serve(app, host=args.host, port=args.port)

    except ImportError as e:
        print(f"[Error] Flask not installed: {e}")
        print("[Info] Install with: pip install flask")


def launch_telegram_bot(config, args):
    """Запуск Telegram бота"""
    if not TELEGRAM_AVAILABLE:
        print("[Error] Telegram bot dependencies not installed")
        print("[Info] Install with: pip install python-telegram-bot python-dotenv")
        return

    print("\n[Telegram Bot] Starting bot...")

    # Проверяем наличие .env файла
    env_file = Path(".env")

    if not env_file.exists():
        print("[Warning] .env file not found")
        print("[Info] Creating .env template...")

        # Создаем .env файл
        env_template = create_env_template()
        with open(".env", "w", encoding="utf-8") as f:
            f.write(env_template)

        print("Created .env file template")
        print("Please edit .env file and add your bot token")
        print("Get token from @BotFather on Telegram")
        return

    # Загружаем конфигурацию из .env
    try:
        bot_config = create_bot_config_from_env()

        # Переопределение токена если указан в аргументах
        if args.telegram_token:
            bot_config.token = args.telegram_token

        # Проверка токена
        if not bot_config.token or bot_config.token == "your_bot_token_here":
            print("[Error] Telegram bot token is not set in .env file!")
            print("[Info] Edit .env file and set TELEGRAM_BOT_TOKEN variable")
            return

        # Создаем бота
        bot = CycleGANTelegramBot(bot_config, config)

        # Запускаем бота
        bot.run()

    except Exception as e:
        print(f"[Error] Failed to start Telegram bot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="CycleGAN - Complete Pipeline with Telegram Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py --mode train --epochs 100 --batch-size 4

  # Testing
  python main.py --mode test --checkpoint checkpoints/cyclegan_best.pth

  # Inference on single image
  python main.py --mode inference --checkpoint checkpoints/cyclegan_best.pth --image-path test.jpg

  # Batch inference
  python main.py --mode inference --checkpoint checkpoints/cyclegan_best.pth --input-dir images/ --output-dir results/

  # Streamlit web interface
  python main.py --mode inference --checkpoint checkpoints/cyclegan_best.pth --web-interface streamlit

  # Flask API
  python main.py --mode inference --checkpoint checkpoints/cyclegan_best.pth --web-interface api --port 8080

  # Telegram Bot (with .env file)
  # 1. First run: python main.py --create-env
  # 2. Edit .env and add your token
  # 3. Run: python main.py --mode inference --checkpoint checkpoints/cyclegan_best.pth --web-interface telegram

  # Telegram Bot (with token argument)
  python main.py --mode inference --checkpoint checkpoints/cyclegan_best.pth --web-interface telegram --telegram-token YOUR_TOKEN
        """,
    )

    # Основные параметры
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "inference"],
        help="Operation mode",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )

    # Конфигурация
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs for training"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Тестирование
    parser.add_argument(
        "--single-batch", action="store_true", help="Test only single batch"
    )
    parser.add_argument(
        "--num-samples", type=int, default=None, help="Number of samples for testing"
    )

    # Инференс
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to single image for inference",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory for batch inference",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for batch inference",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="A_to_B",
        choices=["A_to_B", "B_to_A", "cycle"],
        help="Transformation direction",
    )

    # Веб-интерфейсы
    parser.add_argument(
        "--web-interface",
        type=str,
        default=None,
        choices=["streamlit", "api", "telegram", "simple"],
        help="Web interface type",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Server port (8501 for Streamlit, 5000 for Flask)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Telegram Bot специфичные параметры
    parser.add_argument(
        "--telegram-token",
        type=str,
        default="",
        help="Telegram bot token (overrides .env file)",
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create .env template file for Telegram bot",
    )
    parser.add_argument(
        "--env-file", type=str, default="", help="Path to .env configuration file"
    )

    # Производительность
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmark",
    )

    args = parser.parse_args()

    # Создание .env файла если запрошено
    if args.create_env:
        if TELEGRAM_AVAILABLE:
            env_template = create_env_template()
            with open(".env", "w", encoding="utf-8") as f:
                f.write(env_template)
            print("Created .env template file")
            print("Please edit .env file and add your bot token")
            print("Get token from @BotFather on Telegram")
        else:
            print("[Error] Telegram bot dependencies not installed")
            print("[Info] Install with: pip install python-telegram-bot python-dotenv")
        return

    # Установка сида
    set_seed(args.seed)

    # Конфигурация
    config = Config()

    # Переопределение параметров
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.dataset:
        config.dataset_path = args.dataset
    if args.log_dir:
        config.log_dir = args.log_dir

    # Автонастройка портов
    if args.web_interface == "api" and args.port == 8501:
        args.port = 5000  # Flask default port

    # Установка пути к .env файлу
    if args.env_file and args.env_file != ".env":
        os.environ["DOTENV_CONFIG"] = args.env_file

    # Отображение конфигурации
    config.display()

    # Выбор режима
    if args.mode == "train":
        train_mode(config, args)
    elif args.mode == "test":
        test_mode(config, args)
    elif args.mode == "inference":
        inference_mode(config, args)


if __name__ == "__main__":
    main()
