# cuda_test.py
import sys
import time

import torch


def test_cuda_setup():
    """Тестирование установки CUDA"""
    print("=" * 50)
    print("Проверка установки CUDA для чат-бота")
    print("=" * 50)

    # Проверка доступности CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступна: {cuda_available}")

    if cuda_available:
        # Информация о GPU
        gpu_count = torch.cuda.device_count()
        print(f"Количество GPU: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name}")
            print(f"  Память: {gpu_memory:.2f} GB")

        # Тест производительности
        print("\nТест производительности...")
        x = torch.randn(10000, 1000)
        y = torch.randn(1000, 10000)

        # CPU
        start_time = time.time()
        z_cpu = torch.matmul(x, y)
        cpu_time = time.time() - start_time

        # GPU
        x_gpu = x.cuda()
        y_gpu = y.cuda()

        start_time = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()  # Ожидание завершения вычислений на GPU
        gpu_time = time.time() - start_time

        print(f"Время вычислений на CPU: {cpu_time:.4f} сек")
        print(f"Время вычислений на GPU: {gpu_time:.4f} сек")
        print(f"Ускорение: {cpu_time/gpu_time:.2f}x")

    else:
        print(" CUDA не доступна. Убедитесь, что:")
        print("   - Установлены драйверы NVIDIA")
        print("   - Установлен CUDA Toolkit")
        print("   - Установлен cuDNN")
        print("   - Видеокарта поддерживает CUDA")


if __name__ == "__main__":
    test_cuda_setup()
