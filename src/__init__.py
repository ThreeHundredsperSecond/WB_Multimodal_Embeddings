from .data_preparation import prepare_data_and_images
from .train_model import train_model
import os

def run_pipeline(parquet_file, raw_images_dir, processed_data_dir, processed_images_dir, model_save_dir, project_name="fine-tuning-ruclip", num_epochs=2, batch_size=32, lr=1e-6, default_image_size=(384, 384)):
    """
    Запускает весь пайплайн от подготовки данных до обучения модели.

    Args:
        parquet_file (str): Путь к файлу Parquet.
        raw_images_dir (str): Директория с исходными изображениями.
        processed_data_dir (str): Директория для сохранения обработанных данных.
        processed_images_dir (str): Директория для сохранения уменьшенных изображений.
        model_save_dir (str): Директория для сохранения обученной модели.
        project_name (str): Название проекта для wandb.
        num_epochs (int, optional): Количество эпох для обучения. По умолчанию 2.
        batch_size (int, optional): Размер пакета для обучения. По умолчанию 32.
        lr (float, optional): Начальная скорость обучения. По умолчанию 1e-6.
        default_image_size (tuple, optional): Размер изображения по умолчанию (ширина, высота). По умолчанию (384, 384).
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # Подготовка данных
    prepare_data_and_images(
        file_path=parquet_file,
        base_image_path=raw_images_dir,
        processed_data_path=processed_data_dir,
        processed_images_path=processed_images_dir,
        default_image_size=default_image_size
    )

    # Путь к обработанным данным
    processed_data_csv = os.path.join(processed_data_dir, 'processed_data.csv')

    # Обучение модели
    train_model(
        file_path=parquet_file,
        base_image_path=raw_images_dir,
        processed_data_path=processed_data_csv,
        model_save_path=model_save_dir,
        project_name=project_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr
    )

__all__ = ["run_pipeline", "prepare_data_and_images", "train_model"]