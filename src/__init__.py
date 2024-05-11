from .data_preparation import prepare_data_and_images
from .train_model import train_model
from .generate_embeddings import generate_embeddings
import os
import subprocess

def run_pipeline(parquet_file, raw_images_dir, processed_data_dir, processed_images_dir, 
                 model_save_dir, embeddings_dir, results_dir, project_name="fine-tuning-ruclip", 
                 num_epochs=2, batch_size=32, lr=1e-6, default_image_size=(384, 384), emb_batch_size=32):
    """
    Запускает весь пайплайн от подготовки данных до обучения модели, генерации эмбеддингов и вычисления метрик.

    Args:
        parquet_file (str): Путь к файлу Parquet.
        raw_images_dir (str): Директория с исходными изображениями.
        processed_data_dir (str): Директория для сохранения обработанных данных.
        processed_images_dir (str): Директория для сохранения уменьшенных изображений.
        model_save_dir (str): Директория для сохранения обученной модели.
        embeddings_dir (str): Директория для сохранения эмбеддингов.
        results_dir (str): Директория для сохранения результатов метрик.
        project_name (str): Название проекта для wandb.
        num_epochs (int): Количество эпох для обучения.
        batch_size (int): Размер пакета для обучения.
        lr (float): Начальная скорость обучения.
        default_image_size (tuple): Размер изображения по умолчанию (ширина, высота).
        emb_batch_size (int): Размер пакета для генерации эмбеддингов.
    """

    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Подготовка данных и изображений
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

    # Путь к дообученной модели
    fine_tuned_model_path = os.path.join(model_save_dir, f'ruCLIP_model_epoch{num_epochs}.pth')

    # Путь к файлу эмбеддингов
    output_embeddings_path = os.path.join(embeddings_dir, 'embeddings.npz')

    # Генерация эмбеддингов
    generate_embeddings(
        fine_tuned_model_path=fine_tuned_model_path,
        processed_data_csv=processed_data_csv,
        output_path=output_embeddings_path,
        batch_size=emb_batch_size
    )

    # Путь к файлу с результатами метрик
    output_metrics_path = os.path.join(results_dir, 'metrics_results.csv')

    # Конфигурация для compute_metrics
    compute_metrics_script = 'src/compute_metrics.py'
    model_names = ['ruCLIP']
    emb_paths = [output_embeddings_path]
    tasks = ['Male/Female', 'Adult/Child', 'IsAdult', 'Multiclass']
    dfs = [
        os.path.join(processed_data_dir, 'male_female_labels.csv'),
        os.path.join(processed_data_dir, 'adult_child_labels.csv'),
        os.path.join(processed_data_dir, 'is_adult_labels.csv'),
        os.path.join(processed_data_dir, 'multiclass_labels.csv')
    ]

    # Подготовка меток для всех задач
    label_preparation_scripts = [
        'prepare_male_female_dataset.py',
        'prepare_adult_child_dataset.py',
        'prepare_is_adult_dataset.py',
        'prepare_multiclass_dataset.py'
    ]
    
    # Подготовить метки для каждой из задач
    for prep_script, label_file in zip(label_preparation_scripts, dfs):
        subprocess.run([
            'python', os.path.join('src', prep_script),
            '--input_file', processed_data_csv,
            '--output_file', label_file
        ], check=True)

    # Вызов скрипта compute_metrics.py с нужными параметрами
    subprocess.run([
        'python', compute_metrics_script,
        '--model_names', *model_names,
        '--emb_paths', *emb_paths,
        '--tasks', *tasks,
        '--dfs', *dfs,
        '--save_path', output_metrics_path
    ], check=True)

__all__ = ["run_pipeline", "prepare_data_and_images", "train_model", "generate_embeddings"]
