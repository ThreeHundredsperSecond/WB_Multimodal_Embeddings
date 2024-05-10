 # Исходный код для пайплайна
### Суммарное описание скриптов

**1. Скрипт `data_preprocessing.py`:**

Этот скрипт предназначен для предварительной обработки данных. Он выполняет следующие задачи:
- Загружает и обрабатывает изображения, изменяя их размер до заданного формата.
- Загружает и обрабатывает данные из файла формата Parquet.
- Удаляет строки с отсутствующими заголовками и описаниями.
- Преобразует идентификаторы изображений в пути к файлам изображений.
- Сохраняет обработанные данные в формате CSV.

**Пример использования:**
```bash
python data_preprocessing.py --file_path <path_to_parquet_file> --base_image_path <path_to_images> --processed_data_path <path_to_save_processed_data> --processed_images_path <path_to_save_resized_images> --default_image_size 384 384
```

**2. Скрипт `train_model.py`:**

Этот скрипт предназначен для обучения модели ruCLIP на предварительно обработанных данных. Он выполняет следующие задачи:
- Загружает модель ruCLIP и необходимые компоненты.
- Загружает обработанные данные из CSV-файла, созданного в `data_preprocessing.py`.
- Создает DataLoader для обучения модели.
- Настраивает оптимизатор и планировщик обучения.
- Обучает модель на основе данных, отслеживая прогресс с помощью tqdm и wandb.
- Сохраняет обученную модель после каждой эпохи.

**Пример использования:**
```bash
python train_model.py --file_path <path_to_parquet_file> --base_image_path <path_to_images> --processed_data_path <path_to_processed_csv> --model_save_path <path_to_save_model> --project_name fine-tuning-ruclip --num_epochs 2 --batch_size 32 --lr 1e-6
```

### Взаимосвязь между скриптами

1. **`data_preprocessing.py`** подготавливает данные и изображения, которые затем используются в **`train_model.py`**:
    - **Обработанные данные:** Сохраняются в файл CSV, содержащий пути к изображениям и соответствующие заголовки.
    - **Измененные изображения:** Сохраняются в указанной директории с уменьшенным размером.

2. **`train_model.py`** считывает обработанные данные из файла CSV и обучает модель с использованием подготовленных изображений.

### Запуск всех скриптов вместе

1. **Предварительная обработка данных:**
   Запустите `data_preprocessing.py`, чтобы подготовить данные:
   ```bash
   python data_preprocessing.py --file_path <path_to_parquet_file> --base_image_path <path_to_images> --processed_data_path <path_to_save_processed_data> --processed_images_path <path_to_save_resized_images> --default_image_size 384 384
   ```
   **Пример:**
   ```bash
   python data_preprocessing.py --file_path data/input_data.parquet --base_image_path images/raw --processed_data_path data/processed --processed_images_path images/processed --default_image_size 384 384
   ```

2. **Обучение модели:**
   После завершения предварительной обработки, запустите `train_model.py` для обучения модели:
   ```bash
   python train_model.py --file_path <path_to_parquet_file> --base_image_path <path_to_images> --processed_data_path <path_to_processed_csv> --model_save_path <path_to_save_model> --project_name fine-tuning-ruclip --num_epochs 2 --batch_size 32 --lr 1e-6
   ```
   **Пример:**
   ```bash
   python train_model.py --file_path data/input_data.parquet --base_image_path images/raw --processed_data_path data/processed/processed_data.csv --model_save_path models --project_name fine-tuning-ruclip --num_epochs 2 --batch_size 32 --lr 1e-6
   ```
