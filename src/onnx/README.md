

# **`convert_to_onnx`**

Основные шаги, выполняемые скриптом:

1. Загрузка модели ruCLIP с использованием функции `ruclip.load()`. 

2. Если указан путь к файлу с fine-tuned моделью (`fine_tuned_model_path`), то загружаются веса из этого файла и применяются к модели.

3. Создаются dummy-входы для визуальной и текстовой частей модели. Для визуальной части создается случайный тензор размером `[1, 3, 384, 384]`, а для текстовой части - случайный тензор размером `[1, 77]` (предполагается, что максимальная длина текста составляет 77 токенов).

4. Создается экземпляр класса `clip_onnx` (предполагается, что этот класс определен в файле `clip_onnx.py`), который принимает модель ruCLIP и пути для сохранения визуальной и текстовой частей модели в формате ONNX.

5. Вызывается метод `convert2onnx()` экземпляра `clip_onnx`, который выполняет конвертацию модели в формат ONNX, используя созданные dummy-входы. Флаг `verbose=True` включает подробный вывод информации о процессе конвертации.

6. После успешной конвертации выводятся сообщения о путях сохранения визуальной и текстовой частей модели в формате ONNX.

Скрипт также использует `argparse` для обработки аргументов командной строки. Можно указать следующие аргументы:
- `--fine_tuned_model_path`: путь к файлу с fine-tuned моделью (необязательный аргумент).
- `visual_path`: путь для сохранения визуальной части модели в формате ONNX.
- `textual_path`: путь для сохранения текстовой части модели в формате ONNX.

Пример использования:

```
python convert_ruclip_to_onnx.py "visual_model.onnx" "textual_model.onnx" --fine_tuned_model_path "fine_tuned_model.pth"

```

# **`onnx_generate_embeddings`**

Этот скрипт предназначен для генерации вложений (embeddings) изображений и текстов с использованием модели ruCLIP в формате ONNX. Он предоставляет удобный способ получения векторных представлений для изображений и соответствующих им текстовых описаний.

**Основные возможности:**

1. **Генерация вложений изображений и текстов**: Скрипт позволяет загрузить предварительно обученные модели CLIP и использовать их для генерации вложений изображений и текстов.

2. **Поддержка ONNX**: Скрипт предоставляет возможность конвертировать модели CLIP в формат ONNX для более эффективного использования на различных платформах.

3. **Обработка данных**: Предоставляются функции и классы для обработки изображений и текстов, включая возможность работы с наборами данных.

4. **Конфигурируемость**: Пользователь может настроить параметры обработки данных и конфигурацию моделей CLIP с помощью аргументов командной строки.

**Использование:**

Для использования скрипта необходимо выполнить команду в командной строке, указав следующие обязательные аргументы:

- `visual_onnx_path`: Путь к предварительно обученной модели CLIP для обработки изображений в формате ONNX.
- `textual_onnx_path`: Путь к предварительно обученной модели CLIP для обработки текстов в формате ONNX.
- `processed_data_csv`: Путь к CSV-файлу с обработанными данными, содержащими изображения и соответствующие текстовые описания.
- `output_path`: Путь для сохранения сгенерированных вложений в формате `.npz`.
- `batch_size` (опционально): Размер пакета для обработки данных (по умолчанию 10).

Пример использования:

```
python onnx_generate_embeddings.py --visual_onnx_path "visual_model.onnx" --textual_onnx_path "textual_model.onnx" --processed_data_csv "data.csv" --output_path "embeddings.npz" --batch_size 32
```

После выполнения скрипта будет создан файл `embeddings.npz`, содержащий вложения для изображений и текстов, готовые к использованию в дальнейшем анализе и приложениях.
