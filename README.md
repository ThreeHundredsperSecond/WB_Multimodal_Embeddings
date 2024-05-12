# WB_Multimodal_Embeddings
### 1. **Структура Репозитория**
```
WB_Multimodal_Embeddings/
│
├── data/                                     # Директория для хранения данных
│   
│   
│
├── models/                                   # Директория для сохранения обученных моделей
│
├── notebooks/                                # Jupyter notebooks для исследования и прототипирования
│   └── main_notebook.ipynb                   # Ваш исходный ноутбук
│
├── src/                                      # Исходный код для пайплайна
│   ├── __init__.py                           # Запускает все скрипты последовательно 
│   ├── data_preparation.py                   # Скрипт для подготовки данных
│   ├── train_model.py                        # Скрипт для обучения модели
│   ├── generate_embeddings.py                # Скрипт для получения embeddings   
│   ├── compute_metrics.py                    # Скрипт для подсчета метрик
│   └── metrics_scripts/                      # Директория для скриптов вычисления метрик
│       ├── compute_metrics_logreg_binary.py  # Скрипт для вычисления метрик для бинарной классификации логистической регрессии 
│       ├── compute_metrics_mlp_binary.py     # Скрипт для вычисления метрик для бинарной классификации MLP регрессии 
│       ├── compute_metrics_mlp_multiclass.py # Скрипт для вычисления метрик для многоклассовой классификации MLP
│       ├── compute_metrics_multiclass_logreg.py  # Скрипт для вычисления метрик для многоклассовой классификации логистической регрессии
│       ├── prepare_adult_child_dataset.py    # Скрипт для подготовки датасета "Взрослый/Ребенок"
│       ├── prepare_is_adult_dataset.py       # Скрипт для подготовки датасета "18+"
│       ├── prepare_male_female_dataset.py    # Скрипт для подготовки датасета "Мужчина/Женщина"
│       └── prepare_multiclass_dataset.py     # Скрипт для подготовки датасета для многоклассовой классификации
│
├── requirements.txt                          # Файл с зависимостями
├── .gitignore                                # Файл для исключения из репозитория
└── README.md                                 # Описание проекта
```
