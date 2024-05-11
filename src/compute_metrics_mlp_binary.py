import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")

'''
## Обзор

`compute_metrics_mlp_binary.py` — это скрипт для вычисления метрик (F1, точность, полнота) бинарной классификации с использованием многослойного перцептрона (MLP) 
или на основе предварительно вычисленных эмбеддингов текста и изображений. Скрипт поддерживает работу с различными типами данных,
включая текст, изображения, их комбинации и PCA-преобразованные данные.

## Установка зависимостей


## Параметры командной строки

Скрипт поддерживает следующие аргументы командной строки для настройки процесса вычисления метрик:

- `--model_names`: Список имен моделей (например, для различения результатов в выходном файле).
- `--emb_path`: Пути к файлам с эмбеддингами (`.npz`), содержащими эмбеддинги текста (`text`) и изображений (`img`).
- `--tasks`: Список имен задач (используется для метаинформации в выходном файле).
- `--dfs`: Пути к CSV-файлам с данными, где каждый файл должен содержать колонку `target` с метками классов.
- `--save_path`: Путь к файлу, в который будут сохранены результаты вычислений.
- `--average`: Метод усреднения для вычисления метрик (поддерживаются `binary`, `micro`, `macro`, `weighted`).

## Пример использования

```bash
python compute_metrics_mlp_binary.py --model_names model1 model2 --emb_path path/to/embeddings1.npz path/to/embeddings2.npz --tasks Task1 Task2 --dfs path/to/task1_labels.csv path/to/task2_labels.csv --save_path path/to/results.csv --average binary
```   

В этом примере скрипт вычислит метрики для двух задач (`Task1` и `Task2`), используя данные из соответствующих файлов эмбеддингов и меток классов. Результаты будут сохранены в `path/to/results.csv`.

## Описание работы скрипта

1. **Загрузка данных:** Для каждой задачи скрипт загружает соответствующие эмбеддинги текста и изображений из указанных файлов `.npz`, а также метки классов из CSV-файлов.
2. **Обработка данных:** Выполняется нормализация данных, их конкатенация и, при необходимости, применение PCA.
3. **Классификация:** Для каждого типа данных (текст, изображение, комбинация, PCA) выполняется обучение и валидация модели MLPClassifier, вычисляются метрики.
4. **Сохранение результатов:** Результаты вычислений (средние значения F1, точности и полноты) для каждого типа данных сохраняются в указанный файл.
'''

class ContentClassifier:
    def __init__(self, n_splits=5, problem='binary'):
        self.n_splits = n_splits
        self.problem = problem

    def _concatenate_features(self, X):
        return np.concatenate(X, axis=1) if isinstance(X, list) else X

    def fit(self, X, y, one_score=True):
        X = self._concatenate_features(X)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=5000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average=self.problem)
            precision = precision_score(y_val, y_pred, average=self.problem)
            recall = recall_score(y_val, y_pred, average=self.problem)
            scores.append((f1, precision, recall))

        return (np.mean(scores, axis=0) if one_score else scores)

def calculate_metrics(X_text, X_img, y, model_name, task, data_type, average='binary'):
    scaler = StandardScaler()

    classifier_text = ContentClassifier(problem=average)
    classifier_img = ContentClassifier(problem=average)
    classifier_combined = ContentClassifier(problem=average)
    classifier_pca = ContentClassifier(problem=average)

    X_combined = np.concatenate([X_text, X_img], axis=1)
    X_pca = PCA(n_components=512).fit_transform(scaler.fit_transform(X_combined))

    datasets = [X_text, X_img, X_combined, X_pca]
    results = {}

    for classifier, data, dtype in zip([classifier_text, classifier_img, classifier_combined, classifier_pca], 
                                       datasets, 
                                       data_type):
        avg_f1, avg_precision, avg_recall = classifier.fit(data, y)
        results[dtype] = {'Average F1-score': avg_f1, 'Average Precision': avg_precision, 'Average Recall': avg_recall}

    results = {**{'Model': model_name, 'Task': task}, **results}
    return results

def get_metrics(model_names, emb_path, tasks, dfs, save_path, average='binary'):
    result_df = pd.DataFrame()
    
    for model, emb, task, df in zip(model_names, emb_path, tasks, dfs):
        embeddings = np.load(emb)
        X_text = embeddings['text']
        X_img = embeddings['img']
        y = df['target'].values
        
        metrics = calculate_metrics(X_text, X_img, y, model, task, ['Text', 'Image', 'Combined', 'PCA'], average)
        
        for data_type, scores in metrics.items():
            if data_type in ['Model', 'Task']:
                continue
            result_df = result_df.append({
                'Model': metrics['Model'],
                'Task': metrics['Task'],
                'Data Type': data_type,
                **scores
            }, ignore_index=True)

    result_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute metrics for binary classification using MLP Classifier')
    parser.add_argument('--model_names', nargs='+', required=True, help='Names of the models')
    parser.add_argument('--emb_path', nargs='+', required=True, help='Paths to embedding .npz files')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks')
    parser.add_argument('--dfs', nargs='+', required=True, help='Paths to CSV files containing labels for each task')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results CSV')
    parser.add_argument('--average', type=str, default='binary', choices=['binary', 'micro', 'macro', 'weighted'], help='Averaging method for F1, Precision, Recall')

    args = parser.parse_args()

    # Подготовка данных: загрузка целевых значений из CSV файлов
    dfs = [pd.read_csv(df_path) for df_path in args.dfs]
    for df in dfs:
        if 'target' not in df.columns:
            raise ValueError("Each dataframe must include a 'target' column.")

    get_metrics(
        model_names=args.model_names,
        emb_path=args.emb_path,
        tasks=args.tasks,
        dfs=dfs,
        save_path=args.save_path,
        average=args.average
    )