import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")
'''
## Обзор

`compute_metrics_multiclass_logreg.py` — это скрипт для вычисления метрик мультиклассовой классификации с использованием логистической регрессии. Он предназначен для работы с эмбеддингами текста и изображений, их комбинациями и преобразованными через PCA данными. Скрипт позволяет оценить эффективность различных подходов классификации на множестве категорий.


Убедитесь, что все зависимости установлены в вашем окружении перед запуском скрипта.

## Параметры командной строки

- `--model_names` (обязательно): Список имен моделей.
- `--emb_path` (обязательно): Пути к файлам с эмбеддингами (`.npz`), разделенные пробелами.
- `--tasks` (обязательно): Список имен задач.
- `--dfs` (обязательно): Пути к CSV-файлам с данными для каждой задачи. Каждый файл должен содержать колонку `category` с категориями.
- `--save_path` (обязательно): Путь для сохранения результирующего файла с метриками (CSV).

## Пример использования

```bash
python compute_metrics_multiclass_logreg.py --model_names model1 model2 --emb_path path/to/embeddings1.npz path/to/embeddings2.npz --tasks Task1 Task2 --dfs path/to/task1_labels.csv path/to/task2_labels.csv --save_path path/to/results.csv
```

Этот пример произведет вычисление метрик для двух моделей (`model1` и `model2`) на основе данных, содержащихся в файлах эмбеддингов и меток для двух задач (`Task1` и `Task2`). Результаты будут сохранены в файл `path/to/results.csv`.

## Описание работы скрипта

1. **Загрузка и подготовка данных:** Для каждой задачи скрипт загружает соответствующие эмбеддинги и метки классов. Метки классов кодируются в числовой формат.
2. **Вычисление метрик:** Скрипт обрабатывает текстовые и изображенные данные, их комбинации, а также данные, преобразованные через PCA, для каждой задачи. Для каждого типа данных вычисляются метрики мультиклассовой классификации (F1-мера, точность, полнота) с использованием логистической регрессии.
3. **Сохранение результатов:** Результаты вычислений сохраняются в указанный файл CSV.

## Важные моменты

- Каждый CSV файл с метками должен содержать колонку `category`, которая будет использоваться для кодирования меток.
- Убедитесь, что пути к файлам и имена задач указаны корректно и соответствуют вашим данным.
- Скрипт предполагает, что эмбеддинги для текста и изображений уже предварительно вычислены и сохранены в `.npz` файлах.
'''

class ContentClassifier:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.model = None

    def _concatenate_features(self, X):
        if isinstance(X, list):
            return np.concatenate(X, axis=1)
        else:
            return X

    def fit(self, X, y, average='weighted', one_score=True):
        X = self._concatenate_features(X)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(class_weight='balanced', max_iter=5000, multi_class='multinomial', solver='lbfgs')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average=average)
            precision = precision_score(y_val, y_pred, average=average)
            recall = recall_score(y_val, y_pred, average=average)
            scores.append((f1, precision, recall))

        if one_score:
            avg_f1 = sum(score[0] for score in scores) / len(scores)
            avg_precision = sum(score[1] for score in scores) / len(scores)
            avg_recall = sum(score[2] for score in scores) / len(scores)
            return avg_f1, avg_precision, avg_recall
        else:
            return scores

def calculate_metrics(X_text, X_img, y):
    scaler = StandardScaler()
    results = {}
    data_types = ['Text', 'Image', 'Combined', 'PCA']

    X_combined = np.concatenate([X_text, X_img], axis=1)
    X_pca = PCA(n_components=min(512, X_combined.shape[1])).fit_transform(scaler.fit_transform(X_combined))

    datasets = [X_text, X_img, X_combined, X_pca]

    for data_type, dataset in zip(data_types, datasets):
        classifier = ContentClassifier()
        avg_f1, avg_precision, avg_recall = classifier.fit(dataset, y)
        results[data_type] = {
            'Average F1-score': avg_f1,
            'Average Precision': avg_precision,
            'Average Recall': avg_recall
        }

    return results

def get_metrics(model_names, emb_path, tasks, dfs, save_path):
    columns = ['Model', 'Task', 'Data Type', 'Average F1-score', 'Average Precision', 'Average Recall']
    result_df = pd.DataFrame(columns=columns)

    total_iterations = len(model_names) * len(tasks)
    progress_bar = tqdm(total=total_iterations, desc='Progress')

    for model_index, model_name in enumerate(model_names):
        embeddings = np.load(emb_path[model_index])
        text_embeddings = embeddings['text']
        img_embeddings = embeddings['img']

        for task_index, task in enumerate(tasks):
            df = dfs[task_index]
            y = df['label_encoded'].values

            metrics_results = calculate_metrics(text_embeddings, img_embeddings, y)

            for data_type, metrics in metrics_results.items():
                result_df = result_df.append({
                    'Model': model_name,
                    'Task': task,
                    'Data Type': data_type,
                    **metrics
                }, ignore_index=True)

            progress_bar.update(1)

    progress_bar.close()
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute multiclass metrics using logistic regression')
    parser.add_argument('--model_names', nargs='+', required=True, help='List of model names')
    parser.add_argument('--emb_path', nargs='+', required=True, help='Paths to embedding files')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of task names')
    parser.add_argument('--dfs', nargs='+', required=True, help='Paths to CSV files containing labels for each task')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results CSV')

    args = parser.parse_args()

   

    get_metrics(
        model_names=args.model_names,
        emb_path=args.emb_path,
        tasks=args.tasks,
        dfs=dfs,
        save_path=args.save_path
    )
