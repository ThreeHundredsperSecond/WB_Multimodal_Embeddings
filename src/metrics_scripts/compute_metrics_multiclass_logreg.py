
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

def calculate_metrics(X_text, X_img, y, model_name, task, average='weighted'): # Добавили аргумент average
    scaler = StandardScaler()
    results = {}
    data_types = ['Text', 'Image', 'Combined', 'PCA']

    X_combined = np.concatenate([X_text, X_img], axis=1)
    X_pca = PCA(n_components=min(10, X_combined.shape[1])).fit_transform(scaler.fit_transform(X_combined))

    datasets = [X_text, X_img, X_combined, X_pca]

    for data_type, dataset in zip(data_types, datasets):
        classifier = ContentClassifier()
        avg_f1, avg_precision, avg_recall = classifier.fit(dataset, y, average=average)
        results[data_type] = {
            'Average F1-score': avg_f1,
            'Average Precision': avg_precision,
            'Average Recall': avg_recall
        }

    return {**{'Model': model_name, 'Task': task}, **results}

def get_metrics(model_names, emb_path, tasks, dfs, save_path, average='weighted', raw_df_paths=None, output_label_filenames=None):
    """
    Вычисляет метрики и опционально создает и сохраняет метки классов.

    Args:
        model_names (list): Список имен моделей.
        emb_path (list): Список путей к файлам .npz с embeddings.
        tasks (list): Список имен задач.
        dfs (list): Список путей к CSV файлам с метками классов для каждой задачи.
        save_path (str): Путь для сохранения CSV файла с результатами.
        average (str, optional): Метод усреднения для F1, Precision, Recall. По умолчанию 'weighted'.
        raw_df_paths (list, optional): Список путей к CSV файлам с исходными данными.
        output_label_filenames (list, optional): Список имен CSV файлов для сохранения созданных меток классов.
    """
    results = []

    for model_name, emb_file, task, df_file in zip(model_names, emb_path, tasks, dfs):
        embeddings = np.load(emb_file)
        X_text = embeddings['text']
        X_img = embeddings['img']

        df = pd.read_csv(df_file)
        if 'target' not in df.columns:
            raise ValueError(f"Each dataframe must include a 'target' column, but {df_file} does not.")

        y = LabelEncoder().fit_transform(df['target'])

        metrics = calculate_metrics(X_text, X_img, y, model_name, task, average=average)
        results.append(metrics)

    result_df = pd.DataFrame(results)
    result_df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    # Создание и сохранение меток классов (опционально)
    if raw_df_paths and output_label_filenames:
        if len(raw_df_paths) != len(output_label_filenames):
            raise ValueError("Количество путей к CSV файлам с исходными данными должно совпадать с количеством имен выходных файлов для меток.")

        for raw_df_path, output_filename in zip(raw_df_paths, output_label_filenames):
            df = pd.read_csv(raw_df_path)
            create_and_save_labels(df, emb_path[0], output_filename) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute metrics for multiclass classification using MLP Classifier')
    parser.add_argument('--model_names', nargs='+', required=True, help='Names of the models')
    parser.add_argument('--emb_path', nargs='+', required=True, help='Paths to embedding .npz files')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks')
    parser.add_argument('--dfs', nargs='+', required=True, help='Paths to CSV files containing labels for each task')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results CSV')



    args = parser.parse_args()

    get_metrics(
        model_names=args.model_names,
        emb_path=args.emb_path,
        tasks=args.tasks,
        dfs=args.dfs,
        save_path=args.save_path,
    
    )
