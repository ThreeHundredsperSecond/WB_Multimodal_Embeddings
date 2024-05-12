import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")



class ContentClassifier:
    def __init__(self, n_splits=5, problem='binary'):
        self.n_splits = n_splits
        self.model = None
        self.task = problem

    def _concatenate_features(self, X):
        if isinstance(X, list):
            return np.concatenate(X, axis=1)
        else:
            return X

    def fit(self, X, y, one_score=True):
        average = self.task
        X = self._concatenate_features(X)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(class_weight='balanced', max_iter=5000)
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
    results = {}
    data_types = ['Text', 'Image', 'Combined', 'PCA']

    scaler = StandardScaler()
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

def get_metrics(emb_path, tasks, df_paths, save_path, model_names):
    result_df = pd.DataFrame()
    progress_bar = tqdm(total=len(model_names) * len(tasks), desc='Computing metrics')

    for model_name, emb_path, df_paths in zip(model_names, emb_path, df_paths):
        embeddings = np.load(emb_path)
        text_embeddings = embeddings['text']
        img_embeddings = embeddings['img']

        for task, df_path in zip(tasks, df_paths):
            df = pd.read_csv(df_path)
            X_text = text_embeddings[df['index'].values]
            X_img = img_embeddings[df['index'].values]
            y = df['target'].values

            metrics_results = calculate_metrics(X_text, X_img, y)

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
    parser = argparse.ArgumentParser(description='Compute metrics for logistic regression models')
    parser.add_argument('--emb_path', nargs='+', required=True, help='Paths to embedding files')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of task names')
    parser.add_argument('--df_paths', nargs='+', required=True, help='Paths to CSV files containing datasets for each task')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results CSV')
    parser.add_argument('--model_names', nargs='+', required=True, help='Names of models corresponding to embeddings')

    args = parser.parse_args()

    # Обработка каждого пути к файлам входных данных как списка для поддержки нескольких задач
    df_paths_processed = [df_paths.split(',') for df_paths in args.df_paths]

    get_metrics(
        emb_path=args.emb_path,
        tasks=args.tasks,
        df_paths=df_paths_processed,
        save_path=args.save_path,
        model_names=args.model_names
    )
