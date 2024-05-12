import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")
# Документация `compute_metrics_mlp_multiclass.py`



class ContentClassifier:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def _concatenate_features(self, X):
        return np.concatenate(X, axis=1) if isinstance(X, list) else X

    def fit(self, X, y, average='weighted'):
        X = self._concatenate_features(X)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='adam', max_iter=5000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average=average)
            precision = precision_score(y_val, y_pred, average=average)
            recall = recall_score(y_val, y_pred, average=average)
            scores.append((f1, precision, recall))

        avg_f1 = np.mean([score[0] for score in scores])
        avg_precision = np.mean([score[1] for score in scores])
        avg_recall = np.mean([score[2] for score in scores])
        return avg_f1, avg_precision, avg_recall

def calculate_metrics(X_text, X_img, y, model_name, task, average='weighted'):
    scaler = StandardScaler()

    X_text = scaler.fit_transform(X_text)
    X_img = scaler.fit_transform(X_img)
    X_combined = np.concatenate([X_text, X_img], axis=1)

    classifier_text = ContentClassifier()
    classifier_img = ContentClassifier()
    classifier_combined = ContentClassifier()

    results = {}
    for classifier, data, data_type in zip(
        [classifier_text, classifier_img, classifier_combined],
        [X_text, X_img, X_combined],
        ['Text', 'Image', 'Combined']
    ):
        avg_f1, avg_precision, avg_recall = classifier.fit(data, y, average=average)
        results[data_type] = {
            'Average F1-score': avg_f1,
            'Average Precision': avg_precision,
            'Average Recall': avg_recall
        }

    return {**{'Model': model_name, 'Task': task}, **results}

def get_metrics(model_names, emb_path, tasks, dfs, save_path, average='weighted'):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute metrics for multiclass classification using MLP Classifier')
    parser.add_argument('--model_names', nargs='+', required=True, help='Names of the models')
    parser.add_argument('--emb_path', nargs='+', required=True, help='Paths to embedding .npz files')
    parser.add_argument('--tasks', nargs='+', required=True, help='List of tasks')
    parser.add_argument('--dfs', nargs='+', required=True, help='Paths to CSV files containing labels for each task')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save results CSV')
    parser.add_argument('--average', type=str, default='weighted', choices=['micro', 'macro', 'weighted'], help='Averaging method for F1, Precision, Recall')

    args = parser.parse_args()

    get_metrics(
        model_names=args.model_names,
        emb_path=args.emb_path,
        tasks=args.tasks,
        dfs=args.dfs,
        save_path=args.save_path,
        average=args.average
    )
