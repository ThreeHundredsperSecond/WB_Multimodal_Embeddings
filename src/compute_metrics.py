import os
import subprocess
import argparse

def run_preparation_script(script_path, input_file, output_file):
    """Запускает скрипт подготовки данных."""
    subprocess.run(['python', script_path, '--input_file', input_file, '--output_file', output_file], check=True)

def run_compute_metrics_script(script_path, model_names, emb_paths, tasks, dfs, save_path, average=None):
    """Запускает скрипт вычисления метрик."""
    command = [
        'python', script_path,
        '--model_names', *model_names,
        '--emb_path', *emb_paths,
        '--tasks', *tasks,
        '--dfs', *dfs,
        '--save_path', save_path
    ]
    if average:
        command += ['--average', average]
    subprocess.run(command, check=True)

def main(args):
    # Подготовка данных для Male/Female
    run_preparation_script(
        'prepare_male_female_dataset.py',
        args.input_file,
        'male_female_labels.csv'
    )

    # Подготовка данных для Adult/Child
    run_preparation_script(
        'prepare_adult_child_dataset.py',
        args.input_file,
        'adult_child_labels.csv'
    )

    # Подготовка данных для IsAdult
    run_preparation_script(
        'prepare_is_adult_dataset.py',
        args.input_file,
        'is_adult_labels.csv'
    )

    # Подготовка данных для Multiclass
    run_preparation_script(
        'prepare_multiclass_dataset.py',
        args.input_file,
        'multiclass_labels.csv'
    )

    # Вычисление метрик для Male/Female с MLP и LogReg
    run_compute_metrics_script(
        'compute_metrics_mlp_binary.py',
        args.model_names,
        args.emb_paths,
        ['Male/Female'],
        ['male_female_labels.csv'],
        'results_mlp_binary.csv',
        'binary'
    )

    run_compute_metrics_script(
        'compute_metrics_logreg_binary.py',
        args.model_names,
        args.emb_paths,
        ['Male/Female'],
        ['male_female_labels.csv'],
        'results_logreg_binary.csv',
        'binary'
    )

    # Вычисление метрик для Adult/Child с MLP и LogReg
    run_compute_metrics_script(
        'compute_metrics_mlp_binary.py',
        args.model_names,
        args.emb_paths,
        ['Adult/Child'],
        ['adult_child_labels.csv'],
        'results_adult_child_mlp.csv',
        'binary'
    )

    run_compute_metrics_script(
        'compute_metrics_logreg_binary.py',
        args.model_names,
        args.emb_paths,
        ['Adult/Child'],
        ['adult_child_labels.csv'],
        'results_adult_child_logreg.csv',
        'binary'
    )

    # Вычисление метрик для IsAdult с MLP и LogReg
    run_compute_metrics_script(
        'compute_metrics_mlp_binary.py',
        args.model_names,
        args.emb_paths,
        ['IsAdult'],
        ['is_adult_labels.csv'],
        'results_is_adult_mlp.csv',
        'binary'
    )

    run_compute_metrics_script(
        'compute_metrics_logreg_binary.py',
        args.model_names,
        args.emb_paths,
        ['IsAdult'],
        ['is_adult_labels.csv'],
        'results_is_adult_logreg.csv',
        'binary'
    )

    # Вычисление метрик для Multiclass с MLP и LogReg
    run_compute_metrics_script(
        'compute_metrics_mlp_multiclass.py',
        args.model_names,
        args.emb_paths,
        ['Multiclass'],
        ['multiclass_labels.csv'],
        'results_multiclass_mlp.csv',
        'weighted'
    )

    run_compute_metrics_script(
        'compute_metrics_multiclass_logreg.py',
        args.model_names,
        args.emb_paths,
        ['Multiclass'],
        ['multiclass_labels.csv'],
        'results_multiclass_logreg.csv'
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute all metrics for various classification tasks")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the test dataframe file')
    parser.add_argument('--emb_paths', type=str, nargs='+', required=True, help='Paths to the embedding files')
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help='Names of the models')

    args = parser.parse_args()
    main(args)