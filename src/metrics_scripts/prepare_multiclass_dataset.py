import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import argparse

def create_and_save_labels(df, embedding_file, output_filename):
    """
    Создает метки классов для DataFrame, выбирая embeddings по индексам из DataFrame,
    и сохраняет метки в CSV файл.

    Args:
        df (pd.DataFrame): DataFrame с индексами и метками классов.
        embedding_file (str): Путь к файлу с embeddings.
        output_filename (str): Имя CSV файла для сохранения меток.
    """

    # Загрузка embeddings
    embeddings = np.load(embedding_file)
    text_embeddings = embeddings['text']
    img_embeddings = embeddings['img']

    # Создание LabelEncoder
    label_encoder = LabelEncoder()

    # Проверка наличия столбца "target"
    if 'target' not in df.columns:
        raise ValueError("DataFrame must include a 'target' column.")

    # Преобразование меток классов в числовые
    y = label_encoder.fit_transform(df['target'])

    # Выборка embeddings по индексам из DataFrame
    X_text = text_embeddings[df['index'].values]
    X_img = img_embeddings[df['index'].values]

    # Создание DataFrame для меток классов
    df_y = pd.DataFrame({
        'index': df.index,
        'target': y
    })

    # Сохранение меток в CSV файл
    df_y.to_csv(output_filename, index=False)
    print(f"Labels saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Создает метки классов для DataFrame и сохраняет их в CSV файл.')
    parser.add_argument('--df_paths', nargs='+', required=True, help='Пути к CSV файлам с данными.')
    parser.add_argument('--embedding_file', type=str, required=True, help='Путь к файлу с embeddings.')
    parser.add_argument('--output_filenames', nargs='+', required=True, 
                        help='Имена CSV файлов для сохранения меток. Количество имен должно совпадать с количеством DataFrame.')

    args = parser.parse_args()

    # Проверка количества аргументов
    if len(args.df_paths) != len(args.output_filenames):
        raise ValueError("Количество путей к CSV файлам должно совпадать с количеством имен выходных файлов.")

    # Обработка каждого DataFrame
    for df_path, output_filename in zip(args.df_paths, args.output_filenames):
        df = pd.read_csv(df_path)
        create_and_save_labels(df, args.embedding_file, output_filename)
