import pandas as pd
import os
from PIL import Image
import cv2
from tqdm import tqdm
import argparse


def load_image(image_path, default_size):
    """
    Загружает изображение с помощью OpenCV в формате BGR.

    Args:
        image_path (str): Путь к изображению.
        default_size (tuple): Размер изображения по умолчанию.

    Returns:
        Image: Изображение в формате PIL.
    """
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    image_pil.thumbnail(default_size)
    return image_pil


def resize_images(image_paths, output_directory, default_size):
    """
    Изменяет размер изображений и сохраняет их в выходной директории.

    Args:
        image_paths (list): Список путей к изображениям.
        output_directory (str): Директория для сохранения обработанных изображений.
        default_size (tuple): Размер изображения по умолчанию.
    """
    os.makedirs(output_directory, exist_ok=True)

    progress_bar = tqdm(total=len(image_paths), unit='image')

    for image_path in image_paths:
        image = load_image(image_path, default_size)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_directory, filename)
        image.save(output_path)
        progress_bar.set_description(f"Processing {filename}")
        progress_bar.update(1)

    progress_bar.close()
    print("Все изображения обработаны и сохранены.")


def load_and_process_data(file_path, base_image_path):
    """
    Загружает данные из файла Parquet, удаляет строки с отсутствующими заголовками,
    удаляет лишние пробелы и строки с пропущенными описаниями, а также удаляет изображения с определенным идентификатором.

    Args:
        file_path (str): Путь к файлу Parquet.
        base_image_path (str): Базовый путь к директории с изображениями.

    Returns:
        DataFrame: Обработанный DataFrame с данными.
    """
    df = pd.read_parquet(file_path)
    df = df[~df['title'].isna()]

    def remove_space(df):
        df['title'] = df['title'].replace(r'^\s*$', 'No info', regex=True)
        df['description'] = df['description'].replace(
            r'^\s*$', 'No info', regex=True)
        df = df[~df['description'].isna()]
        return df

    df = remove_space(df)

    def convert_to_path(num):
        return f'{base_image_path}/{num}.jpg'

    df['image_path'] = df['nm'].apply(convert_to_path)

    # Сохраняем только колонки 'title' и 'image_path'
    df = df[['title', 'image_path']]

    return df


def prepare_data_and_images(file_path, base_image_path, processed_data_path, processed_images_path, default_image_size=(384, 384)):
    """
    Подготавливает данные и изменяет размер изображений.

    Args:
        file_path (str): Путь к файлу Parquet.
        base_image_path (str): Базовый путь к директории с изображениями.
        processed_data_path (str): Директория для сохранения обработанных данных.
        processed_images_path (str): Директория для сохранения обработанных изображений.
        default_image_size (tuple): Размер изображения по умолчанию.
    """
    os.makedirs(processed_data_path, exist_ok=True)

    data = load_and_process_data(file_path, base_image_path)
    resize_images(data['image_path'].tolist(),
                  processed_images_path, default_image_size)

    # Сохраняем обработанные данные, содержащие только 'title' и 'image_path'
    data.to_csv(os.path.join(processed_data_path,
                'processed_data.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data preparation script')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to the input Parquet file')
    parser.add_argument('--base_image_path', type=str,
                        required=True, help='Base path for raw images')
    parser.add_argument('--processed_data_path', type=str, required=True,
                        help='Path to the directory for processed data')
    parser.add_argument('--processed_images_path', type=str,
                        required=True, help='Path to the directory for resized images')
    parser.add_argument('--default_image_size', type=int, nargs=2,
                        default=(384, 384), help='Default image size (width height)')

    args = parser.parse_args()

    prepare_data_and_images(
        file_path=args.file_path,
        base_image_path=args.base_image_path,
        processed_data_path=args.processed_data_path,
        processed_images_path=args.processed_images_path,
        default_image_size=tuple(args.default_image_size)
    )
