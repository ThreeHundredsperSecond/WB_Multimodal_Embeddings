import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import wandb
import argparse
import ruclip
import pandas as pd


class ImageTitleDataset:
    def __init__(self, processor, list_image_path, list_txt):
        """
        Класс Dataset для загрузки изображений и соответствующих текстовых описаний.

        Args:
            processor: Обработчик текста (например, токенизатор).
            list_image_path (list): Список путей к изображениям.
            list_txt (list): Список текстовых описаний.
        """
        self.processor = processor
        self.image_path = list_image_path
        self.title = list_txt

    def __len__(self):
        """Возвращает общее количество пар изображение-текст."""
        return len(self.title)

    def __getitem__(self, idx):
        """
        Получает пару изображение-текст по индексу.

        Args:
            idx (int): Индекс пары.

        Returns:
            tuple: Кортеж, содержащий тензор изображения и тензор текста.
        """
        image = Image.open(self.image_path[idx])
        text = self.title[idx]
        data_dict = self.processor(text=text, images=[image])

        # Получаем тензор для изображений
        image = data_dict['pixel_values']
        # Тензор с IDs для текста
        text = data_dict['input_ids']

        return image, text


def collate_fn(batch):
    """
    Функция collate_fn для DataLoader.

    Args:
        batch (list): Список пар изображение-текст.

    Returns:
        tuple: Кортеж, содержащий тензоры изображений и текста.
    """
    images, texts = zip(*batch)
    images = torch.stack(images, dim=0).squeeze(1)
    texts = torch.stack(texts, dim=0).squeeze(1)
    return images, texts


def create_dataloader(processor, list_image_path, list_txt, batch_size, shuffle=True, num_workers=-1):
    """
    Создает DataLoader для набора данных ImageTitleDataset.

    Args:
        processor: Обработчик текста.
        list_image_path (list): Список путей к изображениям.
        list_txt (list): Список текстовых описаний.
        batch_size (int): Размер пакета.
        shuffle (bool, optional): Флаг перемешивания данных. По умолчанию True.
        num_workers (int, optional): Количество рабочих процессов для загрузки данных.
                                     По умолчанию -1, что означает использование всех доступных CPU.

    Returns:
        DataLoader: Объект DataLoader для набора данных.
    """
    dataset = ImageTitleDataset(processor, list_image_path, list_txt)

    if num_workers == -1:
        num_workers = os.cpu_count()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader


def load_and_process_data(processed_data_path):
    """
    Загружает данные из обработанного CSV-файла.

    Args:
        processed_data_path (str): Путь к CSV-файлу с обработанными данными.

    Returns:
        DataFrame: DataFrame с обработанными данными.
    """
    df = pd.read_csv(processed_data_path)
    return df


def train_model(file_path, base_image_path, processed_data_path, model_save_path, project_name, num_epochs=2, batch_size=32, lr=1e-6):
    """
    Обучает модель ruCLIP на данных.

    Args:
        file_path (str): Путь к файлу Parquet.
        base_image_path (str): Базовый путь к изображениями.
        processed_data_path (str): Путь к обработанным данным.
        model_save_path (str): Директория для сохранения обученной модели.
        project_name (str): Название проекта для wandb.
        num_epochs (int, optional): Количество эпох обучения. По умолчанию 2.
        batch_size (int, optional): Размер пакета. По умолчанию 32.
        lr (float, optional): Начальная скорость обучения. По умолчанию 1e-6.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)

    data = load_and_process_data(processed_data_path)
    list_image_path = data['image_path'].tolist()
    list_txt = data['title'].tolist()

    dataloader = create_dataloader(processor, list_image_path, list_txt, batch_size)

    wandb.login()
    run = wandb.init(project=project_name, name=f"training-ruclip-{num_epochs}epoch")

    clip.to(device)
    optimizer = optim.Adam(clip.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()

            images, texts = batch
            images = images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = clip.forward(input_ids=texts, pixel_values=images)

            ground_truth = torch.arange(images.size(0), device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            current_time = datetime.now().strftime("%H:%M:%S")
            current_lr = scheduler.get_last_lr()[0]

            wandb.log({"Epoch": epoch + 1, "Batch": i + 1, "Loss": total_loss.item(), "Time": current_time, "LR": current_lr, "BS": batch_size})

            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}/{len(dataloader)}, Loss: {total_loss.item():.6f}, LR: {current_lr:.7f}, Time: {current_time}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': clip.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_save_path, f'ruCLIP_model_epoch{epoch + 1}.pth'))

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for ruCLIP model')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input Parquet file')
    parser.add_argument('--base_image_path', type=str, required=True, help='Base path for raw images')
    parser.add_argument('--processed_data_path', type=str, required=True, help='Path to the directory for processed data')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to the directory for saving trained model')
    parser.add_argument('--project_name', type=str, default='fine-tuning-ruclip', help='Project name for wandb')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for optimizer')

    args = parser.parse_args()

    train_model(
        file_path=args.file_path,
        base_image_path=args.base_image_path,
        processed_data_path=args.processed_data_path,
        model_save_path=args.model_save_path,
        project_name=args.project_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )