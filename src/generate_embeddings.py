import torch
import ruclip
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse


class EmbeddingPipeline:
    def __init__(self, base_model_name='ruclip-vit-base-patch32-384', fine_tuned_model_path=None, device=None, quiet=True):
        """Initialize the Pipeline with either a base or fine-tuned model.

        Args:
            base_model_name (str): The base model name to load.
            fine_tuned_model_path (str, optional): Path to fine-tuned model checkpoint.
            device (str, optional): Device to use ('cuda' or 'cpu').
            quiet (bool): If True, suppress progress bars in ruclip.Predictor.
        """
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)

        # Load the base model and processor
        self.model, self.processor = ruclip.load(base_model_name, device=self.device)

        # Load the fine-tuned model if the path is provided
        if fine_tuned_model_path:
            checkpoint = torch.load(fine_tuned_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        self.predictor = ruclip.Predictor(self.model, self.processor, self.device, quiet=quiet)

    class TextDataset(Dataset):
        """Dataset for pairing text and image paths."""
        def __init__(self, df):
            self.texts = df['title'].values
            self.image_paths = df['image_path'].values

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            image_path = self.image_paths[idx]
            return text, image_path

    def get_embeddings(self, df, batch_size=10):
        """Get image and text embeddings from a DataFrame.

        Args:
            df (DataFrame): DataFrame containing 'title' and 'image_path' columns.
            batch_size (int): Batch size for processing.

        Returns:
            tuple: Image embeddings, Text embeddings.
        """
        dataset = self.TextDataset(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        img_embeddings = []
        text_embeddings = []

        with tqdm(total=len(dataloader), desc="Processing batches", leave=True) as pbar:
            for texts, image_paths in dataloader:
                images = [Image.open(path) for path in image_paths]
                with torch.no_grad():
                    text_emb = self.predictor.get_text_latents(texts)
                    img_emb = self.predictor.get_image_latents(images)
                img_embeddings.append(img_emb.cpu().numpy())
                text_embeddings.append(text_emb.cpu().numpy())
                pbar.update(1)

        img_embeddings = np.vstack(img_embeddings)
        text_embeddings = np.vstack(text_embeddings)

        return img_embeddings, text_embeddings


def generate_embeddings(fine_tuned_model_path, processed_data_csv, output_path, batch_size=10):
    """
    Generate embeddings for images and texts.

    Args:
        fine_tuned_model_path (str): Path to the fine-tuned model checkpoint.
        processed_data_csv (str): Path to the processed data CSV file.
        output_path (str): Path to save generated embeddings.
        batch_size (int): Batch size for processing.
    """
    df = pd.read_csv(processed_data_csv)

    pipeline = EmbeddingPipeline(fine_tuned_model_path=fine_tuned_model_path, quiet=True)
    img_embeddings, text_embeddings = pipeline.get_embeddings(df, batch_size=batch_size)

    np.savez(output_path, text=text_embeddings, img=img_embeddings)

    print(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for images and texts')
    parser.add_argument('--fine_tuned_model_path', type=str, required=True, help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--processed_data_csv', type=str, required=True, help='Path to the processed data CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save generated embeddings (npz file)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')

    args = parser.parse_args()

    generate_embeddings(
        fine_tuned_model_path=args.fine_tuned_model_path,
        processed_data_csv=args.processed_data_csv,
        output_path=args.output_path,
        batch_size=args.batch_size
    )