
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
import ruclip
from PIL import Image
import os
import argparse
import torch.nn.functional as F
from torch import nn
import onnxruntime
import onnx
from torch import nn
from onnxruntime.quantization import quantize_dynamic, QuantType


class Textual(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.token_embedding = model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # needs .float() before .argmax(  ) to work
        x = x[torch.arange(x.shape[0]), text.float().argmax(dim=-1)] @ self.text_projection

        return x


def attention(self, x: torch.Tensor):
    # onnx doesn't like multi_head_attention_forward so this is a reimplementation
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    q, k, v = (torch.einsum("tbh, oh -> tbo", x, self.attn.in_proj_weight) + self.attn.in_proj_bias).contiguous().chunk(
        3, dim=-1)
    tgt_len = q.shape[0]
    bsz = q.shape[1]
    num_heads = self.attn.num_heads
    head_dim = q.shape[2] // num_heads
    attn_output = scaled_dot_product_attention(
        q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1),
        k.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1),
        v.reshape(tgt_len, bsz * num_heads, head_dim).transpose(0, 1), self.attn_mask, 0.0
    )
    attn_output = attn_output.transpose(0, 1).contiguous().view(q.shape)
    attn_output = F.linear(attn_output, self.attn.out_proj.weight, self.attn.out_proj.bias)
    return attn_output

def scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p):
    if attn_mask is None:
        attn_weight = torch.softmax(Q @ K.transpose(-2, -1) / Q.size(-1)**0.5, dim=-1)
    else:
        attn_weight = torch.softmax(Q @ K.transpose(-2, -1) / Q.size(-1)**0.5 + attn_mask[None, ...], dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p) # this is always 0.0 in CLIP so I comment it out.
    return attn_weight @ V


DEFAULT_EXPORT = dict(input_names=['input'], output_names=['output'],
                      export_params=True, verbose=False, opset_version=12,
                      do_constant_folding=True,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
class clip_converter(nn.Module):
    def __init__(self, model, visual_path: str = "clip_visual.onnx",
                 textual_path: str = "clip_textual.onnx"):
        super().__init__()
        self.model = model
        self.visual_path = visual_path
        self.textual_path = textual_path
        self.visual_flag = False
        self.textual_flag = False
        self.logit_scale = self.model.logit_scale.exp()

        self.model.eval()
        for x in self.model.parameters():
            x.requires_grad = False

    def quantization(self, mode: str = "dynamic"):
        assert mode in ["dynamic"]
        if mode == "dynamic":
            model_quant_visual = f"{self.visual_path}.quant"
            quantize_dynamic(self.visual_path,
                             model_quant_visual,
                             weight_type=QuantType.QUInt8)
            self.visual_path = model_quant_visual

            model_quant_textual = f"{self.textual_path}.quant"
            quantize_dynamic(self.textual_path,
                             model_quant_textual,
                             weight_type=QuantType.QUInt8)
            self.textual_path = model_quant_textual

    def torch_export(self, model, dummy_input, path: str, export_params=DEFAULT_EXPORT):
        torch.onnx.export(model, dummy_input, path, **export_params)

    def onnx_checker(self, path: str):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        del model

    def convert_visual(self, dummy_input, wrapper=lambda x: x,
                       export_params=DEFAULT_EXPORT):
        visual = wrapper(self.model.visual)
        self.torch_export(visual, dummy_input, self.visual_path,
                          export_params=export_params)
        self.onnx_checker(self.visual_path)

    def convert_textual(self, dummy_input, wrapper=Textual,
                        export_params=DEFAULT_EXPORT):
        textual = wrapper(self.model)
        self.torch_export(textual, dummy_input, self.textual_path,
                          export_params=export_params)
        self.onnx_checker(self.textual_path)

    def convert2onnx(self, visual_input=None, textual_input=None, verbose=True,
                     visual_wrapper=lambda x: x,
                     textual_wrapper=Textual,
                     visual_export_params=DEFAULT_EXPORT,
                     textual_export_params=DEFAULT_EXPORT):
        isinstance_visual_input = isinstance(visual_input, (torch.Tensor))
        isinstance_textual_input = isinstance(textual_input, (torch.Tensor))

        if (not isinstance_visual_input) and (not isinstance_textual_input):
            raise Exception("[CLIP ONNX] Please, choose a dummy input")
        elif not isinstance_visual_input:
            print("[CLIP ONNX] Convert only textual model")
        elif not isinstance_textual_input:
            print("[CLIP ONNX] Convert only visual model")

        if isinstance_visual_input:
            self.visual_flag = True
            if verbose:
                print("[CLIP ONNX] Start convert visual model")
            self.convert_visual(visual_input, visual_wrapper, visual_export_params)
            if verbose:
                print("[CLIP ONNX] Start check visual model")
            self.onnx_checker(self.visual_path)

        if isinstance_textual_input:
            self.textual_flag = True
            if verbose:
                print("[CLIP ONNX] Start convert textual model")
            self.convert_textual(textual_input, textual_wrapper, textual_export_params)
            if verbose:
                print("[CLIP ONNX] Start check textual model")
            self.onnx_checker(self.textual_path)

        if verbose:
            print("[CLIP ONNX] Models converts successfully")

class clip_onnx(clip_converter):
    def __init__(self, model=None,
                 visual_path: str = "clip_visual.onnx",
                 textual_path: str = "clip_textual.onnx"):
        if not isinstance(model, (type(None))):
            super().__init__(model, visual_path, textual_path)
        else:
            print("[CLIP ONNX] Load mode")

    def load_onnx(self, visual_path=None, textual_path=None, logit_scale=None):
        if visual_path and textual_path:
            if not logit_scale:
                raise Exception("For this mode logit_scale must be specified. Example: model.logit_scale.exp()")
            self.logit_scale = logit_scale
        if visual_path:
            self.visual_path = visual_path
            self.visual_flag = True
        if textual_path:
            self.textual_path = textual_path
            self.textual_flag = True

    def start_sessions(self, providers=['TensorrtExecutionProvider',
                                        'CUDAExecutionProvider',
                                        'CPUExecutionProvider']):
        if self.visual_flag:
            self.visual_session = onnxruntime.InferenceSession(self.visual_path,
                                                               providers=providers)
        if self.textual_flag:
            self.textual_session = onnxruntime.InferenceSession(self.textual_path,
                                                                providers=providers)

    def visual_run(self, onnx_image):
        onnx_input_image = {self.visual_session.get_inputs()[0].name: onnx_image}
        visual_output, = self.visual_session.run(None, onnx_input_image)
        return visual_output

    def textual_run(self, onnx_text):
        onnx_input_text = {self.textual_session.get_inputs()[0].name: onnx_text}
        textual_output, = self.textual_session.run(None, onnx_input_text)
        return textual_output

    def __call__(self, image, text, device: str = "cpu"):
        assert self.visual_flag and self.textual_flag
        image_features = torch.from_numpy(self.visual_run(image)).to(device)
        text_features = torch.from_numpy(self.textual_run(text)).to(device)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def encode_image(self, image):
        return self.visual_run(image)

    def encode_text(self, text):
        return self.textual_run(text)

providers = ["CUDAExecutionProvider"]  # Если GPU доступен
if not ort.get_available_providers():
    providers = ["CPUExecutionProvider"] # Если GPU недоступен, используем CPU

# Для корректной работы onnx
def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]

    # Преобразование к тензорам и заполнение
    images = torch.stack(images, dim=0)
    texts = pad_sequence(texts, batch_first=True, padding_value=1)

    return images, texts

class ImageTitleDataset(Dataset):
    def __init__(self, dataframe, processor):
        """
        Класс Dataset для загрузки изображений и соответствующих текстовых описаний из DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame, содержащий пути к изображениям и текстовые описания.
            processor: Обработчик текста и изображений (например, токенизатор ruCLIP).
        """
        self.processor = processor
        self.image_paths = dataframe['image_path'].tolist()
        self.titles = dataframe['title'].tolist()

    def __len__(self):
        """Возвращает общее количество пар изображение-текст."""
        return len(self.titles)

    def __getitem__(self, idx):
        """
        Получает пару изображение-текст по индексу.

        Args:
            idx (int): Индекс пары.

        Returns:
            tuple: Кортеж, содержащий тензор изображения и тензор текста.
        """
        image = Image.open(self.image_paths[idx])
        text = self.titles[idx]
        data_dict = self.processor(text=[text], images=[image], return_tensors='pt', padding=True)
        
        # Получаем тензор для изображений
        image_tensor = data_dict['pixel_values'][0]
        # Тензор с IDs для текста
        text_tensor = data_dict['input_ids'][0]
        return image_tensor, text_tensor 

class EmbeddingGenerator:
    def __init__(self, visual_onnx_path, textual_onnx_path, logit_scale=np.exp(1.0)):
        self.clip_model = clip_onnx()  
        self.clip_model.load_onnx(
            visual_path=visual_onnx_path,
            textual_path=textual_onnx_path,
            logit_scale=logit_scale,
        )
        self.clip_model.start_sessions(providers=providers)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, self.processor = ruclip.load("ruclip-vit-base-patch32-384", device=self.device )  # Загружаем модель на GPU

    def get_embeddings(self, dataframe, batch_size=32):
        image_embeddings = []
        text_embeddings = []

        dataset = ImageTitleDataset(dataframe, self.processor)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
        )

        with tqdm(total=len(dataloader), desc="Processing batches", leave=True) as pbar:
            for images, texts in dataloader:
                # Перенос данных на GPU
                images = images 
                texts = texts

                # 
                images_onnx = images.numpy().astype(np.float32)  
                texts_onnx = texts.numpy()[::-1].astype(np.int64) 

                # Получение эмбеддингов
                image_embedding = self.clip_model.encode_image(images_onnx)
                text_embedding = self.clip_model.encode_text(texts_onnx)

                image_embeddings.append(image_embedding)
                text_embeddings.append(text_embedding)

                pbar.update(1)

        return np.concatenate(image_embeddings), np.concatenate(text_embeddings)

def generate_embeddings(visual_onnx_path, textual_onnx_path, processed_data_csv, output_path, batch_size=10):
    """
    Generate embeddings for images and texts.

    Args:
        visual_onnx_path (str): Path to the visual ONNX model.
        textual_onnx_path (str): Path to the textual ONNX model.
        processed_data_csv (str): Path to the processed data CSV file.
        output_path (str): Path to save generated embeddings.
        batch_size (int): Batch size for processing.
    """
    df = pd.read_csv(processed_data_csv)

    pipeline = EmbeddingGenerator(visual_onnx_path, textual_onnx_path, quiet=True)
    img_embeddings, text_embeddings = pipeline.get_embeddings(df, batch_size=batch_size)

    np.savez(output_path, text=text_embeddings, img=img_embeddings)

    print(f"Embeddings saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate embeddings for images and texts')
    parser.add_argument('--visual_onnx_path', type=str, required=True, help='Path to the visual ONNX model')
    parser.add_argument('--textual_onnx_path', type=str, required=True, help='Path to the textual ONNX model')
    parser.add_argument('--processed_data_csv', type=str, required=True, help='Path to the processed data CSV file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save generated embeddings (npz file)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')

    args = parser.parse_args()

    generate_embeddings(
        visual_onnx_path=args.visual_onnx_path,
        textual_onnx_path=args.textual_onnx_path,
        processed_data_csv=args.processed_data_csv,
        output_path=args.output_path,
        batch_size=args.batch_size
    )