from typing import Optional, Tuple, List
import os
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from args import args_define
from pathlib import Path
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as FT
import PIL
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

args = args_define.args

@torch.no_grad()
def extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 16, dress_type = None) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    if dress_type is None:
        dir_path = f'feature/{args.dataset}/{args.type}/'
        index_names_path = f'feature/{args.dataset}/index_names.npy'
        index_features_path = f'feature/{args.dataset}/{args.type}/index_features.pt'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        np.save(index_names_path, index_names)
        torch.save(index_features, index_features_path)
    else:
        dir_path_1 = f'feature/{args.dataset}/{args.type}/{dress_type}'
        dir_path_2 = f'feature/{args.dataset}/{dress_type}'
        if not os.path.exists(dir_path_1):
            os.makedirs(dir_path_1)
        if not os.path.exists(dir_path_2):
            os.makedirs(dir_path_2)
        index_names_path = f'feature/{args.dataset}/{dress_type}/index_names.npy'
        index_features_path = f'feature/{args.dataset}/{args.type}/{dress_type}/index_features.pt'
        np.save(index_names_path, index_names)
        torch.save(index_features, index_features_path)
    return index_features, index_names


def contrastive_loss(v1: torch.Tensor, v2: torch.Tensor, temperature: float) -> torch.Tensor:
    # Based on https://github.com/NVlabs/PALAVRA/blob/main/utils/nv.py
    v1 = F.normalize(v1, dim=1)
    v2 = F.normalize(v2, dim=1)

    numerator = torch.exp(torch.diag(torch.inner(v1, v2)) / temperature)
    numerator = torch.cat((numerator, numerator), 0)
    joint_vector = torch.cat((v1, v2), 0)
    pairs_product = torch.exp(torch.mm(joint_vector, joint_vector.t()) / temperature)
    denominator = torch.sum(pairs_product - pairs_product * torch.eye(joint_vector.shape[0]).to(device), 0)

    loss = -torch.mean(torch.log(numerator / denominator))

    return loss


class Phi(nn.Module):
    """
    Textual Inversion Phi network.
    Takes as input the visual features of an image and outputs the pseudo-work embedding.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()
def extract_pseudo_tokens_with_phi(clip_model: CLIP, phi: Phi, dataset: Dataset) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts pseudo tokens from a dataset using a CLIP model and a phi model
    """
    data_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=10, pin_memory=False,
                             collate_fn=collate_fn)
    predicted_tokens = []
    names_list = []
    print(f"Extracting tokens using phi model")
    for batch in tqdm(data_loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        image_features = clip_model.encode_image(images)

        if phi:
            batch_predicted_tokens = phi(image_features)
            predicted_tokens.append(batch_predicted_tokens.cpu())
        names_list.extend(names)

    if phi:
        predicted_tokens = torch.vstack(predicted_tokens)
        torch.save(predicted_tokens, f'feature/{args.dataset}/{args.type}/predicted_tokens.pt')
    np.save(f'feature/{args.dataset}/names_list.npy', names_list)
    return predicted_tokens, names_list


class CustomTensorDataset(Dataset):
    """
    Custom Tensor Dataset which yields image_features and image_names
    """

    def __init__(self, images: torch.Tensor, names: torch.Tensor):
        self.images = images
        self.names = names

    def __getitem__(self, index) -> dict:
        return {'image': self.images[index],
                'image_name': self.names[index]
                }

    def __len__(self):
        return len(self.images)


def get_templates():
    """
    Return a list of templates
    Same templates as in PALAVRA: https://arxiv.org/abs/2204.01694
    """
    return [
        "This is a photo of a {}",
        "This photo contains a {}",
        "A photo of a {}",
        "This is an illustration of a {}",
        "This illustration contains a {}",
        "An illustrations of a {}",
        "This is a sketch of a {}",
        "This sketch contains a {}",
        "A sketch of a {}",
        "This is a diagram of a {}",
        "This diagram contains a {}",
        "A diagram of a {}",
        "A {}",
        "We see a {}",
        "{}",
        "We see a {} in this photo",
        "We see a {} in this image",
        "We see a {} in this illustration",
        "We see a {} photo",
        "We see a {} image",
        "We see a {} illustration",
        "{} photo",
        "{} image",
        "{} illustration",
    ]

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

class TargetPad:
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FT.pad(image, padding, 0, 'constant')


def targetpad_transform(target_ratio: float, dim: int) -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")


def encode_with_pseudo_tokens(clip_model: CLIP, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    if args.is_image_tokens:
        _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
        cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
        first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

        if pseudo_tokens.shape[0] == x.shape[0]:
            if len(pseudo_tokens.shape) == 2:
                pseudo_tokens = pseudo_tokens.unsqueeze(1)
            x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
                x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
        else:
            first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
            rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
            x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
                x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x