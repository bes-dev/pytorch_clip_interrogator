"""
Copyright 2023 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from typing import Union, List
import torch
import torch.nn.functional as F
# CLIP Model
from transformers import CLIPProcessor, CLIPModel
# FAISS vocab
from .vocab import Vocab
# utils
import numpy as np
import PIL
from addict import Dict as addict
from .utils import *


def preprocess_vocabulary(
        clip: CLIPModel,
        clip_processor: CLIPProcessor,
        batch_size: int = 64,
        device: str = "cpu"
) -> addict:
    """ Build prompt vocabulary.

        Args:
            batch_size (int): batch size.
        Returns:
            addict vocabulary.
        """
    vocab = {}
    for name in ["artists", "flavors", "mediums", "movements", "sites"]:
        vocab[name] = Vocab.from_corpus(
            os.path.join(res_path("data"), f"{name}.txt"),
            clip,
            clip_processor,
            batch_size,
            device
        )
    return addict(vocab)


class CLIPInterrogator:
    """ CLIP Interrogator

    Args:
        clip_model (str): CLIP model name.
        device (str): target device.
        torch_dtype (torch.dtype): target type.
    """
    def __init__(
            self,
            clip: CLIPModel,
            clip_processor: CLIPProcessor,
            vocab: Vocab,
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32
    ):
        # params
        self.device = device
        self.torch_dtype = torch_dtype
        # models
        self.clip = clip
        self.clip_processor = clip_processor
        self.vocab = vocab

    def __call__(
            self,
            images: Union[PIL.Image.Image, List[PIL.Image.Image]],
            caption: Union[str, List[str]],
            max_flavors: int = 3
    ) -> List[str]:
        """ CLIP Interrogate.

        Args:
            images (PIL.Image.Image): input image.
            caption (str): initial image caption.
            max_flavors (int): max flavors (default: 3).
        Returns:
            Caption of the image.
        """
        if not isinstance(caption, list):
            caption = [caption]
        image_features = self._image_to_features(images)
        # interrogate
        medium = to_list(self.vocab.mediums(image_features, 1))
        artist = to_list(self.vocab.artists(image_features, 1))
        movement = to_list(self.vocab.movements(image_features, 1))
        flaves = to_list(self.vocab.flavors(image_features, max_flavors))

        output = []
        for i in range(len(caption)):
            if caption[i].startswith(medium[i]):
                prompt = f"{caption[i]}, {artist[i]}, {movement[i]}, {', '.join(flaves[i])}"
            else:
                prompt = f"{caption[i]}, {medium[i]} {artist[i]}, {movement[i]}, {', '.join(flaves[i])}"
            output.append(prompt)

        return output

    @torch.inference_mode()
    def _image_to_features(
            self,
            images: Union[PIL.Image.Image, List[PIL.Image.Image]]
    ) -> np.ndarray:
        """ Image to CLIP features.

        Args:
            images (Union[PIL.Image.Image, List[PIL.Image.Image]]): input images.
        Returns:
            CLIP embeddings.
        """
        if not isinstance(images, list):
            images = [images]
        inputs = self.clip_processor(images=images, return_tensors="pt")
        image_features = self.clip.get_image_features(**to_device(inputs, self.device, dtype=self.torch_dtype))
        image_features = F.normalize(image_features, p=2, dim=1).float()
        return image_features.cpu().numpy()

    def save_pretrained(self, path: str) -> None:
        """ Save pretrained interrogator to disk.

        Args:
            path (str): path.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        for key, val in self.vocab.items():
            val.save_pretrained(os.path.join(path, key))
        self.clip_processor.save_pretrained(os.path.join(path, "clip_processor"))
        self.clip.save_pretrained(os.path.join(path, "clip"))

    @classmethod
    def from_pretrained(
            cls,
            path: str,
            torch_dtype: torch.dtype = torch.float32,
            device: str = "cpu"
    ):
        """ Load pretrained interrogator from disk.

        Args:
            path (str): path.
        """
        vocab = {}
        for name in ["artists", "flavors", "mediums", "movements", "sites"]:
            vocab[name] = Vocab.from_pretrained(
                os.path.join(path, name)
            )
        vocab = addict(vocab)
        clip_processor = CLIPProcessor.from_pretrained(os.path.join(path, "clip_processor"))
        clip = CLIPModel.from_pretrained(os.path.join(path, "clip"), torch_dtype=torch_dtype).to(device)
        clip.eval()
        return cls(clip, clip_processor, vocab, device, torch_dtype)

    @classmethod
    def load_model(
            cls,
            clip_model: str = "openai/clip-vit-base-patch32",
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32,
            batch_size: int = 64
    ):
        """ Load pretrained interrogator from disk.

        Args:
            path (str): path.
        """
        clip = CLIPModel.from_pretrained(clip_model, torch_dtype=torch_dtype).to(device)
        clip.eval()
        clip_processor = CLIPProcessor.from_pretrained(clip_model)
        vocab = preprocess_vocabulary(clip, clip_processor, batch_size, device)
        return cls(clip, clip_processor, vocab, device, torch_dtype)
