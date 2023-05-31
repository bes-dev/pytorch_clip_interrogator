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
import PIL
from PIL import Image
from .blip import BLIP
from .clip_interrogator import CLIPInterrogator


class PromptEngineer:
    """ Find prompt that best match to input image.

    Args:
        blip_model (str): BLIP model name.
        clip_model (str): CLIP model name.
        device (str): target device.
        torch_dtype (torch.dtype): target type.
    """
    def __init__(self, blip, clip_interrogator):
        self.blip = blip
        self.clip_interrogator = clip_interrogator

    def __call__(
            self,
            images: Union[PIL.Image.Image, List[PIL.Image.Image]],
            max_flavors: int = 3,
            interrogate: bool = True
    ):
        """ Find prompt that best match to input image.

        Args:
            image (PIL.Image.Image): input image.
            max_flavors (int): max flavors.
            interrogate (bool): use clip interrogate estimator.
        Returns:
            Caption of the image.
        """
        caption = self.blip(images)
        if interrogate:
            caption = self.clip_interrogator(
                caption=caption,
                images=images,
                max_flavors=max_flavors
            )
        return caption

    @classmethod
    def load_model(
            cls,
            blip_model: str = "Salesforce/blip-image-captioning-large",
            clip_model: str = "openai/clip-vit-base-patch32",
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32
    ):
        """ Load pretrained prompt engineer.

        Args:
            path (str): path.
        """
        blip = BLIP.load_model(
            blip_model=blip_model,
            device=device,
            torch_dtype=torch_dtype
        )
        clip_interrogator = CLIPInterrogator.load_model(
            clip_model=clip_model,
            device=device,
            torch_dtype=torch_dtype
        )
        return cls(blip, clip_interrogator)

    def save_pretrained(self, path: str) -> None:
        """ Save pretrained prompt engineer to disk.

        Args:
            path (str): path.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.blip.save_pretrained(os.path.join(path, "blip"))
        self.clip_interrogator.save_pretrained(os.path.join(path, "clip_interrogator"))

    @classmethod
    def from_pretrained(
            cls,
            path: str,
            torch_dtype: torch.dtype = torch.float32,
            device: str = "cpu"
    ):
        """ Load pretrained prompt engineer from disk.

        Args:
            path (str): path.
        """
        blip = BLIP.from_pretrained(
            os.path.join(path, "blip"),
            torch_dtype=torch_dtype,
            device=device
        )
        clip_interrogator = CLIPInterrogator.from_pretrained(
            os.path.join(path, "clip_interrogator"),
            torch_dtype=torch_dtype,
            device=device
        )
        return cls(blip, clip_interrogator)
