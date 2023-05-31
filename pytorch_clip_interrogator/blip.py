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
from typing import Union, List, Optional, Dict
import torch
# BLIP Model
from transformers import BlipProcessor, BlipForConditionalGeneration
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
except:
    print("BLIP2 model is not available. \nPlease, try to install latest version of transformers library.")
# utils
import PIL
from PIL import Image
import json


class BLIP:
    """ BLIP Image-To-Text model

    Args:
        blip_model (str): BLIP model name.
        device (str): target device.
        torch_dtype (torch.dtype): target type.
    """
    def __init__(
            self,
            blip,
            blip_processor,
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        # params
        self.device = device
        self.torch_dtype = torch_dtype
        # model
        self.blip = blip
        self.blip_processor = blip_processor

    @torch.inference_mode()
    def __call__(
            self,
            image: Union[PIL.Image.Image, List[PIL.Image.Image]],
    ) -> str:
        """ Generate caption using BLIP model.

        Args:
            image (PIL.Image.Image): input image.
        Returns:
            Caption of the image.
        """
        pixel_values = self._prepare_inputs(image)
        preds = self.blip.generate(pixel_values=pixel_values)
        captions = []
        for i in range(preds.size(0)):
            caption = self.blip_processor.decode(preds[i], skip_special_tokens=True)
            captions.append(caption.replace("\n", ""))
        return captions

    def _prepare_inputs(
            self,
            images: Union[PIL.Image.Image, List[PIL.Image.Image]],
    ) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]
        pixel_values = None
        for image in images:
            _pixel_values = self.blip_processor(image, return_tensors="pt")["pixel_values"]
            if pixel_values is not None:
                pixel_values = torch.cat([pixel_values, _pixel_values], dim=0)
            else:
                pixel_values = _pixel_values
        return pixel_values.to(device=self.device, dtype=self.torch_dtype)

    @classmethod
    def load_model(
            cls,
            blip_model: str = "Salesforce/blip-image-captioning-large",
            device: str = "cpu",
            torch_dtype: torch.dtype = torch.float32
    ):
        """ Load pretrained BLIP model.

        Args:
            blip_model (str): BLIP model name.
            device (str): target device.
            torch_dtype (torch.dtype): target type.
        """
        print(f"load BLIP model: {blip_model}...")
        if "blip2" in blip_model:
            blip = Blip2ForConditionalGeneration.from_pretrained(blip_model, torch_dtype=torch_dtype).to(device)
            blip_processor = Blip2Processor.from_pretrained(blip_model)
        else:
            blip = BlipForConditionalGeneration.from_pretrained(blip_model, torch_dtype=torch_dtype).to(device)
            blip_processor = BlipProcessor.from_pretrained(blip_model)
        blip.eval()
        return cls(blip, blip_processor, device, torch_dtype)

    def save_pretrained(self, path: str) -> None:
        """ Save pretrained interrogator to disk.

        Args:
            path (str): path.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.blip.save_pretrained(os.path.join(path, "blip"))
        self.blip_processor.save_pretrained(os.path.join(path, "blip_processor"))

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
        with open(os.path.join(path, "blip", "config.json")) as f:
            cfg = json.load(f)
            if "blip2" in cfg["architectures"][0].lower():
                blip = Blip2ForConditionalGeneration.from_pretrained(os.path.join(path, "blip"), torch_dtype=torch_dtype).to(device)
                blip_processor = Blip2Processor.from_pretrained(os.path.join(path, "blip_processor"))
            else:
                blip = BlipForConditionalGeneration.from_pretrained(os.path.join(path, "blip"), torch_dtype=torch_dtype).to(device)
                blip_processor = BlipProcessor.from_pretrained(os.path.join(path, "blip_processor"))
            blip.eval()
        return cls(blip, blip_processor, device, torch_dtype)
